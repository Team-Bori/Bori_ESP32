/*
 * tflm_runtime: generic TFLite Micro runtime for BTF1 model packages.
 *
 * Serial protocol v2 (docs/PROTOCOL_v2.md), a superset of v1:
 *   - every machine-readable line is one JSON object terminated by '\n';
 *     other lines (ESP_LOG, bootloader, TFLM messages) must be ignored by the host
 *   - 1-byte commands: m, i, b, p (v1), a (eval on the package samples), l (labels)
 *   - periodic metrics are measured by a task on the other core, so commands that do
 *     not run the model (m, l, p, frames) are answered while a slow model is measured
 *   - binary frames (start with 0xA5 0x5A) for test inputs sent by the server
 *     and for changing the baud rate
 */
#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

#include "esp_attr.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_timer.h"

#include "bori_common.h"
#include "runtime.h"
#include "serial_link.h"

#define PROTOCOL_VERSION 2
#define FIRMWARE_ID "tflm_runtime"
#define METRICS_INTERVAL_MS 5000

/* Benchmarks run for a time budget instead of a fixed count (models range from ~10 us to ~1 s). */
#define BENCH_BUDGET_US 2000000
#define BENCH_MAX_ITERATIONS 1000
#define METRICS_BUDGET_US 200000
#define METRICS_MAX_ITERATIONS 100
#define WARMUP_BUDGET_US 100000
#define WARMUP_MAX_ITERATIONS 10
/* Long loops sleep one tick this often so the idle task (task watchdog) can run. */
#define YIELD_EVERY_US 500000

/* Binary frames */
#define FRAME_SYNC0 0xA5
#define FRAME_SYNC1 0x5A
#define FRAME_PING 0x01
#define FRAME_SET_BAUD 0x02
#define FRAME_TEST_BEGIN 0x10
#define FRAME_TEST_SAMPLE 0x11
#define FRAME_TEST_END 0x12
#define FRAME_GAP_TIMEOUT_MS 1000
#define FRAME_SMALL_PAYLOAD 32
#define BAUD_CONFIRM_MS 3000
#define TEST_IDLE_TIMEOUT_MS 60000

/* Output values listed in inference / test_result messages (keeps lines < 1 KB). */
#define MAX_REPORTED_OUTPUTS 16
/* Periodic metrics run on the APP CPU; only the main task prints. */
#define METRICS_TASK_STACK 8192
#define METRICS_TASK_CORE 1
/* 'l' lists labels until the line would exceed this many characters. */
#define LABELS_MAX_LINE 1000
/* Longest label text copied into a test_result line. */
#define MAX_RESULT_LABEL_CHARS 64
/* info line limit (without "
") and the room kept for "},"memory":{...}}" after the labels. */
#define INFO_MAX_LINE 1023
#define INFO_TAIL_RESERVE 160

static const char *TAG = "TFLM_APP";

static bool g_periodic_enabled = true;

/*
 * The interpreter (input tensor, Invoke) is used by the main task (commands, tests) and by
 * the metrics task. Whoever runs the model holds this lock; printing stays in the main task.
 */
static SemaphoreHandle_t s_interp_lock;

static void interp_lock(void)
{
    xSemaphoreTake(s_interp_lock, portMAX_DELAY);
}

static void interp_unlock(void)
{
    xSemaphoreGive(s_interp_lock);
}

/* Crash guard: survives a panic/watchdog reset, not a power cycle. */
#define GUARD_LOADING 0x4C4F4144u /* "LOAD" */
static RTC_NOINIT_ATTR uint32_t s_load_guard;

/* ------------------------------------------------------------------ */
/* small helpers                                                       */

static void print_error(const char *code, const char *message)
{
    bori_print_error(TAG, code, message);
}

static void print_error_seq(const char *code, int seq, const char *fmt, ...)
{
    char msg[128];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    printf("{\"type\":\"error\",\"code\":\"%s\",\"seq\":%d,\"message\":", code, seq);
    bori_print_json_string(msg);
    printf("}\n");
    fflush(stdout);
    ESP_LOGE(TAG, "[error] %s (seq %d): %s", code, seq, msg);
}

static void print_float(float v)
{
    if (isnan(v) || isinf(v)) {
        printf("null");
    } else {
        printf("%.6g", (double)v);
    }
}

/* Prints a tensor descriptor object; returns the characters written. */
static int print_tensor(const char *name, const btf_tensor_desc_t *t)
{
    int n = printf("\"%s\":{\"shape\":[", name);
    for (int i = 0; i < t->ndim; ++i) {
        n += printf(i ? ",%" PRId32 : "%" PRId32, t->dims[i]);
    }
    n += printf("],\"dtype\":\"%s\",\"scale\":%.6g,\"zero_point\":%" PRId32 ",\"bytes\":%" PRIu32 "}",
                btf_dtype_name(t->dtype), (double)t->scale, t->zero_point, t->bytes);
    return n;
}

static void maybe_yield(int64_t *last_yield_us)
{
    const int64_t now = esp_timer_get_time();
    if (now - *last_yield_us >= YIELD_EVERY_US) {
        vTaskDelay(1);
        *last_yield_us = esp_timer_get_time();
    }
}

/* ------------------------------------------------------------------ */
/* prediction                                                          */

struct Prediction {
    int pred;          /* class index (classification / binary_score), -1 for regression */
    float score;       /* top-1 score (classification) or raw score (binary_score) */
    bool has_expected;
    bool correct;
    float abs_err;     /* regression: max |output - expected| */
};

static void evaluate_output(const uint8_t *expected_le, Prediction *p)
{
    const btf_package_t &pkg = g_rt.pkg;
    const uint32_t n = runtime_output_count();
    p->pred = -1;
    p->score = 0.0f;
    p->has_expected = expected_le != nullptr;
    p->correct = false;
    p->abs_err = 0.0f;

    switch (pkg.task) {
    case BTF_TASK_CLASSIFICATION: {
        int best = 0;
        float best_v = runtime_output_value(0);
        for (uint32_t i = 1; i < n; ++i) {
            const float v = runtime_output_value(i);
            if (v > best_v) {
                best_v = v;
                best = (int)i;
            }
        }
        p->pred = best;
        p->score = best_v;
        if (expected_le) {
            p->correct = (int)lroundf(btf_read_f32(expected_le)) == best;
        }
        break;
    }
    case BTF_TASK_BINARY_SCORE: {
        p->score = runtime_output_value(0);
        p->pred = p->score >= pkg.task_param ? 1 : 0;
        if (expected_le) {
            p->correct = (int)lroundf(btf_read_f32(expected_le)) == p->pred;
        }
        break;
    }
    default: { /* regression */
        if (expected_le) {
            float worst = 0.0f;
            for (uint32_t i = 0; i < n; ++i) {
                const float err = fabsf(runtime_output_value(i) - btf_read_f32(expected_le + 4 * i));
                if (err > worst) {
                    worst = err;
                }
            }
            p->abs_err = worst;
            p->correct = worst <= pkg.task_param;
        }
        break;
    }
    }
}

static void print_prediction_fields(const Prediction &p, const uint8_t *expected_le)
{
    const btf_package_t &pkg = g_rt.pkg;
    printf("\"task\":\"%s\"", btf_task_name(pkg.task));
    if (pkg.task != BTF_TASK_REGRESSION) {
        printf(",\"pred\":%d", p.pred);
        const char *label = btf_label(&pkg, (uint32_t)p.pred);
        if (label != nullptr) {
            printf(",\"label\":");
            bori_print_json_string(label);
        }
        printf(",\"score\":");
        print_float(p.score);
        if (expected_le) {
            printf(",\"expected\":%ld", lroundf(btf_read_f32(expected_le)));
        }
    } else if (expected_le) {
        printf(",\"expected\":[");
        for (uint32_t i = 0; i < pkg.label_dim && i < MAX_REPORTED_OUTPUTS; ++i) {
            if (i) putchar(',');
            print_float(btf_read_f32(expected_le + 4 * i));
        }
        printf("],\"abs_err\":");
        print_float(p.abs_err);
    }
    if (expected_le) {
        printf(",\"correct\":%s", p.correct ? "true" : "false");
    }
}

static void print_outputs(void)
{
    const uint32_t n = runtime_output_count();
    printf("\"output\":[");
    for (uint32_t i = 0; i < n && i < MAX_REPORTED_OUTPUTS; ++i) {
        if (i) putchar(',');
        print_float(runtime_output_value(i));
    }
    printf("],\"output_count\":%" PRIu32, n);
}

/* ------------------------------------------------------------------ */
/* latency statistics                                                  */

struct LatencyStats {
    uint32_t count;
    int64_t total_us;
    int64_t min_us;
    int64_t max_us;

    void reset() { count = 0; total_us = 0; min_us = INT64_MAX; max_us = 0; }
    void add(int64_t us)
    {
        ++count;
        total_us += us;
        if (us < min_us) min_us = us;
        if (us > max_us) max_us = us;
    }
    double avg() const { return count ? (double)total_us / (double)count : 0.0; }
    void print() const
    {
        printf("\"total_us\":%" PRId64 ",\"avg_us\":%.3f,\"min_us\":%" PRId64 ",\"max_us\":%" PRId64,
               total_us, avg(), count ? min_us : 0, max_us);
    }
};

/*
 * Runs the current input repeatedly for up to budget_us (at least once, at most
 * max_iterations). Only Invoke() time is counted. Returns false if an Invoke failed.
 */
static bool run_timed(int64_t budget_us, uint32_t max_iterations, LatencyStats *stats)
{
    stats->reset();
    int64_t last_yield = esp_timer_get_time();
    const int64_t start = last_yield;
    while (stats->count < max_iterations) {
        int64_t us;
        if (!runtime_invoke(&us)) {
            return false;
        }
        stats->add(us);
        if (esp_timer_get_time() - start >= budget_us) {
            break;
        }
        maybe_yield(&last_yield);
    }
    return true;
}

static bool warm_up(void)
{
    LatencyStats s;
    return run_timed(WARMUP_BUDGET_US, WARMUP_MAX_ITERATIONS, &s);
}

/* ------------------------------------------------------------------ */
/* boot / info                                                         */

static void print_load_error(void)
{
    if (g_rt.error_code != nullptr) {
        print_error(g_rt.error_code, g_rt.error_message);
    }
}

/* Length of s once printed by bori_print_json_string (quotes included). */
static size_t json_string_len(const char *s)
{
    size_t n = 2;
    for (const unsigned char *p = (const unsigned char *)s; *p; ++p) {
        if (*p == '"' || *p == '\\' || *p == '\n' || *p == '\r' || *p == '\t') {
            n += 2;
        } else if (*p < 0x20) {
            n += 6;
        } else {
            n += 1;
        }
    }
    return n;
}

/*
 * info must stay within INFO_MAX_LINE characters. Everything except the label list
 * has a bounded size, so labels go last in "model" and are dropped (label_count and
 * labels_truncated remain) when they would not fit.
 */
static void print_info(void)
{
    const btf_package_t &pkg = g_rt.pkg;
    int n = 0;

    n += printf("{\"type\":\"info\",\"proto\":%d,\"firmware_id\":\"%s\",", PROTOCOL_VERSION, FIRMWARE_ID);
    n += bori_print_board_fields();
    n += printf(",\"baud\":%" PRIu32 ",\"registered_ops\":%d,", link_baud(), runtime_registered_ops());
    if (g_rt.loaded) {
        n += printf("\"supported_models\":[\"%s\"],", pkg.model_id);
    } else {
        n += printf("\"supported_models\":[],");
    }

    n += printf("\"model\":{\"runtime\":\"tflm\",\"loaded\":%s", g_rt.loaded ? "true" : "false");
    if (g_rt.loaded) {
        n += printf(",\"id\":\"%s\",\"task\":\"%s\",", pkg.model_id, btf_task_name(pkg.task));
        n += print_tensor("input", &pkg.input);
        n += printf(",");
        n += print_tensor("output", &pkg.output);
        n += printf(",\"arena_bytes\":%" PRIu32 ",\"arena_alloc_bytes\":%u,\"arena_used_bytes\":%u,"
                    "\"eval_samples\":%" PRIu32 ",\"ops\":%" PRIu32 ",\"size_bytes\":%" PRIu32 ","
                    "\"checksum\":\"0x%08" PRIx32 "\"",
                    pkg.arena_bytes, (unsigned)g_rt.arena_alloc_bytes, (unsigned)g_rt.arena_used_bytes,
                    pkg.eval_count, g_rt.model_ops, pkg.total_size, pkg.crc32);
    } else if (g_rt.error_code != nullptr) {
        n += printf(",\"load_error\":\"%s\"", g_rt.error_code);
    }
    n += printf(",\"max_arena_bytes\":%u", (unsigned)g_rt.boot_largest_free);
    if (g_rt.partition_size) {
        n += printf(",\"partition_offset\":%" PRIu32 ",\"partition_bytes\":%" PRIu32,
                    g_rt.partition_offset, g_rt.partition_size);
    }
    if (g_rt.loaded) {
        n += printf(",\"label_count\":%" PRIu32, pkg.label_count);
        if (pkg.labels != nullptr) {
            size_t labels_len = 12; /* ,"labels":[] */
            for (uint32_t i = 0; i < pkg.label_count; ++i) {
                labels_len += json_string_len(btf_label(&pkg, i)) + (i ? 1 : 0);
            }
            if ((size_t)n + labels_len + INFO_TAIL_RESERVE <= INFO_MAX_LINE) {
                printf(",\"labels\":[");
                for (uint32_t i = 0; i < pkg.label_count; ++i) {
                    if (i) putchar(',');
                    bori_print_json_string(btf_label(&pkg, i));
                }
                printf("]");
            } else {
                printf(",\"labels_truncated\":true");
            }
        }
    }
    printf("},");
    bori_print_memory_fields();
    printf("}\n");
    fflush(stdout);

    bori_log_board(TAG);
    if (g_rt.loaded) {
        ESP_LOGI(TAG, "  model  : %s (%s) loaded, checksum 0x%08" PRIx32 ", arena %u used / %u allocated, %" PRIu32 " eval samples",
                 pkg.model_id, btf_task_name(pkg.task), pkg.crc32,
                 (unsigned)g_rt.arena_used_bytes, (unsigned)g_rt.arena_alloc_bytes, pkg.eval_count);
    } else {
        ESP_LOGW(TAG, "  model  : NOT loaded (%s: %s)", g_rt.error_code ? g_rt.error_code : "-",
                 g_rt.error_message);
    }
    ESP_LOGI(TAG, "  largest free block before the arena: %u bytes, link %" PRIu32 " baud",
             (unsigned)g_rt.boot_largest_free, link_baud());
    bori_log_memory(TAG);
}

/* ------------------------------------------------------------------ */
/* commands                                                            */

static bool require_model(void)
{
    if (!g_rt.loaded) {
        char msg[200];
        snprintf(msg, sizeof(msg), "no model loaded (%s)",
                 g_rt.error_code ? g_rt.error_code : "no package");
        print_error("model_not_loaded", msg);
        return false;
    }
    return true;
}

static void run_inference(void)
{
    if (!require_model()) {
        return;
    }
    const btf_package_t &pkg = g_rt.pkg;
    runtime_set_demo_input();
    int64_t us;
    if (!runtime_invoke(&us)) {
        print_error("invoke_failed", "Invoke() failed on the demo input");
        return;
    }
    const uint8_t *expected = pkg.demo_input ? pkg.demo_expected : nullptr;
    Prediction p;
    evaluate_output(expected, &p);

    printf("{\"type\":\"inference\",\"input\":\"%s\",", pkg.demo_input ? "demo" : "zeros");
    print_prediction_fields(p, expected);
    printf(",\"latency_us\":%" PRId64 ",", us);
    print_outputs();
    putchar(',');
    bori_print_memory_fields();
    printf("}\n");
    fflush(stdout);

    if (pkg.task == BTF_TASK_REGRESSION) {
        ESP_LOGI(TAG, "[inference] output[0] %.4f in %" PRId64 " us", (double)runtime_output_value(0), us);
    } else {
        const char *label = btf_label(&pkg, (uint32_t)p.pred);
        ESP_LOGI(TAG, "[inference] predicted %d%s%s%s (score %.3f)%s in %" PRId64 " us",
                 p.pred, label ? " (" : "", label ? label : "", label ? ")" : "", (double)p.score,
                 expected ? (p.correct ? ", correct" : ", WRONG") : "", us);
    }
    bori_log_memory(TAG);
}

static void run_benchmark(void)
{
    if (!require_model()) {
        return;
    }
    runtime_set_demo_input();
    const size_t heap_before = esp_get_free_heap_size();
    LatencyStats s;
    if (!warm_up() || !run_timed(BENCH_BUDGET_US, BENCH_MAX_ITERATIONS, &s)) {
        print_error("invoke_failed", "Invoke() failed during the benchmark");
        return;
    }
    const size_t heap_after = esp_get_free_heap_size();
    const double avg = s.avg();

    printf("{\"type\":\"bench\",\"iterations\":%" PRIu32 ",", s.count);
    s.print();
    printf(",\"fps\":%.3f,\"budget_ms\":%d,\"heap_before\":%u,\"heap_after\":%u,",
           avg > 0 ? 1000000.0 / avg : 0.0, BENCH_BUDGET_US / 1000,
           (unsigned)heap_before, (unsigned)heap_after);
    bori_print_memory_fields();
    printf("}\n");
    fflush(stdout);

    ESP_LOGI(TAG, "[bench] %" PRIu32 " runs: avg %.1f us (min %" PRId64 ", max %" PRId64 "), %.1f FPS",
             s.count, avg, s.min_us, s.max_us, avg > 0 ? 1000000.0 / avg : 0.0);
    bori_log_memory(TAG);
}

/* Accuracy bookkeeping shared by 'a' (package samples) and streamed tests. */
struct EvalAccumulator {
    uint32_t samples;
    uint32_t labeled;
    uint32_t correct;
    uint32_t invoke_errors;
    double abs_err_sum;
    LatencyStats latency;

    void reset()
    {
        samples = labeled = correct = invoke_errors = 0;
        abs_err_sum = 0.0;
        latency.reset();
    }

    void print_fields() const
    {
        printf("\"samples\":%" PRIu32 ",\"labeled\":%" PRIu32 ",\"correct\":%" PRIu32 ",\"accuracy\":",
               samples, labeled, correct);
        if (labeled) {
            printf("%.4f", (double)correct / (double)labeled);
        } else {
            printf("null");
        }
        if (g_rt.pkg.task == BTF_TASK_REGRESSION && labeled) {
            printf(",\"mean_abs_err\":%.6g", abs_err_sum / (double)labeled);
        }
        printf(",\"invoke_errors\":%" PRIu32 ",\"error_rate\":%.4f,", invoke_errors,
               samples ? (double)invoke_errors / (double)samples : 0.0);
        latency.print();
    }
};

static void run_eval(void)
{
    if (!require_model()) {
        return;
    }
    const btf_package_t &pkg = g_rt.pkg;
    if (pkg.eval_count == 0) {
        print_error("no_eval_samples", "the package has no evaluation samples");
        return;
    }
    EvalAccumulator acc;
    acc.reset();
    uint8_t *in = runtime_input();
    const uint32_t expected_stride = (uint32_t)pkg.label_dim * 4u;
    int64_t last_yield = esp_timer_get_time();
    const int64_t wall_start = last_yield;

    for (uint32_t i = 0; i < pkg.eval_count; ++i) {
        memcpy(in, pkg.eval_inputs + (size_t)i * pkg.input.bytes, pkg.input.bytes);
        ++acc.samples;
        int64_t us;
        if (!runtime_invoke(&us)) {
            ++acc.invoke_errors;
            continue;
        }
        acc.latency.add(us);
        const uint8_t *expected = pkg.eval_expected + (size_t)i * expected_stride;
        Prediction p;
        evaluate_output(expected, &p);
        ++acc.labeled;
        acc.correct += p.correct ? 1 : 0;
        acc.abs_err_sum += p.abs_err;
        maybe_yield(&last_yield);
    }
    const int64_t wall_ms = (esp_timer_get_time() - wall_start) / 1000;

    printf("{\"type\":\"eval\",\"source\":\"package\",\"model\":\"%s\",\"task\":\"%s\",",
           pkg.model_id, btf_task_name(pkg.task));
    acc.print_fields();
    printf(",\"wall_ms\":%" PRId64 ",", wall_ms);
    bori_print_memory_fields();
    printf("}\n");
    fflush(stdout);

    ESP_LOGI(TAG, "[eval] %" PRIu32 "/%" PRIu32 " correct (%.2f%%), avg %.1f us, %" PRIu32 " invoke errors, %" PRId64 " ms",
             acc.correct, acc.labeled, acc.labeled ? 100.0 * acc.correct / acc.labeled : 0.0,
             acc.latency.avg(), acc.invoke_errors, wall_ms);
    bori_log_memory(TAG);
}

enum { METRICS_IDLE = 0, METRICS_RUNNING = 1, METRICS_DONE = 2 };

static struct {
    TaskHandle_t task;
    volatile int state;
    LatencyStats stats;
    bool ok;
} s_metrics;

/* Measures the demo input for the metrics time budget whenever the main loop asks. */
static void metrics_task(void *arg)
{
    (void)arg;
    while (true) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        interp_lock();
        runtime_set_demo_input();
        s_metrics.ok = run_timed(METRICS_BUDGET_US, METRICS_MAX_ITERATIONS, &s_metrics.stats);
        interp_unlock();
        __atomic_store_n(&s_metrics.state, METRICS_DONE, __ATOMIC_RELEASE);
    }
}

/* Prints a metrics line; `measured` false when there is no model to measure. */
static void print_metrics(bool measured)
{
    static uint32_t seq = 0;
    ++seq;
    const int64_t uptime_ms = esp_timer_get_time() / 1000;
    const LatencyStats &s = s_metrics.stats;
    const bool ok = measured && s_metrics.ok;

    printf("{\"type\":\"metrics\",\"seq\":%" PRIu32 ",\"uptime_ms\":%" PRId64 ",\"model_loaded\":%s,",
           seq, uptime_ms, g_rt.loaded ? "true" : "false");
    if (g_rt.loaded) {
        if (ok) {
            const double avg = s.avg();
            printf("\"iterations\":%" PRIu32 ",\"avg_us\":%.3f,\"fps\":%.3f,\"min_us\":%" PRId64 ",\"max_us\":%" PRId64 ",",
                   s.count, avg, avg > 0 ? 1000000.0 / avg : 0.0, s.min_us, s.max_us);
        }
        printf("\"invoke_errors\":%" PRIu32 ",\"arena_used_bytes\":%u,",
               g_rt.invoke_errors, (unsigned)g_rt.arena_used_bytes);
    }
    bori_print_memory_fields();
    printf("}\n");
    fflush(stdout);

    if (!g_rt.loaded) {
        ESP_LOGW(TAG, "[metrics #%" PRIu32 "] uptime %" PRId64 " s, model NOT loaded", seq, uptime_ms / 1000);
    } else if (!ok) {
        ESP_LOGW(TAG, "[metrics #%" PRIu32 "] Invoke() failed", seq);
    } else {
        ESP_LOGI(TAG, "[metrics #%" PRIu32 "] uptime %" PRId64 " s, avg %.1f us (%" PRIu32 " runs)",
                 seq, uptime_ms / 1000, s.avg(), s.count);
    }
}

static void print_labels(void)
{
    if (!require_model()) {
        return;
    }
    const btf_package_t &pkg = g_rt.pkg;
    /* {"type":"labels","count":N,"labels":[...],"shown":K,"truncated":B} */
    int n = printf("{\"type\":\"labels\",\"count\":%" PRIu32 ",\"labels\":[", pkg.label_count);
    uint32_t shown = 0;
    for (; shown < pkg.label_count; ++shown) {
        const char *label = btf_label(&pkg, shown);
        const size_t need = json_string_len(label) + (shown ? 1 : 0);
        if ((size_t)n + need + 40 > LABELS_MAX_LINE) {
            break;
        }
        if (shown) {
            putchar(',');
        }
        bori_print_json_string(label);
        n += (int)need;
    }
    printf("],\"shown\":%" PRIu32 ",\"truncated\":%s}\n", shown, shown < pkg.label_count ? "true" : "false");
    fflush(stdout);
}

static void set_periodic(bool enabled)
{
    g_periodic_enabled = enabled;
    printf("{\"type\":\"periodic\",\"enabled\":%s,\"interval_ms\":%d}\n",
           enabled ? "true" : "false", METRICS_INTERVAL_MS);
    fflush(stdout);
    ESP_LOGI(TAG, "[periodic] metrics report %s (every %d ms)", enabled ? "ON" : "OFF", METRICS_INTERVAL_MS);
}

/* ------------------------------------------------------------------ */
/* binary frames: baud rate and streamed tests                         */

static struct {
    bool active;
    bool has_labels;
    bool report_outputs;
    uint32_t expected_count;
    int last_seq;
    int64_t last_activity_us;
    EvalAccumulator acc;
    char last_result[768]; /* resent unchanged if the host repeats a sample frame */
} s_test;

static struct {
    bool pending;
    uint32_t previous;
    int64_t deadline_us;
} s_baud;

static uint32_t rd32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/* seq < 0: not a reply to a frame (session timed out or was replaced). */
static void print_test_summary(bool aborted, int seq)
{
    const btf_package_t &pkg = g_rt.pkg;
    printf("{\"type\":\"test_summary\",");
    if (seq >= 0) {
        printf("\"seq\":%d,", seq);
    }
    printf("\"source\":\"stream\",\"model\":\"%s\",\"task\":\"%s\",",
           pkg.model_id, btf_task_name(pkg.task));
    if (aborted) {
        printf("\"aborted\":true,");
    }
    if (s_test.expected_count) {
        printf("\"expected_samples\":%" PRIu32 ",", s_test.expected_count);
    }
    s_test.acc.print_fields();
    putchar(',');
    bori_print_memory_fields();
    printf("}\n");
    fflush(stdout);

    ESP_LOGI(TAG, "[test] %s: %" PRIu32 " samples, %" PRIu32 "/%" PRIu32 " correct, avg %.1f us, %" PRIu32 " invoke errors",
             aborted ? "aborted" : "done", s_test.acc.samples, s_test.acc.correct, s_test.acc.labeled,
             s_test.acc.latency.avg(), s_test.acc.invoke_errors);
}

static void end_test(bool aborted, int seq)
{
    print_test_summary(aborted, seq);
    s_test.active = false;
}

/* Appends printf-style text to s_test.last_result. */
static size_t s_result_len;
static void res_printf(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void res_printf(const char *fmt, ...)
{
    if (s_result_len >= sizeof(s_test.last_result)) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    const int n = vsnprintf(s_test.last_result + s_result_len, sizeof(s_test.last_result) - s_result_len, fmt, ap);
    va_end(ap);
    if (n > 0) {
        s_result_len += (size_t)n;
    }
}

static void res_float(float v)
{
    if (isnan(v) || isinf(v)) {
        res_printf("null");
    } else {
        res_printf("%.6g", (double)v);
    }
}

static void handle_test_sample(int seq, const uint8_t *expected_le, bool invoke_ok, int64_t us)
{
    const btf_package_t &pkg = g_rt.pkg;
    const uint32_t index = s_test.acc.samples;
    ++s_test.acc.samples;
    s_result_len = 0;

    res_printf("{\"type\":\"test_result\",\"seq\":%d,\"index\":%" PRIu32, seq, index);
    if (!invoke_ok) {
        ++s_test.acc.invoke_errors;
        res_printf(",\"ok\":false,\"error\":\"invoke_failed\"}");
    } else {
        s_test.acc.latency.add(us);
        Prediction p;
        evaluate_output(expected_le, &p);
        if (expected_le) {
            ++s_test.acc.labeled;
            s_test.acc.correct += p.correct ? 1 : 0;
            s_test.acc.abs_err_sum += p.abs_err;
        }
        res_printf(",\"ok\":true,\"latency_us\":%" PRId64, us);
        if (pkg.task != BTF_TASK_REGRESSION) {
            res_printf(",\"pred\":%d", p.pred);
            const char *label = btf_label(&pkg, (uint32_t)p.pred);
            if (label) {
                /* Packages can come from users: escape, and cap the length so the line stays valid JSON. */
                res_printf(",\"label\":\"");
                int shown = 0;
                for (const char *c = label; *c && shown < MAX_RESULT_LABEL_CHARS; ++c, ++shown) {
                    if (*c == '"' || *c == '\\') res_printf("\\%c", *c);
                    else if ((unsigned char)*c >= 0x20) res_printf("%c", *c);
                }
                res_printf("\"");
            }
            res_printf(",\"score\":");
            res_float(p.score);
            if (expected_le) {
                res_printf(",\"expected\":%ld", lroundf(btf_read_f32(expected_le)));
            }
        } else if (expected_le) {
            res_printf(",\"abs_err\":");
            res_float(p.abs_err);
        }
        if (expected_le) {
            res_printf(",\"correct\":%s", p.correct ? "true" : "false");
        }
        if (s_test.report_outputs || pkg.task == BTF_TASK_REGRESSION) {
            const uint32_t n = runtime_output_count();
            res_printf(",\"output\":[");
            for (uint32_t i = 0; i < n && i < MAX_REPORTED_OUTPUTS; ++i) {
                if (i) res_printf(",");
                res_float(runtime_output_value(i));
            }
            res_printf("]");
        }
        res_printf("}");
    }
    s_test.last_seq = seq;
    printf("%s\n", s_test.last_result);
    fflush(stdout);
}

/* Reads one frame after the first sync byte (0xA5) has been received. */
static void handle_frame(void)
{
    if (link_read_byte(100) != FRAME_SYNC1) {
        return; /* stray 0xA5: ignore, like other non-ASCII bytes */
    }
    uint8_t hdr[6];
    uint32_t crc = 0;
    if (!link_read_exact(hdr, sizeof(hdr), FRAME_GAP_TIMEOUT_MS, &crc)) {
        print_error_seq("frame_timeout", -1, "frame header incomplete");
        return;
    }
    const uint8_t type = hdr[0];
    const int seq = hdr[1];
    const uint32_t len = rd32(hdr + 2);

    uint8_t small[FRAME_SMALL_PAYLOAD];
    uint8_t labels[BTF_MAX_LABEL_DIM * 4];
    const btf_package_t &pkg = g_rt.pkg;
    const uint32_t label_bytes = s_test.has_labels ? (uint32_t)pkg.label_dim * 4u : 0u;
    const char *reject = nullptr; /* error code; payload is still drained to stay in sync */
    char reject_msg[96] = "";
    bool to_tensor = false;

    if (type == FRAME_TEST_SAMPLE) {
        if (!g_rt.loaded) {
            reject = "model_not_loaded";
        } else if (!s_test.active) {
            reject = "test_not_started";
        } else if (len != pkg.input.bytes + label_bytes) {
            reject = "input_size_mismatch";
            snprintf(reject_msg, sizeof(reject_msg), "payload %u bytes, expected %u (input %u + labels %u)",
                     (unsigned)len, (unsigned)(pkg.input.bytes + label_bytes),
                     (unsigned)pkg.input.bytes, (unsigned)label_bytes);
        } else {
            to_tensor = true;
        }
    } else if (len > FRAME_SMALL_PAYLOAD) {
        reject = "frame_too_large";
    }
    if (len > 4u * 1024u * 1024u) {
        /* Nonsense length (corrupted header): resynchronise instead of draining. */
        link_discard_until_idle(100, 3000);
        print_error_seq("frame_too_large", seq, "length %u", (unsigned)len);
        return;
    }

    bool ok;
    if (to_tensor) {
        ok = link_read_exact(runtime_input(), pkg.input.bytes, FRAME_GAP_TIMEOUT_MS, &crc) &&
             link_read_exact(labels, label_bytes, FRAME_GAP_TIMEOUT_MS, &crc);
    } else {
        ok = link_read_exact(reject ? nullptr : small, len, FRAME_GAP_TIMEOUT_MS, &crc);
    }
    uint8_t crc_bytes[4];
    if (!ok || !link_read_exact(crc_bytes, 4, FRAME_GAP_TIMEOUT_MS, nullptr)) {
        link_discard_until_idle(100, 3000);
        print_error_seq("frame_timeout", seq, "frame payload incomplete");
        return;
    }
    if (rd32(crc_bytes) != crc) {
        link_discard_until_idle(100, 3000);
        print_error_seq("crc_error", seq, "frame crc 0x%08x, computed 0x%08x",
                        (unsigned)rd32(crc_bytes), (unsigned)crc);
        return;
    }

    /* A valid frame proves the host talks at our current baud rate. */
    s_baud.pending = false;
    s_test.last_activity_us = esp_timer_get_time();

    if (reject != nullptr) {
        print_error_seq(reject, seq, "%s", reject_msg[0] ? reject_msg : reject);
        return;
    }

    switch (type) {
    case FRAME_PING:
        printf("{\"type\":\"pong\",\"seq\":%d,\"baud\":%" PRIu32 "}\n", seq, link_baud());
        fflush(stdout);
        break;

    case FRAME_SET_BAUD: {
        const uint32_t baud = len >= 4 ? rd32(small) : 0;
        if (!link_baud_supported(baud)) {
            print_error_seq("bad_baud", seq, "unsupported baud %u (115200, 230400, 460800, 921600)", (unsigned)baud);
            break;
        }
        const uint32_t previous = link_baud();
        printf("{\"type\":\"baud\",\"seq\":%d,\"baud\":%" PRIu32 ",\"previous\":%" PRIu32 ",\"confirm_ms\":%d}\n",
               seq, baud, previous, BAUD_CONFIRM_MS);
        link_set_baud(baud);
        if (baud != previous) {
            s_baud.pending = true;
            s_baud.previous = previous;
            s_baud.deadline_us = esp_timer_get_time() + (int64_t)BAUD_CONFIRM_MS * 1000;
        }
        break;
    }

    case FRAME_TEST_BEGIN: {
        if (!g_rt.loaded) {
            print_error_seq("model_not_loaded", seq, "no model loaded");
            break;
        }
        if (s_test.active) {
            end_test(true, -1);
        }
        /* A metrics measurement may still be running: let it finish before samples are
         * streamed straight into the input tensor (no new one starts during the session). */
        interp_lock();
        interp_unlock();
        s_test.expected_count = len >= 4 ? rd32(small) : 0;
        s_test.has_labels = len >= 5 && small[4] != 0;
        s_test.report_outputs = len >= 6 && small[5] != 0;
        s_test.active = true;
        s_test.last_seq = -1;
        s_test.acc.reset();
        printf("{\"type\":\"test_begin\",\"seq\":%d,\"model\":\"%s\",\"task\":\"%s\",\"input_bytes\":%" PRIu32 ","
               "\"label_dim\":%u,\"has_labels\":%s,\"max_payload\":%" PRIu32 "}\n",
               seq, pkg.model_id, btf_task_name(pkg.task), pkg.input.bytes, pkg.label_dim,
               s_test.has_labels ? "true" : "false",
               pkg.input.bytes + (s_test.has_labels ? (uint32_t)pkg.label_dim * 4u : 0u));
        fflush(stdout);
        ESP_LOGI(TAG, "[test] started: %" PRIu32 " samples expected, labels %s", s_test.expected_count,
                 s_test.has_labels ? "yes" : "no");
        break;
    }

    case FRAME_TEST_SAMPLE: {
        if (seq == s_test.last_seq) {
            /* The host did not get our answer and resent the same sample. */
            printf("%s\n", s_test.last_result);
            fflush(stdout);
            break;
        }
        int64_t us = 0;
        interp_lock();
        const bool invoke_ok = runtime_invoke(&us);
        interp_unlock();
        handle_test_sample(seq, s_test.has_labels ? labels : nullptr, invoke_ok, us);
        break;
    }

    case FRAME_TEST_END:
        if (!s_test.active) {
            print_error_seq("test_not_started", seq, "no test session");
            break;
        }
        end_test(false, seq);
        break;

    default:
        print_error_seq("unknown_frame", seq, "frame type 0x%02x", type);
        break;
    }
}

/* ------------------------------------------------------------------ */

static void load_model_with_guard(void)
{
    const esp_reset_reason_t reason = esp_reset_reason();
    const char *skip = nullptr;
    if (s_load_guard == GUARD_LOADING && bori_reset_reason_is_fault(reason)) {
        skip = "the previous boot crashed while loading or first running this package; "
               "write another package (the board stays usable)";
    }
    s_load_guard = skip ? 0 : GUARD_LOADING;

    runtime_load(skip);

    if (g_rt.loaded) {
        /* First run doubles as a self-test before the guard is cleared. */
        runtime_set_demo_input();
        int64_t us;
        if (!runtime_invoke(&us)) {
            ESP_LOGW(TAG, "self-test Invoke() failed");
        } else {
            ESP_LOGI(TAG, "self-test inference: %" PRId64 " us", us);
        }
    }
    s_load_guard = 0;
}

/* Byte read while skipping an escape sequence that turned out not to be one. */
static int s_pending_byte = -1;

/*
 * Terminal keys (arrows, Home, F-keys) send ESC [ ... or ESC O ...; drop the whole sequence
 * instead of answering unknown_command for '[' and 'A'. A lone ESC is ignored.
 */
static void skip_escape_sequence(void)
{
    const int next = link_read_byte(30);
    if (next != '[' && next != 'O') {
        s_pending_byte = next;  /* not a sequence: handle that byte normally (-1 = nothing) */
        return;
    }
    for (int i = 0; i < 8; ++i) {
        const int b = link_read_byte(30);
        if (b < 0 || (b >= 0x40 && b <= 0x7E)) {
            return;  /* final byte of the sequence */
        }
    }
}

extern "C" void app_main(void)
{
    link_init();
    s_interp_lock = xSemaphoreCreateMutex();
    bori_print_boot(TAG, PROTOCOL_VERSION, FIRMWARE_ID);

    load_model_with_guard();
    print_load_error();
    if (g_rt.loaded &&
        xTaskCreatePinnedToCore(metrics_task, "metrics", METRICS_TASK_STACK, nullptr, 1,
                                &s_metrics.task, METRICS_TASK_CORE) != pdPASS) {
        s_metrics.task = nullptr;
        ESP_LOGE(TAG, "metrics task not created; periodic metrics run without measurement");
    }

    printf("{\"type\":\"ready\",\"model_loaded\":%s,\"commands\":\"i,b,m,p,a,l\",\"frames\":true,"
           "\"periodic\":%s,\"interval_ms\":%d}\n",
           g_rt.loaded ? "true" : "false", g_periodic_enabled ? "true" : "false", METRICS_INTERVAL_MS);
    fflush(stdout);
    ESP_LOGI(TAG, "[ready] model %s. commands: i=inference, b=benchmark, m=info, p=toggle metrics, "
             "a=evaluate package samples, l=labels; binary frames for streamed tests",
             g_rt.loaded ? "loaded" : "NOT loaded");

    int64_t next_report_us = esp_timer_get_time() + (int64_t)METRICS_INTERVAL_MS * 1000;

    while (true) {
        const int64_t now = esp_timer_get_time();

        if (s_baud.pending && now >= s_baud.deadline_us) {
            s_baud.pending = false;
            link_set_baud(s_baud.previous);
            printf("{\"type\":\"baud\",\"baud\":%" PRIu32 ",\"reverted\":true}\n", s_baud.previous);
            fflush(stdout);
            ESP_LOGW(TAG, "[baud] no host traffic at the new rate; back to %" PRIu32, s_baud.previous);
        }
        if (s_test.active && now - s_test.last_activity_us > (int64_t)TEST_IDLE_TIMEOUT_MS * 1000) {
            end_test(true, -1);
        }
        if (__atomic_load_n(&s_metrics.state, __ATOMIC_ACQUIRE) == METRICS_DONE) {
            print_metrics(true);
            s_metrics.state = METRICS_IDLE;
            next_report_us = esp_timer_get_time() + (int64_t)METRICS_INTERVAL_MS * 1000;
        }
        if (g_periodic_enabled && !s_test.active && !s_baud.pending &&
            s_metrics.state == METRICS_IDLE && now >= next_report_us) {
            if (s_metrics.task != nullptr) {
                s_metrics.state = METRICS_RUNNING;  /* printed when the task is done */
                xTaskNotifyGive(s_metrics.task);
            } else {
                print_metrics(false);
                next_report_us = esp_timer_get_time() + (int64_t)METRICS_INTERVAL_MS * 1000;
            }
        }

        int ch;
        if (s_pending_byte >= 0) {
            ch = s_pending_byte;
            s_pending_byte = -1;
        } else {
            ch = link_read_byte(20);
        }
        if (ch < 0) {
            continue;
        }
        if (ch == 0x1B) {
            skip_escape_sequence();
            continue;
        }
        if (ch == FRAME_SYNC0) {
            handle_frame();
            continue;
        }

        bool known = true;
        switch (ch) {
        case 'i':
            interp_lock();
            run_inference();
            interp_unlock();
            break;
        case 'b':
            interp_lock();
            run_benchmark();
            interp_unlock();
            break;
        case 'm':
            print_info();
            break;
        case 'a':
            interp_lock();
            run_eval();
            interp_unlock();
            break;
        case 'l':
            print_labels();
            break;
        case 'p':
            set_periodic(!g_periodic_enabled);
            next_report_us = esp_timer_get_time() + (int64_t)METRICS_INTERVAL_MS * 1000;
            break;
        default: {
            known = false;
            // Ignore whitespace, control bytes and non-ASCII noise (same as v1).
            if (ch <= ' ' || ch >= 0x7F) {
                break;
            }
            char message[48];
            if (ch == '"' || ch == '\\') {
                snprintf(message, sizeof(message), "got 0x%02X; use i, b, m, p, a or l", ch);
            } else {
                snprintf(message, sizeof(message), "got '%c' (0x%02X); use i, b, m, p, a or l", ch, ch);
            }
            print_error("unknown_command", message);
            break;
        }
        }
        if (known) {
            s_baud.pending = false;
        }
    }
}
