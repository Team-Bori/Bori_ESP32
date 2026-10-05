#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_partition.h"
#include "esp_system.h"
#include "esp_timer.h"

#include "bori_common.h"
#include "mlp.h"
#include "model.h"

/*
 * Serial protocol (see PROTOCOL.md).
 * Every machine-readable line is a single JSON object terminated by '\n'.
 * Any other line (ESP_LOG output, bootloader messages) must be ignored by the host.
 *
 * Each JSON message is followed by a human-readable ESP_LOGI summary of the same
 * content for people watching the serial monitor. Lower CONFIG_LOG_DEFAULT_LEVEL
 * to WARN to silence them.
 */
#define PROTOCOL_VERSION 1
#define FIRMWARE_ID "mlp"
#define MODEL_ID "mlp_64_16_10_int8"
#define BENCH_ITERATIONS 1000
#define METRICS_INTERVAL_MS 5000
#define METRICS_ITERATIONS 100

static const char *TAG = "MLP_APP";

static mlp_model_t g_model;
static bool g_model_loaded = false;
static bool g_periodic_enabled = true;

static const int8_t DEMO_INPUT[MLP_INPUT_SIZE] = {
    0, 0, 64, 95, 111, 95, 24, 0,
    0, 0, 95, 40, 0, 24, 0, 0,
    0, 0, 127, 16, 32, 8, 0, 0,
    0, 32, 127, 111, 95, 119, 32, 0,
    0, 0, 32, 0, 0, 64, 64, 0,
    0, 8, 0, 0, 0, 87, 40, 0,
    0, 48, 111, 8, 16, 119, 8, 0,
    0, 0, 64, 111, 127, 32, 0, 0
};

static const int DEMO_LABEL = 5;

static void log_memory(void)
{
    bori_log_memory(TAG);
}

static void print_error(const char *code, const char *message)
{
    bori_print_error(TAG, code, message);
}

static void print_boot(void)
{
    bori_print_boot(TAG, PROTOCOL_VERSION, FIRMWARE_ID);
}

static void print_info(void)
{
    const esp_partition_t *part = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, BORI_MODEL_PARTITION_SUBTYPE, BORI_MODEL_PARTITION_LABEL);

    printf("{\"type\":\"info\",\"proto\":%d,\"firmware_id\":\"%s\",", PROTOCOL_VERSION, FIRMWARE_ID);
    bori_print_board_fields();
    printf(",");

    printf("\"supported_models\":[\"%s\"],", MODEL_ID);

    printf("\"model\":{\"id\":\"%s\",\"loaded\":%s,\"dims\":[%d,%d,%d],"
           "\"quantization\":\"int8\"",
           MODEL_ID, g_model_loaded ? "true" : "false",
           MLP_INPUT_SIZE, MLP_HIDDEN_SIZE, MLP_OUTPUT_SIZE);
    if (g_model_loaded) {
        const size_t payload =
            g_model.header.w1_bytes + g_model.header.b1_bytes +
            g_model.header.w2_bytes + g_model.header.b2_bytes;
        printf(",\"size_bytes\":%u,\"checksum\":\"0x%08" PRIx32 "\"",
               (unsigned)(sizeof(g_model.header) + payload), g_model.header.checksum);
    }
    if (part != NULL) {
        printf(",\"partition_offset\":%" PRIu32 ",\"partition_bytes\":%" PRIu32,
               part->address, part->size);
    }
    printf("},");

    bori_print_memory_fields();
    printf("}\n");
    fflush(stdout);

    bori_log_board(TAG);
    if (g_model_loaded) {
        ESP_LOGI(TAG, "  model  : %s (%d-%d-%d, int8) loaded, checksum 0x%08" PRIx32,
                 MODEL_ID, MLP_INPUT_SIZE, MLP_HIDDEN_SIZE, MLP_OUTPUT_SIZE,
                 g_model.header.checksum);
    } else {
        ESP_LOGW(TAG, "  model  : %s NOT loaded", MODEL_ID);
    }
    if (part != NULL) {
        ESP_LOGI(TAG, "  model partition: 0x%06" PRIx32 ", %" PRIu32 " bytes",
                 part->address, part->size);
    }
    log_memory();
}

static void run_inference(void)
{
    if (!g_model_loaded) {
        print_error("model_not_loaded", "no valid model in the model partition");
        return;
    }

    int32_t logits[MLP_OUTPUT_SIZE];
    const int64_t start = esp_timer_get_time();
    const int pred = mlp_predict(&g_model, DEMO_INPUT, logits);
    const int64_t elapsed_us = esp_timer_get_time() - start;

    printf("{\"type\":\"inference\",\"input\":\"demo\",\"pred\":%d,\"expected\":%d,"
           "\"correct\":%s,\"latency_us\":%" PRId64 ",\"logits\":[",
           pred, DEMO_LABEL, pred == DEMO_LABEL ? "true" : "false", elapsed_us);
    for (int i = 0; i < MLP_OUTPUT_SIZE; ++i) {
        printf(i == 0 ? "%" PRId32 : ",%" PRId32, logits[i]);
    }
    printf("],");
    bori_print_memory_fields();
    printf("}\n");
    fflush(stdout);

    ESP_LOGI(TAG, "[inference] predicted %d (expected %d, %s) in %" PRId64 " us",
             pred, DEMO_LABEL, pred == DEMO_LABEL ? "correct" : "WRONG", elapsed_us);
    log_memory();
}

/* Runs `iterations` timed inferences (after a short warm-up) and returns the total time. */
static int64_t measure_inference_us(int iterations)
{
    int32_t logits[MLP_OUTPUT_SIZE];
    volatile int sink = 0;

    // Warm up caches before timing.
    for (int i = 0; i < 10; ++i) {
        sink ^= mlp_predict(&g_model, DEMO_INPUT, logits);
    }

    const int64_t start = esp_timer_get_time();
    for (int i = 0; i < iterations; ++i) {
        sink ^= mlp_predict(&g_model, DEMO_INPUT, logits);
    }
    const int64_t total_us = esp_timer_get_time() - start;

    (void)sink;
    return total_us;
}

static void run_benchmark(void)
{
    if (!g_model_loaded) {
        print_error("model_not_loaded", "no valid model in the model partition");
        return;
    }

    const size_t heap_before = esp_get_free_heap_size();
    const int64_t total_us = measure_inference_us(BENCH_ITERATIONS);
    const size_t heap_after = esp_get_free_heap_size();
    const double avg_us = (double)total_us / (double)BENCH_ITERATIONS;

    printf("{\"type\":\"bench\",\"iterations\":%d,\"total_us\":%" PRId64 ","
           "\"avg_us\":%.3f,\"fps\":%.3f,\"heap_before\":%u,\"heap_after\":%u,",
           BENCH_ITERATIONS, total_us, avg_us, 1000000.0 / avg_us,
           (unsigned)heap_before, (unsigned)heap_after);
    bori_print_memory_fields();
    printf("}\n");
    fflush(stdout);

    ESP_LOGI(TAG, "[bench] %d runs in %" PRId64 " us -> avg %.3f us, %.1f FPS, heap change %d bytes",
             BENCH_ITERATIONS, total_us, avg_us, 1000000.0 / avg_us,
             (int)heap_after - (int)heap_before);
    log_memory();
}

/* Periodic performance report, emitted every METRICS_INTERVAL_MS while enabled. */
static void report_metrics(void)
{
    static uint32_t seq = 0;
    ++seq;

    const int64_t uptime_ms = esp_timer_get_time() / 1000;

    printf("{\"type\":\"metrics\",\"seq\":%" PRIu32 ",\"uptime_ms\":%" PRId64 ","
           "\"model_loaded\":%s,",
           seq, uptime_ms, g_model_loaded ? "true" : "false");

    if (g_model_loaded) {
        const int64_t total_us = measure_inference_us(METRICS_ITERATIONS);
        const double avg_us = (double)total_us / (double)METRICS_ITERATIONS;

        printf("\"iterations\":%d,\"avg_us\":%.3f,\"fps\":%.3f,",
               METRICS_ITERATIONS, avg_us, 1000000.0 / avg_us);
        bori_print_memory_fields();
        printf("}\n");
        fflush(stdout);

        ESP_LOGI(TAG, "[metrics #%" PRIu32 "] uptime %" PRId64 " s, avg %.3f us, %.1f FPS (%d runs)",
                 seq, uptime_ms / 1000, avg_us, 1000000.0 / avg_us, METRICS_ITERATIONS);
    } else {
        bori_print_memory_fields();
        printf("}\n");
        fflush(stdout);

        ESP_LOGW(TAG, "[metrics #%" PRIu32 "] uptime %" PRId64 " s, model NOT loaded",
                 seq, uptime_ms / 1000);
    }
    log_memory();
}

static void set_periodic(bool enabled)
{
    g_periodic_enabled = enabled;

    printf("{\"type\":\"periodic\",\"enabled\":%s,\"interval_ms\":%d}\n",
           enabled ? "true" : "false", METRICS_INTERVAL_MS);
    fflush(stdout);

    ESP_LOGI(TAG, "[periodic] metrics report %s (every %d ms)",
             enabled ? "ON" : "OFF", METRICS_INTERVAL_MS);
}

void app_main(void)
{
    print_boot();

    g_model_loaded = mlp_model_load(&g_model);
    if (!g_model_loaded) {
        // Stay responsive so the host can still query status via 'm'.
        print_error("model_load_failed",
                    "flash a valid model.bin to the model partition (0x200000)");
    }

    printf("{\"type\":\"ready\",\"model_loaded\":%s,\"commands\":\"i,b,m,p\","
           "\"periodic\":%s,\"interval_ms\":%d}\n",
           g_model_loaded ? "true" : "false",
           g_periodic_enabled ? "true" : "false", METRICS_INTERVAL_MS);
    fflush(stdout);

    ESP_LOGI(TAG, "[ready] model %s. commands: i=inference, b=benchmark, m=board/model info, "
             "p=toggle periodic metrics (every %d ms)",
             g_model_loaded ? "loaded" : "NOT loaded", METRICS_INTERVAL_MS);

    int64_t next_report_us = esp_timer_get_time() + (int64_t)METRICS_INTERVAL_MS * 1000;

    while (1) {
        if (g_periodic_enabled && esp_timer_get_time() >= next_report_us) {
            report_metrics();
            next_report_us = esp_timer_get_time() + (int64_t)METRICS_INTERVAL_MS * 1000;
        }

        const int ch = getchar();
        switch (ch) {
        case 'i':
            run_inference();
            break;
        case 'b':
            run_benchmark();
            break;
        case 'm':
            print_info();
            break;
        case 'p':
            set_periodic(!g_periodic_enabled);
            next_report_us = esp_timer_get_time() + (int64_t)METRICS_INTERVAL_MS * 1000;
            break;
        case EOF:
            clearerr(stdin);
            vTaskDelay(pdMS_TO_TICKS(20));
            break;
        default:
            // Ignore whitespace, control bytes (terminal escape sequences) and
            // non-ASCII noise such as 0x00/0xFF glitches when the port opens.
            if (ch <= ' ' || ch >= 0x7F) {
                break;
            }
            char message[48];
            if (ch == '"' || ch == '\\') {
                // Keep the JSON string valid: show only the byte value.
                snprintf(message, sizeof(message), "got 0x%02X; use i, b, m or p", ch);
            } else {
                snprintf(message, sizeof(message), "got '%c' (0x%02X); use i, b, m or p", ch, ch);
            }
            print_error("unknown_command", message);
            break;
        }
    }
}
