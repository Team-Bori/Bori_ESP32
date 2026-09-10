#include <stdio.h>
#include <inttypes.h>
#include <stdint.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_system.h"
#include "esp_heap_caps.h"

#include "mlp.h"
#include "model.h"

static const char *TAG = "MLP";
static mlp_model_t g_model;

// Fixed demo input from the original prototype. Input values are quantized 0..127.
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

static void print_logits(const int32_t logits[MLP_OUTPUT_SIZE])
{
    printf("logits:");
    for (int i = 0; i < MLP_OUTPUT_SIZE; ++i) {
        printf(" %" PRId32, logits[i]);
    }
    printf("\n");
}

static void run_once(void)
{
    int32_t logits[MLP_OUTPUT_SIZE];
    int64_t start = esp_timer_get_time();
    int pred = mlp_predict(&g_model, DEMO_INPUT, logits);
    int64_t elapsed = esp_timer_get_time() - start;

    printf("RESULT pred=%d expected=%d latency_us=%" PRId64 "\n",
           pred, DEMO_LABEL, elapsed);
    print_logits(logits);
}

static void run_benchmark(void)
{
    const int iterations = 1000;
    int32_t logits[MLP_OUTPUT_SIZE];
    volatile int sink = 0;

    for (int i = 0; i < 10; ++i) {
        sink ^= mlp_predict(&g_model, DEMO_INPUT, logits);
    }

    size_t heap_before = esp_get_free_heap_size();
    size_t internal_before = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    int64_t start = esp_timer_get_time();

    for (int i = 0; i < iterations; ++i) {
        sink ^= mlp_predict(&g_model, DEMO_INPUT, logits);
    }

    int64_t total_us = esp_timer_get_time() - start;
    size_t heap_after = esp_get_free_heap_size();
    size_t internal_after = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);

    double avg_us = (double)total_us / (double)iterations;
    double fps = 1000000.0 / avg_us;

    const size_t model_bytes =
        g_model.header.w1_bytes + g_model.header.b1_bytes +
        g_model.header.w2_bytes + g_model.header.b2_bytes;

    printf("BENCH iterations=%d total_us=%" PRId64
           " avg_us=%.3f fps=%.3f sink=%d\n",
           iterations, total_us, avg_us, fps, sink);
    printf("MEM model_bytes=%u heap_before=%u heap_after=%u "
           "internal_before=%u internal_after=%u\n",
           (unsigned)model_bytes,
           (unsigned)heap_before,
           (unsigned)heap_after,
           (unsigned)internal_before,
           (unsigned)internal_after);
}

void app_main(void)
{
    if (!mlp_model_load(&g_model)) {
        ESP_LOGE(TAG, "No valid model. Flash model.bin first.");
        printf("MLP_ERROR model_not_loaded\n");
        return;
    }

    ESP_LOGI(TAG, "ESP32 INT8 MLP ready (64 -> 16 -> 10)");
    printf("MLP_READY model=digits64x16x10\n");
    printf("Commands: i = inference, b = benchmark\n");

    while (1) {
        int ch = getchar();
        if (ch == 'i') {
            run_once();
        } else if (ch == 'b') {
            run_benchmark();
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}
