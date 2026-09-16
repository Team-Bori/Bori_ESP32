#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_partition.h"
#include "esp_system.h"
#include "esp_timer.h"

#include "mlp.h"
#include "model.h"

static const char *TAG = "MLP_APP";
static mlp_model_t g_model;

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
    printf("LOGITS");

    for (int i = 0; i < MLP_OUTPUT_SIZE; ++i) {
        printf(" %" PRId32, logits[i]);
    }

    printf("\n");
}

static void print_system_info(void)
{
    const esp_partition_t *model = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, 0x40, "model");

    printf("SYSTEM chip=ESP32\n");
    printf("SYSTEM free_heap=%u internal_heap=%u min_free_heap=%u\n",
           (unsigned)esp_get_free_heap_size(),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)esp_get_minimum_free_heap_size());

    if (model != NULL) {
        printf("SYSTEM model_partition_offset=0x%06" PRIx32 " size=%u\n",
               model->address, (unsigned)model->size);
    }
}

static void run_once(void)
{
    int32_t logits[MLP_OUTPUT_SIZE];

    const int64_t start = esp_timer_get_time();
    const int pred = mlp_predict(&g_model, DEMO_INPUT, logits);
    const int64_t elapsed_us = esp_timer_get_time() - start;

    printf("RESULT pred=%d expected=%d latency_us=%" PRId64 "\n", pred, DEMO_LABEL, elapsed_us);
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

    const size_t heap_before = esp_get_free_heap_size();
    const size_t internal_before = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);

    const int64_t start = esp_timer_get_time();

    for (int i = 0; i < iterations; ++i) {
        sink ^= mlp_predict(&g_model, DEMO_INPUT, logits);
    }

    const int64_t total_us = esp_timer_get_time() - start;
    const size_t heap_after = esp_get_free_heap_size();
    const size_t internal_after = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);

    const double avg_us = (double)total_us / (double)iterations;
    const double fps = 1000000.0 / avg_us;
    const size_t model_bytes =
        g_model.header.w1_bytes + g_model.header.b1_bytes +
        g_model.header.w2_bytes + g_model.header.b2_bytes;

    printf("BENCH iterations=%d total_us=%" PRId64
           " avg_us=%.3f fps=%.3f sink=%d\n",
           iterations, total_us, avg_us, fps, sink);
    printf("MEM model_payload_bytes=%u model_total_bytes=%u\n",
           (unsigned)model_bytes,
           (unsigned)(sizeof(g_model.header) + model_bytes));
    printf("MEM heap_before=%u heap_after=%u "
           "internal_before=%u internal_after=%u\n",
           (unsigned)heap_before,
           (unsigned)heap_after,
           (unsigned)internal_before,
           (unsigned)internal_after);
}

void app_main(void)
{
    printf("\n=== ESP32 MLP AI DEMO ===\n");
    printf("MODEL architecture=64-16-10 quantization=INT8\n");

    print_system_info();

    if (!mlp_model_load(&g_model)) {
        ESP_LOGE(TAG, "No valid model. Flash model.bin to the model partition.");
        printf("ERROR model_not_loaded\n");
        printf("HINT flash model partition only; firmware does not need to change.\n");
        return;
    }

    mlp_model_print_info(&g_model);

    ESP_LOGI(TAG, "MLP ready");
    printf("READY commands: i=inference, b=benchmark, m=model-info\n");

    while (1) {
        const int ch = getchar();

        switch (ch) {
        case 'i':
            run_once();
            break;
        case 'b':
            run_benchmark();
            break;
        case 'm':
            print_system_info();
            mlp_model_print_info(&g_model);
            break;
        default:
            break;
        }

        vTaskDelay(pdMS_TO_TICKS(10));
    }
}
