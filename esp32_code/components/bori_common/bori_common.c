#include "bori_common.h"

#include <inttypes.h>
#include <stdio.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_app_desc.h"
#include "esp_chip_info.h"
#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "esp_log.h"

const char *bori_reset_reason_name(esp_reset_reason_t reason)
{
    switch (reason) {
    case ESP_RST_POWERON:   return "poweron";
    case ESP_RST_EXT:       return "external";
    case ESP_RST_SW:        return "software";
    case ESP_RST_PANIC:     return "panic";
    case ESP_RST_INT_WDT:   return "int_wdt";
    case ESP_RST_TASK_WDT:  return "task_wdt";
    case ESP_RST_WDT:       return "wdt";
    case ESP_RST_DEEPSLEEP: return "deepsleep";
    case ESP_RST_BROWNOUT:  return "brownout";
    case ESP_RST_SDIO:      return "sdio";
    default:                return "unknown";
    }
}

bool bori_reset_reason_is_fault(esp_reset_reason_t reason)
{
    return reason == ESP_RST_PANIC || reason == ESP_RST_INT_WDT ||
           reason == ESP_RST_TASK_WDT || reason == ESP_RST_WDT ||
           reason == ESP_RST_BROWNOUT;
}

void bori_print_boot(const char *tag, int proto, const char *firmware_id)
{
    const esp_app_desc_t *app = esp_app_get_description();
    const esp_reset_reason_t reason = esp_reset_reason();

    printf("{\"type\":\"boot\",\"proto\":%d,", proto);
    if (firmware_id != NULL) {
        printf("\"firmware_id\":\"%s\",", firmware_id);
    }
    printf("\"fw_version\":\"%s\",\"idf_version\":\"%s\","
           "\"reset_reason\":\"%s\",\"fault_reset\":%s}\n",
           app->version, app->idf_ver,
           bori_reset_reason_name(reason),
           bori_reset_reason_is_fault(reason) ? "true" : "false");
    fflush(stdout);

    ESP_LOGI(tag, "[boot] protocol v%d, firmware %s%s%s, ESP-IDF %s",
             proto, firmware_id ? firmware_id : "", firmware_id ? " " : "",
             app->version, app->idf_ver);
    if (bori_reset_reason_is_fault(reason)) {
        ESP_LOGW(tag, "  reset reason: %s (previous run crashed)", bori_reset_reason_name(reason));
    } else {
        ESP_LOGI(tag, "  reset reason: %s", bori_reset_reason_name(reason));
    }
}

int bori_print_memory_fields(void)
{
    return printf("\"memory\":{\"heap_free\":%u,\"heap_min_free\":%u,\"heap_total\":%u,"
           "\"internal_free\":%u,\"stack_free\":%u}",
           (unsigned)esp_get_free_heap_size(),
           (unsigned)esp_get_minimum_free_heap_size(),
           (unsigned)heap_caps_get_total_size(MALLOC_CAP_DEFAULT),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)(uxTaskGetStackHighWaterMark(NULL) * sizeof(StackType_t)));
}

void bori_log_memory(const char *tag)
{
    ESP_LOGI(tag, "  memory : heap free %u / total %u bytes (min free %u), internal free %u, stack free %u",
             (unsigned)esp_get_free_heap_size(),
             (unsigned)heap_caps_get_total_size(MALLOC_CAP_DEFAULT),
             (unsigned)esp_get_minimum_free_heap_size(),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)(uxTaskGetStackHighWaterMark(NULL) * sizeof(StackType_t)));
}

static uint32_t flash_size_bytes(void)
{
    uint32_t flash_size = 0;
    if (esp_flash_get_size(NULL, &flash_size) != ESP_OK) {
        flash_size = 0;
    }
    return flash_size;
}

int bori_print_board_fields(void)
{
    esp_chip_info_t chip;
    esp_chip_info(&chip);
    const esp_app_desc_t *app = esp_app_get_description();

    return printf("\"board\":{\"chip\":\"ESP32\",\"cores\":%d,\"revision\":%d,"
           "\"cpu_freq_mhz\":%d,\"flash_bytes\":%" PRIu32 ",\"psram_bytes\":%u,"
           "\"wifi\":%s,\"bt\":%s,\"ai_accelerator\":false,"
           "\"fw_version\":\"%s\",\"idf_version\":\"%s\"}",
           chip.cores, chip.revision, CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ, flash_size_bytes(),
           (unsigned)heap_caps_get_total_size(MALLOC_CAP_SPIRAM),
           (chip.features & CHIP_FEATURE_WIFI_BGN) ? "true" : "false",
           (chip.features & (CHIP_FEATURE_BT | CHIP_FEATURE_BLE)) ? "true" : "false",
           app->version, app->idf_ver);
}

void bori_log_board(const char *tag)
{
    esp_chip_info_t chip;
    esp_chip_info(&chip);
    const esp_app_desc_t *app = esp_app_get_description();

    ESP_LOGI(tag, "[info] ESP32 rev %d.%d, %d cores @ %d MHz, flash %" PRIu32 " KB, PSRAM %u KB, AI accelerator: no",
             chip.revision / 100, chip.revision % 100, chip.cores,
             CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ, flash_size_bytes() / 1024,
             (unsigned)(heap_caps_get_total_size(MALLOC_CAP_SPIRAM) / 1024));
    ESP_LOGI(tag, "  firmware: %s (ESP-IDF %s)", app->version, app->idf_ver);
}

void bori_print_json_string(const char *s)
{
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)s; p != NULL && *p != '\0'; ++p) {
        switch (*p) {
        case '"':  fputs("\\\"", stdout); break;
        case '\\': fputs("\\\\", stdout); break;
        case '\n': fputs("\\n", stdout); break;
        case '\r': fputs("\\r", stdout); break;
        case '\t': fputs("\\t", stdout); break;
        default:
            if (*p < 0x20) {
                printf("\\u%04x", *p);
            } else {
                putchar(*p);
            }
            break;
        }
    }
    putchar('"');
}

void bori_print_error(const char *tag, const char *code, const char *message)
{
    printf("{\"type\":\"error\",\"code\":\"%s\",\"message\":", code);
    bori_print_json_string(message);
    printf("}\n");
    fflush(stdout);

    ESP_LOGE(tag, "[error] %s: %s", code, message);
}

static uint32_t s_crc_table[256];
static bool s_crc_table_ready = false;

uint32_t bori_crc32(uint32_t crc, const void *data, size_t len)
{
    if (!s_crc_table_ready) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            }
            s_crc_table[i] = c;
        }
        s_crc_table_ready = true;
    }

    const uint8_t *p = (const uint8_t *)data;
    crc = ~crc;
    for (size_t i = 0; i < len; ++i) {
        crc = s_crc_table[(crc ^ p[i]) & 0xFFu] ^ (crc >> 8);
    }
    return ~crc;
}
