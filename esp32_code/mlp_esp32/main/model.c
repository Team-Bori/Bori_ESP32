#include "model.h"

#include <inttypes.h>
#include <string.h>

#include "esp_err.h"
#include "esp_log.h"
#include "esp_partition.h"

static const char *TAG = "MLP_MODEL";

static uint32_t fnv1a32(const uint8_t *data, size_t len)
{
    uint32_t hash = 2166136261u;

    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= 16777619u;
    }

    return hash;
}

static bool header_is_valid(const mlp_model_header_t *h, size_t payload_size)
{
    if (memcmp(h->magic, MLP_MODEL_MAGIC, 4) != 0) return false;
    if (h->version != MLP_MODEL_VERSION) return false;

    if (h->input_size != MLP_INPUT_SIZE ||
        h->hidden_size != MLP_HIDDEN_SIZE ||
        h->output_size != MLP_OUTPUT_SIZE) {
        return false;
    }

    const size_t expected_w1 = sizeof(((mlp_model_t *)0)->w1);
    const size_t expected_b1 = sizeof(((mlp_model_t *)0)->b1);
    const size_t expected_w2 = sizeof(((mlp_model_t *)0)->w2);
    const size_t expected_b2 = sizeof(((mlp_model_t *)0)->b2);

    if (h->w1_bytes != expected_w1 ||
        h->b1_bytes != expected_b1 ||
        h->w2_bytes != expected_w2 ||
        h->b2_bytes != expected_b2) {
        return false;
    }

    if (h->input_scale <= 0.0f ||
        h->w1_scale <= 0.0f ||
        h->hidden_scale <= 0.0f ||
        h->w2_scale <= 0.0f) {
        return false;
    }

    const size_t expected_payload = expected_w1 + expected_b1 + expected_w2 + expected_b2;

    return payload_size == expected_payload;
}

bool mlp_model_load(mlp_model_t *model)
{
    if (model == NULL) {
        return false;
    }

    const esp_partition_t *partition = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, 0x40, "model");

    if (partition == NULL) {
        ESP_LOGE(TAG,
                 "model partition not found; flash model.bin at 0x200000 "
                 "after flashing the firmware");
        return false;
    }

    const size_t payload_size =
        sizeof(model->w1) + sizeof(model->b1) +
        sizeof(model->w2) + sizeof(model->b2);
    const size_t total_size = sizeof(model->header) + payload_size;

    if (partition->size < total_size) {
        ESP_LOGE(TAG, "model partition too small: %u bytes", (unsigned)partition->size);
        return false;
    }

    ESP_LOGI(TAG, "model partition: address=0x%" PRIx32 " size=%u",
             partition->address, (unsigned)partition->size);

    memset(model, 0, sizeof(*model));

    esp_err_t err = esp_partition_read(partition, 0, &model->header, sizeof(model->header));
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "header read failed: %s", esp_err_to_name(err));
        return false;
    }

    if (!header_is_valid(&model->header, payload_size)) {
        ESP_LOGE(TAG,
                 "invalid model header: magic=%.4s version=%" PRIu32
                 " dims=%u-%u-%u payload=%" PRIu32 "+%" PRIu32 "+%" PRIu32 "+%" PRIu32,
                 model->header.magic,
                 model->header.version,
                 model->header.input_size,
                 model->header.hidden_size,
                 model->header.output_size,
                 model->header.w1_bytes,
                 model->header.b1_bytes,
                 model->header.w2_bytes,
                 model->header.b2_bytes);
        return false;
    }

    uint8_t *payload = (uint8_t *)model->w1;
    err = esp_partition_read(partition, sizeof(model->header), payload, payload_size);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "payload read failed: %s", esp_err_to_name(err));
        return false;
    }

    const uint32_t checksum = fnv1a32(payload, payload_size);
    if (checksum != model->header.checksum) {
        ESP_LOGE(TAG,
                 "checksum mismatch: expected=0x%08" PRIx32
                 " actual=0x%08" PRIx32,
                 model->header.checksum, checksum
        );
        return false;
    }

    ESP_LOGI(TAG, "model loaded: total=%u payload=%u checksum=0x%08" PRIx32,
             (unsigned)total_size, 
             (unsigned)payload_size, checksum
    );

    return true;
}

void mlp_model_print_info(const mlp_model_t *model)
{
    if (model == NULL) return;

    const size_t payload_size =
        model->header.w1_bytes + model->header.b1_bytes +
        model->header.w2_bytes + model->header.b2_bytes;

    printf("MODEL dims=%u-%u-%u total_bytes=%u payload_bytes=%u\n",
           model->header.input_size,
           model->header.hidden_size,
           model->header.output_size,
           (unsigned)(sizeof(model->header) + payload_size),
           (unsigned)payload_size);
    printf("MODEL scales input=%.8f w1=%.8f hidden=%.8f w2=%.8f\n",
           (double)model->header.input_scale,
           (double)model->header.w1_scale,
           (double)model->header.hidden_scale,
           (double)model->header.w2_scale);
    printf("MODEL checksum=0x%08" PRIx32 "\n", model->header.checksum);
}
