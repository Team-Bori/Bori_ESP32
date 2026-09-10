#pragma once

#include <stdbool.h>
#include <stdint.h>

#define MLP_INPUT_SIZE 64
#define MLP_HIDDEN_SIZE 16
#define MLP_OUTPUT_SIZE 10

#define MLP_MODEL_MAGIC "MLP1"
#define MLP_MODEL_VERSION 1u

/* Fixed-size header stored at the beginning of model.bin. */
typedef struct {
    char magic[4];
    uint32_t version;
    uint16_t input_size;
    uint16_t hidden_size;
    uint16_t output_size;
    uint16_t reserved;
    float input_scale;
    float w1_scale;
    float hidden_scale;
    float w2_scale;
    uint32_t w1_bytes;
    uint32_t b1_bytes;
    uint32_t w2_bytes;
    uint32_t b2_bytes;
    uint32_t checksum;
} mlp_model_header_t;

_Static_assert(sizeof(mlp_model_header_t) == 52, "Unexpected model header size");

typedef struct {
    mlp_model_header_t header;
    int8_t w1[MLP_INPUT_SIZE * MLP_HIDDEN_SIZE];
    int32_t b1[MLP_HIDDEN_SIZE];
    int8_t w2[MLP_HIDDEN_SIZE * MLP_OUTPUT_SIZE];
    int32_t b2[MLP_OUTPUT_SIZE];
} mlp_model_t;

bool mlp_model_load(mlp_model_t *model);
