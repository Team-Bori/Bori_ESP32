#pragma once

/*
 * BTF1 model package (docs/PACKAGE_FORMAT.md).
 * The package is memory-mapped from the model partition; nothing is copied to RAM.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BTF_MAGIC "BTF1"
#define BTF_FORMAT_VERSION 1
#define BTF_HEADER_MIN_SIZE 64
#define BTF_SECTION_ENTRY_SIZE 16
#define BTF_TENSOR_DESC_SIZE 40
#define BTF_MAX_DIMS 6
#define BTF_MODEL_ID_LEN 32
#define BTF_TFLITE_ALIGN 16
#define BTF_MAX_LABEL_DIM 64

enum btf_task {
    BTF_TASK_CLASSIFICATION = 0,
    BTF_TASK_REGRESSION = 1,
    BTF_TASK_BINARY_SCORE = 2,
};

enum btf_section_type {
    BTF_SEC_TFLITE = 1,
    BTF_SEC_TENSORS = 2,
    BTF_SEC_LABELS = 3,
    BTF_SEC_DEMO_INPUT = 4,
    BTF_SEC_DEMO_EXPECTED = 5,
    BTF_SEC_EVAL_INPUTS = 6,
    BTF_SEC_EVAL_EXPECTED = 7,
};

enum btf_dtype {
    BTF_DTYPE_INT8 = 1,
    BTF_DTYPE_UINT8 = 2,
    BTF_DTYPE_INT16 = 3,
    BTF_DTYPE_INT32 = 4,
    BTF_DTYPE_FLOAT32 = 5,
};

typedef struct {
    uint8_t dtype;
    uint8_t ndim;
    int32_t dims[BTF_MAX_DIMS];
    float scale;
    int32_t zero_point;
    uint32_t bytes;
} btf_tensor_desc_t;

typedef struct {
    /* Whole package as mapped in the address space. */
    const uint8_t *base;
    uint32_t total_size;
    uint32_t crc32;

    char model_id[BTF_MODEL_ID_LEN + 1];
    uint8_t task;
    uint16_t label_dim;
    uint32_t arena_bytes; /* 0 = auto (largest free internal block) */
    float task_param;     /* binary_score: threshold, regression: tolerance */

    const uint8_t *tflite;
    uint32_t tflite_size;

    btf_tensor_desc_t input;
    btf_tensor_desc_t output;

    const char *labels; /* label_count NUL-terminated strings, back to back */
    uint32_t labels_size;
    uint32_t label_count;

    const uint8_t *demo_input;   /* input.bytes, or NULL */
    const uint8_t *demo_expected; /* label_dim floats (little endian), or NULL */

    const uint8_t *eval_inputs;  /* eval_count * input.bytes */
    const uint8_t *eval_expected; /* eval_count * label_dim floats (little endian) */
    uint32_t eval_count;
} btf_package_t;

/*
 * Parses and validates a package already mapped at `base` (size `mapped_size`).
 * Returns NULL on success, or a static error message. *code is set to the protocol
 * error code ("package_invalid" or "checksum_mismatch").
 */
const char *btf_parse(const uint8_t *base, uint32_t mapped_size, btf_package_t *pkg, const char **code);

/* Reads a little-endian float from a possibly unaligned address. */
float btf_read_f32(const uint8_t *p);

const char *btf_task_name(uint8_t task);
const char *btf_dtype_name(uint8_t dtype);
uint32_t btf_dtype_size(uint8_t dtype);

/* Returns the i-th label or NULL. */
const char *btf_label(const btf_package_t *pkg, uint32_t index);

#ifdef __cplusplus
}
#endif
