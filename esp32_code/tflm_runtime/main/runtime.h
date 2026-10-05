#pragma once

/* TFLite Micro runtime for one BTF1 package in the model partition. */

#include <stddef.h>
#include <stdint.h>

#include "package.h"

struct RuntimeState {
    bool loaded;
    /* Set when loading failed: protocol error code and message. */
    const char *error_code;
    char error_message[160];

    btf_package_t pkg;
    uint32_t partition_offset;
    uint32_t partition_size;

    size_t boot_largest_free;  /* largest internal 8-bit block before the arena was allocated */
    size_t arena_alloc_bytes;  /* bytes actually allocated for the arena */
    size_t arena_used_bytes;   /* TFLM arena_used_bytes() after AllocateTensors */
    uint32_t model_ops;        /* operator codes used by the model */
    uint32_t invoke_errors;    /* failed Invoke() calls since boot */
};

extern RuntimeState g_rt;

/* Number of ops registered in the resolver (ops.def). */
int runtime_registered_ops(void);

/*
 * Maps the model partition, validates the package, allocates the arena and the
 * interpreter. On failure g_rt.loaded is false and error_code/error_message are set.
 * `skip_reason` non-NULL means do not load (crash guard); it becomes the error message.
 */
void runtime_load(const char *skip_reason);

uint8_t *runtime_input(void);
size_t runtime_input_bytes(void);
uint32_t runtime_output_count(void);
float runtime_output_value(uint32_t index);

/* Fills the input with the demo sample, or with the quantized zero point when absent. */
void runtime_set_demo_input(void);

/* One Invoke(); returns false on failure (and counts it). *elapsed_us is the Invoke time. */
bool runtime_invoke(int64_t *elapsed_us);
