#pragma once

/*
 * Code shared by the Bori ESP32 firmwares (mlp_esp32, tflm_runtime).
 *
 * Every helper that prints JSON writes a fragment to stdout without a trailing
 * newline, so the caller composes one complete JSON object per line.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "esp_system.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Name of the data partition (type data, subtype 0x40) that holds the model. */
#define BORI_MODEL_PARTITION_LABEL "model"
#define BORI_MODEL_PARTITION_SUBTYPE 0x40

const char *bori_reset_reason_name(esp_reset_reason_t reason);

/* true when the previous reset was a panic, watchdog or brownout. */
bool bori_reset_reason_is_fault(esp_reset_reason_t reason);

/* Prints the "boot" message. firmware_id may be NULL (omitted from the JSON). */
void bori_print_boot(const char *tag, int proto, const char *firmware_id);

/* Prints "memory":{...} (protocol v1 fields). Returns the characters written. */
int bori_print_memory_fields(void);

/* Human-readable memory summary at INFO level. */
void bori_log_memory(const char *tag);

/* Prints "board":{...}. Returns the characters written. */
int bori_print_board_fields(void);

/* Human-readable board summary at INFO level. */
void bori_log_board(const char *tag);

/* Prints s as a JSON string literal (with quotes), escaping as needed. */
void bori_print_json_string(const char *s);

/* Prints {"type":"error","code":...,"message":...} and logs it. */
void bori_print_error(const char *tag, const char *code, const char *message);

/* Standard CRC-32 (IEEE 802.3, same as Python zlib.crc32). Start with crc = 0. */
uint32_t bori_crc32(uint32_t crc, const void *data, size_t len);

#ifdef __cplusplus
}
#endif
