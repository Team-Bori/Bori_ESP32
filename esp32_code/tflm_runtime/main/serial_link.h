#pragma once

/*
 * UART0 console with the UART driver installed, so the host can send binary
 * frames (docs/PROTOCOL_v2.md, "binary frames") and the baud rate can change at
 * run time. stdout (printf, ESP_LOG) keeps working through the same UART.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LINK_DEFAULT_BAUD 115200

void link_init(void);

/* Returns the next byte, or -1 if nothing arrived within timeout_ms. */
int link_read_byte(uint32_t timeout_ms);

/*
 * Reads exactly n bytes into dst (dst may be NULL to discard them).
 * Fails if the gap between bytes exceeds gap_timeout_ms.
 * If crc is not NULL it is updated with bori_crc32 over the bytes read.
 */
bool link_read_exact(uint8_t *dst, size_t n, uint32_t gap_timeout_ms, uint32_t *crc);

/* Drops incoming bytes until the line has been idle for idle_ms (at most max_ms). */
void link_discard_until_idle(uint32_t idle_ms, uint32_t max_ms);

/* Waits until everything printed so far has left the UART. */
void link_flush_tx(void);

bool link_baud_supported(uint32_t baud);
void link_set_baud(uint32_t baud);
uint32_t link_baud(void);

#ifdef __cplusplus
}
#endif
