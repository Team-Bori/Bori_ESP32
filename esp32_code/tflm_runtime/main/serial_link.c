#include "serial_link.h"

#include <stdio.h>
#include <unistd.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/uart.h"
#include "driver/uart_vfs.h"
#include "esp_timer.h"

#include "bori_common.h"

#define LINK_UART UART_NUM_0
/* Enough to hold ~44 ms of data at 921600 baud while the main loop is busy elsewhere. */
#define LINK_RX_BUFFER 4096

static uint32_t s_baud = LINK_DEFAULT_BAUD;

void link_init(void)
{
    fflush(stdout);
    uart_wait_tx_idle_polling(LINK_UART);
    ESP_ERROR_CHECK(uart_driver_install(LINK_UART, LINK_RX_BUFFER, 0, 0, NULL, 0));
    uart_vfs_dev_use_driver(LINK_UART);
    uart_set_baudrate(LINK_UART, s_baud);
}

static TickType_t ms_to_ticks(uint32_t ms)
{
    TickType_t t = pdMS_TO_TICKS(ms);
    return t == 0 ? 1 : t;
}

int link_read_byte(uint32_t timeout_ms)
{
    uint8_t b;
    const int n = uart_read_bytes(LINK_UART, &b, 1, ms_to_ticks(timeout_ms));
    return n == 1 ? b : -1;
}

bool link_read_exact(uint8_t *dst, size_t n, uint32_t gap_timeout_ms, uint32_t *crc)
{
    uint8_t scratch[64];
    size_t done = 0;
    while (done < n) {
        size_t want = n - done;
        uint8_t *target;
        if (dst != NULL) {
            target = dst + done;
        } else {
            target = scratch;
            if (want > sizeof(scratch)) {
                want = sizeof(scratch);
            }
        }
        /* Ask only for what is already buffered (at least 1 byte) so the gap timeout
         * applies between bytes, not to the whole remaining payload. */
        size_t avail = 0;
        uart_get_buffered_data_len(LINK_UART, &avail);
        if (avail == 0) {
            avail = 1;
        }
        if (want > avail) {
            want = avail;
        }
        const int got = uart_read_bytes(LINK_UART, target, want, ms_to_ticks(gap_timeout_ms));
        if (got <= 0) {
            return false;
        }
        if (crc != NULL) {
            *crc = bori_crc32(*crc, target, (size_t)got);
        }
        done += (size_t)got;
    }
    return true;
}

void link_discard_until_idle(uint32_t idle_ms, uint32_t max_ms)
{
    const int64_t end = esp_timer_get_time() + (int64_t)max_ms * 1000;
    uint8_t buf[64];
    while (esp_timer_get_time() < end) {
        const int got = uart_read_bytes(LINK_UART, buf, sizeof(buf), ms_to_ticks(idle_ms));
        if (got <= 0) {
            return;
        }
    }
}

void link_flush_tx(void)
{
    fflush(stdout);
    fsync(fileno(stdout));
    uart_wait_tx_done(LINK_UART, ms_to_ticks(1000));
}

bool link_baud_supported(uint32_t baud)
{
    return baud == 115200 || baud == 230400 || baud == 460800 || baud == 921600;
}

void link_set_baud(uint32_t baud)
{
    link_flush_tx();
    uart_set_baudrate(LINK_UART, baud);
    s_baud = baud;
    /* Bytes received around the switch are garbage at one of the two rates. */
    uart_flush_input(LINK_UART);
}

uint32_t link_baud(void)
{
    return s_baud;
}
