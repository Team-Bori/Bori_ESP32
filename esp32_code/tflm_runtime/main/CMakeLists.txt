idf_component_register(
    SRCS "main.cc" "runtime.cc" "package.c" "serial_link.c"
    INCLUDE_DIRS "."
    REQUIRES bori_common esp_partition esp_timer esp_driver_uart vfs spi_flash heap
)
