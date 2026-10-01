idf_component_register(
    SRCS "main.c" "mlp.c" "model.c"
    INCLUDE_DIRS "."
    REQUIRES esp_partition esp_timer esp_app_format spi_flash
)
