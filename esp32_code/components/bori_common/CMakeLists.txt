idf_component_register(
    SRCS "bori_common.c"
    INCLUDE_DIRS "include"
    REQUIRES esp_system
    PRIV_REQUIRES esp_app_format spi_flash esp_hw_support heap log freertos
)
