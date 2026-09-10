idf_component_register(
    SRCS "main.c" "mlp.c" "model.c"
    INCLUDE_DIRS "."
    REQUIRES esp_partition
)
