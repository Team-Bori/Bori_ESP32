cmake_minimum_required(VERSION 3.16)

# Shared code (bori_common) used by both firmwares.
set(EXTRA_COMPONENT_DIRS "${CMAKE_CURRENT_LIST_DIR}/../components")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)

project(tflm_runtime)

# Same linker search path workaround as mlp_esp32 (ESP-IDF 5.3.1 generates
# memory.ld and sections.ld in build/esp-idf/esp_system/ld/).
if(TARGET ${PROJECT_NAME}.elf)
    target_link_directories(
        ${PROJECT_NAME}.elf
        PRIVATE
        "${CMAKE_BINARY_DIR}/esp-idf/esp_system/ld"
    )
endif()
