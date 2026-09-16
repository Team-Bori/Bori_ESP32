cmake_minimum_required(VERSION 3.16)

set(EXTRA_COMPONENT_DIRS "")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)

project(mlp_esp32)

# ESP-IDF 5.3.1 generates memory.ld and sections.ld here:
# build/esp-idf/esp_system/ld/
#
# Add this directory directly to the final application linker.
if(TARGET ${PROJECT_NAME}.elf)
    target_link_directories(
        ${PROJECT_NAME}.elf
        PRIVATE
        "${CMAKE_BINARY_DIR}/esp-idf/esp_system/ld"
    )
endif()