cmake_minimum_required(VERSION 3.17)
project(TFLiteImageClassification)

set(CMAKE_CXX_STANDARD 14)

# OpenCV Integration
if(WIN32)
    set(OpenCV_DIR C:/Users/Administrator/Karthick/Softwares/Libs/opencv/build)
endif()
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# TensorFlow Lite Integration
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/include/)
add_executable(${PROJECT_NAME} main.cpp)

if(WIN32)
    find_library(TFLITE tensorflowlite.dll.if HINTS ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/libs/windows_x64/)
    find_library(FLATBUFFERS flatbuffers  HINTS ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/libs/windows_x64/)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${TFLITE} ${FLATBUFFERS} ${OpenCV_LIBS})
endif()

if(UNIX AND NOT APPLE )
    ADD_LIBRARY(tensorflowlite SHARED IMPORTED)
    set_property(TARGET tensorflowlite PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/libs/linux_x64/libtensorflowlite.so)
    target_link_libraries(TFLiteCheck tensorflowlite ${OpenCV_LIBS})
endif()

if(APPLE)
    ADD_LIBRARY(tensorflowlite SHARED IMPORTED)
    set_property(TARGET tensorflowlite PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/libs/mac/libtensorflowlite.dylib)
    target_link_libraries(TFLiteCheck tensorflowlite ${OpenCV_LIBS})
endif()


if(WIN32)
    add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/libs/windows_x64/tensorflowlite.dll ${CMAKE_BINARY_DIR}/Release/
    )

    add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${OpenCV_DIR}/x64/vc15/bin/opencv_world452.dll ${CMAKE_BINARY_DIR}/Release/
    )
endif()
