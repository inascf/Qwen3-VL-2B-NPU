cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(VLM_NPU)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fexceptions -pthread -O3")

include(CheckIncludeFile)

# --- Find OpenCV
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv4)
include_directories(${OPENCV_INCLUDE_DIRS})
link_directories(${OPENCV_LIBRARY_DIRS})

# --- Rockchip libraries
set(RK_LIB_PATH "/usr/local/lib" CACHE PATH "Path to Rockchip libraries")
link_directories(${RK_LIB_PATH})

# --- Project include dir
include_directories(include)

# ── Shared inference source ─────────────────────────────────────────────────
set(RKLLM_SOURCES src/RK35llm.cpp)

# ── CLI target (original) ───────────────────────────────────────────────────
add_executable(VLM_NPU src/main.cpp ${RKLLM_SOURCES})
target_link_libraries(VLM_NPU
    ${OPENCV_LIBRARIES}
    ${RK_LIB_PATH}/librknnrt.so
    ${RK_LIB_PATH}/librkllmrt.so
    -fopenmp -ldl -lpthread -lgomp)

# ── Server target ───────────────────────────────────────────────────────────
include(FetchContent)

# cpp-httplib (header-only)
FetchContent_Declare(httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.18.0)
FetchContent_MakeAvailable(httplib)

# nlohmann/json (header-only)
FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3)
FetchContent_MakeAvailable(nlohmann_json)

add_executable(VLM_server
    src/server_main.cpp
    src/server.cpp
    ${RKLLM_SOURCES})

target_link_libraries(VLM_server
    httplib::httplib
    nlohmann_json::nlohmann_json
    ${OPENCV_LIBRARIES}
    ${RK_LIB_PATH}/librknnrt.so
    ${RK_LIB_PATH}/librkllmrt.so
    -fopenmp -ldl -lpthread -lgomp)

set(EXECUTABLE_OUTPUT_PATH "../")
