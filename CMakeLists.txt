cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(VLM_NPU)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fexceptions -pthread -O3")

# --- Rockchip libraries
set(RK_LIB_PATH "/usr/local/lib" CACHE PATH "Path to Rockchip libraries")
link_directories(${RK_LIB_PATH})

# --- Project headers (include/ + bundled third-party)
include_directories(include third_party third_party/nlohmann)

# ── Shared inference source ─────────────────────────────────────────────────
set(RKLLM_SOURCES src/RK35llm.cpp)

set(RK_LIBS
    ${RK_LIB_PATH}/librknnrt.so
    ${RK_LIB_PATH}/librkllmrt.so
    -fopenmp -ldl -lpthread -lgomp)

# ── CLI target ───────────────────────────────────────────────────────────────
add_executable(VLM_NPU src/main.cpp ${RKLLM_SOURCES})
target_link_libraries(VLM_NPU ${RK_LIBS})

# ── Server target ────────────────────────────────────────────────────────────
add_executable(VLM_server src/server_main.cpp src/server.cpp ${RKLLM_SOURCES})
target_link_libraries(VLM_server ${RK_LIBS})

set(EXECUTABLE_OUTPUT_PATH "../")
