cmake_minimum_required(VERSION 3.16)

project(rasp_compiler LANGUAGES CXX)

# ── 全局设置 ──────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type (Debug|Release|RelWithDebInfo)" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ── 选项 ─────────────────────────────────────────────────
option(RASP_BUILD_TESTS  "Build unit tests"        ON)
option(RASP_ASAN         "Enable AddressSanitizer"  OFF)

# ── 公共编译选项 ──────────────────────────────────────────
add_library(rasp_compile_flags INTERFACE)
target_compile_options(rasp_compile_flags INTERFACE
  -Wall -Wextra -Wpedantic
  $<$<CONFIG:Debug>:-g3 -O0>
  $<$<CONFIG:Release>:-O3 -DNDEBUG>
  $<$<CONFIG:RelWithDebInfo>:-O2 -g -DNDEBUG>
)
if(RASP_ASAN)
  target_compile_options(rasp_compile_flags INTERFACE -fsanitize=address,undefined)
  target_link_options(rasp_compile_flags INTERFACE -fsanitize=address,undefined)
endif()

# ── 头文件路径 ────────────────────────────────────────────
add_library(rasp_includes INTERFACE)
target_include_directories(rasp_includes INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

# ── 核心库：rasp_ir ───────────────────────────────────────
# OBJECT 库确保所有 .o 被直接链入，避免静态库链接器丢弃
# op_registry_init.cpp 的 static-init 块依赖此行为
add_library(rasp_ir OBJECT
  src/driver/compiler_driver.cpp
  src/high_level_ir/op_registry_init.cpp
  src/high_level_ir/ir_module.cpp
  src/high_level_ir/hl_expr_visitor.cpp
  src/high_level_ir/hl_expr_mutator.cpp
  src/low_level_ir/buffer.cpp
  src/low_level_ir/stmt.cpp
  src/low_level_ir/ll_expr_visitor.cpp
  src/low_level_ir/ll_expr_mutator.cpp
  src/lowering/op_lowering_rules.cpp
  src/lowering/lowering.cpp
  src/codegen/codegen.cpp
  src/runtime/tensor.cpp
  src/runtime/runtime.cpp
  src/runtime/reference_runtime.cpp
  src/pass/pass_manager.cpp
  src/pass/analysis/type_inference.cpp
  src/pass/opt/constant_folding.cpp
  src/pass/opt/dead_code_elimination.cpp
  src/pass/opt/operator_fusion.cpp
)
target_link_libraries(rasp_ir
  PUBLIC  rasp_includes
  PRIVATE rasp_compile_flags
          dl
)

# ── 主可执行文件 ──────────────────────────────────────────
add_executable(rasp_compiler src/rpc.cpp)
target_link_libraries(rasp_compiler
  PRIVATE rasp_ir rasp_compile_flags
)

# ── 测试 ─────────────────────────────────────────────────
if(RASP_BUILD_TESTS)
  enable_testing()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
    add_subdirectory(tests)
  endif()
endif()
