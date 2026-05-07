#!/usr/bin/env bash
# build.sh — rasp_compiler 构建脚本
# 用法: ./build.sh [选项]
#   -r, --release      Release 构建（默认 Debug）
#   -c, --clean        清理构建目录后重新配置
#   -j N, --jobs N     并行数（默认 nproc）
#       --asan         开启 AddressSanitizer
#       --no-tests     不构建测试
#       --test         构建完成后运行 ctest
#       --run [ARGS]   构建完成后运行 rasp_compiler
set -euo pipefail

# ── 默认参数 ──────────────────────────────────────────────
BUILD_TYPE="Debug"
CLEAN=0
JOBS=$(nproc)
ASAN="OFF"
BUILD_TESTS="ON"
RUN_TESTS=0
RUN_AFTER=0
RUN_ARGS=()

# ── 参数解析 ──────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--release)   BUILD_TYPE="Release";      shift ;;
    -c|--clean)     CLEAN=1;                   shift ;;
    -j|--jobs)      JOBS="$2";                 shift 2 ;;
       --asan)      ASAN="ON";                 shift ;;
        --no-tests)  BUILD_TESTS="OFF";         shift ;;
       --test)      RUN_TESTS=1;               shift ;;
       --run)       RUN_AFTER=1;               shift
                    while [[ $# -gt 0 && "$1" != --* ]]; do
                      RUN_ARGS+=("$1"); shift
                    done ;;
    -h|--help)
      sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "未知选项: $1" >&2; exit 1 ;;
  esac
done

BUILD_DIR="build/$(echo "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')"
BINARY="${BUILD_DIR}/rasp_compiler"

# ── 清理 ─────────────────────────────────────────────────
if [[ $CLEAN -eq 1 && -d "$BUILD_DIR" ]]; then
  echo "清理 ${BUILD_DIR} ..."
  rm -rf "$BUILD_DIR"
fi

# ── 配置 ─────────────────────────────────────────────────
cmake -S . -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DRASP_ASAN="$ASAN" \
  -DRASP_BUILD_TESTS="$BUILD_TESTS" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# compile_commands.json 软链接（供 clangd / IDE 使用）
if [[ -f "${BUILD_DIR}/compile_commands.json" ]]; then
  ln -sf "${BUILD_DIR}/compile_commands.json" compile_commands.json
fi

# ── 构建 ─────────────────────────────────────────────────
cmake --build "$BUILD_DIR" -j"$JOBS"

# compile_commands.json 可能在 build 后才生成
if [[ -f "${BUILD_DIR}/compile_commands.json" ]]; then
  ln -sf "${BUILD_DIR}/compile_commands.json" compile_commands.json
fi

# ── 结果 ─────────────────────────────────────────────────
echo
echo "✓ 构建完成 [${BUILD_TYPE}]"
echo "  二进制:  ${BINARY}"
echo "  调试:    gdb --args ${BINARY} <input.json>"
echo "  ASan:    RASP_ASAN=ON ./build.sh --asan"

if [[ $RUN_TESTS -eq 1 ]]; then
  echo
  echo "运行测试..."
  echo "────────────────────────────────────────"
  ctest --test-dir "$BUILD_DIR" --output-on-failure
fi

if [[ $RUN_AFTER -eq 1 ]]; then
  echo
  echo "运行: ${BINARY} ${RUN_ARGS[*]:-}"
  echo "────────────────────────────────────────"
  "${BINARY}" "${RUN_ARGS[@]:-}"
fi
