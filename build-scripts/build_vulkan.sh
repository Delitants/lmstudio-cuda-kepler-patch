#!/usr/bin/env bash
set -euo pipefail

# Minimal upstream-safe Vulkan build script for llama.cpp.

BUILD_DIR="${BUILD_DIR:-build-vulkan}"
ROOT_DIR="${1:-$(pwd)}"

if [[ ! -f "${ROOT_DIR}/CMakeLists.txt" ]]; then
    echo "[ERROR] Could not find CMakeLists.txt in: ${ROOT_DIR}"
    echo "Run this script from the llama.cpp root or pass the source path as arg 1."
    exit 1
fi

if command -v nproc >/dev/null 2>&1; then
    JOBS="$(nproc)"
else
    JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
fi

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_TOOLS=OFF \
  -DGGML_VULKAN=ON \
  -DGGML_VULKAN_COOPMAT_GLSLC_SUPPORT=OFF \
  -DGGML_VULKAN_COOPMAT2_GLSLC_SUPPORT=OFF \
  -DGGML_NATIVE=OFF \
  -DGGML_AVX=ON \
  -DGGML_AVX2=OFF \
  -DGGML_FMA=OFF \
  -DGGML_F16C=OFF \
  -DGGML_BMI2=OFF

cmake --build "${BUILD_DIR}" --config Release -j"${JOBS}"

echo "Vulkan build complete. Output libraries are in: ${BUILD_DIR}/bin"
