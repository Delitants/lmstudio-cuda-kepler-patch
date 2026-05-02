#!/usr/bin/env bash
set -euo pipefail

# LM Studio Unlocked Backend - Tesla K40 / Kepler build script
# Builds llama.cpp shared libs for Linux + LM Studio:
# - GPU: NVIDIA Tesla K40 (SM 3.5)
# - CPU: AVX1 baseline
#
# Important: Kepler is not supported by CUDA 12+, so use CUDA 11.x.

BUILD_DIR="${BUILD_DIR:-build-cuda-k40-avx1}"
CUDA_ARCH="${CUDA_ARCH:-35}"
ROOT_DIR="${1:-$(pwd)}"

pick_tool() {
    for tool in "$@"; do
        if command -v "${tool}" >/dev/null 2>&1; then
            echo "${tool}"
            return 0
        fi
    done
    return 1
}

if [[ ! -f "${ROOT_DIR}/CMakeLists.txt" ]]; then
    echo "[ERROR] Could not find CMakeLists.txt in: ${ROOT_DIR}"
    echo "Run this script from the llama.cpp root or pass the source path as arg 1."
    exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
    echo "[ERROR] cmake is required."
    exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
    echo "[ERROR] nvcc not found. Install CUDA Toolkit 11.x first."
    exit 1
fi

NVCC_VERSION="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
if [[ -z "${NVCC_VERSION}" ]]; then
    echo "[ERROR] Could not detect nvcc version."
    exit 1
fi

NVCC_MAJOR="${NVCC_VERSION%%.*}"
if (( NVCC_MAJOR >= 12 )); then
    echo "[ERROR] Detected nvcc ${NVCC_VERSION}. Tesla K40 (sm_35) needs CUDA 11.x."
    echo "Install CUDA 11.8 and re-run this script."
    exit 1
fi

if command -v nproc >/dev/null 2>&1; then
    JOBS="$(nproc)"
else
    JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
fi

CC_BIN="${CC:-$(pick_tool gcc-11 gcc-10 gcc || true)}"
CXX_BIN="${CXX:-$(pick_tool g++-11 g++-10 g++ || true)}"

if [[ -z "${CC_BIN}" || -z "${CXX_BIN}" ]]; then
    echo "[ERROR] Could not find gcc/g++."
    exit 1
fi

echo "=== Building llama.cpp for Tesla K40 (sm_${CUDA_ARCH}) + AVX1 ==="
echo "Source directory: ${ROOT_DIR}"
echo "Build directory: ${BUILD_DIR}"
echo "nvcc version: ${NVCC_VERSION}"
echo "Host compiler: ${CC_BIN}/${CXX_BIN}"

rm -rf "${BUILD_DIR}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_C_COMPILER="${CC_BIN}" \
  -DCMAKE_CXX_COMPILER="${CXX_BIN}" \
  -DCMAKE_CUDA_COMPILER=nvcc \
  -DCMAKE_CUDA_HOST_COMPILER="${CXX_BIN}" \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_TOOLS=OFF \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FORCE_CUBLAS=ON \
  -DGGML_CUDA_FA=OFF \
  -DGGML_CUDA_GRAPHS=OFF \
  -DGGML_CUDA_NO_VMM=ON \
  -DGGML_AVX=ON \
  -DGGML_AVX2=OFF \
  -DGGML_FMA=OFF

cmake --build "${BUILD_DIR}" --config Release -j"${JOBS}"

echo
echo "Tesla K40 build complete."
echo "Copy the generated libggml*.so files from ${BUILD_DIR}/bin into a copied LM Studio CUDA backend folder."
