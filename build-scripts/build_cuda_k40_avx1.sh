#!/usr/bin/env bash
set -euo pipefail

# LM Studio CUDA Backend Build Script — Kepler (K40/K80) + Modern Multi-GPU
# Builds llama.cpp shared libs for Linux + LM Studio:
# - GPU: NVIDIA Tesla K40/K80 (SM 3.5/3.7) and co-existing modern GPUs
# - CPU: AVX baseline (no AVX2 required)
#
# IMPORTANT:
# - Kepler is NOT supported by CUDA 12+. You MUST use CUDA 11.x (recommended 11.8).
# - The llama.cpp source MUST be checked out to tag b8861 (LM Studio v2.14.0).
# - A source patch is automatically applied to fix batched GEMM on Kepler.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="${SCRIPT_DIR}/ggml-cuda-kepler-batched-gemm.patch"

BUILD_DIR="${BUILD_DIR:-build-cuda-k40-avx1}"
CUDA_ARCH="${CUDA_ARCH:-35;37;61}"
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
    echo "Usage: $0 /path/to/llama.cpp"
    exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
    echo "[ERROR] cmake is required."
    exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
    echo "[ERROR] nvcc not found. Install CUDA Toolkit 11.x first."
    echo "  Ubuntu/Debian: sudo apt install cuda-toolkit-11-8"
    exit 1
fi

NVCC_VERSION="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
if [[ -z "${NVCC_VERSION}" ]]; then
    echo "[ERROR] Could not detect nvcc version."
    exit 1
fi

NVCC_MAJOR="${NVCC_VERSION%%.*}"
if (( NVCC_MAJOR >= 12 )); then
    echo "[ERROR] Detected nvcc ${NVCC_VERSION}. Tesla K40/K80 (sm_35/sm_37) needs CUDA 11.x."
    echo "  Install CUDA 11.8 and ensure /usr/local/cuda-11.8/bin is first in PATH."
    exit 1
fi

# Verify GCC compatibility (CUDA 11.8 needs gcc <= 11)
CC_BIN="${CC:-$(pick_tool gcc-11 gcc-10 gcc || true)}"
CXX_BIN="${CXX:-$(pick_tool g++-11 g++-10 g++ || true)}"

if [[ -z "${CC_BIN}" || -z "${CXX_BIN}" ]]; then
    echo "[ERROR] Could not find a suitable gcc/g++ for CUDA ${NVCC_VERSION}."
    echo "  Install gcc-11 / g++-11: sudo apt install gcc-11 g++-11"
    exit 1
fi

echo "=== LM Studio CUDA Backend Builder (Kepler + Multi-GPU) ==="
echo "Source directory: ${ROOT_DIR}"
echo "Build directory:  ${BUILD_DIR}"
echo "nvcc version:     ${NVCC_VERSION}"
echo "Host compiler:    ${CC_BIN} / ${CXX_BIN}"
echo "CUDA arch list:   ${CUDA_ARCH}"
echo ""

# Check tag/commit
cd "${ROOT_DIR}"
CURRENT_TAG="$(git describe --tags --exact-match 2>/dev/null || git rev-parse --short HEAD)"
EXPECTED_TAG="b8861"
if [[ "${CURRENT_TAG}" != "${EXPECTED_TAG}" ]]; then
    echo "[WARN] llama.cpp is at '${CURRENT_TAG}', but LM Studio v2.14.0 expects '${EXPECTED_TAG}'."
    echo "  Run the following in your llama.cpp source folder before building:"
    echo "    git fetch --tags origin"
    echo "    git checkout ${EXPECTED_TAG}"
    read -rp "  Continue anyway? [y/N] " yn
    [[ "${yn,,}" == "y" ]] || exit 1
fi

# Apply patch
if [[ -f "${PATCH_FILE}" ]]; then
    echo "[INFO] Applying Kepler batched GEMM patch..."
    if git apply --check "${PATCH_FILE}" 2>/dev/null; then
        git apply "${PATCH_FILE}"
        echo "[INFO] Patch applied."
    else
        echo "[WARN] Patch could not be applied cleanly (already applied?). Skipping."
    fi
else
    echo "[WARN] Patch file not found: ${PATCH_FILE}"
fi

if command -v nproc >/dev/null 2>&1; then
    JOBS="$(nproc)"
else
    JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
fi

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
  -DGGML_CUDA_FA=OFF \
  -DGGML_CUDA_GRAPHS=OFF \
  -DGGML_CUDA_NO_VMM=ON \
  -DGGML_AVX=ON \
  -DGGML_AVX2=OFF \
  -DGGML_FMA=OFF

cmake --build "${BUILD_DIR}" --config Release -j"${JOBS}"

echo ""
echo "=== BUILD COMPLETE ==="
echo "Shared libraries are in: ${BUILD_DIR}/bin/"
echo ""
echo "Files to copy into your LM Studio backend folder:"
echo "  - ${BUILD_DIR}/bin/libggml-base.so"
echo "  - ${BUILD_DIR}/bin/libggml-cpu.so"
echo "  - ${BUILD_DIR}/bin/libggml-cuda.so"
echo "  - ${BUILD_DIR}/bin/libggml.so"
echo "  - ${BUILD_DIR}/bin/libllama.so"
