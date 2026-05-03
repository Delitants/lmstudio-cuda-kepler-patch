# CUDA Backend for Tesla K40 / K80 (Kepler) in LM Studio

This guide covers building and installing a **CUDA** backend for LM Studio that supports NVIDIA Kepler datacenter GPUs (Tesla K40, K80) on modern Linux distributions.

## Why CUDA instead of Vulkan?

- **Higher throughput**: Native cuBLAS GEMM is faster than Vulkan compute on NVIDIA hardware.
- **Better compatibility**: The CUDA path is more mature in `llama.cpp`.
- **Multi-GPU support**: Easily co-exist with newer GPUs (e.g., GTX 1070 + K40).

## Supported Hardware

| GPU | Architecture | Compute Capability | Status |
|-----|-------------|-------------------|--------|
| Tesla K40 | Kepler | 3.5 | Supported |
| Tesla K80 | Kepler | 3.7 | Supported |
| GTX 1070 / 1080 | Pascal | 6.1 | Supported (multi-arch build) |
| GTX 980 Ti | Maxwell | 5.2 | Likely compatible (untested) |

---

## Prerequisites

### 1. NVIDIA Driver (470.x only)

Kepler GPUs require **driver 470.x or earlier**. Newer drivers drop Kepler support.

**Recommended:** `NVIDIA-Linux-x86_64-470.256.02.run`

Download from NVIDIA and install in runlevel 3:

```bash
sudo systemctl set-default multi-user.target
sudo reboot
# After reboot:
sudo systemctl stop gdm3
chmod +x NVIDIA-Linux-x86_64-470.256.02.run
sudo ./NVIDIA-Linux-x86_64-470.256.02.run --no-opengl-files
sudo systemctl set-default graphical.target
sudo reboot
```

### 2. Disable Wayland

Wayland and the 470 driver do not coexist well. Switch to **X11**:

```bash
sudo nano /etc/gdm3/custom.conf
# Uncomment: WaylandEnable=false
sudo reboot
```

### 3. Install CUDA 11.8

Kepler is dropped in CUDA 12. Use CUDA 11.8:

```bash
# Ubuntu / Debian (apt method)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-11-8
```

Add to your `~/.bashrc`:

```bash
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

**Verify:**

```bash
nvcc --version
# Expected: release 11.8
```

### 4. Host Compiler (gcc-11)

CUDA 11.8 does not support gcc-15. Install gcc-11:

```bash
sudo apt install gcc-11 g++-11
```

### 5. glibc Patch (Ubuntu 25.10+ only)

Modern glibc (2.41+) breaks CUDA 11.8 math headers. Run this once:

```bash
sudo sed -i 's/__MATHCALL_NARROW (\(__\|__f\|__w\|__f128\)/__MATHCALL_NARROW ( __\/__f\/__w\/__f128 /g' /usr/local/cuda-11.8/targets/x86_64-linux/include/crt/math_functions.h
```

---

## Build llama.cpp for Kepler

### 1. Get the exact source LM Studio expects

LM Studio v2.14.0 uses llama.cpp tag `b8861`:

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
git fetch --tags origin
git checkout b8861
```

### 2. Run the build script from this repo

```bash
cd /path/to/this/repo
chmod +x build-scripts/build_cuda_k40_avx1.sh
./build-scripts/build_cuda_k40_avx1.sh /path/to/llama.cpp
```

This will:
- Verify your CUDA version
- Apply the Kepler batched-GEMM patch automatically
- Build for `sm_35;sm_37;sm_61` (add more archs if you have other GPUs)
- Produce libraries in `llama.cpp/build-cuda-k40-avx1/bin/`

**To target only Kepler:**

```bash
CUDA_ARCH="35;37" ./build-scripts/build_cuda_k40_avx1.sh /path/to/llama.cpp
```

**To include additional GPUs** (e.g., Ampere RTX 3060):

```bash
CUDA_ARCH="35;37;61;75;86" ./build-scripts/build_cuda_k40_avx1.sh /path/to/llama.cpp
```

---

## Integrate into LM Studio

### 1. Locate LM Studio backends

```bash
ls ~/.lmstudio/extensions/backends/
```

You should see something like:
```
llama.cpp-linux-x86_64-nvidia-cuda-avx2-2.14.0
```

### 2. Back up and duplicate the stock CUDA backend

```bash
cd ~/.lmstudio/extensions/backends/
cp -r llama.cpp-linux-x86_64-nvidia-cuda-avx2-2.14.0 \
     llama.cpp-linux-x86_64-nvidia-cuda-avx2-2.14.0.stock-backup

cp -r llama.cpp-linux-x86_64-nvidia-cuda-avx2-2.14.0 \
     llama.cpp-linux-x86_64-nvidia-cuda-avx2-k40k80-2.14.0
```

### 3. Replace only the llama.cpp libraries

```bash
cd llama.cpp-linux-x86_64-nvidia-cuda-avx2-k40k80-2.14.0

cp /path/to/llama.cpp/build-cuda-k40-avx1/bin/libggml-base.so* .
cp /path/to/llama.cpp/build-cuda-k40-avx1/bin/libggml-cpu.so* .
cp /path/to/llama.cpp/build-cuda-k40-avx1/bin/libggml-cuda.so* .
cp /path/to/llama.cpp/build-cuda-k40-avx1/bin/libggml.so* .
cp /path/to/llama.cpp/build-cuda-k40-avx1/bin/libllama.so* .
```

**Do NOT touch:**
- `llm_engine_cuda.node`
- `liblmstudio_bindings_cuda.node`
- `libllm_engine.so`
- `liblmstudiocore.so`
- `libmtmd.so`
- `libggml_llamacpp.so`
- `display-data.json`

### 4. Update `backend-manifest.json`

Use `lm-studio-manifest/backend-manifest-cuda.json` as reference.

Critical changes from stock:

```json
{
  "cpu": {
    "instruction_set_extensions": ["AVX"]
  },
  "name": "llama.cpp-linux-x86_64-nvidia-cuda-avx2-k40k80"
}
```

- `instruction_set_extensions`: Must be `["AVX"]` (not `AVX2`) if your CPU lacks AVX2.
- `name`: Change to something unique so LM Studio shows it correctly.

### 5. Optional: Update `display-data.json`

Edit `display-data.json` and change `displayName` to something friendly:

```json
{
  "displayName": "CUDA llama.cpp PATCHED K40-K80 (Linux)"
}
```

### 6. Restart LM Studio

LM Studio only scans backends at startup. **Close and reopen LM Studio completely.**

---

## Troubleshooting

### `CUBLAS_STATUS_ARCH_MISMATCH` on Kepler

This happens when the code uses `cublasGemmStridedBatchedEx` or `cublasGemmBatchedEx`, which do not exist for sm_35. Our patch replaces these with `cublasSgemmBatched` on Kepler.

If you see this:
1. Make sure the patch was applied before building.
2. Make sure you rebuilt and copied the new `libggml-cuda.so`.

### `undefined symbol: ggml_cuda_op_mul_mat_vec_f`

You built from the wrong llama.cpp commit. Checkout tag `b8861` and rebuild.

### `undefined symbol: ggml_backend_init`

You replaced LM Studio wrapper `.node` files. Only replace the `libggml*.so` and `libllama.so` files.

### LM Studio does not show the backend

- Check that `backend-manifest.json` is valid JSON.
- Ensure `cpu.instruction_set_extensions` matches your CPU.
- **Restart LM Studio** — it does not hot-reload backends.

### CMake fails with "No CUDA toolset found"

Ensure `nvcc` is in PATH:

```bash
export PATH=/usr/local/cuda-11.8/bin:$PATH
```

### Build fails with gcc version errors

CUDA 11.8 requires gcc ≤ 11. Set explicitly:

```bash
CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 ./build-scripts/build_cuda_k40_avx1.sh /path/to/llama.cpp
```

---

## Runtime Tips for Kepler

- **Batch size**: Keep conservative (256–384) to avoid out-of-memory on K40's 12 GB VRAM.
- **Flash Attention**: Disable in LM Studio runtime settings when available. Kepler does not support Flash Attention.
- **Context length**: Start with 2048–4096, then scale up gradually.
- **Mixed GPUs**: If you have a newer GPU alongside K40, LM Studio will automatically assign layers. The multi-arch build ensures both GPUs can run the same kernel code.

---

## License

- Build scripts and patches: MIT
- `llama.cpp`: MIT (upstream)
- LM Studio wrapper binaries: property of Element Labs; not redistributed here
- NVIDIA driver and CUDA: proprietary NVIDIA licenses; not redistributed here
