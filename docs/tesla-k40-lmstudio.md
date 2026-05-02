# Tesla K40 (Kepler) on Linux with LM Studio

This workflow builds `llama.cpp` shared libraries for NVIDIA Tesla K40 (compute capability 3.5) and integrates them into LM Studio.

## Why this is separate from normal CUDA builds

- Tesla K40 is Kepler (`sm_35`).
- CUDA 12.x does not support Kepler.
- Use CUDA 11.x (recommended: 11.8).

## 1) Prerequisites

You need:
- NVIDIA driver that supports your K40
- CUDA Toolkit 11.x (`nvcc --version` must report 11.x)
- `cmake`, `git`, `gcc`, `g++`

Quick checks:
```bash
nvidia-smi
nvcc --version
```

## 2) Build llama.cpp libraries for K40

From this repo folder:

```bash
cd "Generate Backends/Linux"
chmod +x build_cuda_k40_avx1.sh
./build_cuda_k40_avx1.sh /path/to/llama.cpp
```

This generates Linux shared libraries in:
```bash
build-cuda-k40-avx1/bin
```

## 3) Prepare a custom LM Studio backend folder

1. Go to:
```bash
~/.lmstudio/extensions/backends
```
2. Copy an existing Linux CUDA backend folder to a new folder name ending in `avx1`.
3. Keep all `.node` files unchanged.

## 4) Replace only ggml shared libraries

Copy these files from your K40 build output into the copied backend folder:
- `libggml-base.so`
- `libggml-cpu.so`
- `libggml-cuda.so`

Do not replace LM Studio wrapper files (`*.node` and backend launcher binaries).

## 5) Update `backend-manifest.json`

In your copied backend folder:
- Set `cpu.instruction_set_extensions` to `["AVX"]`
- Set `gpu.targets` to `["3.5"]`
- Update `name` to your custom backend name
- If needed, set `minimum_driver_version` to match your CUDA 11.x baseline

## 6) Runtime tips for Kepler

- Keep batch sizes conservative (for example 256-384) to reduce instability.
- Disable Flash Attention in runtime settings when available.
- Start with smaller GGUF models and lower context length, then scale up gradually.
