# Building llama.cpp on Linux with CUDA + AVX1 (Pascal and newer)

This guide is for NVIDIA GPUs with compute capability 6.1+ (Pascal to Ada).

If you need Tesla K40 (Kepler / SM 3.5), use:
- [tesla-k40-lmstudio.md](tesla-k40-lmstudio.md)
- `build_cuda_k40_avx1.sh`

## Install dependencies

### Arch / Artix
```bash
sudo pacman -S base-devel gcc cmake git cuda
```

### Ubuntu / Debian
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git
# Install CUDA toolkit (12.x for modern GPUs)
```

## Build

Run from this repo folder and pass your `llama.cpp` path:

```bash
cd "Generate Backends/Linux"
./build_cuda_avx1.sh /path/to/llama.cpp
```

The script builds shared libraries for LM Studio and targets:
- 61 (Pascal)
- 70/72 (Volta)
- 75 (Turing)
- 80/86/87 (Ampere)
- 89 (Ada)

## Install into LM Studio

1. Copy an existing Linux CUDA backend folder and rename it to an `avx1` variant.
2. Replace only the generated `libggml*.so` files in that copied folder.
3. Keep `.node` files and wrapper binaries unchanged.
4. Update `backend-manifest.json`:
- `cpu.instruction_set_extensions` -> `["AVX"]`
- `gpu.targets` -> architectures you compiled
- `name` -> your backend name
