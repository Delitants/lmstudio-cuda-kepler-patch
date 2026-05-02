# Patched Vulkan Runtime for NVIDIA Tesla K40 / K80 (Kepler)

This repository enables NVIDIA Kepler-architecture datacenter GPUs (Tesla K40, K80, and compatible models) to run LLM inference through the **Vulkan backend** in LM Studio and upstream `llama.cpp`.

Kepler is no longer supported by CUDA 12.x, and the stock LM Studio Vulkan backend is compiled for AVX2-only systems. This project provides build scripts, CMake patches, and LM Studio backend manifests to restore compatibility.

---

## Supported Hardware

| GPU | Architecture | Compute Capability | Status |
|-----|-------------|-------------------|--------|
| Tesla K40 | Kepler | 3.5 | Supported |
| Tesla K80 | Kepler | 3.7 | Supported |
| GRID K520 | Kepler | 3.0 | Likely compatible (untested) |
| GTX 780 Ti | Kepler | 3.5 | Likely compatible (untested) |

---

## Critical: NVIDIA Driver Version

Kepler GPUs require **driver 470.x or earlier**. Newer drivers (495+) drop Kepler support entirely.

**Recommended driver:** `NVIDIA-Linux-x86_64-470.256.02.run`

### Driver download

Because NVIDIA drivers are proprietary and cannot be redistributed here, download the official installer from NVIDIA:

- **File:** `NVIDIA-Linux-x86_64-470.256.02.run`
- **SHA-256:** `d6451862deb695bb0447f3b7cd6268f73e81168c10e2c10597ff3fa01349b1de`
- **Download:** https://www.nvidia.com/Download/driverResults.aspx/216951/en-us/

### Driver installation (runlevel 3)

```bash
# 1. Disable Nouveau and switch to text mode
sudo systemctl set-default multi-user.target
sudo reboot

# 2. After reboot, stop display manager
sudo systemctl stop gdm3   # or sddm, lightdm

# 3. Run the installer
chmod +x NVIDIA-Linux-x86_64-470.256.02.run
sudo ./NVIDIA-Linux-x86_64-470.256.02.run --no-opengl-files

# 4. Re-enable graphical target
sudo systemctl set-default graphical.target
sudo reboot
```

---

## Disable Wayland on Modern Ubuntu

Wayland and the 470-series proprietary driver do not coexist well. You must switch to **X11**.

### Ubuntu 22.04 / 24.04 / newer

Edit the GDM configuration:

```bash
sudo nano /etc/gdm3/custom.conf
```

Uncomment or add the line:

```ini
[daemon]
WaylandEnable=false
```

Save and reboot:

```bash
sudo reboot
```

Verify you are on X11:

```bash
echo $XDG_SESSION_TYPE
# Expected output: x11
```

If you use **SDDM** (KDE Plasma) or **LightDM**, the equivalent setting is typically in:

- `/etc/sddm.conf` or `/etc/sddm.conf.d/` — set `DisplayServer=x11`
- `/etc/lightdm/lightdm.conf` — no extra change needed, X11 is default

---

## Build the Vulkan Backend

### 1. Prerequisites

```bash
# Ubuntu / Debian
sudo apt update
sudo apt install -y build-essential cmake git vulkan-tools libvulkan-dev

# Verify Vulkan is visible to the NVIDIA driver
vulkaninfo | grep "deviceName"
# Should list your K40/K80
```

### 2. Get llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
git checkout b2xxx   # Use a stable tag near LM Studio's version (e.g., b3700-b3800)
```

### 3. Apply the CMake patch

```bash
git apply build-scripts/cmake_vulkan_fix.patch
```

This patch disables `GGML_VULKAN_MXFP`, which is not yet implemented for the Vulkan backend and causes missing-symbol build failures.

### 4. Build

```bash
cd /path/to/this/repo
./build-scripts/build_vulkan.sh /path/to/llama.cpp
```

Output libraries will be in `llama.cpp/build-vulkan/bin/`:
- `libggml-base.so`
- `libggml-cpu.so`
- `libggml-vulkan.so`
- `libllama.so`

---

## Integrate into LM Studio

### 1. Locate LM Studio backends

```bash
ls ~/.lmstudio/extensions/backends/
```

### 2. Back up the stock Vulkan backend

```bash
cd ~/.lmstudio/extensions/backends/
cp -r llama.cpp-linux-x86_64-vulkan-avx2-2.14.0 \
     llama.cpp-linux-x86_64-vulkan-avx2-2.14.0.stock-backup-before-k40k80
```

### 3. Create the K40/K80 backend folder

```bash
cp -r llama.cpp-linux-x86_64-vulkan-avx2-2.14.0 \
     llama.cpp-linux-x86_64-vulkan-avx2-k40k80-2.14.0
```

### 4. Replace only the llama.cpp shared libraries

Do **not** replace `.node` files or LM Studio wrapper binaries.

```bash
cd llama.cpp-linux-x86_64-vulkan-avx2-k40k80-2.14.0
cp /path/to/llama.cpp/build-vulkan/bin/libggml*.so .
cp /path/to/llama.cpp/build-vulkan/bin/libllama.so .
```

### 5. Update `backend-manifest.json`

Use the provided manifest in `lm-studio-manifest/backend-manifest.json` as a reference.

Key changes from stock:

```json
{
  "cpu": {
    "instruction_set_extensions": ["AVX"]
  },
  "name": "llama.cpp-linux-x86_64-vulkan-avx2-k40k80"
}
```

The critical change is switching `instruction_set_extensions` from `["AVX2"]` to `["AVX"]`, which allows the backend to load on older CPUs that lack AVX2 (common in K40/K80 server platforms).

### 6. Restart LM Studio

Launch LM Studio, open **Runtime > Loaded Models**, and verify the loaded backend name reflects your custom folder.

---

## CUDA Alternative (Native, Faster)

If you have CUDA 11.x installed, you can build a native CUDA backend instead of Vulkan. This typically yields higher throughput on Kepler.

```bash
./build-scripts/build_cuda_k40_avx1.sh /path/to/llama.cpp
```

Requirements:
- CUDA Toolkit 11.8 (nvcc 11.x)
- `gcc-10` or `gcc-11` as host compiler

See `docs/tesla-k40-lmstudio.md` for full CUDA integration steps.

---

## Demo Screenshots

See `screenshots/` for LM Studio running on a Tesla K40 via the patched Vulkan backend.

---

## Troubleshooting

### `vkCreateInstance` fails
- Ensure `vulkan-tools` and the NVIDIA Vulkan ICD are installed.
- Check `ls /usr/share/vulkan/icd.d/` for `nvidia_icd.json`.
- Run `vulkaninfo` to confirm the GPU is enumerated.

### LM Studio shows "Backend failed to load"
- Verify `cpu.instruction_set_extensions` is `["AVX"]` and not `["AVX2"]` in your manifest.
- Check LM Studio logs for missing `.so` symbols.

### Build fails with MXFP4 errors
- You skipped the CMake patch. Re-apply `cmake_vulkan_fix.patch` before building.

### Driver 470 installation aborts on modern kernels
- Pass `--disable-nouveau` and `--no-drm` flags if needed.
- For kernels newer than 6.5, you may need the [nvidia-470 dkms patch](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/others) or use the distro-packaged `nvidia-driver-470` if available.

---

## License

- Build scripts and patches: MIT (or your preferred license)
- `llama.cpp`: MIT (upstream)
- LM Studio wrapper binaries and `.node` files: property of Element Labs / LM Studio; not redistributed here
- NVIDIA driver: proprietary NVIDIA license; not redistributed here
