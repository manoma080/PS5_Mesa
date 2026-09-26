# RADV on the PlayStation 5

This is my fork of Mesa 26.2.0 (branch `ps5-port`). It runs RADV, Mesa's
Vulkan driver for AMD GPUs, on the PlayStation 5's GPU from a homebrew title.
RADV itself is unchanged. The fork adds a *winsys* for the console, the layer
RADV keeps the kernel behind, plus the build switches that keep Linux's DRM
out of a console build.

It is route B of `PS5_Vulkan/docs/VULKAN_1_4_PLAN.md`. The goal is a genuine
Vulkan 1.4 driver that passes the Khronos CTS on the console. ps5vk remains
the shipping driver until this one passes the same titles.

## The winsys (`src/amd/vulkan/winsys/ps5`)

| Part | What it does |
| --- | --- |
| `radv_ps5_platform.c` | GPU-visible direct memory. The CPU and the GPU see each allocation at one address. 32-bit buffers go in the shaders' 4 GiB window, other buffers in a 256 GiB region at 0x4000000000. It also handles submission through `sceAgcDriverSubmitDcb` plus the suspend point, and CPU cache flushes. A host build models all of this, so RADV runs on a PC for everything that doesn't need the GPU. |
| `radv_ps5_winsys.c` | The GPU description, in the amdgpu kernel's own form, completed by `ac_gpu_info`'s derivations. It is a GFX10.3 shader core addressed as the console's tile maps were measured: sixteen pipes and no RB+, which is the Navi10 configuration. It also holds the process-wide queue, which the completion markers track. |
| `radv_ps5_cs.c` | Command streams kept in CPU memory. A submission copies all of them into one AGC submission, because the console faults a PM4 `INDIRECT_BUFFER` into title memory. The submission ends with a cache-flushing `RELEASE_MEM` that writes its sequence number. |
| `radv_ps5_bo.c` | Buffers: one direct-memory allocation each, always resident. |
| `radv_ps5_sync.c` | A binary sync object signalled by a submission's sequence number. Mesa's runtime builds timeline semaphores over it. |

What the winsys relies on was measured on the console in `PS5_Vulkan`, where
ps5vk's runs established it. The values it has not yet measured, the shader
engine and compute-unit counts, are marked where they are set.

## Building

`meson setup` with `-Dradv-winsys=ps5`, for the console with
`PS5_Vulkan/tooling/radv/ps5-cross.ini` and a static default library.
`PS5_Vulkan/tools/build-radv.sh` builds the revision `PS5_Vulkan` pins and
records it, and `PS5_Vulkan/tools/radv-link.sh` links the archive into a
title. The payload SDK fork's platform layer supplies the AGC declarations
and the libc functions the console doesn't export.

A Linux host build of the same option (a shared ICD for the Khronos loader)
models the console. It runs API-level tests and the CTS's API groups on a PC.

## Licence

Mesa's licences apply. The files this fork adds are MIT, like the code
around them.
