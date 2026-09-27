/*
 * Copyright © 2026 Mihawk
 *
 * SPDX-License-Identifier: MIT
 */

#include "radv_ps5_winsys.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "addrlib/src/amdgpu_asic_addr.h"
#include "tools/radv_debug.h"
#include "util/os_time.h"
#include "util/u_atomic.h"
#include "util/u_math.h"
#include "ac_linux_drm.h"
#include "sid.h"

/* ------------------------------------------------------------------ queue */

static struct radv_ps5_queue radv_ps5_queue_storage;
static struct radv_ps5_queue *radv_ps5_queue_instance;
static once_flag radv_ps5_queue_once = ONCE_FLAG_INIT;

static void
radv_ps5_queue_create(void)
{
   struct radv_ps5_queue *const queue = &radv_ps5_queue_storage;
   if (!radv_ps5_platform_init())
      return;
   /* The words the GPU runs and the marker it writes live in the window, where
    * ps5vk's submission buffer has always been. */
   if (!radv_ps5_memory_alloc(RADV_PS5_RING_BYTES, RADV_PS5_LARGE_BYTES, true, &queue->ring))
      return;
   if (!radv_ps5_memory_alloc(RADV_PS5_PAGE_BYTES, RADV_PS5_PAGE_BYTES, true, &queue->marker_memory)) {
      radv_ps5_memory_free(&queue->ring);
      return;
   }
   queue->marker = (volatile uint32_t *)queue->marker_memory.cpu;
   *queue->marker = 0;
   radv_ps5_cpu_flush((const void *)queue->marker, sizeof(*queue->marker));
   simple_mtx_init(&queue->submit_lock, mtx_plain);
   mtx_init(&queue->sync_lock, mtx_plain);
   cnd_init(&queue->sync_cond);
   radv_ps5_queue_instance = queue;
}

struct radv_ps5_queue *
radv_ps5_queue_get(void)
{
   call_once(&radv_ps5_queue_once, radv_ps5_queue_create);
   return radv_ps5_queue_instance;
}

uint64_t
radv_ps5_queue_poll(struct radv_ps5_queue *queue)
{
   const uint64_t submitted = p_atomic_read(&queue->submitted_seq);
   /* The marker holds the low 32 bits of the newest sequence the GPU has run.
    * Its CPU cache line may hold an older value, so it is evicted before the
    * read, as ps5vk's marker poll does. */
   radv_ps5_cpu_flush((const void *)queue->marker, sizeof(*queue->marker));
   const uint32_t marker = *queue->marker;
   /* The GPU never lags the CPU by 2^32 submissions, so the newest sequence
    * with these low bits that is not past the submitted one is the one. */
   const uint64_t completed = submitted - (uint32_t)((uint32_t)submitted - marker);
   uint64_t known = p_atomic_read(&queue->completed_seq);
   while (completed > known) {
      const uint64_t seen = p_atomic_cmpxchg(&queue->completed_seq, known, completed);
      if (seen == known) {
         known = completed;
         break;
      }
      known = seen;
   }
   return known;
}

/* A marker that makes no progress for this long while work is outstanding
 * means the GPU is not running it: the device is lost (ps5vk gives up after 2 s;
 * a conformance run's heaviest cases need more). */
#define RADV_PS5_STALL_NS (UINT64_C(10) * 1000000000)
/* Before sleeping, the marker is checked in a loop for this long: most waits
 * end well inside a millisecond (PS5_Vulkan R36). */
#define RADV_PS5_SPIN_NS UINT64_C(1500000)

bool
radv_ps5_queue_wait_seq(struct radv_ps5_queue *queue, uint64_t seq, uint64_t abs_timeout_ns)
{
   uint64_t completed = radv_ps5_queue_poll(queue);
   if (completed >= seq)
      return true;
   uint64_t now = os_time_get_nano();
   const uint64_t spin_until = now + RADV_PS5_SPIN_NS;
   uint64_t progress_at = now;
   uint64_t last = completed;
   for (;;) {
      completed = radv_ps5_queue_poll(queue);
      if (completed >= seq)
         return true;
      now = os_time_get_nano();
      if (now >= abs_timeout_ns)
         return false;
      if (completed != last) {
         last = completed;
         progress_at = now;
      } else if (now - progress_at > RADV_PS5_STALL_NS) {
         fprintf(stderr, "radv/ps5: the GPU has not reached submission %" PRIu64 " (at %" PRIu64 ") in %u s\n", seq,
                 completed, (unsigned)(RADV_PS5_STALL_NS / 1000000000));
         return false;
      }
      if (now < spin_until)
         __builtin_ia32_pause();
      else
         radv_ps5_sleep_us(1000);
   }
}

/* ----------------------------------------------------------- the GPU itself */

/* The console's GPU as the winsys describes it to RADV, in the form the amdgpu
 * kernel reports a GPU (Mesa's recorded devices, src/amd/common/amdgpu_devices.c,
 * use the same form for AddrLib's tests), then completed by ac_gpu_info's own
 * derivations.
 *
 * Measured on my console (PS5_Vulkan):
 * - shaders compiled by ACO for GFX10.3 (CHIP_NAVI21) run: every psbc stage
 *   since M5 A, and compute in waves of 32;
 * - tiled images follow AddrLib's GFX10 swizzles with sixteen pipes, a 256-byte
 *   pipe interleave and no RB+ -- the Navi10 configuration -- texel for texel in
 *   both 64 KiB modes (docs/HARDWARE_FINDINGS.md, tools/mip-layout-oracle.cpp);
 * - the GPU timestamp counts at 100 MHz (V0-query, pid 143);
 * - a full-screen draw's occlusion count is a sixteenth of its samples, which
 *   is one counter per render backend if there are sixteen (to be proved by S6).
 *
 * Not yet measured, and marked where they are set: the shader engine and
 * compute-unit counts (the public PS5 figures: two shader engines, 36 active
 * compute units of 40) and the render-backend layout. Probe S3 measures them;
 * until then they only size scratch and a few hardware limits. */
#define RADV_PS5_EXTERNAL_REV 0x28 /* the NAVI21 range: GFX10.3 */
/* GFX1013's range: AddrLib's non-RB+ GFX10 swizzles, with the depth/stencil
 * mipmap fix Navi10 lacks. With a Navi10 revision a 256x256 D16 image with
 * mips was laid out in three 64 KiB blocks, its 128x128 level in the mip
 * tail, while the depth block wrote level 0 into a fourth (a write past the
 * image: dEQP-VK.glsl.texture_functions.texture.sampler2dshadow_*). */
#define RADV_PS5_ADDRLIB_REV 0x82
#define RADV_PS5_GB_ADDR_CONFIG 0x00100044

static void
radv_ps5_describe_gpu(struct radeon_info *info, bool compiler_compat_mode)
{
   const struct drm_amdgpu_info_hw_ip gfx_ip = {
      .hw_ip_version_major = 10,
      .hw_ip_version_minor = 3,
      .ib_start_alignment = 32,
      .ib_size_alignment = 32,
      .available_rings = 0x1,
      .ip_discovery_version = 0xa0300,
   };
   struct drm_amdgpu_info_device dev = {
      .device_id = 0,
      .chip_rev = 0,
      .external_rev = RADV_PS5_EXTERNAL_REV,
      .family = FAMILY_NV,
      /* Provisional until S3. */
      .num_shader_engines = 2,
      .num_shader_arrays_per_engine = 2,
      .cu_active_number = 36,
      .cu_ao_mask = 0x1ff,
      .cu_bitmap = {{0x1ff, 0x1ff}, {0x1ff, 0x1ff}},
      .cu_ao_bitmap = {{0x1ff, 0x1ff}, {0x1ff, 0x1ff}},
      .num_cu_per_sh = 10,
      .enabled_rb_pipes_mask = 0xffff,
      .num_rb_pipes = 16,
      .num_tcc_blocks = 16,
      /* The 100 MHz timestamp clock, in kHz. */
      .gpu_counter_freq = 100000,
      .max_engine_clock = UINT64_C(2230000),
      .min_engine_clock = UINT64_C(500000),
      .max_memory_clock = UINT64_C(1750000),
      .min_memory_clock = UINT64_C(96000),
      .num_hw_gfx_contexts = 8,
      .ids_flags = 0,
      .virtual_address_offset = UINT64_C(0x200000),
      .virtual_address_max = RADV_PS5_GPU_ADDRESS_LIMIT,
      .virtual_address_alignment = RADV_PS5_PAGE_BYTES,
      .pte_fragment_size = 2097152,
      .gart_page_size = RADV_PS5_PAGE_BYTES,
      .vram_type = AMDGPU_VRAM_TYPE_GDDR6,
      .vram_bit_width = 256,
      .gc_double_offchip_lds_buf = 1,
      .wave_front_size = 32,
      .num_shader_visible_vgprs = 1024,
      .gs_vgt_table_depth = 32,
      .gs_prim_buffer_depth = 1792,
      .max_gs_waves_per_vgt = 32,
   };
   const uint64_t pool = radv_ps5_memory_pool_bytes();
   const struct drm_amdgpu_memory_info memory = {
      .vram = {.total_heap_size = pool},
      .cpu_accessible_vram = {.total_heap_size = pool},
      .gtt = {0},
   };
   const struct amdgpu_gpu_info gpu_info = {
      .gb_addr_cfg = RADV_PS5_GB_ADDR_CONFIG,
   };

   memset(info, 0, sizeof(*info));
   /* No kernel stands behind this description; the version is the lowest
    * RADV's amdgpu path accepts, so no feature is assumed from a newer kernel. */
   info->drm_major = 3;
   info->drm_minor = 54;
   info->is_amdgpu = false;

   ac_fill_hw_ip_info(info, &dev, AMD_IP_GFX, &gfx_ip);
   info->ip[AMD_IP_GFX].num_instances = 1;
   ac_identify_chip(info, &dev);
   snprintf(info->marketing_name, sizeof(info->marketing_name), "PlayStation 5 GPU");
   ac_fill_memory_info(info, &dev, &memory);
   ac_fill_hw_info(info, &dev);
   ac_fill_tiling_info(info, &gpu_info);
   ac_fill_feature_info(info, &dev);
   ac_fill_bug_info(info);
   ac_fill_tess_info(info);
   ac_fill_compiler_info(info, &dev, compiler_compat_mode);
   /* The integer dot-product instructions do not compute what they should
    * here: NGG's workgroup repack sums its counts with v_dot4_u32_u8, and
    * with it a geometry shader whose vertex count is not constant, and a
    * tessellated patch that NGG culling repacked, drew nothing. Without
    * them (the v_msad_u8 fallback, as on NAVI10) both draw every texel (the
    * RADV smoke title's geometry and tessellation checks). */
   info->compiler_info.has_accelerated_dot_product = false;
   /* Every wave of a shader using scratch_* instructions faulted (MEMVIOL) a
    * few instructions in, where it had set FLAT_SCRATCH with s_setreg; the
    * same shaders through buffer instructions, as on GFX8, run (the smoke
    * title's scratch checks, graphicsfuzz's large private arrays). */
   info->compiler_info.has_flat_scratch = false;
   /* A 128-invocation workgroup of wave64 reported one subgroup where two ran
    * (dEQP-VK.subgroups.multiple_dispatches.uniform_subgroup_size): TG_SIZE's
    * GFX10.3 wave ID (bits 20-24) reads 0, as on GFX10.1, whose ordered wave
    * ID serves instead. */
   info->compiler_info.has_cs_wave_id = false;
   /* Every acceleration structure build faulted the GPU (a write to an
    * unmapped page far past every buffer the process had, even for an empty
    * top level: dEQP-VK.ray_query.acceleration_structures.empty.*.gpu_built),
    * and host builds are not offered, so no ray tracing is reported until the
    * build runs. */
   info->compiler_info.has_image_bvh_intersect_ray = false;

   /* What the console's layout needs beyond the NAVI21 defaults. */
   info->chip_external_rev = RADV_PS5_ADDRLIB_REV;
   /* A legacy GS hung the GPU every time (dEQP-VK.geometry with
    * RADV_DEBUG=nongg: 31 of 33 cases), and neither AGC library exports a
    * way to set the GS rings it needs, as sceAgcDriverSetTFRing does the
    * tessellation factor ring. */
   info->has_legacy_gs = false;
   /* An INDIRECT_BUFFER into title memory faulted the GPU (PS5_Vulkan B8), so
    * the command buffers device-generated commands write cannot run. */
   info->has_gpu_written_ibs = false;
   /* No exported function holds the GPU at a stable power state
    * (radv_ps5_ctx_set_pstate refuses every state but none), so the profiling
    * lock VK_KHR_performance_query takes always failed
    * (dEQP-VK.query_pool.performance_query.*: VK_ERROR_UNKNOWN). */
   info->has_perf_counters = false;
   /* The second of two triangles read its per-vertex inputs rotated, (v5, v3,
    * v4) for (v3, v4, v5), with or without ROTATE_PC_PTR: this GPU's
    * parameter cache is GFX10.1's, which upstream RADV does not report
    * VK_KHR_fragment_shader_barycentric for (the RADV smoke title's
    * barycentric pair; dEQP-VK.fragment_shading_barycentric.data, triangles). */
   info->has_ps_strict_vertex_order = false;
   info->rbplus_allowed = false;
   /* A depth-only image cleared to 0 read back 1 after a draw whose
    * fragments were all discarded (dEQP-VK.dynamic_state.*.discard.depth;
    * with RADV_DEBUG=nohiz or nofastclears it passed): the TC-compatible
    * HTILE clear bug Mesa records for GFX8 and GFX1013 (the BC-250, whose
    * missing dot products this GPU shares too). This turns on its
    * workaround, ZRANGE_PRECISION 0 after a clear to 0. */
   info->has_htile_tc_z_clear_bug_without_stencil = true;
   info->has_htile_tc_z_clear_bug_with_stencil = true;
   info->has_dedicated_vram = true;
   info->all_vram_visible = true;
   info->address32_hi = RADV_PS5_ADDRESS32_HI;
   /* Timeline semaphores: the runtime builds them over the binary sync type
    * (vk_sync_timeline), with threaded submission for waits before signals. */
   info->has_timeline_syncobj = true;
   info->kernel_has_modifiers = false;
   info->has_sparse = false;
   info->has_sparse_image_3d = false;
   info->has_vm_always_valid = true;
   /* buffer_from_ptr imports nothing yet: whether a CPU allocation can be
    * made visible to the GPU is a probe still to run. */
   info->has_userptr = false;
   /* The shaders are GFX10.3's but the colour block renders E5B9G9R9 wrong:
    * every blit into it with a non-zero colour read back incorrect
    * (dEQP-VK.api.copy_and_blit.*.blit_image.all_formats.color.*.
    * e5b9g9r9_ufloat_pack32, 500 cases), while the zero colours of an
    * a8_unorm source passed. */
   info->has_rgb9e5_color_target = false;
   info->max_submitted_ibs[AMD_IP_GFX] = 1;

   /* ac_query_gpu_info's own derivations, which have no kernel input. */
   info->scratch_wavesize_granularity_shift = 10;
   info->scratch_wavesize_granularity = BITFIELD_BIT(info->scratch_wavesize_granularity_shift);
   const unsigned max_waves_per_tg = 32;
   info->max_scratch_waves =
      MAX2(32 * info->max_good_cu_per_sa * info->max_sa_per_se * info->num_se, max_waves_per_tg);
   info->has_scratch_base_registers = false;
   info->max_gflops = 128 * info->num_cu * info->max_gpu_freq_mhz / 1000;
   info->memory_bandwidth_gbps =
      DIV_ROUND_UP(info->memory_freq_mhz_effective * info->memory_bus_width / 8, 1000);
   info->instr_prefetch_distance = 3;
   info->se_tile_repeat = 32 * info->max_se;
}

void
radv_ps5_winsys_heap_usage(uint64_t *pool_bytes, uint64_t *available_bytes, uint64_t *driver_bytes)
{
   *pool_bytes = radv_ps5_memory_pool_bytes();
   *available_bytes = radv_ps5_memory_available_bytes();
   *driver_bytes = p_atomic_read(&radv_ps5_allocated_bytes);
}

VkResult
radv_ps5_winsys_query_info(uint64_t debug_flags, struct radeon_winsys_info *info)
{
   memset(info, 0, sizeof(*info));
   if (!radv_ps5_queue_get())
      return VK_ERROR_INITIALIZATION_FAILED;
   radv_ps5_describe_gpu(&info->base, !(debug_flags & RADV_DEBUG_NO_CACHE_COMPAT));
   info->syncobj_sync_type = radv_ps5_sync_type;
   /* One queue, which a title's submissions all reach: low and medium, the
    * priorities an application may always have, both run there. A priority is
    * a scheduling hint, and with one queue there is nothing to order. */
   info->global_priority_mask = BITFIELD_BIT(RADEON_CTX_PRIORITY_LOW) | BITFIELD_BIT(RADEON_CTX_PRIORITY_MEDIUM);
   /* GPU memory and the queue's syncs have no file descriptor to share. */
   info->has_external_fd = false;
   return VK_SUCCESS;
}

/* -------------------------------------------------------------- the winsys */

static uint64_t
radv_ps5_winsys_query_value(struct radeon_winsys *rws, enum radeon_value_id value)
{
   struct radv_ps5_winsys *ws = radv_ps5_winsys(rws);
   switch (value) {
   case RADEON_ALLOCATED_VRAM:
   case RADEON_ALLOCATED_VRAM_VIS:
   case RADEON_VRAM_USAGE:
   case RADEON_VRAM_VIS_USAGE:
      return p_atomic_read(&ws->allocated_vram);
   case RADEON_ALLOCATED_GTT:
   case RADEON_GTT_USAGE:
      return p_atomic_read(&ws->allocated_gtt);
   default:
      /* The console's clocks and counters are not the title's to read. */
      return 0;
   }
}

static bool
radv_ps5_winsys_read_registers(struct radeon_winsys *rws, unsigned reg_offset, unsigned num_registers, uint32_t *out)
{
   (void)rws;
   (void)reg_offset;
   (void)num_registers;
   (void)out;
   return false;
}

static bool
radv_ps5_winsys_query_gpuvm_fault(struct radeon_winsys *rws, struct radv_winsys_gpuvm_fault_info *fault_info)
{
   (void)rws;
   (void)fault_info;
   return false;
}

static VkResult
radv_ps5_ctx_create(struct radeon_winsys *rws, enum radeon_ctx_priority priority, struct radeon_winsys_ctx **rctx)
{
   /* The priorities reported (global_priority_mask) are the ones a context is
    * made with; a higher one is denied, as the extension allows
    * (dEQP-VK.api.device_init.create_device_global_priority*). */
   if (priority != RADEON_CTX_PRIORITY_MEDIUM && priority != RADEON_CTX_PRIORITY_LOW)
      return VK_ERROR_NOT_PERMITTED;
   struct radv_ps5_ctx *const ctx = calloc(1, sizeof(*ctx));
   if (!ctx)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   ctx->ws = radv_ps5_winsys(rws);
   ctx->priority = priority;
   *rctx = (struct radeon_winsys_ctx *)ctx;
   return VK_SUCCESS;
}

static void
radv_ps5_ctx_set_tess_factor_ring(struct radeon_winsys_ctx *rctx, uint64_t va, uint32_t size)
{
   struct radv_ps5_ctx *const ctx = (struct radv_ps5_ctx *)rctx;
   ctx->tess_factor_ring_va = va;
   ctx->tess_factor_ring_size = size;
}

static void
radv_ps5_ctx_destroy(struct radeon_winsys_ctx *rctx)
{
   free(rctx);
}

static bool
radv_ps5_ctx_wait_idle(struct radeon_winsys_ctx *rctx, enum amd_ip_type ip_type, int ring_index)
{
   (void)ip_type;
   (void)ring_index;
   struct radv_ps5_queue *const queue = ((struct radv_ps5_ctx *)rctx)->ws->queue;
   return radv_ps5_queue_wait_seq(queue, p_atomic_read(&queue->submitted_seq), OS_TIMEOUT_INFINITE);
}

static int
radv_ps5_ctx_set_pstate(struct radeon_winsys_ctx *rctx, uint32_t pstate)
{
   (void)rctx;
   return pstate == RADEON_CTX_PSTATE_NONE ? 0 : -1;
}

static void
radv_ps5_winsys_destroy(struct radeon_winsys *rws)
{
   free(rws);
}

static int
radv_ps5_winsys_get_fd(struct radeon_winsys *rws)
{
   (void)rws;
   return -1;
}

static struct util_sync_provider *
radv_ps5_winsys_get_sync_provider(struct radeon_winsys *rws)
{
   (void)rws;
   return NULL;
}

static int
radv_ps5_winsys_reserve_vmid(struct radeon_winsys *rws)
{
   (void)rws;
   return -1;
}

static void
radv_ps5_winsys_unreserve_vmid(struct radeon_winsys *rws)
{
   (void)rws;
}

VkResult
radv_ps5_winsys_create(const struct radeon_info *info, uint64_t debug_flags, uint64_t perftest_flags,
                       struct radeon_winsys **winsys)
{
   struct radv_ps5_queue *const queue = radv_ps5_queue_get();
   if (!queue)
      return VK_ERROR_INITIALIZATION_FAILED;
   struct radv_ps5_winsys *const ws = calloc(1, sizeof(*ws));
   if (!ws)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   memcpy(&ws->info, info, sizeof(ws->info));
   ws->debug_flags = debug_flags;
   ws->perftest_flags = perftest_flags;
   ws->queue = queue;

   ws->base.destroy = radv_ps5_winsys_destroy;
   ws->base.query_value = radv_ps5_winsys_query_value;
   ws->base.read_registers = radv_ps5_winsys_read_registers;
   ws->base.query_gpuvm_fault = radv_ps5_winsys_query_gpuvm_fault;
   ws->base.ctx_create = radv_ps5_ctx_create;
   ws->base.ctx_destroy = radv_ps5_ctx_destroy;
   ws->base.ctx_set_tess_factor_ring = radv_ps5_ctx_set_tess_factor_ring;
   ws->base.ctx_wait_idle = radv_ps5_ctx_wait_idle;
   ws->base.ctx_set_pstate = radv_ps5_ctx_set_pstate;
   ws->base.get_fd = radv_ps5_winsys_get_fd;
   ws->base.get_sync_provider = radv_ps5_winsys_get_sync_provider;
   ws->base.copy_sync_payloads = radv_ps5_sync_copy_payloads;
   ws->base.reserve_vmid = radv_ps5_winsys_reserve_vmid;
   ws->base.unreserve_vmid = radv_ps5_winsys_unreserve_vmid;
   radv_ps5_bo_init_functions(ws);
   radv_ps5_cs_init_functions(ws);

   *winsys = &ws->base;
   return VK_SUCCESS;
}
