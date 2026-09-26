/*
 * Copyright © 2026 Mihawk
 *
 * SPDX-License-Identifier: MIT
 */

#include "radv_ps5_winsys.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util/os_time.h"
#include "util/u_atomic.h"

/* A buffer is one direct-memory allocation, mapped for the CPU and the GPU at
 * the same address for as long as it lives, so it is always resident, its GPU
 * address is its CPU address, and mapping it is a view of it (PS5_Vulkan B3). A
 * 32-bit buffer lies in the shaders' address window; any other anywhere below
 * 2^47 (R86-R88). */

uint64_t radv_ps5_allocated_bytes;

/* Memory reports (VK_EXT_device_memory_report) name a buffer by an id that is
 * never 0 and never reused. */
static uint64_t radv_ps5_next_obj_id;

static VkResult
radv_ps5_buffer_create(struct radeon_winsys *rws, uint64_t size, unsigned alignment, enum radeon_bo_domain domain,
                       enum radeon_bo_flag flags, unsigned priority, uint64_t replay_address,
                       struct radeon_winsys_bo **out_bo)
{
   (void)priority;
   struct radv_ps5_winsys *const ws = radv_ps5_winsys(rws);
   *out_bo = NULL;

   /* Sparse buffers need page-table control a title does not have, and a
    * buffer cannot be placed at an address an earlier run chose; the physical
    * device reports neither (has_sparse_vm_mappings, capture replay). */
   if (flags & RADEON_FLAG_VIRTUAL)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   if (replay_address)
      return VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS;

   struct radv_ps5_bo *const bo = calloc(1, sizeof(*bo));
   if (!bo)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   if (!radv_ps5_memory_alloc(size, alignment, flags & RADEON_FLAG_32BIT, &bo->memory)) {
      free(bo);
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }
   bo->ws = ws;
   bo->flags = flags;
   bo->base.va = (uint64_t)(uintptr_t)bo->memory.cpu;
   bo->base.size = size;
   bo->base.is_local = true;
   bo->base.use_global_list = true;
   bo->base.initial_domain = domain;
   bo->base.obj_id = p_atomic_inc_return(&radv_ps5_next_obj_id);

   /* Direct memory arrives with whatever the pool held; RADV asks for zeroes
    * where it relies on them. The GPU reads through the CPU's writes only once
    * they are in memory. */
   if (flags & RADEON_FLAG_ZERO_VRAM) {
      memset(bo->memory.cpu, 0, size);
      radv_ps5_cpu_flush(bo->memory.cpu, size);
   }

   p_atomic_add(&radv_ps5_allocated_bytes, bo->memory.bytes);
   if (domain & RADEON_DOMAIN_VRAM)
      p_atomic_add(&ws->allocated_vram, bo->memory.bytes);
   else
      p_atomic_add(&ws->allocated_gtt, bo->memory.bytes);

   *out_bo = &bo->base;
   return VK_SUCCESS;
}

static void
radv_ps5_buffer_destroy(struct radeon_winsys *rws, struct radeon_winsys_bo *base)
{
   struct radv_ps5_winsys *const ws = radv_ps5_winsys(rws);
   struct radv_ps5_bo *const bo = radv_ps5_bo(base);
   if (!bo)
      return;
   p_atomic_add(&radv_ps5_allocated_bytes, -(int64_t)bo->memory.bytes);
   if (bo->base.initial_domain & RADEON_DOMAIN_VRAM)
      p_atomic_add(&ws->allocated_vram, -(int64_t)bo->memory.bytes);
   else
      p_atomic_add(&ws->allocated_gtt, -(int64_t)bo->memory.bytes);
   radv_ps5_memory_free(&bo->memory);
   free(bo);
}

static void *
radv_ps5_buffer_map(struct radeon_winsys *rws, struct radeon_winsys_bo *base, bool use_fixed_addr, void *fixed_addr)
{
   (void)rws;
   /* A placed mapping would need a second CPU mapping of the memory. */
   if (use_fixed_addr)
      return NULL;
   (void)fixed_addr;
   return radv_ps5_bo(base)->memory.cpu;
}

static void
radv_ps5_buffer_unmap(struct radeon_winsys *rws, struct radeon_winsys_bo *base, bool replace)
{
   (void)rws;
   (void)base;
   (void)replace;
}

static VkResult
radv_ps5_buffer_from_ptr(struct radeon_winsys *rws, void *pointer, uint64_t size, unsigned priority,
                         struct radeon_winsys_bo **out_bo)
{
   (void)rws;
   (void)pointer;
   (void)size;
   (void)priority;
   *out_bo = NULL;
   return VK_ERROR_INVALID_EXTERNAL_HANDLE;
}

static VkResult
radv_ps5_buffer_from_fd(struct radeon_winsys *rws, int fd, unsigned priority, struct radeon_winsys_bo **out_bo,
                        uint64_t *alloc_size)
{
   (void)rws;
   (void)fd;
   (void)priority;
   (void)alloc_size;
   *out_bo = NULL;
   return VK_ERROR_INVALID_EXTERNAL_HANDLE;
}

static bool
radv_ps5_buffer_get_fd(struct radeon_winsys *rws, struct radeon_winsys_bo *bo, int *fd)
{
   (void)rws;
   (void)bo;
   (void)fd;
   return false;
}

static bool
radv_ps5_buffer_get_flags_from_fd(struct radeon_winsys *rws, int fd, enum radeon_bo_domain *domains,
                                  enum radeon_bo_flag *flags)
{
   (void)rws;
   (void)fd;
   (void)domains;
   (void)flags;
   return false;
}

static void
radv_ps5_buffer_set_metadata(struct radeon_winsys *rws, struct radeon_winsys_bo *base, struct radeon_bo_metadata *md)
{
   (void)rws;
   radv_ps5_bo(base)->metadata = *md;
}

static void
radv_ps5_buffer_get_metadata(struct radeon_winsys *rws, struct radeon_winsys_bo *base, struct radeon_bo_metadata *md)
{
   (void)rws;
   *md = radv_ps5_bo(base)->metadata;
}

static VkResult
radv_ps5_buffer_virtual_bind(struct radeon_winsys *rws, struct radeon_winsys_bo *parent, uint64_t offset,
                             uint64_t size, struct radeon_winsys_bo *bo, uint64_t bo_offset)
{
   (void)rws;
   (void)parent;
   (void)offset;
   (void)size;
   (void)bo;
   (void)bo_offset;
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

static VkResult
radv_ps5_buffer_make_resident(struct radeon_winsys *rws, struct radeon_winsys_bo *bo, bool resident)
{
   (void)rws;
   (void)bo;
   (void)resident;
   return VK_SUCCESS;
}

static bool
radv_ps5_bo_wait_for_idle(struct radeon_winsys *rws, struct radeon_winsys_bo *bo)
{
   (void)bo;
   struct radv_ps5_queue *const queue = radv_ps5_winsys(rws)->queue;
   return radv_ps5_queue_wait_seq(queue, p_atomic_read(&queue->submitted_seq), OS_TIMEOUT_INFINITE);
}

static void
radv_ps5_dump_bo_ranges(struct radeon_winsys *rws, FILE *file)
{
   (void)rws;
   (void)file;
}

static void
radv_ps5_dump_bo_log(struct radeon_winsys *rws, FILE *file)
{
   (void)rws;
   (void)file;
}

void
radv_ps5_bo_init_functions(struct radv_ps5_winsys *ws)
{
   ws->base.buffer_create = radv_ps5_buffer_create;
   ws->base.buffer_destroy = radv_ps5_buffer_destroy;
   ws->base.buffer_map = radv_ps5_buffer_map;
   ws->base.buffer_unmap = radv_ps5_buffer_unmap;
   ws->base.buffer_from_ptr = radv_ps5_buffer_from_ptr;
   ws->base.buffer_from_fd = radv_ps5_buffer_from_fd;
   ws->base.buffer_get_fd = radv_ps5_buffer_get_fd;
   ws->base.buffer_get_flags_from_fd = radv_ps5_buffer_get_flags_from_fd;
   ws->base.buffer_set_metadata = radv_ps5_buffer_set_metadata;
   ws->base.buffer_get_metadata = radv_ps5_buffer_get_metadata;
   ws->base.buffer_virtual_bind = radv_ps5_buffer_virtual_bind;
   ws->base.buffer_make_resident = radv_ps5_buffer_make_resident;
   ws->base.bo_wait_for_idle = radv_ps5_bo_wait_for_idle;
   ws->base.dump_bo_ranges = radv_ps5_dump_bo_ranges;
   ws->base.dump_bo_log = radv_ps5_dump_bo_log;
}
