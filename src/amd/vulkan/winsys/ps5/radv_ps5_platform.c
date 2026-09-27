/*
 * Copyright © 2026 Mihawk
 *
 * SPDX-License-Identifier: MIT
 */

#include "radv_ps5_platform.h"

#include <assert.h>
#include <errno.h>
#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "c11/threads.h"
#include "util/macros.h"
#include "util/simple_mtx.h"
#include "util/u_math.h"

#if defined(__PROSPERO__)
#include <ps5platform/agc.h>
#include <ps5platform/kernel.h>
#else
#include <sys/mman.h>
#endif

/* A first-fit allocator over a range of fixed-size granules, one bit each. The
 * console places buffers the GPU reaches through full addresses in a region it
 * maps at the address asked for (PS5_Vulkan R88); the host model places the
 * address window's buffers the same way. */
struct radv_ps5_granules {
   simple_mtx_t lock;
   uint64_t base;
   uint64_t granule_bytes;
   uint32_t count;
   uint64_t *used;
};

static bool
radv_ps5_granules_init(struct radv_ps5_granules *g, uint64_t base, uint64_t bytes, uint64_t granule_bytes)
{
   simple_mtx_init(&g->lock, mtx_plain);
   g->base = base;
   g->granule_bytes = granule_bytes;
   g->count = (uint32_t)(bytes / granule_bytes);
   g->used = calloc(DIV_ROUND_UP(g->count, 64), sizeof(uint64_t));
   return g->used != NULL;
}

static void
radv_ps5_granules_mark(struct radv_ps5_granules *g, uint32_t first, uint32_t count, bool used)
{
   for (uint32_t i = first; i < first + count; i++) {
      const uint64_t bit = UINT64_C(1) << (i % 64);
      if (used)
         g->used[i / 64] |= bit;
      else
         g->used[i / 64] &= ~bit;
   }
}

/* count free granules in a row, aligned to align_granules; UINT32_MAX if none. */
static uint32_t
radv_ps5_granules_take(struct radv_ps5_granules *g, uint32_t count, uint32_t align_granules)
{
   if (count == 0 || count > g->count)
      return UINT32_MAX;
   align_granules = MAX2(align_granules, 1);
   simple_mtx_lock(&g->lock);
   uint32_t start = 0;
   while (start + count <= g->count) {
      uint32_t run = 0;
      while (run < count && !(g->used[(start + run) / 64] & (UINT64_C(1) << ((start + run) % 64))))
         run++;
      if (run == count) {
         radv_ps5_granules_mark(g, start, count, true);
         simple_mtx_unlock(&g->lock);
         return start;
      }
      start = align(start + run + 1, align_granules);
   }
   simple_mtx_unlock(&g->lock);
   return UINT32_MAX;
}

static void
radv_ps5_granules_give(struct radv_ps5_granules *g, uint32_t first, uint32_t count)
{
   if (count == 0)
      return;
   simple_mtx_lock(&g->lock);
   radv_ps5_granules_mark(g, first, count, false);
   simple_mtx_unlock(&g->lock);
}

/* The device-memory region: 256 GiB at 0x4000000000, handed out in 2 MiB
 * granules. The GPU reads and writes the whole direct-memory pool there
 * (PS5_Vulkan R86-R88). */
#define RADV_PS5_REGION_BASE UINT64_C(0x4000000000)
#define RADV_PS5_REGION_BYTES (UINT64_C(256) << 30)

static struct radv_ps5_granules radv_ps5_region;
static once_flag radv_ps5_once = ONCE_FLAG_INIT;
static bool radv_ps5_ready;

uint64_t
radv_ps5_now_ns(void)
{
   struct timespec now;
   clock_gettime(CLOCK_MONOTONIC, &now);
   return (uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec;
}

void
radv_ps5_sleep_us(unsigned microseconds)
{
   const struct timespec delay = {
      .tv_sec = microseconds / 1000000u,
      .tv_nsec = (long)(microseconds % 1000000u) * 1000l,
   };
   nanosleep(&delay, NULL);
}

void
radv_ps5_cpu_flush(const void *address, size_t bytes)
{
   if (bytes == 0)
      return;
   const uintptr_t line = 64;
   uintptr_t at = (uintptr_t)address & ~(line - 1);
   const uintptr_t end = (uintptr_t)address + bytes;
   __builtin_ia32_mfence();
   for (; at < end; at += line)
      __builtin_ia32_clflush((const void *)at);
   __builtin_ia32_mfence();
}

static bool
radv_ps5_window_contains(uint64_t address, uint64_t bytes)
{
   return address >= RADV_PS5_WINDOW_BASE && bytes <= RADV_PS5_WINDOW_BYTES &&
          address - RADV_PS5_WINDOW_BASE <= RADV_PS5_WINDOW_BYTES - bytes;
}

#if defined(__PROSPERO__)

/* ------------------------------------------------------------------ console */

/* GPU-visible direct memory as ps5vk and the test runner allocate it: type 12,
 * mapped for CPU and GPU read and write. */
#define RADV_PS5_DIRECT_TYPE 12
#define RADV_PS5_PROTECTION                                                                        \
   (PS5_KERNEL_PROT_CPU_READ | PS5_KERNEL_PROT_CPU_WRITE | PS5_KERNEL_PROT_GPU_READ |              \
    PS5_KERNEL_PROT_GPU_WRITE)

static int32_t radv_ps5_agc_result = -1;

static void
radv_ps5_platform_once(void)
{
   radv_ps5_agc_result = sceAgcInit(PS5_AGC_INIT_VERSION);
   if (radv_ps5_agc_result != 0)
      fprintf(stderr, "radv/ps5: sceAgcInit(%u) failed: 0x%08x\n", PS5_AGC_INIT_VERSION,
              (unsigned)radv_ps5_agc_result);
   radv_ps5_ready = radv_ps5_agc_result == 0 &&
                    radv_ps5_granules_init(&radv_ps5_region, RADV_PS5_REGION_BASE, RADV_PS5_REGION_BYTES,
                                           RADV_PS5_LARGE_BYTES);
}

bool
radv_ps5_platform_runs_gpu(void)
{
   return true;
}

uint64_t
radv_ps5_memory_pool_bytes(void)
{
   const int64_t bytes = sceKernelGetDirectMemorySize();
   return bytes > 0 ? (uint64_t)bytes : 0;
}

uint64_t
radv_ps5_memory_available_bytes(void)
{
   const int64_t pool = sceKernelGetDirectMemorySize();
   int64_t start = -1;
   size_t available = 0;
   if (pool <= 0 || sceKernelAvailableDirectMemorySize(0, pool, RADV_PS5_PAGE_BYTES, &start, &available) != 0)
      return 0;
   return available;
}

bool
radv_ps5_memory_alloc(uint64_t bytes, uint64_t alignment, bool window32, struct radv_ps5_memory *out)
{
   *out = (struct radv_ps5_memory){.physical = -1};
   bytes = align64(MAX2(bytes, 1), RADV_PS5_PAGE_BYTES);
   alignment = MAX2(alignment, PS5_KERNEL_DIRECT_ALIGNMENT);
   if (bytes >= RADV_PS5_LARGE_BYTES)
      alignment = MAX2(alignment, RADV_PS5_LARGE_BYTES);
   alignment = util_next_power_of_two64(alignment);

   int64_t physical = -1;
   if (sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), bytes, alignment,
                                     RADV_PS5_DIRECT_TYPE, &physical) != 0)
      return false;

   /* A window buffer goes where the kernel puts a mapping it is given no
    * address for, which is the window (PS5_Vulkan R86); anything else goes in
    * the device-memory region, at the address asked for. */
   void *hint = NULL;
   uint32_t granule = 0, granules = 0;
   if (!window32) {
      granules = (uint32_t)DIV_ROUND_UP(bytes, RADV_PS5_LARGE_BYTES);
      granule = radv_ps5_granules_take(&radv_ps5_region, granules,
                                       (uint32_t)MAX2(alignment / RADV_PS5_LARGE_BYTES, 1));
      if (granule == UINT32_MAX)
         granules = 0;
      else
         hint = (void *)(uintptr_t)(RADV_PS5_REGION_BASE + (uint64_t)granule * RADV_PS5_LARGE_BYTES);
   }

   void *address = hint;
   int32_t result = sceKernelMapDirectMemory(&address, bytes, RADV_PS5_PROTECTION, 0, physical, alignment);
   if (result != 0 && hint != NULL) {
      radv_ps5_granules_give(&radv_ps5_region, granule, granules);
      granules = 0;
      address = NULL;
      result = sceKernelMapDirectMemory(&address, bytes, RADV_PS5_PROTECTION, 0, physical, alignment);
   }
   if (result == 0 && address != hint) {
      radv_ps5_granules_give(&radv_ps5_region, granule, granules);
      granules = 0;
   }

   const uint64_t va = (uint64_t)(uintptr_t)address;
   const bool placed = result == 0 && address != NULL &&
                       (window32 ? radv_ps5_window_contains(va, bytes)
                                 : va < RADV_PS5_GPU_ADDRESS_LIMIT && bytes <= RADV_PS5_GPU_ADDRESS_LIMIT - va);
   if (!placed) {
      if (result == 0 && address != NULL)
         sceKernelMunmap(address, bytes);
      radv_ps5_granules_give(&radv_ps5_region, granule, granules);
      sceKernelReleaseDirectMemory(physical, bytes);
      return false;
   }

   *out = (struct radv_ps5_memory){
      .cpu = address,
      .bytes = bytes,
      .physical = physical,
      .granule = granule,
      .granules = granules,
   };
   return true;
}

void
radv_ps5_memory_free(struct radv_ps5_memory *memory)
{
   if (!memory->cpu)
      return;
   const int32_t unmapped = sceKernelMunmap(memory->cpu, memory->bytes);
   if (unmapped != 0)
      fprintf(stderr, "radv/ps5: sceKernelMunmap(%p, %" PRIu64 ") failed: 0x%08x\n", (void *)memory->cpu,
              memory->bytes, (unsigned)unmapped);
   radv_ps5_granules_give(&radv_ps5_region, memory->granule, memory->granules);
   if (memory->physical >= 0)
      sceKernelReleaseDirectMemory(memory->physical, memory->bytes);
   *memory = (struct radv_ps5_memory){.physical = -1};
}

int
radv_ps5_submit(uint32_t *words, uint32_t count, volatile uint32_t *marker, uint32_t marker_value)
{
   (void)marker;
   (void)marker_value;
   radv_ps5_cpu_flush(words, (size_t)count * sizeof(uint32_t));
   struct ps5_agc_submit_description description = {
      .words = words,
      .word_count = count,
   };
   int32_t result = sceAgcDriverSubmitDcb(&description);
   if (result != 0)
      return result;
   /* Without the suspend point the console starts a submission up to a
    * refresh late (PS5_Vulkan R68). */
   return sceAgcSuspendPoint();
}

int
radv_ps5_set_tess_factor_ring(uint64_t va, uint32_t size)
{
   return sceAgcDriverSetTFRing((uintptr_t)va, size);
}

int
radv_ps5_set_hs_offchip_param(uint32_t granularity, uint32_t buffering)
{
   return sceAgcDriverSetHsOffchipParam(granularity, buffering);
}

bool
radv_ps5_memory_map_at(const struct radv_ps5_memory *memory, void *address)
{
   if (memory->physical < 0)
      return false;
   void *at = address;
   const int32_t result = sceKernelMapDirectMemory(&at, memory->bytes, PS5_KERNEL_PROT_CPU_READ | PS5_KERNEL_PROT_CPU_WRITE,
                                                   PS5_KERNEL_MAP_FIXED, memory->physical, PS5_KERNEL_PAGE_SIZE);
   if (result == 0 && at != address) {
      sceKernelMunmap(at, memory->bytes);
      return false;
   }
   return result == 0;
}

void
radv_ps5_memory_unmap_at(void *address, uint64_t bytes, bool reserve)
{
   if (reserve) {
      /* A reservation laid over the mapping replaces it, as a fixed mapping does. */
      void *at = address;
      if (sceKernelReserveVirtualRange(&at, bytes, PS5_KERNEL_MAP_FIXED, PS5_KERNEL_PAGE_SIZE) == 0 && at == address)
         return;
      fprintf(stderr, "radv/ps5: a placed mapping at %p could not be left reserved\n", address);
   }
   sceKernelMunmap(address, bytes);
}

#else

/* --------------------------------------------------------------- host model */

/* The window is reserved at the console's address, so 32-bit pointers carry
 * the same high word on both; buffers outside it are ordinary mappings. */
static struct radv_ps5_granules radv_ps5_window;
#define RADV_PS5_WINDOW_GRANULE UINT64_C(0x10000)

static void
radv_ps5_platform_once(void)
{
   void *const window = mmap((void *)(uintptr_t)RADV_PS5_WINDOW_BASE, RADV_PS5_WINDOW_BYTES, PROT_NONE,
                             MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE, -1, 0);
   if (window != (void *)(uintptr_t)RADV_PS5_WINDOW_BASE) {
      fprintf(stderr, "radv/ps5: the host model could not reserve the address window: %s\n", strerror(errno));
      if (window != MAP_FAILED)
         munmap(window, RADV_PS5_WINDOW_BYTES);
      return;
   }
   radv_ps5_ready = radv_ps5_granules_init(&radv_ps5_window, RADV_PS5_WINDOW_BASE, RADV_PS5_WINDOW_BYTES,
                                           RADV_PS5_WINDOW_GRANULE);
}

bool
radv_ps5_platform_runs_gpu(void)
{
   return false;
}

uint64_t
radv_ps5_memory_pool_bytes(void)
{
   /* The console's pool is about 12 GiB; the model reports the same. */
   return UINT64_C(12) << 30;
}

uint64_t
radv_ps5_memory_available_bytes(void)
{
   return radv_ps5_memory_pool_bytes();
}

bool
radv_ps5_memory_alloc(uint64_t bytes, uint64_t alignment, bool window32, struct radv_ps5_memory *out)
{
   *out = (struct radv_ps5_memory){.physical = -1};
   bytes = align64(MAX2(bytes, 1), RADV_PS5_PAGE_BYTES);
   if (window32) {
      const uint32_t count = (uint32_t)DIV_ROUND_UP(bytes, RADV_PS5_WINDOW_GRANULE);
      const uint32_t first = radv_ps5_granules_take(
         &radv_ps5_window, count, (uint32_t)MAX2(util_next_power_of_two64(alignment) / RADV_PS5_WINDOW_GRANULE, 1));
      if (first == UINT32_MAX)
         return false;
      uint8_t *const cpu = (uint8_t *)(uintptr_t)(RADV_PS5_WINDOW_BASE + (uint64_t)first * RADV_PS5_WINDOW_GRANULE);
      const uint64_t span = (uint64_t)count * RADV_PS5_WINDOW_GRANULE;
      if (mprotect(cpu, span, PROT_READ | PROT_WRITE) != 0) {
         radv_ps5_granules_give(&radv_ps5_window, first, count);
         return false;
      }
      *out = (struct radv_ps5_memory){.cpu = cpu, .bytes = span, .physical = -1, .granule = first, .granules = count};
      return true;
   }
   alignment = util_next_power_of_two64(MAX2(alignment, RADV_PS5_PAGE_BYTES));
   /* Over-allocate so the start can be aligned, then trim. */
   const uint64_t span = bytes + alignment;
   uint8_t *const raw = mmap(NULL, span, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
   if (raw == MAP_FAILED)
      return false;
   uint8_t *const cpu = (uint8_t *)align64((uint64_t)(uintptr_t)raw, alignment);
   if (cpu > raw)
      munmap(raw, cpu - raw);
   const uint8_t *const end = raw + span;
   if (cpu + bytes < end)
      munmap(cpu + bytes, end - (cpu + bytes));
   *out = (struct radv_ps5_memory){.cpu = cpu, .bytes = bytes, .physical = -1};
   return true;
}

void
radv_ps5_memory_free(struct radv_ps5_memory *memory)
{
   if (!memory->cpu)
      return;
   if (radv_ps5_window_contains((uint64_t)(uintptr_t)memory->cpu, memory->bytes)) {
      /* Back to reserved, and zero when it is handed out again. */
      mmap(memory->cpu, memory->bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, -1, 0);
      radv_ps5_granules_give(&radv_ps5_window, memory->granule, memory->granules);
   } else {
      munmap(memory->cpu, memory->bytes);
   }
   *memory = (struct radv_ps5_memory){.physical = -1};
}

int
radv_ps5_submit(uint32_t *words, uint32_t count, volatile uint32_t *marker, uint32_t marker_value)
{
   (void)words;
   (void)count;
   /* Nothing runs the words on a PC; the submission completes at once. */
   *marker = marker_value;
   return 0;
}

int
radv_ps5_set_tess_factor_ring(uint64_t va, uint32_t size)
{
   (void)va;
   (void)size;
   return 0;
}

int
radv_ps5_set_hs_offchip_param(uint32_t granularity, uint32_t buffering)
{
   (void)granularity;
   (void)buffering;
   return 0;
}

bool
radv_ps5_memory_map_at(const struct radv_ps5_memory *memory, void *address)
{
   /* The host model's memory is anonymous: nothing maps it twice. */
   (void)memory;
   (void)address;
   return false;
}

void
radv_ps5_memory_unmap_at(void *address, uint64_t bytes, bool reserve)
{
   (void)address;
   (void)bytes;
   (void)reserve;
}

#endif

bool
radv_ps5_platform_init(void)
{
   call_once(&radv_ps5_once, radv_ps5_platform_once);
   return radv_ps5_ready;
}
