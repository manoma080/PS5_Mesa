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
#include "util/u_math.h"
#include "sid.h"

/* Command streams. The console faults a PM4 INDIRECT_BUFFER into title memory
 * (PS5_Vulkan B8), so nothing is chained: a stream's words stay in CPU memory,
 * in chunks that never move, and a submission copies every stream it carries
 * into the queue's ring as one AGC submission, which ends with the packet that
 * writes the submission's sequence number to the marker. */

#define RADV_PS5_CS_INITIAL_DW (20 * 1024)
#define RADV_PS5_CS_MAX_CHUNK_DW (4u << 20)

static bool
radv_ps5_cs_alloc(struct radv_ps5_cs *cs, uint32_t max_dw)
{
   uint32_t *const words = malloc((size_t)max_dw * sizeof(uint32_t));
   if (!words)
      return false;
   cs->base.buf = words;
   cs->base.max_dw = max_dw;
   cs->base.cdw = 0;
   cs->base.reserved_dw = 0;
   return true;
}

static struct ac_cmdbuf *
radv_ps5_cs_create(struct radeon_winsys *rws, enum amd_ip_type ip_type, bool is_secondary)
{
   struct radv_ps5_cs *const cs = calloc(1, sizeof(*cs));
   if (!cs)
      return NULL;
   cs->ws = radv_ps5_winsys(rws);
   cs->ip_type = ip_type;
   cs->is_secondary = is_secondary;
   cs->status = VK_SUCCESS;
   util_dynarray_init(&cs->chunks, NULL);
   if (!radv_ps5_cs_alloc(cs, RADV_PS5_CS_INITIAL_DW)) {
      free(cs);
      return NULL;
   }
   return &cs->base;
}

static void
radv_ps5_cs_free_chunks(struct radv_ps5_cs *cs)
{
   util_dynarray_foreach (&cs->chunks, struct radv_ps5_cs_chunk, chunk)
      free(chunk->words);
   util_dynarray_clear(&cs->chunks);
   cs->words_in_chunks = 0;
}

static void
radv_ps5_cs_destroy(struct ac_cmdbuf *base)
{
   struct radv_ps5_cs *const cs = radv_ps5_cs(base);
   radv_ps5_cs_free_chunks(cs);
   util_dynarray_fini(&cs->chunks);
   free(cs->base.buf);
   free(cs);
}

static enum radeon_bo_domain
radv_ps5_cs_domain(const struct radeon_winsys *rws)
{
   (void)rws;
   return RADEON_DOMAIN_GTT;
}

static void
radv_ps5_cs_grow(struct ac_cmdbuf *base, size_t min_size)
{
   struct radv_ps5_cs *const cs = radv_ps5_cs(base);
   if (cs->status != VK_SUCCESS) {
      cs->base.cdw = 0;
      return;
   }
   /* The current chunk is finished as it stands: every packet RADV writes
    * checks for its space first, so none straddles two chunks. */
   const struct radv_ps5_cs_chunk done = {
      .words = cs->base.buf,
      .cdw = cs->base.cdw,
      .max_dw = cs->base.max_dw,
   };
   const uint64_t wanted = MAX2((uint64_t)min_size + 16, (uint64_t)cs->base.max_dw * 2);
   const uint32_t max_dw = (uint32_t)MAX2(MIN2(wanted, RADV_PS5_CS_MAX_CHUNK_DW), (uint64_t)min_size + 16);
   struct radv_ps5_cs_chunk *const slot = util_dynarray_grow(&cs->chunks, struct radv_ps5_cs_chunk, 1);
   if (slot)
      *slot = done;
   if (!slot || !radv_ps5_cs_alloc(cs, max_dw)) {
      if (slot)
         (void)util_dynarray_pop(&cs->chunks, struct radv_ps5_cs_chunk);
      cs->base.buf = done.words;
      cs->base.max_dw = done.max_dw;
      cs->base.cdw = 0;
      cs->status = VK_ERROR_OUT_OF_HOST_MEMORY;
      return;
   }
   cs->words_in_chunks += done.cdw;
}

static void
radv_ps5_cs_emit_nops(struct radv_ps5_cs *cs, unsigned count)
{
   if (count == 0)
      return;
   if (count == 1) {
      cs->base.buf[cs->base.cdw++] = PKT3_NOP_PAD;
      return;
   }
   /* One NOP packet whose body fills the rest. */
   cs->base.buf[cs->base.cdw] = PKT3(PKT3_NOP, count - 2, 0);
   cs->base.cdw += count;
}

static void
radv_ps5_cs_pad(struct ac_cmdbuf *base, unsigned leave_dw_space)
{
   struct radv_ps5_cs *const cs = radv_ps5_cs(base);
   const uint32_t mask = cs->ws->info.ip[cs->ip_type].ib_pad_dw_mask;
   const uint32_t unaligned = (cs->base.cdw + leave_dw_space) & mask;
   if (unaligned)
      radv_ps5_cs_emit_nops(cs, mask + 1 - unaligned);
}

static VkResult
radv_ps5_cs_finalize(struct ac_cmdbuf *base)
{
   struct radv_ps5_cs *const cs = radv_ps5_cs(base);
   assert(cs->base.cdw <= cs->base.reserved_dw || cs->status != VK_SUCCESS);
   return cs->status;
}

static void
radv_ps5_cs_reset(struct ac_cmdbuf *base)
{
   struct radv_ps5_cs *const cs = radv_ps5_cs(base);
   radv_ps5_cs_free_chunks(cs);
   cs->base.cdw = 0;
   cs->base.reserved_dw = 0;
   cs->status = VK_SUCCESS;
}

static bool
radv_ps5_cs_chain(struct ac_cmdbuf *cs, struct ac_cmdbuf *next_cs, bool pre_ena)
{
   (void)cs;
   (void)next_cs;
   (void)pre_ena;
   return false;
}

static void
radv_ps5_cs_unchain(struct ac_cmdbuf *cs)
{
   (void)cs;
}

static void
radv_ps5_cs_add_buffer(struct ac_cmdbuf *cs, struct radeon_winsys_bo *bo)
{
   /* Every buffer is resident for as long as it lives. */
   (void)cs;
   (void)bo;
}

static uint32_t
radv_ps5_cs_words(const struct radv_ps5_cs *cs)
{
   return cs->words_in_chunks + cs->base.cdw;
}

/* Appends count words to a stream, growing it as needed. */
static void
radv_ps5_cs_append(struct radv_ps5_cs *cs, const uint32_t *words, uint32_t count)
{
   while (count && cs->status == VK_SUCCESS) {
      if (cs->base.cdw == cs->base.max_dw)
         radv_ps5_cs_grow(&cs->base, MIN2(count, RADV_PS5_CS_MAX_CHUNK_DW - 16));
      const uint32_t room = cs->base.max_dw - cs->base.cdw;
      const uint32_t n = MIN2(room, count);
      memcpy(cs->base.buf + cs->base.cdw, words, (size_t)n * sizeof(uint32_t));
      cs->base.cdw += n;
      cs->base.reserved_dw = MAX2(cs->base.reserved_dw, cs->base.cdw);
      words += n;
      count -= n;
   }
}

static void
radv_ps5_cs_execute_secondary(struct ac_cmdbuf *parent_base, struct ac_cmdbuf *child_base, bool allow_ib2)
{
   (void)allow_ib2;
   struct radv_ps5_cs *const parent = radv_ps5_cs(parent_base);
   struct radv_ps5_cs *const child = radv_ps5_cs(child_base);
   if (parent->status != VK_SUCCESS || child->status != VK_SUCCESS)
      return;
   /* Keep the child's words together in one chunk of the parent, as the
    * copies a packet's space check reserved would be. */
   const uint32_t total = radv_ps5_cs_words(child);
   if (parent->base.max_dw - parent->base.cdw < total)
      radv_ps5_cs_grow(&parent->base, total);
   util_dynarray_foreach (&child->chunks, struct radv_ps5_cs_chunk, chunk)
      radv_ps5_cs_append(parent, chunk->words, chunk->cdw);
   radv_ps5_cs_append(parent, child->base.buf, child->base.cdw);
}

static void
radv_ps5_cs_execute_ib(struct ac_cmdbuf *cs, struct radeon_winsys_bo *bo, const uint64_t va, const uint32_t cdw,
                       const bool predicate)
{
   (void)bo;
   (void)va;
   (void)cdw;
   (void)predicate;
   /* A GPU-generated command buffer runs as an INDIRECT_BUFFER, which the
    * console faults; the physical device does not report device-generated
    * commands. */
   fprintf(stderr, "radv/ps5: an indirect buffer was recorded; the console cannot run one\n");
   radv_ps5_cs(cs)->status = VK_ERROR_FEATURE_NOT_PRESENT;
}

static void
radv_ps5_cs_chain_dgc_ib(struct ac_cmdbuf *cs, uint64_t va, uint32_t cdw, uint64_t trailer_va, const bool predicate)
{
   (void)va;
   (void)trailer_va;
   radv_ps5_cs_execute_ib(cs, NULL, 0, cdw, predicate);
}

static void
radv_ps5_cs_dump(struct ac_cmdbuf *cs, FILE *file, const int *trace_ids, int trace_id_count,
                 enum radv_cs_dump_type type)
{
   (void)cs;
   (void)file;
   (void)trace_ids;
   (void)trace_id_count;
   (void)type;
}

static void
radv_ps5_cs_annotate(struct ac_cmdbuf *cs, const char *marker)
{
   (void)cs;
   (void)marker;
}

/* ------------------------------------------------------------- submission */

/* The completion packet: one RELEASE_MEM of CACHE_FLUSH_AND_INV_TS_EVENT
 * (event 20, index 5) that flushes the colour and depth caches, writes back and
 * invalidates L2 and the vector caches (the GCR actions 0x30c), and only then
 * writes the sequence number's low 32 bits -- the packet ps5vk ends every
 * submission with (PS5_Vulkan R90), and the one the amdgpu kernel's fences are. */
#define RADV_PS5_COMPLETION_WORDS 8
#define RADV_PS5_COMPLETION_EVENT 20u
#define RADV_PS5_COMPLETION_GCR 0x30cu

static void
radv_ps5_completion_words(uint32_t *words, uint64_t marker_address, uint32_t value)
{
   words[0] = PKT3(PKT3_RELEASE_MEM, 6, 0);
   words[1] = (RADV_PS5_COMPLETION_GCR << 12) | (5u << 8) | RADV_PS5_COMPLETION_EVENT;
   words[2] = 0x20000000u; /* DATA_SEL(1): the 32-bit value; DST_SEL(0): memory */
   words[3] = (uint32_t)marker_address;
   words[4] = (uint32_t)(marker_address >> 32);
   words[5] = value;
   words[6] = 0;
   words[7] = 0;
}

/* Forgets the ring records the GPU has finished with. Called with the submit
 * lock held. */
static void
radv_ps5_queue_retire(struct radv_ps5_queue *queue)
{
   const uint64_t completed = radv_ps5_queue_poll(queue);
   while (queue->in_flight_count) {
      struct radv_ps5_in_flight *const oldest = &queue->in_flight[queue->in_flight_first];
      if (oldest->seq > completed)
         break;
      radv_ps5_memory_free(&oldest->dedicated);
      queue->in_flight_first = (queue->in_flight_first + 1) % RADV_PS5_MAX_IN_FLIGHT;
      queue->in_flight_count--;
   }
}

/* Words for a submission of count words: a ring range the GPU no longer reads,
 * or a buffer of its own for one too large for the ring. Called with the submit
 * lock held; records the claim for seq. */
static uint32_t *
radv_ps5_queue_claim(struct radv_ps5_queue *queue, uint32_t count, uint64_t seq)
{
   radv_ps5_queue_retire(queue);
   while (queue->in_flight_count == RADV_PS5_MAX_IN_FLIGHT) {
      const uint64_t oldest = queue->in_flight[queue->in_flight_first].seq;
      if (!radv_ps5_queue_wait_seq(queue, oldest, OS_TIMEOUT_INFINITE))
         return NULL;
      radv_ps5_queue_retire(queue);
   }
   struct radv_ps5_in_flight *const claim =
      &queue->in_flight[(queue->in_flight_first + queue->in_flight_count) % RADV_PS5_MAX_IN_FLIGHT];
   *claim = (struct radv_ps5_in_flight){.seq = seq, .dedicated = {.physical = -1}};

   if (count > RADV_PS5_RING_WORDS / 2) {
      if (!radv_ps5_memory_alloc((uint64_t)count * 4, RADV_PS5_LARGE_BYTES, true, &claim->dedicated))
         return NULL;
      queue->in_flight_count++;
      return (uint32_t *)claim->dedicated.cpu;
   }

   /* Streams start on a 64-byte line and follow one another round the ring. */
   uint32_t start = align(queue->ring_head, 16);
   if (start + count > RADV_PS5_RING_WORDS)
      start = 0;
   const uint32_t end = start + count;
   /* Wait for every earlier stream still in the range to be run. */
   for (uint32_t i = 0; i < queue->in_flight_count; i++) {
      const struct radv_ps5_in_flight *const record =
         &queue->in_flight[(queue->in_flight_first + i) % RADV_PS5_MAX_IN_FLIGHT];
      if (record->dedicated.cpu || record->end <= start || end <= record->start)
         continue;
      if (!radv_ps5_queue_wait_seq(queue, record->seq, OS_TIMEOUT_INFINITE))
         return NULL;
   }
   radv_ps5_queue_retire(queue);
   claim->start = start;
   claim->end = end;
   /* The claim is recorded after the retire, which may have moved the first
    * record; rebuild it in place at the end of the list. */
   struct radv_ps5_in_flight *const slot =
      &queue->in_flight[(queue->in_flight_first + queue->in_flight_count) % RADV_PS5_MAX_IN_FLIGHT];
   *slot = (struct radv_ps5_in_flight){.seq = seq, .start = start, .end = end, .dedicated = {.physical = -1}};
   queue->in_flight_count++;
   queue->ring_head = end;
   return (uint32_t *)queue->ring.cpu + start;
}

static uint32_t *
radv_ps5_copy_streams(uint32_t *at, struct ac_cmdbuf **streams, unsigned count)
{
   for (unsigned i = 0; i < count; i++) {
      const struct radv_ps5_cs *const cs = radv_ps5_cs(streams[i]);
      util_dynarray_foreach (&cs->chunks, struct radv_ps5_cs_chunk, chunk) {
         memcpy(at, chunk->words, (size_t)chunk->cdw * sizeof(uint32_t));
         at += chunk->cdw;
      }
      memcpy(at, cs->base.buf, (size_t)cs->base.cdw * sizeof(uint32_t));
      at += cs->base.cdw;
   }
   return at;
}

static uint32_t
radv_ps5_count_streams(struct ac_cmdbuf **streams, unsigned count)
{
   uint32_t words = 0;
   for (unsigned i = 0; i < count; i++)
      words += radv_ps5_cs_words(radv_ps5_cs(streams[i]));
   return words;
}

static VkResult
radv_ps5_cs_submit(struct radeon_winsys_ctx *rctx, const struct radv_winsys_submit_info *submit, uint32_t wait_count,
                   const struct vk_sync_wait *waits, uint32_t signal_count, const struct vk_sync_signal *signals)
{
   struct radv_ps5_ctx *const ctx = (struct radv_ps5_ctx *)rctx;
   struct radv_ps5_queue *const queue = ctx->ws->queue;

   if (submit->ip_type != AMD_IP_GFX)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   for (unsigned i = 0; i < submit->cs_count; i++) {
      if (radv_ps5_cs(submit->cs_array[i])->status != VK_SUCCESS)
         return radv_ps5_cs(submit->cs_array[i])->status;
   }

   /* Waits need only their signals submitted: this queue runs in order. */
   uint64_t depends = 0;
   VkResult result = radv_ps5_sync_wait_submitted(wait_count, waits, &depends);
   if (result != VK_SUCCESS)
      return result;

   simple_mtx_lock(&queue->submit_lock);
   if (submit->cs_count == 0) {
      /* Nothing for the GPU: the signals follow what was submitted before. */
      const uint64_t last = queue->submitted_seq;
      simple_mtx_unlock(&queue->submit_lock);
      radv_ps5_sync_signal_seq(signal_count, signals, MAX2(last, depends));
      return VK_SUCCESS;
   }

   /* One AGC submission: the initial preambles (every submission starts from
    * the state they set, as each amdgpu submission does), the streams, the
    * postambles, then the completion. */
   const uint32_t count = radv_ps5_count_streams(submit->initial_preamble_cs, submit->initial_preamble_count) +
                          radv_ps5_count_streams(submit->cs_array, submit->cs_count) +
                          radv_ps5_count_streams(submit->postamble_cs, submit->postamble_count) +
                          RADV_PS5_COMPLETION_WORDS;
   const uint64_t seq = queue->submitted_seq + 1;
   uint32_t *const words = radv_ps5_queue_claim(queue, count, seq);
   if (!words) {
      simple_mtx_unlock(&queue->submit_lock);
      return VK_ERROR_DEVICE_LOST;
   }
   uint32_t *at = radv_ps5_copy_streams(words, submit->initial_preamble_cs, submit->initial_preamble_count);
   at = radv_ps5_copy_streams(at, submit->cs_array, submit->cs_count);
   at = radv_ps5_copy_streams(at, submit->postamble_cs, submit->postamble_count);
   radv_ps5_completion_words(at, (uint64_t)(uintptr_t)queue->marker, (uint32_t)seq);
   at += RADV_PS5_COMPLETION_WORDS;
   assert((uint32_t)(at - words) == count);

   p_atomic_set(&queue->submitted_seq, seq);
   const int submitted = radv_ps5_submit(words, count, queue->marker, (uint32_t)seq);
   simple_mtx_unlock(&queue->submit_lock);
   if (submitted != 0) {
      fprintf(stderr, "radv/ps5: sceAgcDriverSubmitDcb failed: 0x%08x\n", (unsigned)submitted);
      return VK_ERROR_DEVICE_LOST;
   }
   ctx->queue_used[submit->ip_type][submit->queue_index] = true;
   radv_ps5_sync_signal_seq(signal_count, signals, seq);
   return VK_SUCCESS;
}

void
radv_ps5_cs_init_functions(struct radv_ps5_winsys *ws)
{
   ws->base.cs_domain = radv_ps5_cs_domain;
   ws->base.cs_create = radv_ps5_cs_create;
   ws->base.cs_destroy = radv_ps5_cs_destroy;
   ws->base.cs_grow = radv_ps5_cs_grow;
   ws->base.cs_finalize = radv_ps5_cs_finalize;
   ws->base.cs_reset = radv_ps5_cs_reset;
   ws->base.cs_chain = radv_ps5_cs_chain;
   ws->base.cs_unchain = radv_ps5_cs_unchain;
   ws->base.cs_add_buffer = radv_ps5_cs_add_buffer;
   ws->base.cs_execute_secondary = radv_ps5_cs_execute_secondary;
   ws->base.cs_execute_ib = radv_ps5_cs_execute_ib;
   ws->base.cs_chain_dgc_ib = radv_ps5_cs_chain_dgc_ib;
   ws->base.cs_submit = radv_ps5_cs_submit;
   ws->base.cs_dump = radv_ps5_cs_dump;
   ws->base.cs_annotate = radv_ps5_cs_annotate;
   ws->base.cs_pad = radv_ps5_cs_pad;
}
