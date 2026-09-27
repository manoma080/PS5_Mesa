/*
 * Copyright © 2026 Mihawk
 *
 * SPDX-License-Identifier: MIT
 */

/* RADV's OpenCL helpers for geometry shaders run as compute, turned into NIR
 * builder functions by vtn_bindgen2 as poly's are (radv_gs_compute_abi.h). */

#include "compiler/libcl/libcl.h"
#include "poly/geometry.h"
#include "poly/prim.h"
#include "radv_gs_compute_abi.h"

/* A bump allocation from the command buffer's heap, or false when it does
 * not fit: the draw then draws nothing rather than write out of bounds. */
static bool
heap_alloc(global struct poly_heap *heap, uint64_t size_B, uint *offs_B)
{
   if (size_B > heap->size)
      return false;

   uint size = align((uint)size_B, 16);
   uint offs = atomic_fetch_add((volatile atomic_uint *)(&heap->bottom), size);
   if (offs > heap->size - size)
      return false;

   *offs_B = offs;
   return true;
}

void
radv_gs_compute_setup(global struct radv_gs_compute_setup *s)
{
   global uint32_t *draw = s->draw;
   global struct radv_gs_compute_draw *block = s->block;
   global struct poly_vertex_params *vp = block->vertex_params;
   global struct poly_geometry_params *p = block->geometry_params;
   global struct poly_heap *heap = s->heap;
   const bool indexed = s->index_size_B != 0;
   const enum poly_gs_shape shape = s->shape;

   uint vertex_count = draw[0];
   uint instance_count = draw[1];
   if (s->byte_count) {
      uint counter = *s->byte_count;
      vertex_count = counter > s->byte_offset ? (counter - s->byte_offset) / s->byte_stride : 0;
   }
   if (s->draw_count && s->draw_index >= *s->draw_count)
      vertex_count = 0;

   block->first_vertex = indexed ? draw[3] : draw[2];
   block->base_instance = indexed ? draw[4] : draw[3];

   /* Every size in 64 bits: a draw too large for 32-bit counts or for the
    * heap draws nothing. */
   uint64_t prims =
      vertex_count <= INT32_MAX ? u_decomposed_prims_for_vertices(s->prim, vertex_count) : 0;
   uint64_t input_prims = prims * instance_count;
   uint64_t vs_B = (uint64_t)vertex_count * instance_count * util_bitcount64(s->vs_outputs) * 16;
   uint64_t count_B = input_prims * p->count_buffer_stride;
   uint64_t index_B = shape == POLY_GS_SHAPE_DYNAMIC_INDEXED ? input_prims * s->max_indices * 4 : 0;

   uint vs_offs = 0, count_offs = 0, index_offs = 0;
   bool ok = vertex_count && instance_count && vertex_count <= INT32_MAX &&
             input_prims * max(s->max_indices, 1u) <= UINT32_MAX && heap_alloc(heap, vs_B, &vs_offs) &&
             (!count_B || heap_alloc(heap, count_B, &count_offs)) &&
             (!index_B || heap_alloc(heap, index_B, &index_offs));
   if (!ok) {
      vertex_count = 0;
      instance_count = 0;
   }

   poly_vertex_params_set_draw(vp, vertex_count, instance_count);
   poly_geometry_params_set_draw(p, s->prim, shape, s->max_indices, vertex_count, instance_count);

   if (ok) {
      vp->output_buffer = (uintptr_t)(heap->base + vs_offs);
      if (indexed) {
         vp->index_size_B = s->index_size_B;
         vp->index_buffer = draw[2] < s->index_buffer_range_el
                               ? s->index_buffer + (uint64_t)draw[2] * s->index_size_B
                               : block->ro_sink;
         vp->index_buffer_range_el = poly_index_buffer_range_el(s->index_buffer_range_el, draw[2]);
      }
      if (count_B)
         p->count_buffer = (global uint *)(heap->base + count_offs);
      if (index_B) {
         p->output_index_buffer = (global uint *)(heap->base + index_offs);
         p->draw.first_index = index_offs / 4;
      }
   }

   s->vs_groups[0] = (vertex_count + RADV_GS_COMPUTE_WAVE - 1) / RADV_GS_COMPUTE_WAVE;
   s->vs_groups[1] = instance_count;
   s->vs_groups[2] = 1;
   s->gs_groups[0] = (p->grid[0] + RADV_GS_COMPUTE_WAVE - 1) / RADV_GS_COMPUTE_WAVE;
   s->gs_groups[1] = p->grid[1];
   s->gs_groups[2] = 1;
}

/* The first invocation of the workgroup where cond holds, or the workgroup
 * size: the workgroup is one wave, so one ballot sees all of it. */
static uint
first_true(bool cond)
{
   uint4 ballot = sub_group_ballot(cond);
   if (ballot.x)
      return ctz(ballot.x);
   if (ballot.y)
      return 32 + ctz(ballot.y);
   return RADV_GS_COMPUTE_WAVE;
}

/* poly_unroll_geometry (poly/cl/restart.h) for one wave, allocating from the
 * heap without aborting. */
void
radv_gs_compute_unroll(global struct radv_gs_compute_unroll *u)
{
   global uint32_t *in_draw = u->draw;
   global uint32_t *out_draw = u->out_draw;
   global struct poly_heap *heap = u->heap;
   const uint tid = cl_local_id.x;
   const enum mesa_prim mode = u->prim;
   const uint per_prim = mesa_vertices_per_prim(mode);
   const uint size_B = u->index_size_B;

   uint count = in_draw[0];
   if (u->draw_count && u->draw_index >= *u->draw_count)
      count = 0;

   /* Restarts only shorten the output: its bound is the input without any. */
   uint64_t max_prims = count <= INT32_MAX ? u_decomposed_prims_for_vertices(mode, count) : 0;
   uint offs = 0;
   uint ok = 0;
   if (tid == 0)
      ok = count && heap_alloc(heap, max_prims * per_prim * size_B, &offs);
   ok = sub_group_broadcast(ok, 0);
   offs = sub_group_broadcast(offs, 0);

   if (tid == 0) {
      out_draw[0] = 0;
      out_draw[1] = in_draw[1];
      out_draw[2] = offs / size_B;
      out_draw[3] = in_draw[3];
      out_draw[4] = in_draw[4];
   }
   if (!ok)
      return;

   uint64_t in_ptr = in_draw[2] < u->index_buffer_range_el
                        ? u->index_buffer + (uint64_t)in_draw[2] * size_B
                        : u->ro_sink;
   uint in_range_el = poly_index_buffer_range_el(u->index_buffer_range_el, in_draw[2]);
   uint64_t out_ptr = (uintptr_t)(heap->base + offs);

   uint out_prims = 0;
   uint needle = 0;
   while (needle < count) {
      /* The next restart or the end; lanes load in parallel. */
      uint next_restart = needle;
      for (;;) {
         uint idx = next_restart + tid;
         bool restart = idx >= count || poly_load_index(in_ptr, in_range_el, idx, size_B) == u->restart_index;

         uint next_offs = first_true(restart);
         next_restart += next_offs;
         if (next_offs < RADV_GS_COMPUTE_WAVE)
            break;
      }

      /* Up to it, decomposed; lanes store in parallel. */
      uint subcount = next_restart - needle;
      uint subprims = u_decomposed_prims_for_vertices(mode, subcount);
      for (uint i = tid; i < subprims; i += RADV_GS_COMPUTE_WAVE) {
         for (uint vtx = 0; vtx < per_prim; ++vtx) {
            uint id = poly_vertex_id_for_topology(mode, u->flatshade_first, i, vtx, subprims);
            uint x = ((out_prims + i) * per_prim) + vtx;
            uint y = poly_load_index(in_ptr, in_range_el, needle + id, size_B);

            poly_store_index(out_ptr, size_B, x, y);
         }
      }

      out_prims += subprims;
      needle = next_restart + 1;
   }

   if (tid == 0)
      out_draw[0] = out_prims * per_prim;
}

/* The inclusive prefix sum of one word of the count buffer across the draw's
 * input primitives, for their places in the transform feedback buffers: one
 * workgroup of one wave per word. */
void
radv_gs_compute_prefix_sum(global struct poly_geometry_params *p)
{
   const uint word = cl_group_id.x;
   const uint words = p->count_buffer_stride / 4;
   const uint len = p->input_primitives;
   global uint *counts = p->count_buffer;

   uint carry = 0;
   for (uint base = 0; base < len; base += RADV_GS_COMPUTE_WAVE) {
      const uint i = base + cl_local_id.x;
      const uint x = i < len ? counts[i * words + word] : 0;
      const uint sum = sub_group_scan_inclusive_add(x) + carry;
      if (i < len)
         counts[i * words + word] = sum;
      carry = sub_group_broadcast(sum, RADV_GS_COMPUTE_WAVE - 1);
   }
}
