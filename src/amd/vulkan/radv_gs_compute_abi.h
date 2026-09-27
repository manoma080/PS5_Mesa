/*
 * Copyright © 2026 Mihawk
 *
 * SPDX-License-Identifier: MIT
 */

/* What the driver and its OpenCL functions for geometry shaders run as
 * compute share (radv_gs_compute.h, cl/radv_gs_compute.cl). */

#pragma once

#include "compiler/libcl/libcl.h"
#include "poly/geometry.h"

/* The per-draw block every such shader reads through a user SGPR. */
struct radv_gs_compute_draw {
   DEVICE(struct poly_vertex_params) vertex_params;
   DEVICE(struct poly_geometry_params) geometry_params;
   uint64_t ro_sink;        /* where writes that must go nowhere go */
   uint64_t flat_mask;      /* the fragment shader's flat inputs */
   uint32_t input_topology; /* enum mesa_prim of the draw */
   uint32_t provoking_last;
   uint32_t first_vertex;
   uint32_t base_instance;
   uint32_t draw_id;
   uint32_t padding;
} PACKED;
static_assert(sizeof(struct radv_gs_compute_draw) == 14 * 4, "struct radv_gs_compute_draw must be 14 words");

/* A draw whose counts live in memory (indirect, byte count, or unrolled for
 * primitive restart) is set up by one invocation of radv_gs_compute_setup:
 * it sizes the passes, allocates their buffers from the command buffer's
 * heap and writes the workgroup counts of the dispatches that follow. When
 * the heap is full the draw draws nothing. */
struct radv_gs_compute_setup {
   DEVICE(uint32_t) draw;       /* VkDraw[Indexed]IndirectCommand */
   DEVICE(uint32_t) draw_count; /* the count of an indirect count draw, or 0 */
   DEVICE(uint32_t) byte_count; /* the counter of a byte count draw, or 0 */
   DEVICE(struct radv_gs_compute_draw) block;
   DEVICE(struct poly_heap) heap;
   uint64_t index_buffer;
   uint64_t vs_outputs;
   uint32_t index_buffer_range_el;
   uint32_t index_size_B; /* 0 when not indexed */
   uint32_t draw_index;
   uint32_t byte_offset;
   uint32_t byte_stride;
   uint32_t prim; /* enum mesa_prim */
   uint32_t shape;
   uint32_t max_indices;

   /* Written: workgroup counts of the vertex and geometry passes. */
   uint32_t vs_groups[3];
   uint32_t gs_groups[3];
} PACKED;
static_assert(sizeof(struct radv_gs_compute_setup) == 28 * 4, "struct radv_gs_compute_setup must be 28 words");

/* An indexed draw with primitive restart becomes an indexed draw of the
 * decomposed list topology without restarts, in the heap
 * (radv_gs_compute_unroll, one workgroup of one wave). */
struct radv_gs_compute_unroll {
   DEVICE(uint32_t) draw;       /* VkDrawIndexedIndirectCommand */
   DEVICE(uint32_t) draw_count; /* the count of an indirect count draw, or 0 */
   DEVICE(uint32_t) out_draw;   /* the unrolled VkDrawIndexedIndirectCommand */
   DEVICE(struct poly_heap) heap;
   uint64_t index_buffer;
   uint64_t ro_sink;
   uint32_t index_buffer_range_el;
   uint32_t index_size_B;
   uint32_t restart_index;
   uint32_t flatshade_first;
   uint32_t prim; /* enum mesa_prim, before decomposition */
   uint32_t draw_index;
} PACKED;
static_assert(sizeof(struct radv_gs_compute_unroll) == 18 * 4, "struct radv_gs_compute_unroll must be 18 words");

#define RADV_GS_COMPUTE_WAVE 64
