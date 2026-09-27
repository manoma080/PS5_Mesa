/*
 * Copyright © 2026 Mihawk
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef RADV_GS_COMPUTE_H
#define RADV_GS_COMPUTE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "poly/nir/poly_nir.h"
#include "radv_shader.h"
#include "radv_shader_info.h"
#include "sid.h"

/* Geometry shaders the hardware cannot run, run as compute with Mesa's poly
 * lowering (the way Asahi runs every geometry shader). On a GPU without a
 * legacy GS (radeon_info.has_legacy_gs), NGG serves every geometry shader but
 * transform feedback from one (NGG streamout before GFX11 orders its writes
 * with GDS) and a tessellated one amplifying past one subgroup.
 *
 * Such a pipeline has no hardware GS. Per draw:
 * - the vertex shader runs as compute over the unrolled vertex stream and
 *   stores its outputs to memory (poly's GS reads its inputs by position in
 *   that stream);
 * - when the output count is not static and transform feedback or queries
 *   need it, a count pass (which keeps the GS's memory writes) and a prefix
 *   sum run, then the pre-GS setup;
 * - the GS proper runs as compute (writing transform feedback and, for a
 *   dynamic topology, the index buffer to rasterize);
 * - the pipeline's hardware vertex shader is poly's rasterization shader, a
 *   copy of the GS that shades one output vertex per invocation, drawn with
 *   the GS output topology.
 *
 * Every one of those shaders reads one per-draw block, struct
 * radv_gs_compute_draw, through a user SGPR.
 */

struct radv_compiler_info;
struct radv_graphics_state_key;
struct radv_shader_stage;
struct radv_shader_binary;
struct radv_shader_debug_info;

struct radv_gs_compute_draw {
   uint64_t vertex_params;   /* struct poly_vertex_params */
   uint64_t geometry_params; /* struct poly_geometry_params */
   uint64_t ro_sink;         /* where writes that must go nowhere go */
   uint64_t flat_mask;       /* the fragment shader's flat inputs */
   uint32_t input_topology;  /* enum mesa_prim of the draw */
   uint32_t provoking_last;
   uint32_t first_vertex;
   uint32_t base_instance;
   uint32_t draw_id;
   uint32_t padding;
};

#define RADV_GS_COMPUTE_DRAW_OFFSET(field) ((unsigned)offsetof(struct radv_gs_compute_draw, field))

/* enum radv_gs_compute_kind is radv_shader_info.h's. */
#define RADV_GS_COMPUTE_SHADERS 4 /* VS, COUNT, PRE_GS, MAIN */

static inline unsigned
radv_gs_compute_index(enum radv_gs_compute_kind kind)
{
   return kind - RADV_GS_COMPUTE_VS;
}

struct radv_gs_compute_nir {
   nir_shader *nir[RADV_GS_COMPUTE_SHADERS];
   struct poly_gs_info info;
   /* The vertex shader's outputs: the layout of what it stores. */
   uint64_t vs_outputs;
};

/* What a pipeline compile hands back (radv_graphics_shaders_compile). */
struct radv_gs_compute_binaries {
   bool used;
   struct radv_shader_binary *binaries[RADV_GS_COMPUTE_SHADERS];
   struct radv_shader_debug_info debug[RADV_GS_COMPUTE_SHADERS];
   struct poly_gs_info info;
   uint64_t vs_outputs;
};

/* What a pipeline keeps (radv_pipeline.gs_compute). */
struct radv_gs_compute_pipeline {
   struct radv_shader *shaders[RADV_GS_COMPUTE_SHADERS];
   struct poly_gs_info info;
   uint64_t vs_outputs;
};

/* The hardware topology the rasterization copy draws for poly's output mode. */
static inline unsigned
radv_gs_compute_rast_topology(enum mesa_prim mode)
{
   switch (mode) {
   case MESA_PRIM_POINTS:
      return V_008958_DI_PT_POINTLIST;
   case MESA_PRIM_LINE_STRIP:
      return V_008958_DI_PT_LINESTRIP;
   default:
      return V_008958_DI_PT_TRISTRIP;
   }
}

bool radv_gs_compute_wanted(const struct radv_compiler_info *compiler_info,
                            const struct radv_graphics_state_key *gfx_state, const struct radv_shader_stage *stages);

void radv_gs_compute_split(const struct radv_compiler_info *compiler_info, struct radv_shader_stage *stages,
                           struct radv_gs_compute_nir *out);

/* Replaces the system values a geometry shader run as compute reads (poly's,
 * the draw's) with loads from the draw block; the stage's arguments must be
 * declared. */
bool radv_gs_compute_lower_sysvals(nir_shader *nir, const struct radv_compiler_info *compiler_info,
                                   const struct radv_shader_stage *stage);

struct radv_shader_binary *radv_gs_compute_compile(const struct radv_compiler_info *compiler_info,
                                                   const struct radv_graphics_state_key *gfx_state,
                                                   const struct radv_shader_stage *vs_stage,
                                                   enum radv_gs_compute_kind kind, nir_shader *nir,
                                                   struct radv_shader_debug_info *debug);

#endif /* RADV_GS_COMPUTE_H */
