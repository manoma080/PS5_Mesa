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
#include "util/simple_mtx.h"
#include "util/u_dynarray.h"
#include "radv_gs_compute_abi.h"
#include "radv_pipeline_layout.h"
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
struct radv_device;
struct radv_graphics_state_key;
struct radv_shader_stage;
struct radv_shader_binary;
struct radv_shader_debug_info;

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

/* What a pipeline compile hands back (radv_graphics_shaders_compile). With
 * the vertex input unknown when the pipeline is compiled (a dynamic vertex
 * input, which a hardware vertex shader takes through a prolog), the vertex
 * pass is not compiled: its lowered NIR comes back serialized instead, with
 * the vertex stage's key, for radv_gs_compute_deferred_vs_create. */
struct radv_gs_compute_binaries {
   bool used;
   struct radv_shader_binary *binaries[RADV_GS_COMPUTE_SHADERS];
   struct radv_shader_debug_info debug[RADV_GS_COMPUTE_SHADERS];
   struct poly_gs_info info;
   uint64_t vs_outputs;
   void *vs_nir;
   size_t vs_nir_size;
   struct radv_shader_stage_key vs_key;
};

/* A vertex pass compiled at the draw, one variant per vertex input the draws
 * bring: the serialized vertex pass, what compiles it (the vertex stage's key
 * and layout, the pipeline's state key) and the variants built so far. */
struct radv_gs_compute_deferred_vs {
   void *nir;
   size_t nir_size;
   struct radv_shader_stage_key key;
   struct radv_pipeline_layout layout;
   struct radv_graphics_state_key gfx_state;
   simple_mtx_t lock;
   /* struct radv_gs_compute_vs_variant */
   struct util_dynarray variants;
};

struct radv_gs_compute_vs_variant {
   struct radv_graphics_state_key vi_key; /* only vi is compared */
   struct radv_shader *shader;
};

/* What a pipeline keeps (radv_pipeline.gs_compute). shaders[VS] is NULL
 * where deferred_vs builds the vertex pass at the draw. */
struct radv_gs_compute_pipeline {
   struct radv_shader *shaders[RADV_GS_COMPUTE_SHADERS];
   struct radv_gs_compute_deferred_vs *deferred_vs;
   struct poly_gs_info info;
   uint64_t vs_outputs;
};

/* The draw's topology as poly names it, from RADV's dynamic state (which
 * holds the hardware encoding, V_008958_DI_PT_*). */
static inline enum mesa_prim
radv_gs_compute_input_prim(unsigned di_pt)
{
   switch (di_pt) {
   case V_008958_DI_PT_POINTLIST:
      return MESA_PRIM_POINTS;
   case V_008958_DI_PT_LINELIST:
      return MESA_PRIM_LINES;
   case V_008958_DI_PT_LINESTRIP:
      return MESA_PRIM_LINE_STRIP;
   case V_008958_DI_PT_TRILIST:
      return MESA_PRIM_TRIANGLES;
   case V_008958_DI_PT_TRISTRIP:
      return MESA_PRIM_TRIANGLE_STRIP;
   case V_008958_DI_PT_TRIFAN:
      return MESA_PRIM_TRIANGLE_FAN;
   case V_008958_DI_PT_LINELIST_ADJ:
      return MESA_PRIM_LINES_ADJACENCY;
   case V_008958_DI_PT_LINESTRIP_ADJ:
      return MESA_PRIM_LINE_STRIP_ADJACENCY;
   case V_008958_DI_PT_TRILIST_ADJ:
      return MESA_PRIM_TRIANGLES_ADJACENCY;
   case V_008958_DI_PT_TRISTRIP_ADJ:
      return MESA_PRIM_TRIANGLE_STRIP_ADJACENCY;
   default:
      return MESA_PRIM_PATCHES;
   }
}

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

/* The two halves radv_gs_compute_split joins, for shaders compiled apart
 * (shader objects). */
uint64_t radv_gs_compute_lower_vs(nir_shader *vs, const struct radv_shader_stage_key *key);
nir_shader *radv_gs_compute_split_gs(nir_shader *gs, const struct radv_shader_stage_key *key,
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

struct radv_gs_compute_deferred_vs *
radv_gs_compute_deferred_vs_create(struct radv_device *device, const struct radv_gs_compute_binaries *binaries,
                                   const struct radv_pipeline_layout *layout,
                                   const struct radv_graphics_state_key *gfx_state);
void radv_gs_compute_deferred_vs_destroy(struct radv_device *device, struct radv_gs_compute_deferred_vs *deferred);

/* The vertex pass for the vertex input in vi_key (its vi field), compiled on
 * first use; NULL if it cannot be. */
struct radv_shader *radv_gs_compute_deferred_vs_get(struct radv_device *device,
                                                    struct radv_gs_compute_deferred_vs *deferred,
                                                    const struct radv_graphics_state_key *vi_key);

/* The meta compute passes of a draw (cl/radv_gs_compute.cl): setting up one
 * whose counts live in memory, unrolling its primitive restarts, and the
 * prefix sum of its counts for transform feedback. Each one's push constant
 * is the address of its argument block, or of the geometry parameters. */
enum radv_gs_compute_meta {
   RADV_GS_COMPUTE_META_SETUP,
   RADV_GS_COMPUTE_META_UNROLL,
   RADV_GS_COMPUTE_META_PREFIX_SUM,
};

VkResult radv_gs_compute_get_meta_pipeline(struct radv_device *device, enum radv_gs_compute_meta meta,
                                           VkPipeline *pipeline, VkPipelineLayout *layout);

#endif /* RADV_GS_COMPUTE_H */
