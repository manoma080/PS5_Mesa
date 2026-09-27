/*
 * Copyright © 2026 Mihawk
 *
 * SPDX-License-Identifier: MIT
 */

/* Geometry shaders the hardware cannot run, as compute: see radv_gs_compute.h. */

#include "radv_gs_compute.h"

#include "nir/radv_nir.h"
#include "nir_builder.h"
#include "radv_pipeline.h"
#include "radv_shader.h"
#include "radv_shader_args.h"
#include "radv_shader_info.h"

#include "ac_nir.h"
#include "poly/cl/libpoly.h"
#include "poly/geometry.h"
#include "util/u_debug.h"

bool
radv_gs_compute_wanted(const struct radv_compiler_info *compiler_info, const struct radv_graphics_state_key *gfx_state,
                       const struct radv_shader_stage *stages)
{
   const nir_shader *gs = stages[MESA_SHADER_GEOMETRY].nir;

   if (!compiler_info->key.no_legacy_gs || !gs || !stages[MESA_SHADER_VERTEX].nir)
      return false;

   /* So far: a vertex shader feeding the geometry shader, with its vertex input
    * known when the pipeline is compiled. */
   if (stages[MESA_SHADER_TESS_CTRL].nir || stages[MESA_SHADER_TESS_EVAL].nir || gfx_state->vs.has_prolog)
      return false;

   return debug_get_bool_option("RADV_PS5_GS_COMPUTE", false);
}

/* The vertex shader's outputs go to memory, indexed by the invocation's place
 * in the unrolled vertex stream (x) and its instance (y), where poly's
 * geometry shader reads its inputs. */
static bool
lower_vs_output_to_memory(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   const uint64_t outputs = *(const uint64_t *)data;
   b->cursor = nir_instr_remove(&intr->instr);

   const nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   nir_def *location = nir_iadd_imm(b, intr->src[1].ssa, sem.location);
   nir_def *vp = nir_load_vertex_param_buffer_poly(b);
   nir_def *id = nir_load_global_invocation_id(b, 32);
   nir_def *linear = nir_iadd(b, nir_imul(b, nir_channel(b, id, 1), poly_input_vertices(b, vp)), nir_channel(b, id, 0));
   nir_def *addr = poly_vertex_output_address(b, vp, nir_imm_int64(b, outputs), linear, location);

   assert(nir_src_bit_size(intr->src[0]) <= 32);
   addr = nir_iadd_imm(b, addr, nir_intrinsic_component(intr) * 4);
   nir_store_global(b, intr->src[0].ssa, addr, .write_mask = nir_intrinsic_write_mask(intr));
   return true;
}

/* Memory writes of the application's own (buffers, images, device addresses)
 * happen once, in the count pass: the other copies of the geometry shader drop
 * them. poly strips the global and bindless forms; these are RADV's, still
 * derefs and SSBO intrinsics when poly runs. An atomic whose result is used
 * stays. */
static bool
strip_app_memory_writes(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_store_ssbo:
   case nir_intrinsic_image_deref_store:
   case nir_intrinsic_image_store:
   case nir_intrinsic_bindless_image_store:
      nir_instr_remove(&intr->instr);
      return true;

   case nir_intrinsic_store_deref: {
      nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
      if (!nir_deref_mode_is_one_of(deref, nir_var_mem_ssbo | nir_var_mem_global | nir_var_image))
         return false;
      nir_instr_remove(&intr->instr);
      return true;
   }

   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap:
   case nir_intrinsic_deref_atomic:
   case nir_intrinsic_deref_atomic_swap:
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
   case nir_intrinsic_global_atomic:
   case nir_intrinsic_global_atomic_swap:
      if (nir_intrinsic_infos[intr->intrinsic].has_dest && !list_is_empty(&intr->def.uses))
         return false;
      if (intr->intrinsic == nir_intrinsic_deref_atomic || intr->intrinsic == nir_intrinsic_deref_atomic_swap) {
         nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
         if (!nir_deref_mode_is_one_of(deref, nir_var_mem_ssbo | nir_var_mem_global))
            return false;
      }
      nir_instr_remove(&intr->instr);
      return true;

   default:
      return false;
   }
}

static void
strip_side_effects(nir_shader *nir)
{
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, nir, nir_shader_intrinsics_pass, strip_app_memory_writes, nir_metadata_control_flow, NULL);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
   } while (progress);
}

void
radv_gs_compute_split(const struct radv_compiler_info *compiler_info, struct radv_shader_stage *stages,
                      struct radv_gs_compute_nir *out)
{
   struct radv_shader_stage *vs_stage = &stages[MESA_SHADER_VERTEX];
   struct radv_shader_stage *gs_stage = &stages[MESA_SHADER_GEOMETRY];
   nir_shader *vs = vs_stage->nir;
   nir_shader *gs = gs_stage->nir;

   memset(out, 0, sizeof(*out));

   radv_nir_lower_io(vs);
   radv_nir_lower_io(gs);
   NIR_PASS(_, vs, nir_lower_vars_to_ssa);
   NIR_PASS(_, gs, nir_lower_vars_to_ssa);
   radv_optimize_nir(vs, vs_stage->key.optimisations_disabled);
   radv_optimize_nir(gs, gs_stage->key.optimisations_disabled);

   /* The vertex shader: its outputs to memory, where the geometry shader reads
    * them. The mask fixes the layout both sides use. */
   out->vs_outputs = vs->info.outputs_written;
   NIR_PASS(_, vs, nir_shader_intrinsics_pass, lower_vs_output_to_memory, nir_metadata_control_flow,
            &out->vs_outputs);

   /* The geometry shader: poly's count pass, pre-GS setup, the GS proper and
    * the rasterization copy. */
   nir_shader *count = NULL, *rast = NULL, *pre_gs = NULL;
   NIR_PASS(_, gs, poly_nir_lower_gs, &count, &rast, &pre_gs, &out->info);

   /* The count pass keeps the application's memory writes; the others run the
    * geometry shader again and must not repeat them. */
   strip_side_effects(gs);
   strip_side_effects(rast);

   /* poly's bookkeeping lives in shader temporaries: make them SSA as the
    * rest of RADV expects. */
   nir_shader *const lowered[] = {count, pre_gs, gs, rast};
   for (unsigned i = 0; i < ARRAY_SIZE(lowered); i++) {
      if (!lowered[i])
         continue;
      NIR_PASS(_, lowered[i], nir_lower_global_vars_to_local);
      NIR_PASS(_, lowered[i], nir_lower_vars_to_ssa);
   }

   /* The rasterization copy reads the vertex ID RADV lowers right after
    * SPIR-V: poly made it after that. */
   NIR_PASS(_, rast, nir_lower_system_values);

   out->nir[radv_gs_compute_index(RADV_GS_COMPUTE_VS)] = vs;
   out->nir[radv_gs_compute_index(RADV_GS_COMPUTE_COUNT)] = count;
   out->nir[radv_gs_compute_index(RADV_GS_COMPUTE_PRE_GS)] = pre_gs;
   out->nir[radv_gs_compute_index(RADV_GS_COMPUTE_MAIN)] = gs;

   /* The pipeline's hardware vertex shader is the rasterization copy; there is
    * no hardware geometry shader. */
   vs_stage->nir = rast;
   vs_stage->next_stage = MESA_SHADER_FRAGMENT;
   gs_stage->nir = NULL;
   gs_stage->stage = MESA_SHADER_NONE;
}

static nir_def *
load_draw(nir_builder *b, const struct radv_compiler_info *compiler_info, const struct radv_shader_args *args,
          unsigned offset, unsigned num_components)
{
   nir_def *base = nir_pack_64_2x32_split(b, ac_nir_load_arg(b, &args->ac, args->gs_compute_draw),
                                         nir_imm_int(b, compiler_info->hw.address32_hi));
   return ac_nir_load_smem(b, num_components, base, nir_imm_int(b, offset), 4, ACCESS_CAN_SPECULATE);
}

static nir_def *
load_draw64(nir_builder *b, const struct radv_compiler_info *compiler_info, const struct radv_shader_args *args,
            unsigned offset)
{
   return nir_pack_64_2x32(b, load_draw(b, compiler_info, args, offset, 2));
}

struct lower_sysvals_state {
   const struct radv_compiler_info *compiler_info;
   const struct radv_shader_stage *stage;
};

/* The vertex shader run as compute: its vertex is the index-buffer entry at its
 * place in the stream, or that place itself without an index buffer. */
static nir_def *
sw_vs_vertex(nir_builder *b)
{
   nir_def *place = nir_channel(b, nir_load_global_invocation_id(b, 32), 0);
   nir_def *index_size = nir_load_index_size_poly(b);
   nir_def *indexed;
   nir_push_if(b, nir_ine_imm(b, index_size, 0));
   {
      indexed = poly_load_index_buffer(b, nir_load_vertex_param_buffer_poly(b), place, index_size);
   }
   nir_pop_if(b, NULL);
   return nir_if_phi(b, indexed, place);
}

static bool
lower_sw_vs_sysval(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_load_vertex_id_zero_base:
      nir_def_replace(&intr->def, sw_vs_vertex(b));
      return true;
   case nir_intrinsic_load_vertex_id:
      nir_def_replace(&intr->def, nir_iadd(b, sw_vs_vertex(b), nir_load_first_vertex(b)));
      return true;
   case nir_intrinsic_load_instance_id:
      nir_def_replace(&intr->def, nir_channel(b, nir_load_global_invocation_id(b, 32), 1));
      return true;
   case nir_intrinsic_load_is_indexed_draw:
      nir_def_replace(&intr->def, nir_b2i32(b, nir_ine_imm(b, nir_load_index_size_poly(b), 0)));
      return true;
   case nir_intrinsic_load_base_vertex:
      nir_def_replace(&intr->def, nir_bcsel(b, nir_ine_imm(b, nir_load_index_size_poly(b), 0),
                                            nir_load_first_vertex(b), nir_imm_int(b, 0)));
      return true;
   default:
      return false;
   }
}

static bool
lower_sysval(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct lower_sysvals_state *state = data;
   const struct radv_compiler_info *compiler_info = state->compiler_info;
   const struct radv_shader_args *args = &state->stage->args;
   const bool compute = state->stage->stage == MESA_SHADER_COMPUTE;

   b->cursor = nir_before_instr(&intr->instr);

   nir_def *value;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_vertex_param_buffer_poly:
      value = load_draw64(b, compiler_info, args, RADV_GS_COMPUTE_DRAW_OFFSET(vertex_params));
      break;
   case nir_intrinsic_load_geometry_param_buffer_poly:
      value = load_draw64(b, compiler_info, args, RADV_GS_COMPUTE_DRAW_OFFSET(geometry_params));
      break;
   case nir_intrinsic_load_ro_sink_address_poly:
   case nir_intrinsic_load_stat_query_address_poly:
      /* Statistics come with queries (not yet): into the sink meanwhile. */
      value = load_draw64(b, compiler_info, args, RADV_GS_COMPUTE_DRAW_OFFSET(ro_sink));
      break;
   case nir_intrinsic_load_flat_mask:
      value = load_draw64(b, compiler_info, args, RADV_GS_COMPUTE_DRAW_OFFSET(flat_mask));
      break;
   case nir_intrinsic_load_input_topology_poly:
      value = load_draw(b, compiler_info, args, RADV_GS_COMPUTE_DRAW_OFFSET(input_topology), 1);
      break;
   case nir_intrinsic_load_provoking_last:
      value = nir_b2b32(b, nir_ine_imm(b, load_draw(b, compiler_info, args,
                                                     RADV_GS_COMPUTE_DRAW_OFFSET(provoking_last), 1), 0));
      break;
   case nir_intrinsic_ro_to_rw_poly:
      value = intr->src[0].ssa;
      break;
   case nir_intrinsic_load_first_vertex:
      if (!compute)
         return false;
      value = load_draw(b, compiler_info, args, RADV_GS_COMPUTE_DRAW_OFFSET(first_vertex), 1);
      break;
   case nir_intrinsic_load_base_instance:
      if (!compute)
         return false;
      value = load_draw(b, compiler_info, args, RADV_GS_COMPUTE_DRAW_OFFSET(base_instance), 1);
      break;
   case nir_intrinsic_load_draw_id:
      if (!compute)
         return false;
      value = load_draw(b, compiler_info, args, RADV_GS_COMPUTE_DRAW_OFFSET(draw_id), 1);
      break;
   default:
      return false;
   }

   nir_def_replace(&intr->def, value);
   return true;
}

bool
radv_gs_compute_lower_sysvals(nir_shader *nir, const struct radv_compiler_info *compiler_info,
                              const struct radv_shader_stage *stage)
{
   bool progress = false;
   NIR_PASS(progress, nir, poly_nir_lower_sysvals);
   const struct lower_sysvals_state state = {.compiler_info = compiler_info, .stage = stage};
   NIR_PASS(progress, nir, nir_shader_intrinsics_pass, lower_sysval, nir_metadata_control_flow, (void *)&state);
   return progress;
}

/* As RADV's compute shaders, and poly's OpenCL helpers' global IDs without a
 * base (a dispatch here starts at 0). */
static const nir_lower_compute_system_values_options compute_sysval_options = {
   .lower_local_invocation_index = true,
   .has_base_global_invocation_id = false,
   .has_base_workgroup_id = false,
};

struct radv_shader_binary *
radv_gs_compute_compile(const struct radv_compiler_info *compiler_info, const struct radv_graphics_state_key *gfx_state,
                        const struct radv_shader_stage *vs_stage, enum radv_gs_compute_kind kind, nir_shader *nir,
                        struct radv_shader_debug_info *debug)
{
   struct radv_shader_stage stage;
   memset(&stage, 0, sizeof(stage));
   stage.stage = MESA_SHADER_COMPUTE;
   stage.next_stage = MESA_SHADER_NONE;
   stage.entrypoint = "main";
   stage.layout = vs_stage->layout;
   stage.key = vs_stage->key;
   stage.nir = nir;

   /* The vertex shader's attribute fetch needs its vertex input information,
    * gathered while it is still a vertex shader. */
   struct radv_shader_info vs_info;
   if (kind == RADV_GS_COMPUTE_VS) {
      radv_nir_shader_info_init(MESA_SHADER_VERTEX, MESA_SHADER_NONE, &vs_info);
      radv_nir_shader_info_pass(compiler_info, nir, &stage.layout, &stage.key, gfx_state, RADV_PIPELINE_GRAPHICS,
                                false, &vs_info);
   }

   /* Everything runs as compute, 64 invocations a workgroup along x. */
   nir->info.stage = MESA_SHADER_COMPUTE;
   memset(&nir->info.cs, 0, sizeof(nir->info.cs));
   nir->info.workgroup_size[0] = 64;
   nir->info.workgroup_size[1] = 1;
   nir->info.workgroup_size[2] = 1;
   nir->info.workgroup_size_variable = false;
   /* The vertex and geometry shaders keep the wave size of their own stages. */
   nir->info.min_subgroup_size = 64;
   nir->info.max_subgroup_size = 64;
   NIR_PASS(_, nir, nir_lower_compute_system_values, &compute_sysval_options);
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, ac_nir_lower_indirect_derefs);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   radv_optimize_nir(nir, stage.key.optimisations_disabled);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   radv_nir_shader_info_init(MESA_SHADER_COMPUTE, MESA_SHADER_NONE, &stage.info);
   radv_nir_shader_info_pass(compiler_info, nir, &stage.layout, &stage.key, NULL, RADV_PIPELINE_COMPUTE, false,
                             &stage.info);
   stage.info.gs_compute = kind;
   if (kind == RADV_GS_COMPUTE_VS)
      stage.info.vs = vs_info.vs;

   radv_declare_shader_args(compiler_info, NULL, &stage, MESA_SHADER_NONE, debug);
   stage.info.user_sgprs_locs = stage.args.user_sgprs_locs;
   stage.info.inline_push_constant_mask = stage.args.ac.inline_push_const_mask;

   if (kind == RADV_GS_COMPUTE_VS) {
      NIR_PASS(_, nir, radv_nir_lower_vs_inputs, compiler_info, &stage, gfx_state);
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_sw_vs_sysval, nir_metadata_none, NULL);
   }
   radv_gs_compute_lower_sysvals(nir, compiler_info, &stage);
   NIR_PASS(_, nir, nir_lower_compute_system_values, &compute_sysval_options);
   radv_optimize_nir(nir, stage.key.optimisations_disabled);

   radv_postprocess_nir(compiler_info, NULL, &stage);

   struct radv_shader_binary *binary = radv_shader_nir_to_asm(compiler_info, &stage, &nir, 1, NULL);
   radv_shader_dump_asm(compiler_info, debug, binary, &stage.info);
   return binary;
}
