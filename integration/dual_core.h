#pragma once
// Interface exposed by k0/harness.cpp (compiled into libggml-mk.so with
// -DMK_NO_MAIN) for the MK backend's dispatch layer (integration/mk_dispatch.cpp)
// to drive the persistent dual-GPU megakernel over llama's in-place pointers.
// Plain C++ only (included by both an nvcc TU and a g++ TU): no CUDA types.
#include <map>
#include <string>
#include <array>
#include <cstdint>

// GGUF/cache name -> per-GPU device pointer (the R2-extracted llama pointers).
struct MkPtrMap { std::map<std::string, std::array<void *, 2>> p; };

// Stand up the two persistent megakernels once, binding llama's extracted
// per-GPU pointers (weights + KV/state, in place) into the Resolver under the
// split program's expected names; MK-allocate only the scratch. Loads the split
// schedule from program_path. Idempotent: the first call sets up, later calls
// are no-ops.
void mk_dual_setup(const MkPtrMap & pm, const char * program_path,
                   int64_t n_ctx, int64_t n_vocab);

// Run one decode pass at position `pos`: seed the mirrored residual from
// seed0/seed1 (MK#model.input_embed#0's per-GPU data, 5120 f32 — MK's graph does
// the embed on the CPU side), run the megakernel, then copy each GPU's vocab-half
// logits (n_vocab/2 f32) into out0/out1 (the output meta tensor's per-GPU simple
// tensors on GPU0/GPU1). Returns false on timeout or a device error.
bool mk_dual_step(const void * seed0, const void * seed1, int64_t pos,
                  void * out0, void * out1);

bool mk_dual_ready();
void mk_dual_shutdown();
