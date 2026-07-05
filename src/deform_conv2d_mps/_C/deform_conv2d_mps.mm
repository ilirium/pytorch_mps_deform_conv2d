// deform_conv2d_mps.mm — Objective-C++ host for the deformable-conv Metal kernels.
//
// Responsibilities:
//   * compile the MSL source (passed in from Python) into a cached MTLLibrary
//   * expose `add_one` (Phase-0 pipeline check) and `deform_conv2d_forward`
//   * register everything with PyTorch via TORCH_LIBRARY
//
// Dispatch uses the public `torch::mps` API (command buffer + serial dispatch
// queue) so our kernels stay in stream with surrounding ATen ops. Backward is
// intentionally left as a stub; see TODO(Phase 3).

#include <torch/extension.h>
#include <torch/mps.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <string>
#include <unordered_map>

// Extract the MTLBuffer backing an MPS tensor's storage.
static inline id<MTLBuffer> getMTLBufferStorage(const at::Tensor& t) {
    return __builtin_bit_cast(id<MTLBuffer>, t.storage().data());
}

// ---------------------------------------------------------------------------
// Library cache. The MSL source is supplied once from Python (so the shaders
// live as .metal files in the package, not duplicated as C strings here).
// ---------------------------------------------------------------------------
static id<MTLLibrary> g_library = nil;
static id<MTLDevice> g_device = nil;

// Pipeline-state cache: building a PSO is expensive; do it once per function.
// Cleared when the library is (re)compiled.
static std::unordered_map<std::string, id<MTLComputePipelineState>>& pso_cache() {
    static auto* cache =
        new std::unordered_map<std::string, id<MTLComputePipelineState>>();
    return *cache;
}

// The Metal device backing the current MPS stream's command buffer.
static id<MTLDevice> current_device() {
    id<MTLCommandBuffer> cmd_buf = torch::mps::get_command_buffer();
    TORCH_CHECK(cmd_buf != nil, "Could not obtain an MPS command buffer (is MPS available?)");
    return cmd_buf.device;
}

void dcn_compile_library(const std::string& msl_source) {
    @autoreleasepool {
        g_device = current_device();
        TORCH_CHECK(g_device != nil, "No Metal device available");
        NSError* err = nil;
        NSString* src = [NSString stringWithUTF8String:msl_source.c_str()];
        MTLCompileOptions* opts = [MTLCompileOptions new];
        // Phase 3: deformable_col2im scatter-adds into grad_input via
        // atomic_fetch_add_explicit on `device atomic_float*`, which requires
        // MSL >= 3.0 (macOS 13+). The default language version rejects it —
        // that's why the Phase-0 stub dropped atomic<float>.
        opts.languageVersion = MTLLanguageVersion3_0;
        g_library = [g_device newLibraryWithSource:src options:opts error:&err];
        TORCH_CHECK(g_library != nil, "Failed to compile Metal library: ",
                    err ? err.localizedDescription.UTF8String : "unknown error");
        pso_cache().clear();
    }
}

static id<MTLComputePipelineState> pipeline_for(const char* fn_name) {
    TORCH_CHECK(g_library != nil,
                "Metal library not compiled. Call _compile_library() first.");
    auto it = pso_cache().find(fn_name);
    if (it != pso_cache().end()) return it->second;
    @autoreleasepool {
        id<MTLFunction> fn =
            [g_library newFunctionWithName:[NSString stringWithUTF8String:fn_name]];
        TORCH_CHECK(fn != nil, "Metal function not found: ", fn_name);
        NSError* err = nil;
        id<MTLComputePipelineState> pso =
            [g_device newComputePipelineStateWithFunction:fn error:&err];
        TORCH_CHECK(pso != nil, "Failed to build pipeline for ", fn_name, ": ",
                    err ? err.localizedDescription.UTF8String : "unknown error");
        pso_cache().emplace(fn_name, pso);
        return pso;
    }
}

// Mirror of DeformConvParams in deform_conv2d.metal — keep field order in sync.
struct DeformConvParams {
    int batch, channels, height, width;
    int kernel_h, kernel_w, pad_h, pad_w;
    int stride_h, stride_w, dilation_h, dilation_w;
    int out_h, out_w;
    int deformable_groups, channels_per_deformable_group, use_mask;
};

// ---------------------------------------------------------------------------
// Phase 0: add_one — proves source -> library -> dispatch -> tensor works.
// ---------------------------------------------------------------------------
at::Tensor add_one(const at::Tensor& input) {
    TORCH_CHECK(input.device().is_mps(), "add_one expects an MPS tensor");
    auto x = input.contiguous();
    auto out = at::empty_like(x);
    uint32_t n = static_cast<uint32_t>(x.numel());
    if (n == 0) return out;

    id<MTLCommandBuffer> cmd_buf = torch::mps::get_command_buffer();
    dispatch_queue_t q = torch::mps::get_dispatch_queue();
    dispatch_sync(q, ^{
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];
            id<MTLComputePipelineState> pso = pipeline_for("add_one");
            [enc setComputePipelineState:pso];
            [enc setBuffer:getMTLBufferStorage(x)
                    offset:x.storage_offset() * x.element_size() atIndex:0];
            [enc setBuffer:getMTLBufferStorage(out)
                    offset:out.storage_offset() * out.element_size() atIndex:1];
            [enc setBytes:&n length:sizeof(n) atIndex:2];

            NSUInteger tg = MIN((NSUInteger)pso.maxTotalThreadsPerThreadgroup,
                                (NSUInteger)n);
            [enc dispatchThreads:MTLSizeMake(n, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc endEncoding];
            torch::mps::commit();
        }
    });
    return out;
}

// ---------------------------------------------------------------------------
// Phase 3, Step 1: atomic_smoke — runtime proof that MSL 3.0 atomic_float
// scatter-add works before deformable_col2im depends on it. Scatters in[i]
// into out[i % 4]; the slot sums are order-independent for the integer-valued
// inputs the diag ladder feeds it, so the check is exact.
// ---------------------------------------------------------------------------
at::Tensor atomic_smoke(const at::Tensor& input) {
    TORCH_CHECK(input.device().is_mps(), "atomic_smoke expects an MPS tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat,
                "atomic_smoke expects a float32 tensor");
    auto x = input.contiguous();
    // Accumulators must be pre-zeroed (the kernel only adds). at::zeros is an
    // ATen MPS op -> must stay OUTSIDE dispatch_sync (deadlock rule).
    auto out = at::zeros({4}, x.options());
    uint32_t n = static_cast<uint32_t>(x.numel());
    if (n == 0) return out;

    id<MTLCommandBuffer> cmd_buf = torch::mps::get_command_buffer();
    dispatch_queue_t q = torch::mps::get_dispatch_queue();
    dispatch_sync(q, ^{
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];
            id<MTLComputePipelineState> pso = pipeline_for("atomic_smoke");
            [enc setComputePipelineState:pso];
            [enc setBuffer:getMTLBufferStorage(x)
                    offset:x.storage_offset() * x.element_size() atIndex:0];
            [enc setBuffer:getMTLBufferStorage(out)
                    offset:out.storage_offset() * out.element_size() atIndex:1];
            [enc setBytes:&n length:sizeof(n) atIndex:2];

            NSUInteger tg = MIN((NSUInteger)pso.maxTotalThreadsPerThreadgroup,
                                (NSUInteger)n);
            [enc dispatchThreads:MTLSizeMake(n, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc endEncoding];
            torch::mps::commit();
        }
    });
    return out;
}

// ---------------------------------------------------------------------------
// Forward: deformable im2col (Metal) -> GEMM (ATen mm) -> bias.
//
// Phase-1 scope: groups == 1, fp32, contiguous NCHW. Per batch element:
// dispatch deformable_im2col into a reused column buffer, then
// out[n] = weight.view({outC, -1}).mm(columns).view({outC, out_h, out_w}).
//
// Threading note: ATen MPS ops synchronize on the same serial dispatch queue
// we use for encoding, so the `mm` MUST NOT run inside our dispatch_sync
// block (deadlock). Each iteration encodes + commits the im2col, then calls
// `mm` from the caller's thread; both enqueue on the same MPS stream, so
// ordering is preserved. The command buffer is re-fetched every iteration
// because commits recycle it.
// ---------------------------------------------------------------------------
at::Tensor deform_conv2d_forward(
        const at::Tensor& input,    // (N, C, H, W)
        const at::Tensor& weight,   // (outC, C/groups, kh, kw)
        const at::Tensor& offset,   // (N, 2*dg*kh*kw, out_h, out_w)
        const at::Tensor& mask,     // (N, dg*kh*kw, out_h, out_w) or empty
        const c10::optional<at::Tensor>& bias,
        int64_t stride_h, int64_t stride_w,
        int64_t pad_h, int64_t pad_w,
        int64_t dilation_h, int64_t dilation_w,
        int64_t groups, int64_t deformable_groups) {
    // --- Step 1: validation & shape computation -------------------------------
    TORCH_CHECK(input.device().is_mps() && weight.device().is_mps() &&
                offset.device().is_mps(),
                "deform_conv2d_forward: input, weight and offset must be MPS tensors");
    TORCH_CHECK(input.scalar_type() == at::kFloat &&
                weight.scalar_type() == at::kFloat &&
                offset.scalar_type() == at::kFloat,
                "deform_conv2d_forward: only float32 is supported (Phase 1)");
    TORCH_CHECK(input.dim() == 4, "input must be 4-D (N, C, H, W), got ", input.dim());
    TORCH_CHECK(weight.dim() == 4, "weight must be 4-D (outC, C/groups, kh, kw), got ", weight.dim());
    TORCH_CHECK(offset.dim() == 4, "offset must be 4-D (N, 2*dg*kh*kw, oh, ow), got ", offset.dim());
    TORCH_CHECK(groups == 1,
                "deform_conv2d_forward: groups != 1 not supported yet (Phase 5), got ", groups);
    TORCH_CHECK(stride_h > 0 && stride_w > 0, "stride must be positive");
    TORCH_CHECK(dilation_h > 0 && dilation_w > 0, "dilation must be positive");

    auto x = input.contiguous();
    auto w = weight.contiguous();
    auto off = offset.contiguous();

    const int64_t N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    const int64_t outC = w.size(0), kh = w.size(2), kw = w.size(3);
    TORCH_CHECK(w.size(1) * groups == C,
                "weight shape mismatch: expected in-channels ", C,
                ", got ", w.size(1) * groups);

    const int64_t out_h =
        (H + 2 * pad_h - dilation_h * (kh - 1) - 1) / stride_h + 1;
    const int64_t out_w =
        (W + 2 * pad_w - dilation_w * (kw - 1) - 1) / stride_w + 1;
    TORCH_CHECK(out_h > 0 && out_w > 0,
                "calculated output size (", out_h, "x", out_w, ") is non-positive");

    const int64_t dg = deformable_groups;
    TORCH_CHECK(dg > 0, "deformable_groups must be positive, got ", dg);
    TORCH_CHECK(C % dg == 0, "channels (", C,
                ") not divisible by deformable_groups (", dg, ")");
    TORCH_CHECK(off.size(0) == N && off.size(1) == 2 * dg * kh * kw &&
                off.size(2) == out_h && off.size(3) == out_w,
                "offset shape mismatch: expected (", N, ", ", 2 * dg * kh * kw,
                ", ", out_h, ", ", out_w, "), got ", off.sizes());

    const bool use_mask = mask.defined() && mask.numel() > 0;
    at::Tensor msk;
    if (use_mask) {
        TORCH_CHECK(mask.device().is_mps() && mask.scalar_type() == at::kFloat,
                    "mask must be a float32 MPS tensor");
        msk = mask.contiguous();
        TORCH_CHECK(msk.dim() == 4 && msk.size(0) == N &&
                    msk.size(1) == dg * kh * kw &&
                    msk.size(2) == out_h && msk.size(3) == out_w,
                    "mask shape mismatch: expected (", N, ", ", dg * kh * kw,
                    ", ", out_h, ", ", out_w, "), got ", msk.sizes());
    }

    at::Tensor b;
    if (bias.has_value() && bias->defined() && bias->numel() > 0) {
        TORCH_CHECK(bias->device().is_mps() && bias->scalar_type() == at::kFloat,
                    "bias must be a float32 MPS tensor");
        TORCH_CHECK(bias->numel() == outC, "bias must have ", outC,
                    " elements, got ", bias->numel());
        b = bias->contiguous();
    }

    auto output = at::empty({N, outC, out_h, out_w}, x.options());
    const int64_t out_hw = out_h * out_w;
    // Zero-size edge cases: nothing to dispatch (0-thread Metal dispatch is an
    // error path); the empty tensor is already the right answer.
    if (output.numel() == 0) return output;

    // --- Step 2: column buffer & params ---------------------------------------
    // Reused across batch iterations; every im2col dispatch fully overwrites it.
    auto columns = at::empty({C * kh * kw, out_hw}, x.options());

    DeformConvParams params;
    params.batch = 1;  // host loop handles batch
    params.channels = static_cast<int>(C);
    params.height = static_cast<int>(H);
    params.width = static_cast<int>(W);
    params.kernel_h = static_cast<int>(kh);
    params.kernel_w = static_cast<int>(kw);
    params.pad_h = static_cast<int>(pad_h);
    params.pad_w = static_cast<int>(pad_w);
    params.stride_h = static_cast<int>(stride_h);
    params.stride_w = static_cast<int>(stride_w);
    params.dilation_h = static_cast<int>(dilation_h);
    params.dilation_w = static_cast<int>(dilation_w);
    params.out_h = static_cast<int>(out_h);
    params.out_w = static_cast<int>(out_w);
    params.deformable_groups = static_cast<int>(dg);
    params.channels_per_deformable_group = static_cast<int>(C / dg);
    params.use_mask = use_mask ? 1 : 0;

    // Per-image plane sizes (in elements) for pointer-arithmetic batch slicing.
    const int64_t im_plane = C * H * W;
    const int64_t off_plane = 2 * dg * kh * kw * out_hw;
    const int64_t msk_plane = dg * kh * kw * out_hw;

    const int64_t total = C * out_hw;  // one thread per (c, oy, ox)
    if (total == 0) {                  // C == 0: output is all zeros (+ bias)
        output.zero_();
        if (b.defined()) output.add_(b.view({1, outC, 1, 1}));
        return output;
    }

    id<MTLComputePipelineState> pso = pipeline_for("deformable_im2col");
    dispatch_queue_t q = torch::mps::get_dispatch_queue();
    auto w2d = w.view({outC, C * kh * kw});

    // --- Steps 3 & 4: per-batch im2col dispatch + interleaved GEMM ------------
    for (int64_t n = 0; n < N; ++n) {
        // Re-fetch each iteration: the interleaved ATen `mm` may commit and
        // recycle the stream's command buffer.
        id<MTLCommandBuffer> cmd_buf = torch::mps::get_command_buffer();
        TORCH_CHECK(cmd_buf != nil, "Could not obtain an MPS command buffer");

        dispatch_sync(q, ^{
            @autoreleasepool {
                id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:getMTLBufferStorage(x)
                        offset:(x.storage_offset() + n * im_plane) * x.element_size()
                       atIndex:0];
                [enc setBuffer:getMTLBufferStorage(off)
                        offset:(off.storage_offset() + n * off_plane) * off.element_size()
                       atIndex:1];
                if (use_mask) {
                    [enc setBuffer:getMTLBufferStorage(msk)
                            offset:(msk.storage_offset() + n * msk_plane) * msk.element_size()
                           atIndex:2];
                } else {
                    // Placeholder binding so the encoder always has a valid
                    // buffer at index 2; the kernel never reads it (use_mask=0).
                    [enc setBuffer:getMTLBufferStorage(off)
                            offset:off.storage_offset() * off.element_size()
                           atIndex:2];
                }
                [enc setBuffer:getMTLBufferStorage(columns)
                        offset:columns.storage_offset() * columns.element_size()
                       atIndex:3];
                [enc setBytes:&params length:sizeof(params) atIndex:4];

                NSUInteger tg = MIN((NSUInteger)pso.maxTotalThreadsPerThreadgroup,
                                    (NSUInteger)total);
                [enc dispatchThreads:MTLSizeMake((NSUInteger)total, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                [enc endEncoding];
                torch::mps::commit();
            }
        });

        // GEMM for image n — outside the dispatch block (see threading note).
        // Enqueues on the same MPS stream, after the im2col above.
        auto out_n = w2d.mm(columns);                     // (outC, out_hw)
        output.select(0, n).copy_(out_n.view({outC, out_h, out_w}));
    }

    if (b.defined()) output.add_(b.view({1, outC, 1, 1}));
    // No torch::mps::synchronize(): the result is lazily valid like any MPS op.
    return output;
}

TORCH_LIBRARY(deform_conv2d_mps, m) {
    m.def("add_one(Tensor input) -> Tensor");
    m.def("atomic_smoke(Tensor input) -> Tensor");
    m.def(
        "deform_conv2d_forward(Tensor input, Tensor weight, Tensor offset, "
        "Tensor mask, Tensor? bias, int stride_h, int stride_w, int pad_h, "
        "int pad_w, int dilation_h, int dilation_w, int groups, "
        "int deformable_groups) -> Tensor");
}

TORCH_LIBRARY_IMPL(deform_conv2d_mps, MPS, m) {
    m.impl("add_one", TORCH_FN(add_one));
    m.impl("atomic_smoke", TORCH_FN(atomic_smoke));
    m.impl("deform_conv2d_forward", TORCH_FN(deform_conv2d_forward));
}

// pybind: expose the one-time library compile hook to Python.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("_compile_library", &dcn_compile_library,
          "Compile the MSL source into the cached Metal library");
}
