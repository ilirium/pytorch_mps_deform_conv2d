// deform_conv2d_mps.mm — Objective-C++ host for the deformable-conv Metal kernels.
//
// Responsibilities:
//   * compile the MSL source (passed in from Python) into a cached MTLLibrary
//   * expose `add_one` / `atomic_smoke` (pipeline checks),
//     `deform_conv2d_forward`, the single-image backward building blocks
//     (`deformable_col2im`, `deformable_col2im_coord` — also used by the diag
//     ladder) and the fused `deform_conv2d_backward`
//   * register everything with PyTorch via TORCH_LIBRARY
//
// Dispatch uses the public `torch::mps` API (command buffer + serial dispatch
// queue) so our kernels stay in stream with surrounding ATen ops.

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
// Forward: deformable im2col (Metal) -> grouped GEMM (ATen bmm) -> bias.
//
// Scope: fp32, contiguous NCHW. Per batch element: dispatch deformable_im2col
// into a reused column buffer (the kernel fills all C channels regardless of
// groups — it only knows about deformable_groups), then the grouped GEMM
// (Phase 5): view columns as (groups, C/groups*kh*kw, out_hw) — the buffer is
// channel-major, so each group's row block is contiguous — view weight as
// (groups, outC/groups, C/groups*kh*kw), and at::bmm in one dispatch. Both
// are views of contiguous buffers (view() throws rather than copies, so a
// silent per-batch copy can't sneak in). groups == 1 degenerates to the old
// weight.view({outC, -1}).mm(columns).
//
// Threading note: ATen MPS ops synchronize on the same serial dispatch queue
// we use for encoding, so the `bmm` MUST NOT run inside our dispatch_sync
// block (deadlock). Each iteration encodes + commits the im2col, then calls
// `bmm` from the caller's thread; both enqueue on the same MPS stream, so
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
    TORCH_CHECK(groups > 0, "groups must be positive, got ", groups);
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
    TORCH_CHECK(C % groups == 0, "channels (", C,
                ") not divisible by groups (", groups, ")");
    TORCH_CHECK(outC % groups == 0, "out_channels (", outC,
                ") not divisible by groups (", groups, ")");

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

    // Grouped GEMM views (Phase 5). Both `w` and `columns` are contiguous and
    // channel-major, so these are zero-copy views: w (outC, C/g, kh, kw) ->
    // (g, outC/g, C/g*kh*kw); columns (C*kh*kw, out_hw) -> (g, C/g*kh*kw,
    // out_hw). view() would throw if a copy were ever needed.
    const int64_t cpg_kk = (C / groups) * kh * kw;  // rows per group
    auto w_g = w.view({groups, outC / groups, cpg_kk});
    auto cols_g = columns.view({groups, cpg_kk, out_hw});

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

        // Grouped GEMM for image n — outside the dispatch block (threading
        // note above; bmm follows the same serial-queue rule as mm). Enqueues
        // on the same MPS stream, after the im2col above. One dispatch for
        // all groups: (g, outC/g, cpg_kk) @ (g, cpg_kk, out_hw).
        auto out_n = at::bmm(w_g, cols_g);                // (g, outC/g, out_hw)
        output.select(0, n).copy_(out_n.view({outC, out_h, out_w}));
    }

    if (b.defined()) output.add_(b.view({1, outC, 1, 1}));
    // No torch::mps::synchronize(): the result is lazily valid like any MPS op.
    return output;
}

// ---------------------------------------------------------------------------
// Backward building block: deformable_col2im for ONE image (Phase 3, Step 2).
//
// columns: (C*kh*kw, out_h*out_w) column-buffer gradient (already multiplied
//          by weight^T on the Python/ATen side); offset: (2*dg*kh*kw, oh, ow);
//          mask: (dg*kh*kw, oh, ow) or empty. Returns grad_input (C, H, W).
//
// Exposed as its own op so the diag ladder can test the scatter kernel in
// isolation; the fused deform_conv2d_backward (Step 4) reuses it per batch
// element via tensor slices (select(0, n) keeps storage_offset views, which
// the buffer bindings honour).
// ---------------------------------------------------------------------------
at::Tensor deformable_col2im(
        const at::Tensor& columns,
        const at::Tensor& offset,
        const at::Tensor& mask,
        int64_t height, int64_t width,
        int64_t kernel_h, int64_t kernel_w,
        int64_t stride_h, int64_t stride_w,
        int64_t pad_h, int64_t pad_w,
        int64_t dilation_h, int64_t dilation_w,
        int64_t deformable_groups) {
    TORCH_CHECK(columns.device().is_mps() && offset.device().is_mps(),
                "deformable_col2im: columns and offset must be MPS tensors");
    TORCH_CHECK(columns.scalar_type() == at::kFloat &&
                offset.scalar_type() == at::kFloat,
                "deformable_col2im: only float32 is supported");
    TORCH_CHECK(columns.dim() == 2, "columns must be 2-D (C*kh*kw, oh*ow)");
    TORCH_CHECK(offset.dim() == 3, "offset must be 3-D (2*dg*kh*kw, oh, ow)");
    TORCH_CHECK(kernel_h > 0 && kernel_w > 0, "kernel dims must be positive");
    TORCH_CHECK(stride_h > 0 && stride_w > 0, "stride must be positive");
    TORCH_CHECK(dilation_h > 0 && dilation_w > 0, "dilation must be positive");

    auto col = columns.contiguous();
    auto off = offset.contiguous();

    const int64_t kk = kernel_h * kernel_w;
    TORCH_CHECK(col.size(0) % kk == 0,
                "columns rows (", col.size(0), ") not divisible by kh*kw (", kk, ")");
    const int64_t C = col.size(0) / kk;

    const int64_t out_h =
        (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int64_t out_w =
        (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    TORCH_CHECK(col.size(1) == out_h * out_w,
                "columns cols (", col.size(1), ") != out_h*out_w (", out_h * out_w, ")");

    const int64_t dg = deformable_groups;
    TORCH_CHECK(dg > 0, "deformable_groups must be positive, got ", dg);
    TORCH_CHECK(C % dg == 0, "channels (", C,
                ") not divisible by deformable_groups (", dg, ")");
    TORCH_CHECK(off.size(0) == 2 * dg * kk && off.size(1) == out_h &&
                off.size(2) == out_w,
                "offset shape mismatch: expected (", 2 * dg * kk, ", ", out_h,
                ", ", out_w, "), got ", off.sizes());

    const bool use_mask = mask.defined() && mask.numel() > 0;
    at::Tensor msk;
    if (use_mask) {
        TORCH_CHECK(mask.device().is_mps() && mask.scalar_type() == at::kFloat,
                    "mask must be a float32 MPS tensor");
        msk = mask.contiguous();
        TORCH_CHECK(msk.dim() == 3 && msk.size(0) == dg * kk &&
                    msk.size(1) == out_h && msk.size(2) == out_w,
                    "mask shape mismatch: expected (", dg * kk, ", ", out_h,
                    ", ", out_w, "), got ", msk.sizes());
    }

    // The kernel accumulates -> must start from zeros. ATen op: stays OUTSIDE
    // dispatch_sync (deadlock rule).
    auto grad_im = at::zeros({C, height, width}, col.options());

    const int64_t total = C * kk * out_h * out_w;  // one thread per col element
    if (total == 0 || grad_im.numel() == 0) return grad_im;

    DeformConvParams params;
    params.batch = 1;
    params.channels = static_cast<int>(C);
    params.height = static_cast<int>(height);
    params.width = static_cast<int>(width);
    params.kernel_h = static_cast<int>(kernel_h);
    params.kernel_w = static_cast<int>(kernel_w);
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

    id<MTLCommandBuffer> cmd_buf = torch::mps::get_command_buffer();
    TORCH_CHECK(cmd_buf != nil, "Could not obtain an MPS command buffer");
    dispatch_queue_t q = torch::mps::get_dispatch_queue();
    dispatch_sync(q, ^{
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];
            id<MTLComputePipelineState> pso = pipeline_for("deformable_col2im");
            [enc setComputePipelineState:pso];
            [enc setBuffer:getMTLBufferStorage(col)
                    offset:col.storage_offset() * col.element_size() atIndex:0];
            [enc setBuffer:getMTLBufferStorage(off)
                    offset:off.storage_offset() * off.element_size() atIndex:1];
            if (use_mask) {
                [enc setBuffer:getMTLBufferStorage(msk)
                        offset:msk.storage_offset() * msk.element_size() atIndex:2];
            } else {
                // Placeholder so index 2 is always bound; never read (use_mask=0).
                [enc setBuffer:getMTLBufferStorage(off)
                        offset:off.storage_offset() * off.element_size() atIndex:2];
            }
            [enc setBuffer:getMTLBufferStorage(grad_im)
                    offset:grad_im.storage_offset() * grad_im.element_size() atIndex:3];
            [enc setBytes:&params length:sizeof(params) atIndex:4];

            NSUInteger tg = MIN((NSUInteger)pso.maxTotalThreadsPerThreadgroup,
                                (NSUInteger)total);
            [enc dispatchThreads:MTLSizeMake((NSUInteger)total, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc endEncoding];
            torch::mps::commit();
        }
    });
    return grad_im;
}

// ---------------------------------------------------------------------------
// Backward building block: deformable_col2im_coord for ONE image (Phase 3,
// Step 3). Consumes the column-buffer gradient (weight^T @ grad_output) plus
// the forward input, offset and mask; returns (grad_offset, grad_mask).
// grad_mask is an empty tensor when mask is empty (DCNv1).
//
// Every thread writes a unique output element -> no atomics, and at::empty
// would suffice; at::zeros keeps the "grad buffers start zeroed" invariant
// shared with col2im.
// ---------------------------------------------------------------------------
std::tuple<at::Tensor, at::Tensor> deformable_col2im_coord(
        const at::Tensor& columns,   // (C*kh*kw, out_h*out_w)
        const at::Tensor& input,     // (C, H, W)
        const at::Tensor& offset,    // (2*dg*kh*kw, out_h, out_w)
        const at::Tensor& mask,      // (dg*kh*kw, out_h, out_w) or empty
        int64_t kernel_h, int64_t kernel_w,
        int64_t stride_h, int64_t stride_w,
        int64_t pad_h, int64_t pad_w,
        int64_t dilation_h, int64_t dilation_w,
        int64_t deformable_groups) {
    TORCH_CHECK(columns.device().is_mps() && input.device().is_mps() &&
                offset.device().is_mps(),
                "deformable_col2im_coord: columns, input and offset must be MPS tensors");
    TORCH_CHECK(columns.scalar_type() == at::kFloat &&
                input.scalar_type() == at::kFloat &&
                offset.scalar_type() == at::kFloat,
                "deformable_col2im_coord: only float32 is supported");
    TORCH_CHECK(columns.dim() == 2, "columns must be 2-D (C*kh*kw, oh*ow)");
    TORCH_CHECK(input.dim() == 3, "input must be 3-D (C, H, W)");
    TORCH_CHECK(offset.dim() == 3, "offset must be 3-D (2*dg*kh*kw, oh, ow)");
    TORCH_CHECK(kernel_h > 0 && kernel_w > 0, "kernel dims must be positive");
    TORCH_CHECK(stride_h > 0 && stride_w > 0, "stride must be positive");
    TORCH_CHECK(dilation_h > 0 && dilation_w > 0, "dilation must be positive");

    auto col = columns.contiguous();
    auto x = input.contiguous();
    auto off = offset.contiguous();

    const int64_t C = x.size(0), H = x.size(1), W = x.size(2);
    const int64_t kk = kernel_h * kernel_w;
    TORCH_CHECK(col.size(0) == C * kk,
                "columns rows (", col.size(0), ") != C*kh*kw (", C * kk, ")");

    const int64_t out_h =
        (H + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int64_t out_w =
        (W + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    TORCH_CHECK(col.size(1) == out_h * out_w,
                "columns cols (", col.size(1), ") != out_h*out_w (", out_h * out_w, ")");

    const int64_t dg = deformable_groups;
    TORCH_CHECK(dg > 0, "deformable_groups must be positive, got ", dg);
    TORCH_CHECK(C % dg == 0, "channels (", C,
                ") not divisible by deformable_groups (", dg, ")");
    TORCH_CHECK(off.size(0) == 2 * dg * kk && off.size(1) == out_h &&
                off.size(2) == out_w,
                "offset shape mismatch: expected (", 2 * dg * kk, ", ", out_h,
                ", ", out_w, "), got ", off.sizes());

    const bool use_mask = mask.defined() && mask.numel() > 0;
    at::Tensor msk;
    if (use_mask) {
        TORCH_CHECK(mask.device().is_mps() && mask.scalar_type() == at::kFloat,
                    "mask must be a float32 MPS tensor");
        msk = mask.contiguous();
        TORCH_CHECK(msk.dim() == 3 && msk.size(0) == dg * kk &&
                    msk.size(1) == out_h && msk.size(2) == out_w,
                    "mask shape mismatch: expected (", dg * kk, ", ", out_h,
                    ", ", out_w, "), got ", msk.sizes());
    }

    // ATen allocations: OUTSIDE dispatch_sync (deadlock rule).
    auto grad_offset = at::zeros({2 * dg * kk, out_h, out_w}, off.options());
    auto grad_mask = use_mask
        ? at::zeros({dg * kk, out_h, out_w}, off.options())
        : at::empty({0}, off.options());

    const int64_t total = 2 * dg * kk * out_h * out_w;  // one thread per offset elem
    if (total == 0) return std::make_tuple(grad_offset, grad_mask);

    DeformConvParams params;
    params.batch = 1;
    params.channels = static_cast<int>(C);
    params.height = static_cast<int>(H);
    params.width = static_cast<int>(W);
    params.kernel_h = static_cast<int>(kernel_h);
    params.kernel_w = static_cast<int>(kernel_w);
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

    id<MTLCommandBuffer> cmd_buf = torch::mps::get_command_buffer();
    TORCH_CHECK(cmd_buf != nil, "Could not obtain an MPS command buffer");
    dispatch_queue_t q = torch::mps::get_dispatch_queue();
    dispatch_sync(q, ^{
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];
            id<MTLComputePipelineState> pso = pipeline_for("deformable_col2im_coord");
            [enc setComputePipelineState:pso];
            [enc setBuffer:getMTLBufferStorage(col)
                    offset:col.storage_offset() * col.element_size() atIndex:0];
            [enc setBuffer:getMTLBufferStorage(x)
                    offset:x.storage_offset() * x.element_size() atIndex:1];
            [enc setBuffer:getMTLBufferStorage(off)
                    offset:off.storage_offset() * off.element_size() atIndex:2];
            if (use_mask) {
                [enc setBuffer:getMTLBufferStorage(msk)
                        offset:msk.storage_offset() * msk.element_size() atIndex:3];
            } else {
                // Placeholder so index 3 is always bound; never read (use_mask=0).
                [enc setBuffer:getMTLBufferStorage(off)
                        offset:off.storage_offset() * off.element_size() atIndex:3];
            }
            [enc setBuffer:getMTLBufferStorage(grad_offset)
                    offset:grad_offset.storage_offset() * grad_offset.element_size()
                   atIndex:4];
            if (use_mask) {
                [enc setBuffer:getMTLBufferStorage(grad_mask)
                        offset:grad_mask.storage_offset() * grad_mask.element_size()
                       atIndex:5];
            } else {
                // Placeholder (writable target, never written when use_mask=0).
                [enc setBuffer:getMTLBufferStorage(grad_offset)
                        offset:grad_offset.storage_offset() * grad_offset.element_size()
                       atIndex:5];
            }
            [enc setBytes:&params length:sizeof(params) atIndex:6];

            NSUInteger tg = MIN((NSUInteger)pso.maxTotalThreadsPerThreadgroup,
                                (NSUInteger)total);
            [enc dispatchThreads:MTLSizeMake((NSUInteger)total, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc endEncoding];
            torch::mps::commit();
        }
    });
    return std::make_tuple(grad_offset, grad_mask);
}

// ---------------------------------------------------------------------------
// Backward: fused deform_conv2d_backward (Phase 3, Step 4). Mirrors
// torchvision's op: returns (grad_input, grad_offset, grad_mask, grad_weight,
// grad_bias). grad_input/grad_offset/grad_mask come from the Metal kernels;
// grad_weight/grad_bias from plain ATen ops.
//
// Per batch element (threading rule: every ATen op OUTSIDE dispatch_sync;
// the kernel dispatches re-fetch the command buffer internally):
//   1. grad_columns = W^T @ grad_output[n] — grouped (Phase 5): bmm of
//      w_g.transpose(1,2) (g, C/g*kh*kw, outC/g) with grad_output[n] viewed
//      as (g, outC/g, out_hw); result viewed flat as (C*kh*kw, out_hw)
//   2. coord kernel(grad_columns, x[n], off[n], msk[n])
//        -> grad_offset[n], grad_mask[n]
//   3. col2im kernel(grad_columns, off[n], msk[n]) -> grad_input[n]
//      (col2im / col2im_coord consume the full-C column buffer and only
//      know about dg — groups never reaches them)
//   4. im2col kernel(x[n], off[n], msk[n]) -> columns (recomputed, as the
//      reference does — cheaper than saving N column buffers from forward)
//   5. grad_weight += grad_output[n] @ columns^T — grouped: bmm of go_g
//      with cols_g.transpose(1,2), accumulated into (g, outC/g, C/g*kh*kw)
// After the loop: grad_bias = grad_output.sum({0, 2, 3}) if bias defined.
// All grouped views are zero-copy (contiguous buffers; view() throws rather
// than copies); groups == 1 degenerates to the old flat mm math.
//
// Steps 2 and 3 reuse the single-image ops validated by the diag ladder;
// select(0, n) slices stay views (contiguous, storage_offset-based), which
// their buffer bindings honour.
// ---------------------------------------------------------------------------
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
deform_conv2d_backward(
        const at::Tensor& grad_output,  // (N, outC, out_h, out_w)
        const at::Tensor& input,        // (N, C, H, W)
        const at::Tensor& weight,       // (outC, C/groups, kh, kw)
        const at::Tensor& offset,       // (N, 2*dg*kh*kw, out_h, out_w)
        const at::Tensor& mask,         // (N, dg*kh*kw, out_h, out_w) or empty
        const c10::optional<at::Tensor>& bias,
        int64_t stride_h, int64_t stride_w,
        int64_t pad_h, int64_t pad_w,
        int64_t dilation_h, int64_t dilation_w,
        int64_t groups, int64_t deformable_groups) {
    // --- Validation (mirrors forward) + grad_output shape ---------------------
    TORCH_CHECK(grad_output.device().is_mps() && input.device().is_mps() &&
                weight.device().is_mps() && offset.device().is_mps(),
                "deform_conv2d_backward: grad_output, input, weight and offset "
                "must be MPS tensors");
    TORCH_CHECK(grad_output.scalar_type() == at::kFloat &&
                input.scalar_type() == at::kFloat &&
                weight.scalar_type() == at::kFloat &&
                offset.scalar_type() == at::kFloat,
                "deform_conv2d_backward: only float32 is supported (Phase 3)");
    TORCH_CHECK(input.dim() == 4, "input must be 4-D, got ", input.dim());
    TORCH_CHECK(weight.dim() == 4, "weight must be 4-D, got ", weight.dim());
    TORCH_CHECK(offset.dim() == 4, "offset must be 4-D, got ", offset.dim());
    TORCH_CHECK(grad_output.dim() == 4, "grad_output must be 4-D, got ",
                grad_output.dim());
    TORCH_CHECK(groups > 0, "groups must be positive, got ", groups);
    TORCH_CHECK(stride_h > 0 && stride_w > 0, "stride must be positive");
    TORCH_CHECK(dilation_h > 0 && dilation_w > 0, "dilation must be positive");

    auto go = grad_output.contiguous();
    auto x = input.contiguous();
    auto w = weight.contiguous();
    auto off = offset.contiguous();

    const int64_t N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    const int64_t outC = w.size(0), kh = w.size(2), kw = w.size(3);
    TORCH_CHECK(w.size(1) * groups == C,
                "weight shape mismatch: expected in-channels ", C,
                ", got ", w.size(1) * groups);
    TORCH_CHECK(C % groups == 0, "channels (", C,
                ") not divisible by groups (", groups, ")");
    TORCH_CHECK(outC % groups == 0, "out_channels (", outC,
                ") not divisible by groups (", groups, ")");

    const int64_t out_h =
        (H + 2 * pad_h - dilation_h * (kh - 1) - 1) / stride_h + 1;
    const int64_t out_w =
        (W + 2 * pad_w - dilation_w * (kw - 1) - 1) / stride_w + 1;
    const int64_t out_hw = out_h * out_w;
    TORCH_CHECK(go.size(0) == N && go.size(1) == outC &&
                go.size(2) == out_h && go.size(3) == out_w,
                "grad_output shape mismatch: expected (", N, ", ", outC, ", ",
                out_h, ", ", out_w, "), got ", go.sizes());

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

    const bool use_bias = bias.has_value() && bias->defined() &&
                          bias->numel() > 0;

    // --- Allocation (all ATen, all outside dispatch_sync) ---------------------
    // Kernels + weight GEMM accumulate -> grads start zeroed.
    auto grad_input = at::zeros_like(x);
    auto grad_offset = at::zeros_like(off);
    auto grad_mask = use_mask ? at::zeros_like(msk)
                              : at::empty({0}, x.options());
    // Grouped accumulator/views (Phase 5): mirror of the forward's bmm views.
    // All zero-copy (contiguous buffers; view() throws rather than copies).
    const int64_t cpg_kk = (C / groups) * kh * kw;  // rows per group
    auto grad_weight_g =
        at::zeros({groups, outC / groups, cpg_kk}, w.options());
    auto columns = at::empty({C * kh * kw, out_hw}, x.options());
    auto empty_mps = at::empty({0}, x.options());  // mask stand-in (DCNv1)

    auto w_g = w.view({groups, outC / groups, cpg_kk});
    auto cols_g = columns.view({groups, cpg_kk, out_hw});

    // Zero-size edge cases: nothing to dispatch; zeroed grads are the answer.
    const bool skip_loop = (N == 0 || C == 0 || outC == 0 || out_hw == 0);

    if (!skip_loop) {
        // im2col recompute setup (identical to the forward's dispatch).
        DeformConvParams params;
        params.batch = 1;
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

        const int64_t im_plane = C * H * W;
        const int64_t off_plane = 2 * dg * kh * kw * out_hw;
        const int64_t msk_plane = dg * kh * kw * out_hw;
        const int64_t im2col_total = C * out_hw;

        id<MTLComputePipelineState> pso_im2col = pipeline_for("deformable_im2col");
        dispatch_queue_t q = torch::mps::get_dispatch_queue();

        for (int64_t n = 0; n < N; ++n) {
            auto off_n = off.select(0, n);
            auto msk_n = use_mask ? msk.select(0, n) : empty_mps;
            auto go_g = go.select(0, n).view({groups, outC / groups, out_hw});

            // 1. Column-buffer gradient (ATen, outside any dispatch block).
            // Grouped W^T @ grad_out: (g, cpg_kk, outC/g) @ (g, outC/g,
            // out_hw); bmm output is contiguous, so the flat view the
            // kernels consume is zero-copy.
            auto grad_columns = at::bmm(w_g.transpose(1, 2), go_g)
                                    .view({C * kh * kw, out_hw});

            // 2. grad_offset[n], grad_mask[n] (coord kernel; no atomics).
            auto coord_grads = deformable_col2im_coord(
                grad_columns, x.select(0, n), off_n, msk_n,
                kh, kw, stride_h, stride_w, pad_h, pad_w,
                dilation_h, dilation_w, dg);
            grad_offset.select(0, n).copy_(std::get<0>(coord_grads));
            if (use_mask) {
                grad_mask.select(0, n).copy_(std::get<1>(coord_grads));
            }

            // 3. grad_input[n] (col2im kernel; atomic scatter).
            grad_input.select(0, n).copy_(deformable_col2im(
                grad_columns, off_n, msk_n, H, W, kh, kw,
                stride_h, stride_w, pad_h, pad_w,
                dilation_h, dilation_w, dg));

            // 4. Recompute forward columns for image n (same dispatch as the
            // forward pass; command buffer re-fetched — the interleaved ATen
            // ops may have committed and recycled it).
            id<MTLCommandBuffer> cmd_buf = torch::mps::get_command_buffer();
            TORCH_CHECK(cmd_buf != nil, "Could not obtain an MPS command buffer");
            dispatch_sync(q, ^{
                @autoreleasepool {
                    id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];
                    [enc setComputePipelineState:pso_im2col];
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
                        [enc setBuffer:getMTLBufferStorage(off)
                                offset:off.storage_offset() * off.element_size()
                               atIndex:2];
                    }
                    [enc setBuffer:getMTLBufferStorage(columns)
                            offset:columns.storage_offset() * columns.element_size()
                           atIndex:3];
                    [enc setBytes:&params length:sizeof(params) atIndex:4];

                    NSUInteger tg = MIN((NSUInteger)pso_im2col.maxTotalThreadsPerThreadgroup,
                                        (NSUInteger)im2col_total);
                    [enc dispatchThreads:MTLSizeMake((NSUInteger)im2col_total, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                    [enc endEncoding];
                    torch::mps::commit();
                }
            });

            // 5. Accumulate grad_weight (ATen, outside the dispatch block).
            // Grouped grad_out @ columns^T: (g, outC/g, out_hw) @
            // (g, out_hw, cpg_kk) -> (g, outC/g, cpg_kk).
            grad_weight_g.add_(at::bmm(go_g, cols_g.transpose(1, 2)));
        }
    }

    auto grad_weight = grad_weight_g.view({outC, C / groups, kh, kw});
    auto grad_bias = use_bias ? go.sum(at::IntArrayRef{0, 2, 3})
                              : at::empty({0}, x.options());

    return std::make_tuple(grad_input, grad_offset, grad_mask,
                           grad_weight, grad_bias);
}

TORCH_LIBRARY(deform_conv2d_mps, m) {
    m.def("add_one(Tensor input) -> Tensor");
    m.def("atomic_smoke(Tensor input) -> Tensor");
    m.def(
        "deformable_col2im(Tensor columns, Tensor offset, Tensor mask, "
        "int height, int width, int kernel_h, int kernel_w, int stride_h, "
        "int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, "
        "int deformable_groups) -> Tensor");
    m.def(
        "deformable_col2im_coord(Tensor columns, Tensor input, Tensor offset, "
        "Tensor mask, int kernel_h, int kernel_w, int stride_h, int stride_w, "
        "int pad_h, int pad_w, int dilation_h, int dilation_w, "
        "int deformable_groups) -> (Tensor, Tensor)");
    m.def(
        "deform_conv2d_forward(Tensor input, Tensor weight, Tensor offset, "
        "Tensor mask, Tensor? bias, int stride_h, int stride_w, int pad_h, "
        "int pad_w, int dilation_h, int dilation_w, int groups, "
        "int deformable_groups) -> Tensor");
    m.def(
        "deform_conv2d_backward(Tensor grad_output, Tensor input, "
        "Tensor weight, Tensor offset, Tensor mask, Tensor? bias, "
        "int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, "
        "int dilation_w, int groups, int deformable_groups) "
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(deform_conv2d_mps, MPS, m) {
    m.impl("add_one", TORCH_FN(add_one));
    m.impl("atomic_smoke", TORCH_FN(atomic_smoke));
    m.impl("deformable_col2im", TORCH_FN(deformable_col2im));
    m.impl("deformable_col2im_coord", TORCH_FN(deformable_col2im_coord));
    m.impl("deform_conv2d_forward", TORCH_FN(deform_conv2d_forward));
    m.impl("deform_conv2d_backward", TORCH_FN(deform_conv2d_backward));
}

// pybind: expose the one-time library compile hook to Python.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("_compile_library", &dcn_compile_library,
          "Compile the MSL source into the cached Metal library");
}
