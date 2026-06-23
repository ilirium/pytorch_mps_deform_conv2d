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
        g_library = [g_device newLibraryWithSource:src options:opts error:&err];
        TORCH_CHECK(g_library != nil, "Failed to compile Metal library: ",
                    err ? err.localizedDescription.UTF8String : "unknown error");
    }
}

static id<MTLComputePipelineState> pipeline_for(const char* fn_name) {
    TORCH_CHECK(g_library != nil,
                "Metal library not compiled. Call _compile_library() first.");
    @autoreleasepool {
        id<MTLFunction> fn =
            [g_library newFunctionWithName:[NSString stringWithUTF8String:fn_name]];
        TORCH_CHECK(fn != nil, "Metal function not found: ", fn_name);
        NSError* err = nil;
        id<MTLComputePipelineState> pso =
            [g_device newComputePipelineStateWithFunction:fn error:&err];
        TORCH_CHECK(pso != nil, "Failed to build pipeline for ", fn_name, ": ",
                    err ? err.localizedDescription.UTF8String : "unknown error");
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
// Forward: im2col (Metal) -> GEMM (ATen matmul) -> bias.
//
// SCAFFOLD: dispatch deformable_im2col per batch element into a column buffer,
// then `out = (weight.view({outC, -1}) @ columns)` reshaped to
// (N, outC, out_h, out_w); add bias. Wire this up in Phase 1, then validate
// with tests/test_forward.py before flipping _NATIVE_READY in ops.py.
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
    TORCH_CHECK(input.device().is_mps(), "deform_conv2d_forward expects MPS tensors");
    (void)weight; (void)offset; (void)mask; (void)bias;
    (void)stride_h; (void)stride_w; (void)pad_h; (void)pad_w;
    (void)dilation_h; (void)dilation_w; (void)groups; (void)deformable_groups;
    TORCH_CHECK(false,
        "deform_conv2d_forward: native forward not implemented yet (scaffold). "
        "Until implemented, the Python layer falls back to the torchvision CPU reference.");
    return at::Tensor();
}

TORCH_LIBRARY(deform_conv2d_mps, m) {
    m.def("add_one(Tensor input) -> Tensor");
    m.def(
        "deform_conv2d_forward(Tensor input, Tensor weight, Tensor offset, "
        "Tensor mask, Tensor? bias, int stride_h, int stride_w, int pad_h, "
        "int pad_w, int dilation_h, int dilation_w, int groups, "
        "int deformable_groups) -> Tensor");
}

TORCH_LIBRARY_IMPL(deform_conv2d_mps, MPS, m) {
    m.impl("add_one", TORCH_FN(add_one));
    m.impl("deform_conv2d_forward", TORCH_FN(deform_conv2d_forward));
}

// pybind: expose the one-time library compile hook to Python.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("_compile_library", &dcn_compile_library,
          "Compile the MSL source into the cached Metal library");
}
