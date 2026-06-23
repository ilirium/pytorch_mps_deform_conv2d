// deform_conv2d_mps.mm — Objective-C++ host for the deformable-conv Metal kernels.
//
// Responsibilities:
//   * compile the MSL source (passed in from Python) into a cached MTLLibrary
//   * expose `add_one` (Phase-0 pipeline check) and `deform_conv2d_forward`
//   * register everything with PyTorch via TORCH_LIBRARY
//
// Dispatch uses PyTorch's own MPS command buffer/queue so our kernels stay in
// stream with surrounding ATen ops. Backward is intentionally left as a stub;
// see TODO(Phase 3).

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/mps/MPSStream.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mutex>
#include <string>

// ---------------------------------------------------------------------------
// Library cache. The MSL source is supplied once from Python (so the shaders
// live as .metal files in the package, not duplicated as C strings here).
// ---------------------------------------------------------------------------
static id<MTLLibrary> g_library = nil;
static std::once_flag g_lib_once;

static id<MTLDevice> mps_device() {
    // Reuse the device backing the current MPS stream.
    return MPSDevice::getInstance()->device();  // ATen exposes this in MPSDevice.h
}

void dcn_compile_library(const std::string& msl_source) {
    @autoreleasepool {
        id<MTLDevice> device = mps_device();
        NSError* err = nil;
        NSString* src = [NSString stringWithUTF8String:msl_source.c_str()];
        MTLCompileOptions* opts = [MTLCompileOptions new];
        g_library = [device newLibraryWithSource:src options:opts error:&err];
        TORCH_CHECK(g_library != nil, "Failed to compile Metal library: ",
                    err ? err.localizedDescription.UTF8String : "unknown error");
    }
}

static id<MTLComputePipelineState> pipeline_for(const char* fn_name) {
    TORCH_CHECK(g_library != nil,
                "Metal library not compiled. Call _compile_library() first.");
    @autoreleasepool {
        id<MTLDevice> device = mps_device();
        id<MTLFunction> fn =
            [g_library newFunctionWithName:[NSString stringWithUTF8String:fn_name]];
        TORCH_CHECK(fn != nil, "Metal function not found: ", fn_name);
        NSError* err = nil;
        id<MTLComputePipelineState> pso =
            [device newComputePipelineStateWithFunction:fn error:&err];
        TORCH_CHECK(pso != nil, "Failed to build pipeline for ", fn_name);
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

    using namespace at::mps;
    MPSStream* stream = getCurrentMPSStream();
    dispatch_queue_t q = stream->queue();
    dispatch_sync(q, ^{
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
            id<MTLComputePipelineState> pso = pipeline_for("add_one");
            [enc setComputePipelineState:pso];
            [enc setBuffer:getMTLBufferStorage(x)   offset:x.storage_offset()   * x.element_size() atIndex:0];
            [enc setBuffer:getMTLBufferStorage(out) offset:out.storage_offset() * out.element_size() atIndex:1];
            [enc setBytes:&n length:sizeof(n) atIndex:2];

            MTLSize grid = MTLSizeMake(n, 1, 1);
            NSUInteger tg = MIN((NSUInteger)pso.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
            [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        }
    });
    return out;
}

// ---------------------------------------------------------------------------
// Forward: im2col (Metal) -> GEMM (ATen matmul) -> bias.
//
// SCAFFOLD: the im2col dispatch loop is sketched; finish buffer wiring against
// the schema in deform_conv2d.metal during Phase 1, then validate with
// tests/test_forward.py before trusting it.
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
    // TODO(Phase 1): allocate column buffer, dispatch deformable_im2col per
    // batch element, then `out = (weight.view({outC, -1}) @ columns)` reshaped
    // to (N, outC, out_h, out_w); add bias if present.
    TORCH_CHECK(false,
        "deform_conv2d_forward: native forward not implemented yet (scaffold). "
        "Until implemented, the Python layer falls back to the torchvision CPU reference.");
    return at::Tensor();
}

// Helper to extract the MTLBuffer behind an MPS tensor.
static id<MTLBuffer> getMTLBufferStorage(const at::Tensor& t) {
    return __builtin_bit_cast(id<MTLBuffer>, t.storage().data());
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
