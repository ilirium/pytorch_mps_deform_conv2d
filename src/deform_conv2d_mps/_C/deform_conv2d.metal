// deform_conv2d.metal — Metal Shading Language kernels for deformable conv2d.
//
// THREE kernels are needed (everything else reuses torch ops):
//   1. deformable_im2col        (forward gather)        -- IMPLEMENTED (single-group reference)
//   2. deformable_col2im        (backward -> grad input)   -- STUB
//   3. deformable_col2im_coord  (backward -> grad offset/mask) -- STUB
//
// Plus `add_one` — a trivial Phase-0 kernel to validate the .mm -> Metal ->
// tensor pipeline end to end before the deform logic is trusted.
//
// Port reference: torchvision/csrc/ops/cuda/deform_conv2d_kernel.cu
// NOTE: this reference covers the common case (groups == deformable_groups == 1,
// batch handled by the host loop / leading index). Extend the index math for
// groups and deformable_groups as you move past Phase 1 — see TODOs.

#include <metal_stdlib>
#include <metal_atomic>
#include "bilinear.metalh"
using namespace metal;

// ---------------------------------------------------------------------------
// Phase 0: sanity kernel. out[i] = in[i] + 1
// ---------------------------------------------------------------------------
kernel void add_one(
        const device float* in   [[buffer(0)]],
        device float*       out  [[buffer(1)]],
        constant uint&      n    [[buffer(2)]],
        uint                gid  [[thread_position_in_grid]]) {
    if (gid < n) {
        out[gid] = in[gid] + 1.0f;
    }
}

// ---------------------------------------------------------------------------
// Phase 3, Step 1: atomic smoke kernel. Proves the compiled language version
// (MSL >= 3.0) accepts `device atomic_float*` + atomic_fetch_add_explicit —
// the exact primitive deformable_col2im's scatter-add relies on. A language-
// version bump alone can "pass" while atomics still fail, so this must
// compile AND run. Every thread adds in[gid] into out[gid % 4].
// ---------------------------------------------------------------------------
kernel void atomic_smoke(
        const device float*  in   [[buffer(0)]],
        device atomic_float* out  [[buffer(1)]],  // 4 accumulator slots, pre-zeroed
        constant uint&       n    [[buffer(2)]],
        uint                 gid  [[thread_position_in_grid]]) {
    if (gid < n) {
        atomic_fetch_add_explicit(&out[gid % 4], in[gid], memory_order_relaxed);
    }
}

// Parameter block shared by the deform kernels. Keep field order in sync with
// the struct defined host-side in the .mm file.
struct DeformConvParams {
    int batch;
    int channels;       // input channels
    int height, width;  // input spatial
    int kernel_h, kernel_w;
    int pad_h, pad_w;
    int stride_h, stride_w;
    int dilation_h, dilation_w;
    int out_h, out_w;
    int deformable_groups;
    int channels_per_deformable_group;
    int use_mask;       // 1 for DCNv2, 0 for DCNv1
};

// ---------------------------------------------------------------------------
// 1. deformable_im2col  (FORWARD) -- reference implementation
//
// One thread per (channel, out_y, out_x) within a single batch image. The host
// launches it per batch element (or folds batch into the grid). Produces the
// column buffer consumed by the GEMM: columns[c*kh*kw + ki*kw + kj, oy*ow+ox].
// ---------------------------------------------------------------------------
kernel void deformable_im2col(
        const device float* data_im     [[buffer(0)]],  // (C, H, W) for this image
        const device float* data_offset [[buffer(1)]],  // (2*dg*kh*kw, out_h, out_w)
        const device float* data_mask   [[buffer(2)]],  // (dg*kh*kw, out_h, out_w) or unused
        device float*       data_col    [[buffer(3)]],  // (C*kh*kw, out_h*out_w)
        constant DeformConvParams& p    [[buffer(4)]],
        uint gid                        [[thread_position_in_grid]]) {

    int out_hw = p.out_h * p.out_w;
    int total = p.channels * out_hw;   // one thread per (c, oy, ox)
    if ((int)gid >= total) return;

    int ox = (int)gid % p.out_w;
    int oy = ((int)gid / p.out_w) % p.out_h;
    int c  = (int)gid / out_hw;

    int dg = (p.deformable_groups > 0) ? p.deformable_groups : 1;
    int deformable_group_index = c / p.channels_per_deformable_group;
    if (deformable_group_index >= dg) deformable_group_index = dg - 1;

    const device float* im_plane = data_im + c * p.height * p.width;

    // Base of this output column entry for channel c.
    int col_step = p.kernel_h * p.kernel_w;
    device float* col_ptr = data_col + (c * col_step) * out_hw + (oy * p.out_w + ox);

    int in_y0 = oy * p.stride_h - p.pad_h;
    int in_x0 = ox * p.stride_w - p.pad_w;

    for (int ki = 0; ki < p.kernel_h; ++ki) {
        for (int kj = 0; kj < p.kernel_w; ++kj) {
            int k = ki * p.kernel_w + kj;

            int off_h_idx = ((2 * (deformable_group_index * col_step + k)    ) * p.out_h + oy) * p.out_w + ox;
            int off_w_idx = ((2 * (deformable_group_index * col_step + k) + 1) * p.out_h + oy) * p.out_w + ox;
            float off_h = data_offset[off_h_idx];
            float off_w = data_offset[off_w_idx];

            float h_im = in_y0 + ki * p.dilation_h + off_h;
            float w_im = in_x0 + kj * p.dilation_w + off_w;

            float val = bilinear_interpolate(im_plane, p.height, p.width, h_im, w_im);

            if (p.use_mask) {
                int m_idx = ((deformable_group_index * col_step + k) * p.out_h + oy) * p.out_w + ox;
                val *= data_mask[m_idx];
            }

            col_ptr[k * out_hw] = val;
        }
    }
}

// ---------------------------------------------------------------------------
// 2. deformable_col2im  (BACKWARD -> grad_input)  -- STUB
//
// Scatter the column gradient back into grad_input using the same bilinear
// weights as the forward gather. Requires atomic float adds because multiple
// (k, oy, ox) taps can land on the same input pixel.
//
// TODO(Phase 3): port from torchvision deformable_col2im_kernel. Use
//   atomic_fetch_add_explicit on `device atomic_float*` for the four corners.
// ---------------------------------------------------------------------------
// NOTE(Phase 3): grad_im must become `device atomic_float*` for the scatter-
// add. The language version is now MTLLanguageVersion3_0 (set in the .mm,
// Step 1) and `atomic_smoke` above proves atomics compile + run; the signature
// flips when this stub is replaced in Step 2.
kernel void deformable_col2im(
        const device float*  data_col     [[buffer(0)]],
        const device float*  data_offset  [[buffer(1)]],
        const device float*  data_mask    [[buffer(2)]],
        device float*        grad_im      [[buffer(3)]],
        constant DeformConvParams& p      [[buffer(4)]],
        uint gid                          [[thread_position_in_grid]]) {
    // STUB — see TODO above.
    (void)data_col; (void)data_offset; (void)data_mask;
    (void)grad_im; (void)p; (void)gid;
}

// ---------------------------------------------------------------------------
// 3. deformable_col2im_coord  (BACKWARD -> grad_offset, grad_mask)  -- STUB
//
// For each offset element, accumulate the gradient via get_coordinate_weight,
// and (DCNv2) accumulate grad_mask from the plain bilinear sample.
//
// TODO(Phase 3): port from torchvision deformable_col2im_coord_kernel.
// ---------------------------------------------------------------------------
kernel void deformable_col2im_coord(
        const device float*  data_col     [[buffer(0)]],
        const device float*  data_im      [[buffer(1)]],
        const device float*  data_offset  [[buffer(2)]],
        const device float*  data_mask    [[buffer(3)]],
        device float*        grad_offset  [[buffer(4)]],
        device float*        grad_mask    [[buffer(5)]],
        constant DeformConvParams& p      [[buffer(6)]],
        uint gid                          [[thread_position_in_grid]]) {
    // STUB — see TODO above.
    (void)data_col; (void)data_im; (void)data_offset; (void)data_mask;
    (void)grad_offset; (void)grad_mask; (void)p; (void)gid;
}
