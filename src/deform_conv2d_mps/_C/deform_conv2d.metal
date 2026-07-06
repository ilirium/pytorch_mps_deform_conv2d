// deform_conv2d.metal — Metal Shading Language kernels for deformable conv2d.
//
// THREE kernels are needed (everything else reuses torch ops):
//   1. deformable_im2col        (forward gather)        -- IMPLEMENTED (single-group reference)
//   2. deformable_col2im        (backward -> grad input)   -- IMPLEMENTED (Phase 3, Step 2)
//   3. deformable_col2im_coord  (backward -> grad offset/mask) -- IMPLEMENTED (Phase 3, Step 3)
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
// 2. deformable_col2im  (BACKWARD -> grad_input)
//
// Scatter the column gradient back into grad_input using the same bilinear
// weights as the forward gather. One thread per column element; the sampling
// location (h_im, w_im) is recomputed with EXACTLY the forward's index math
// (offset interleaving `2*(dgi*kh*kw + k)` / `+1`), then the gradient is
// atomically added into the up-to-4 corners the forward read from.
// Atomics are required: multiple (k, oy, ox) taps can land on the same input
// pixel. grad_im must be pre-zeroed by the host (the kernel only accumulates).
//
// Ported from torchvision deformable_col2im_kernel (batch handled by the
// host loop, so the reference's `b` index is absent from the decomposition).
// ---------------------------------------------------------------------------
kernel void deformable_col2im(
        const device float*  data_col     [[buffer(0)]],  // (C*kh*kw, out_h*out_w)
        const device float*  data_offset  [[buffer(1)]],  // (2*dg*kh*kw, out_h, out_w)
        const device float*  data_mask    [[buffer(2)]],  // (dg*kh*kw, out_h, out_w) or unused
        device atomic_float* grad_im      [[buffer(3)]],  // (C, H, W), pre-zeroed
        constant DeformConvParams& p      [[buffer(4)]],
        uint gid                          [[thread_position_in_grid]]) {

    int out_hw = p.out_h * p.out_w;
    int total = p.channels * p.kernel_h * p.kernel_w * out_hw;
    if ((int)gid >= total) return;

    // gid is the linear index into data_col:
    //   ((c*kh*kw + ki*kw + kj) * out_h + oy) * out_w + ox
    int ox = (int)gid % p.out_w;
    int oy = ((int)gid / p.out_w) % p.out_h;
    int kj = ((int)gid / out_hw) % p.kernel_w;
    int ki = ((int)gid / (out_hw * p.kernel_w)) % p.kernel_h;
    int c  = (int)gid / (out_hw * p.kernel_w * p.kernel_h);

    int dg = (p.deformable_groups > 0) ? p.deformable_groups : 1;
    int deformable_group_index = c / p.channels_per_deformable_group;
    if (deformable_group_index >= dg) deformable_group_index = dg - 1;

    int col_step = p.kernel_h * p.kernel_w;
    int k = ki * p.kernel_w + kj;

    // Same offset/mask indexing as deformable_im2col — copied, not re-derived.
    int off_h_idx = ((2 * (deformable_group_index * col_step + k)    ) * p.out_h + oy) * p.out_w + ox;
    int off_w_idx = ((2 * (deformable_group_index * col_step + k) + 1) * p.out_h + oy) * p.out_w + ox;
    float off_h = data_offset[off_h_idx];
    float off_w = data_offset[off_w_idx];

    float h_im = oy * p.stride_h - p.pad_h + ki * p.dilation_h + off_h;
    float w_im = ox * p.stride_w - p.pad_w + kj * p.dilation_w + off_w;

    float top_grad = data_col[gid];
    if (p.use_mask) {
        int m_idx = ((deformable_group_index * col_step + k) * p.out_h + oy) * p.out_w + ox;
        top_grad *= data_mask[m_idx];
    }

    device atomic_float* grad_plane = grad_im + c * p.height * p.width;

    // Reference's window scan around the sample point: only cells with
    // |delta| < 1 in both axes get a non-zero bilinear weight.
    int cur_h = (int)floor(h_im);
    int cur_w = (int)floor(w_im);
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            int yp = cur_h + dy;
            int xp = cur_w + dx;
            if (yp >= 0 && yp < p.height && xp >= 0 && xp < p.width &&
                fabs(h_im - (float)yp) < 1.0f &&
                fabs(w_im - (float)xp) < 1.0f) {
                float weight = get_gradient_weight(h_im, w_im, yp, xp,
                                                   p.height, p.width);
                atomic_fetch_add_explicit(&grad_plane[yp * p.width + xp],
                                          weight * top_grad,
                                          memory_order_relaxed);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. deformable_col2im_coord  (BACKWARD -> grad_offset, grad_mask)
//
// One thread per offset element: total = 2*dg*kh*kw*out_h*out_w per image
// (batch via host loop). Each thread owns a unique grad_offset element — and,
// for the h-component (bp_dir == 0) threads, the corresponding grad_mask
// element — so NO atomics are needed.
//
// The thread's offset channel decomposes into (dgi, k, bp_dir); it then loops
// over the channels_per_deformable_group input channels that share this
// offset, accumulating
//   grad_offset += col_grad * mask * d(bilinear)/d(coord)   (get_coordinate_weight)
//   grad_mask   += col_grad * bilinear(im)                  (DCNv2, h-thread only)
//
// Ported from torchvision deformable_col2im_coord_kernel. The fully-OOB
// sentinel (inv_h = inv_w = -2) forces zero contributions; partially-OOB taps
// in (-1, 0) fall through to the per-corner guards in the helpers, matching
// the reference.
// ---------------------------------------------------------------------------
kernel void deformable_col2im_coord(
        const device float*  data_col     [[buffer(0)]],  // (C*kh*kw, out_h*out_w) col grad
        const device float*  data_im      [[buffer(1)]],  // (C, H, W) forward input
        const device float*  data_offset  [[buffer(2)]],  // (2*dg*kh*kw, out_h, out_w)
        const device float*  data_mask    [[buffer(3)]],  // (dg*kh*kw, out_h, out_w) or unused
        device float*        grad_offset  [[buffer(4)]],  // (2*dg*kh*kw, out_h, out_w)
        device float*        grad_mask    [[buffer(5)]],  // (dg*kh*kw, out_h, out_w) or unused
        constant DeformConvParams& p      [[buffer(6)]],
        uint gid                          [[thread_position_in_grid]]) {

    int out_hw = p.out_h * p.out_w;
    int col_step = p.kernel_h * p.kernel_w;
    int offset_channels = 2 * p.deformable_groups * col_step;
    int total = offset_channels * out_hw;
    if ((int)gid >= total) return;

    // gid is the linear index into grad_offset: ((c * out_h) + oy) * out_w + ox
    int ox = (int)gid % p.out_w;
    int oy = ((int)gid / p.out_w) % p.out_h;
    int c  = (int)gid / out_hw;

    int dgi = c / (2 * col_step);
    int offset_c = c - dgi * 2 * col_step;
    int bp_dir = offset_c % 2;   // 0 -> d/dh (y), 1 -> d/dw (x)
    int k = offset_c / 2;        // this thread's kernel tap
    int ki = k / p.kernel_w;
    int kj = k % p.kernel_w;

    // Same offset/mask index expressions as the forward gather — copied.
    int off_h_idx = ((2 * (dgi * col_step + k)    ) * p.out_h + oy) * p.out_w + ox;
    int off_w_idx = ((2 * (dgi * col_step + k) + 1) * p.out_h + oy) * p.out_w + ox;
    float off_h = data_offset[off_h_idx];
    float off_w = data_offset[off_w_idx];

    int m_idx = ((dgi * col_step + k) * p.out_h + oy) * p.out_w + ox;
    float mask_value = 1.0f;
    if (p.use_mask) {
        mask_value = data_mask[m_idx];
    }

    float inv_h = oy * p.stride_h - p.pad_h + ki * p.dilation_h + off_h;
    float inv_w = ox * p.stride_w - p.pad_w + kj * p.dilation_w + off_w;
    // Reference's fully-OOB sentinel: both helpers return 0 at (-2, -2).
    if (inv_h <= -1 || inv_w <= -1 || inv_h >= p.height || inv_w >= p.width) {
        inv_h = inv_w = -2.0f;
    }

    int cpg = p.channels_per_deformable_group;
    float val = 0.0f;
    float mval = 0.0f;
    for (int ch = 0; ch < cpg; ++ch) {
        int c_im = dgi * cpg + ch;
        const device float* im_plane = data_im + c_im * p.height * p.width;
        float col_v = data_col[(c_im * col_step + k) * out_hw + oy * p.out_w + ox];

        float weight = get_coordinate_weight(im_plane, p.height, p.width,
                                             inv_h, inv_w, bp_dir);
        val += mask_value * weight * col_v;

        if (p.use_mask && bp_dir == 0) {
            // Mask grad comes from the h-component thread only (reference).
            mval += col_v * bilinear_interpolate(im_plane, p.height, p.width,
                                                 inv_h, inv_w);
        }
    }

    grad_offset[gid] = val;
    if (p.use_mask && bp_dir == 0) {
        grad_mask[m_idx] = mval;
    }
}
