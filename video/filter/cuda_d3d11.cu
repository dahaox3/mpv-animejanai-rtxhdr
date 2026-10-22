/*
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * CUDA YUV444P16 to full-range RGB10A2. No host pixel transfers.
 * Regenerate cuda_d3d11_ptx.h using TOOLS/cuda_d3d11_ptx.py.
 */
extern "C" __global__ void cuda_d3d11_pack(
    const unsigned short *yp, const unsigned short *up,
    const unsigned short *vp, unsigned *out,
    int ys, int us, int vs, int w, int h,
    float kr, float kb, float yoff, float yscale,
    float coff, float cscale)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    float Y = (((const unsigned short *)((const char *)yp + y * (long long)ys))[x] - yoff) / yscale;
    float U = (((const unsigned short *)((const char *)up + y * (long long)us))[x] - coff) / cscale;
    float V = (((const unsigned short *)((const char *)vp + y * (long long)vs))[x] - coff) / cscale;
    float kg = 1.0f - kr - kb;
    float r = Y + 2.0f * (1.0f - kr) * V;
    float g = Y - 2.0f * kb * (1.0f - kb) / kg * U
                - 2.0f * kr * (1.0f - kr) / kg * V;
    float b = Y + 2.0f * (1.0f - kb) * U;
    unsigned R = __float2uint_rn(fminf(1.0f, fmaxf(0.0f, r)) * 1023.0f);
    unsigned G = __float2uint_rn(fminf(1.0f, fmaxf(0.0f, g)) * 1023.0f);
    unsigned B = __float2uint_rn(fminf(1.0f, fmaxf(0.0f, b)) * 1023.0f);
    out[y * (long long)w + x] = R | (G << 10) | (B << 20) | 0xc0000000u;
}
