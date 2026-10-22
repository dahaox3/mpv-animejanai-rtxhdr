# CUDA / D3D11 RTX HDR bridge

This branch starts from the mpv commit shipped in AnimeJaNai 3.7.0:
`d6d93599d59069b715013a6e77e898056499b905`.

The experimental `cuda-d3d11` video filter converts native AnimeJaNai CUDA
YUV444P16 output to full-range D3D11 RGB10A2, without downloading the picture
to host memory. It can precede `d3d11vpp=nvidia-true-hdr=yes` while the
inference provider remains TensorRT.

Pipeline:

```
copy/software decode -> hwupload_cuda -> animejanai (TensorRT, output-444=yes)
                    -> cuda-d3d11 -> d3d11vpp=nvidia-true-hdr=yes
```

Use `vo=gpu-next` and `gpu-api=d3d11`. The current CUDA VO interop still
requires Vulkan/OpenGL for direct NVDEC output; this first implementation
removes the high-resolution round trip after inference, not every possible
copy before it. The aji ABI and existing inference libraries are unchanged.

The bridge accepts BT.601, BT.709 and BT.2020 non-constant-luminance matrices,
full/limited range, and positive per-plane strides. Output transfer function,
primaries, timestamps, crop and side data follow the source; the bridge only
changes matrix representation, range, bit packing and memory API. It does not
perform HDR conversion itself. HDR sources therefore remain HDR-tagged for
the downstream VPP to bypass SDR-to-HDR conversion.

RGB10A2 CUDA registration has been verified on an RTX 4070 Ti with driver
617.42, but is not listed in NVIDIA's published D3D11 interoperability format
table. Unsupported registration fails with an explicit diagnostic. This
version never silently reduces precision to 8-bit. RGBA16 plus a D3D11 shader
is a possible future compatibility path, not part of this first version.

Textures are allocated with the mpv image pool. Each surface keeps a reference
to its own CUDA session/device, so queued downstream frames can outlive filter
removal or a resolution/context change. Registration occurs at allocation,
not per frame. Mapping coordinates reuse with previously submitted D3D11 work.
The short bridge stream is synchronized before releasing the CUDA input;
native inference retains its existing upstream queue and CUDA graphs.

The PTX is embedded. Ordinary mpv builds need ffnvcodec headers and D3D11,
not a CUDA Toolkit. To regenerate the header after changing the CUDA source:

```
python TOOLS/cuda_d3d11_ptx.py --nvcc /path/to/nvcc
```

The initial generated header uses CUDA 12.8 / compute_75 PTX. It contains
ordinary arithmetic and packing kernels; runtime compilation is done by the
installed NVIDIA driver. Keep the CUDA source and generated header together.

Ordering note: putting RTX HDR at the end of `vf` covers video filters, not
the later GLSL hooks in the VO. Manually enabling HDR does not change that
stage order. A separate future VO change is needed if HDR must also follow
all GLSL processing, with proper subtitle/OSD and color-management handling.
