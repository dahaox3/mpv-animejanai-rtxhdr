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
native inference retains its existing upstream queue and CUDA graphs. The
shared device's ID3D10Multithread lock covers the entire map/write/unmap/sync
ownership interval, including registration and unregistration. Locking only
individual D3D11 calls was insufficient in the first continuous playback test
and caused a device hang; that earlier revision must not be shipped.

The PTX is embedded. Ordinary mpv builds need ffnvcodec headers and D3D11,
not a CUDA Toolkit. To regenerate the header after changing the CUDA source:

```
python TOOLS/cuda_d3d11_ptx.py --nvcc /path/to/nvcc
```

The initial generated header uses CUDA 12.8 / compute_75 PTX. It contains
ordinary arithmetic and packing kernels; runtime compilation is done by the
installed NVIDIA driver. Keep the CUDA source and generated header together.

Ordering scope: the requirement here is native upscaling followed by RTX HDR
at the tail of `vf`. GLSL hooks remain in their existing later VO stage.
This project does not require moving HDR into the VO. Putting HDR at the end
of `vf` does not itself imply a host transfer; a software-only filter inserted
between the hardware filters can still cause an automatic download/upload.

Local validation on RTX 4070 Ti / driver 617.42:

- GPU packing tests cover BT.601/709/2020 NC, full/limited range, padded
  strides, odd dimensions, guard words and opaque alpha. Tested values matched
  the double-precision reference exactly after 10-bit quantization.
- Continuous native inference, bridge and HDR playback and seeking passed
  after the ownership lock fix; output was D3D11 RGB10, BT.2020/PQ, with no
  filter disabling, device-loss errors or automatic host download in the log.
- Controller tests cover AI/HDR toggles, Vulkan/D3D11 changes, file/size/codec
  changes, an HDR10 input, and Manager's explicit/missing default slot.
- Three alternating runs of 1080p24 High10 to 4K, standard Balanced, software
  decode, 800x450 preview and untimed playback: GPU bridge 49.58/49.48/49.57
  fps (median 49.57), copy-back RGB10 bridge 48.30/48.22/48.26 fps (median
  48.26). Both use the same experimental executable and dependencies.

These are synthetic local tests, not a claim of universal driver support or
a frame-rate guarantee. The lack of host pixel-transfer calls in this bridge
and playback logs has not yet been independently verified with a GPU timeline.
The initial copy/software decode and CUDA upload remain as described above.
