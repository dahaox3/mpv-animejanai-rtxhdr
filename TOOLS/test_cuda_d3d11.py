#!/usr/bin/env python3
"""GPU regression checks for packing, range, matrix, stride and bounds.

Requires an NVIDIA driver/GPU. Test-only host transfers are intentional.
No inference model, CUDA Toolkit, Python packages or D3D11 window required.
"""
import ctypes as C
import json
import math
import pathlib
import random
import struct
import sys


def main():
    driver = C.WinDLL('nvcuda.dll') if sys.platform == 'win32' else C.CDLL('libcuda.so.1')
    ptr, u64, integer = C.c_void_p, C.c_uint64, C.c_int

    def api(name, args):
        fn = getattr(driver, name)
        fn.argtypes, fn.restype = args, integer

        def call(*values):
            code = fn(*values)
            if code:
                raise RuntimeError(f'{name}: CUDA error {code}')
        return call

    init = api('cuInit', [C.c_uint])
    device_get = api('cuDeviceGet', [C.POINTER(integer), integer])
    retain = api('cuDevicePrimaryCtxRetain', [C.POINTER(ptr), integer])
    release = api('cuDevicePrimaryCtxRelease_v2', [integer])
    current = api('cuCtxSetCurrent', [ptr])
    module_load = api('cuModuleLoadData', [C.POINTER(ptr), ptr])
    module_free = api('cuModuleUnload', [ptr])
    kernel_get = api('cuModuleGetFunction', [C.POINTER(ptr), ptr, C.c_char_p])
    alloc = api('cuMemAlloc_v2', [C.POINTER(u64), C.c_size_t])
    free = api('cuMemFree_v2', [u64])
    upload = api('cuMemcpyHtoD_v2', [u64, ptr, C.c_size_t])
    download = api('cuMemcpyDtoH_v2', [ptr, u64, C.c_size_t])
    launch = api('cuLaunchKernel', [ptr] + [C.c_uint] * 7 + [ptr, ptr, ptr])
    sync = api('cuCtxSynchronize', [])

    init(0)
    device, context, module, kernel = integer(), ptr(), ptr(), ptr()
    device_get(C.byref(device), 0)
    retain(C.byref(context), device)
    current(context)
    allocations = []
    try:
        root = pathlib.Path(__file__).resolve().parents[1]
        header = (root / 'video/filter/cuda_d3d11_ptx.h').read_text(encoding='utf-8')
        ptx = ''.join(json.loads(line) for line in header.splitlines() if line.startswith('"'))
        text = C.create_string_buffer(ptx.encode('utf-8'))
        module_load(C.byref(module), text)
        kernel_get(C.byref(kernel), module, b'cuda_d3d11_pack')
        w, h, stride = 37, 11, 104
        for size in [stride * h] * 3 + [w * h * 4 + 32]:
            mem = u64()
            alloc(C.byref(mem), size)
            allocations.append(mem)
        samples = random.Random(4711)
        values = [(4096, 32768, 32768), (60160, 32768, 32768),
                  (0, 0, 65535), (65535, 65535, 0)]
        values += [tuple(samples.randrange(65536) for _ in range(3))
                   for _ in range(w * h - len(values))]
        for plane in range(3):
            raw = bytearray([0xa5] * (stride * h))
            for y in range(h):
                for x in range(w):
                    struct.pack_into('<H', raw, y * stride + x * 2, values[y * w + x][plane])
            buf = C.create_string_buffer(bytes(raw))
            upload(allocations[plane], buf, len(raw))

        results = []
        for matrix, kr, kb in [('601', .299, .114), ('709', .2126, .0722),
                               ('2020', .2627, .0593)]:
            for full in [False, True]:
                yoff, yscale = (0., 65535.) if full else (4096., 56064.)
                coff, cscale = 32768., 65535. if full else 57344.
                sentinel = C.create_string_buffer(bytes([0xcd]) * (w * h * 4 + 32))
                upload(allocations[3], sentinel, len(sentinel) - 1)
                params = [*allocations, integer(stride), integer(stride),
                          integer(stride), integer(w), integer(h),
                          *map(C.c_float, [kr, kb, yoff, yscale, coff, cscale])]
                pointers = (ptr * len(params))(*[C.cast(C.byref(v), ptr) for v in params])
                launch(kernel, (w + 31) // 32, (h + 7) // 8, 1, 32, 8, 1,
                       0, None, pointers, None)
                sync()
                out = C.create_string_buffer(w * h * 4 + 32)
                download(out, allocations[3], len(out))
                assert out.raw[-32:] == bytes([0xcd]) * 32, 'kernel wrote past the image'
                max_error = 0
                for i, (yv, uv, vv) in enumerate(values):
                    Y, U, V = (yv - yoff) / yscale, (uv - coff) / cscale, (vv - coff) / cscale
                    kg = 1 - kr - kb
                    expected = [Y + 2 * (1 - kr) * V,
                                Y - 2 * kb * (1 - kb) / kg * U - 2 * kr * (1 - kr) / kg * V,
                                Y + 2 * (1 - kb) * U]
                    expected = [round(max(0, min(1, f)) * 1023) for f in expected]
                    packed, = struct.unpack_from('<I', out.raw, i * 4)
                    actual = [(packed >> bit) & 1023 for bit in [0, 10, 20]]
                    assert packed >> 30 == 3, 'opaque alpha missing'
                    error = max(abs(a - b) for a, b in zip(actual, expected))
                    max_error = max(max_error, error)
                    assert error <= 1, (matrix, full, i, actual, expected)
                results.append({'matrix': matrix, 'range': 'full' if full else 'limited',
                                'pixels': w * h, 'max_error_10bit_codes': max_error})
        print(json.dumps({'status': 'pass', 'odd_width': w, 'padded_stride': stride,
                          'bounds_guard': 'pass', 'cases': results}, indent=2))
    finally:
        for mem in allocations:
            free(mem)
        if module:
            module_free(module)
        release(device)


if __name__ == '__main__':
    main()
