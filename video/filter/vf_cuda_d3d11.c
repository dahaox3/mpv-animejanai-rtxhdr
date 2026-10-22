/*
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Keep native AnimeJaNai CUDA output on the GPU before D3D11 RTX HDR.
 */
#include "config.h"

#include <windows.h>
#include <d3d11.h>
#include <ffnvcodec/dynlink_loader.h>
#include <libavutil/buffer.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/hwcontext_d3d11va.h>
#include <libavutil/mem.h>

#include "common/common.h"
#include "filters/filter.h"
#include "filters/filter_internal.h"
#include "filters/user_filters.h"
#include "video/hwdec.h"
#include "video/fmt-conversion.h"
#include "video/mp_image.h"
#include "video/mp_image_pool.h"
#include "cuda_d3d11_ptx.h"

// Each surface retains this session independently of the filter, so output
// references in the VPP/VO can outlive a seek, reinit or filter removal.
struct session {
    CudaFunctions *cu;
    AVBufferRef *device_ref;
    CUcontext ctx;
    CUstream stream;
    CUevent source_ready;
    CUmodule module;
    CUfunction kernel;
    CUdeviceptr packed;
};

struct surface {
    AVBufferRef *session_ref;
    ID3D11Texture2D *texture;
    CUgraphicsResource resource;
};

struct priv {
    AVBufferRef *d3d_device;
    AVBufferRef *frames;
    AVBufferRef *session_ref;
    struct mp_image_pool *pool;
    int w, h;
};

static bool check_cu(struct mp_filter *f, CUresult ret, const char *op)
{
    if (ret == CUDA_SUCCESS)
        return true;
    struct priv *p = f->priv;
    struct session *s = (void *)p->session_ref->data;
    const char *name = NULL;
    s->cu->cuGetErrorName(ret, &name);
    MP_ERR(f, "%s failed: %s\n", op, name ? name : "unknown CUDA error");
    return false;
}

#define CU_OK(expr) check_cu(f, (expr), #expr)

static void free_session(void *opaque, uint8_t *data)
{
    struct session *s = (void *)data;
    CUcontext dummy;
    if (s->cu && s->ctx &&
        s->cu->cuCtxPushCurrent(s->ctx) == CUDA_SUCCESS)
    {
        if (s->stream)
            s->cu->cuStreamSynchronize(s->stream);
        if (s->packed)
            s->cu->cuMemFree(s->packed);
        if (s->source_ready)
            s->cu->cuEventDestroy(s->source_ready);
        if (s->module)
            s->cu->cuModuleUnload(s->module);
        if (s->stream)
            s->cu->cuStreamDestroy(s->stream);
        s->cu->cuCtxPopCurrent(&dummy);
    }
    av_buffer_unref(&s->device_ref);
    cuda_free_functions(&s->cu);
    av_free(s);
}

static void free_surface(void *opaque, uint8_t *data)
{
    struct surface *surface = (void *)data;
    struct session *s = (void *)surface->session_ref->data;
    CUcontext dummy;
    if (surface->resource &&
        s->cu->cuCtxPushCurrent(s->ctx) == CUDA_SUCCESS)
    {
        s->cu->cuGraphicsUnregisterResource(surface->resource);
        s->cu->cuCtxPopCurrent(&dummy);
    }
    if (surface->texture)
        ID3D11Texture2D_Release(surface->texture);
    av_buffer_unref(&surface->session_ref);
    av_free(surface);
}

static struct mp_image *allocate_surface(void *opaque, int fmt, int w, int h)
{
    struct mp_filter *f = opaque;
    struct priv *p = f->priv;
    struct session *s = (void *)p->session_ref->data;
    AVHWDeviceContext *device = (void *)p->d3d_device->data;
    AVD3D11VADeviceContext *d3d = device->hwctx;
    struct surface *surface = av_mallocz(sizeof(*surface));
    if (!surface)
        return NULL;
    surface->session_ref = av_buffer_ref(p->session_ref);
    if (!surface->session_ref) {
        av_free(surface);
        return NULL;
    }
    D3D11_TEXTURE2D_DESC desc = {
        .Width = w, .Height = h, .MipLevels = 1, .ArraySize = 1,
        .Format = DXGI_FORMAT_R10G10B10A2_UNORM,
        .SampleDesc.Count = 1, .Usage = D3D11_USAGE_DEFAULT,
        .BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET,
    };
    if (FAILED(ID3D11Device_CreateTexture2D(d3d->device, &desc, NULL,
                                           &surface->texture)))
        goto fail;
    CUcontext dummy;
    if (!CU_OK(s->cu->cuCtxPushCurrent(s->ctx)))
        goto fail;
    bool registered = CU_OK(s->cu->cuGraphicsD3D11RegisterResource(
        &surface->resource, (ID3D11Resource *)surface->texture, 0));
    s->cu->cuCtxPopCurrent(&dummy);
    if (!registered) {
        MP_ERR(f, "The driver cannot share RGB10A2 with CUDA. "
                  "Use the explicit copy-back bridge on this driver.\n");
        goto fail;
    }
    struct mp_image *image = mp_image_new_dummy_ref(NULL);
    image->bufs[0] = av_buffer_create((uint8_t *)surface, sizeof(*surface),
                                     free_surface, NULL, 0);
    if (!image->bufs[0]) {
        talloc_free(image);
        goto fail;
    }
    mp_image_setfmt(image, IMGFMT_D3D11);
    mp_image_set_size(image, w, h);
    image->params.hw_subfmt = IMGFMT_X2BGR10;
    image->planes[0] = (void *)surface->texture;
    return image;
fail:
    free_surface(NULL, (uint8_t *)surface);
    return NULL;
}

static void clear_session(struct mp_filter *f)
{
    struct priv *p = f->priv;
    if (p->pool)
        mp_image_pool_clear(p->pool);
    av_buffer_unref(&p->frames);
    av_buffer_unref(&p->session_ref);
    p->w = p->h = 0;
}

static bool configure(struct mp_filter *f, struct mp_image *src)
{
    struct priv *p = f->priv;
    AVHWFramesContext *input = (void *)src->hwctx->data;
    AVCUDADeviceContext *cuda = input->device_ctx->hwctx;
    if (p->session_ref) {
        struct session *old = (void *)p->session_ref->data;
        if (p->w == src->w && p->h == src->h && old->ctx == cuda->cuda_ctx)
            return true;
    }
    clear_session(f);
    struct session *s = av_mallocz(sizeof(*s));
    if (!s)
        return false;
    p->session_ref = av_buffer_create((uint8_t *)s, sizeof(*s),
                                      free_session, NULL, 0);
    if (!p->session_ref) {
        av_free(s);
        return false;
    }
    s->ctx = cuda->cuda_ctx;
    s->device_ref = av_buffer_ref(input->device_ref);
    if (!s->device_ref || cuda_load_functions(&s->cu, f->log) < 0)
        return false;
    CUcontext dummy;
    if (!CU_OK(s->cu->cuCtxPushCurrent(s->ctx)))
        return false;
    bool ok = CU_OK(s->cu->cuStreamCreate(&s->stream, CU_STREAM_NON_BLOCKING)) &&
              CU_OK(s->cu->cuEventCreate(&s->source_ready, CU_EVENT_DISABLE_TIMING)) &&
              CU_OK(s->cu->cuModuleLoadData(&s->module, cuda_d3d11_ptx)) &&
              CU_OK(s->cu->cuModuleGetFunction(&s->kernel, s->module,
                                               "cuda_d3d11_pack")) &&
              CU_OK(s->cu->cuMemAlloc(&s->packed, (size_t)src->w * src->h * 4));
    s->cu->cuCtxPopCurrent(&dummy);
    if (!ok)
        return false;
    p->frames = av_hwframe_ctx_alloc(p->d3d_device);
    if (!p->frames)
        return false;
    AVHWFramesContext *frames = (void *)p->frames->data;
    frames->format = AV_PIX_FMT_D3D11;
    frames->sw_format = AV_PIX_FMT_X2BGR10;
    frames->width = src->w;
    frames->height = src->h;
    AVD3D11VAFramesContext *d3d_frames = frames->hwctx;
    d3d_frames->BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    if (av_hwframe_ctx_init(p->frames) < 0)
        return false;
    p->w = src->w;
    p->h = src->h;
    MP_INFO(f, "GPU bridge: CUDA yuv444p16 -> D3D11 rgb10a2, %dx%d; "
               "no host pixel transfer\n", p->w, p->h);
    return true;
}

static struct mp_image *convert(struct mp_filter *f, struct mp_image *src)
{
    struct priv *p = f->priv;
    if (!src->hwctx || src->imgfmt != IMGFMT_CUDA ||
        src->params.hw_subfmt != pixfmt2imgfmt(AV_PIX_FMT_YUV444P16))
    {
        MP_ERR(f, "Expected CUDA yuv444p16 from animejanai=output-444=yes.\n");
        return NULL;
    }
    AVHWFramesContext *input = (void *)src->hwctx->data;
    if (input->device_ctx->type != AV_HWDEVICE_TYPE_CUDA ||
        src->w <= 0 || src->h <= 0 ||
        src->stride[0] < src->w * 2 || src->stride[1] < src->w * 2 ||
        src->stride[2] < src->w * 2)
        return NULL;
    if (!configure(f, src))
        return NULL;
    struct session *s = (void *)p->session_ref->data;
    struct mp_image *out = mp_image_pool_get(p->pool, IMGFMT_D3D11, p->w, p->h);
    if (!out)
        return NULL;
    struct surface *surface = (void *)out->bufs[0]->data;
    float kr, kb;
    switch (src->params.repr.sys) {
    case PL_COLOR_SYSTEM_BT_601: kr = 0.299f; kb = 0.114f; break;
    case PL_COLOR_SYSTEM_BT_2020_NC: kr = 0.2627f; kb = 0.0593f; break;
    case PL_COLOR_SYSTEM_BT_709: kr = 0.2126f; kb = 0.0722f; break;
    default:
        MP_ERR(f, "Unsupported input YUV matrix.\n");
        talloc_free(out);
        return NULL;
    }
    bool full = src->params.repr.levels == PL_COLOR_LEVELS_FULL;
    float yoff = full ? 0.0f : 4096.0f, yscale = full ? 65535.0f : 56064.0f;
    float coff = 32768.0f, cscale = full ? 65535.0f : 57344.0f;
    int ys = src->stride[0], us = src->stride[1], vs = src->stride[2];
    void *args[] = {&src->planes[0], &src->planes[1], &src->planes[2],
                   &s->packed, &ys, &us, &vs, &p->w, &p->h,
                   &kr, &kb, &yoff, &yscale, &coff, &cscale};
    CUcontext dummy;
    if (!CU_OK(s->cu->cuCtxPushCurrent(s->ctx))) {
        talloc_free(out);
        return NULL;
    }
    AVCUDADeviceContext *cuda = input->device_ctx->hwctx;
    bool mapped = false;
    bool ok = CU_OK(s->cu->cuEventRecord(s->source_ready, cuda->stream)) &&
              CU_OK(s->cu->cuStreamWaitEvent(s->stream, s->source_ready, 0)) &&
              CU_OK(s->cu->cuLaunchKernel(s->kernel,
                    (p->w + 31) / 32, (p->h + 7) / 8, 1, 32, 8, 1,
                    0, s->stream, args, NULL));
    if (ok)
        mapped = ok = CU_OK(s->cu->cuGraphicsMapResources(
            1, &surface->resource, s->stream));
    CUarray array;
    if (ok)
        ok = CU_OK(s->cu->cuGraphicsSubResourceGetMappedArray(
            &array, surface->resource, 0, 0));
    if (ok) {
        CUDA_MEMCPY2D copy = {
            .srcMemoryType = CU_MEMORYTYPE_DEVICE,
            .srcDevice = s->packed, .srcPitch = (size_t)p->w * 4,
            .dstMemoryType = CU_MEMORYTYPE_ARRAY, .dstArray = array,
            .WidthInBytes = (size_t)p->w * 4, .Height = p->h,
        };
        ok = CU_OK(s->cu->cuMemcpy2DAsync(&copy, s->stream));
    }
    if (mapped)
        ok = CU_OK(s->cu->cuGraphicsUnmapResources(
            1, &surface->resource, s->stream)) && ok;
    // Retain src until the bridge finished reading it. Native inference is
    // still pipelined upstream; this waits for the short bridge only.
    ok = CU_OK(s->cu->cuStreamSynchronize(s->stream)) && ok;
    s->cu->cuCtxPopCurrent(&dummy);
    if (!ok) {
        talloc_free(out);
        return NULL;
    }
    mp_image_copy_attributes(out, src);
    out->hwctx = av_buffer_ref(p->frames);
    out->params.imgfmt = IMGFMT_D3D11;
    out->params.hw_subfmt = IMGFMT_X2BGR10;
    out->params.repr.sys = PL_COLOR_SYSTEM_RGB;
    out->params.repr.levels = PL_COLOR_LEVELS_FULL;
    out->params.repr.bits = (struct pl_bit_encoding){
        .sample_depth = 10, .color_depth = 10,
    };
    out->params.chroma_location = PL_CHROMA_UNKNOWN;
    return out;
}

static void process(struct mp_filter *f)
{
    if (!mp_pin_can_transfer_data(f->ppins[1], f->ppins[0]))
        return;
    struct mp_frame frame = mp_pin_out_read(f->ppins[0]);
    if (mp_frame_is_signaling(frame)) {
        mp_pin_in_write(f->ppins[1], frame);
        return;
    }
    struct mp_image *out = frame.type == MP_FRAME_VIDEO
                         ? convert(f, frame.data) : NULL;
    mp_frame_unref(&frame);
    if (!out) {
        mp_filter_internal_mark_failed(f);
        return;
    }
    mp_pin_in_write(f->ppins[1], MAKE_FRAME(MP_FRAME_VIDEO, out));
}

static void destroy(struct mp_filter *f)
{
    struct priv *p = f->priv;
    clear_session(f);
    av_buffer_unref(&p->d3d_device);
}

static const struct mp_filter_info info = {
    .name = "cuda-d3d11", .priv_size = sizeof(struct priv),
    .process = process, .destroy = destroy,
};

static struct mp_filter *create(struct mp_filter *parent, void *options)
{
    talloc_free(options);
    struct mp_filter *f = mp_filter_create(parent, &info);
    if (!f)
        return NULL;
    struct priv *p = f->priv;
    p->pool = mp_image_pool_new(p);
    mp_image_pool_set_allocator(p->pool, allocate_surface, f);
    mp_image_pool_set_lru(p->pool);
    mp_filter_add_pin(f, MP_PIN_IN, "in");
    mp_filter_add_pin(f, MP_PIN_OUT, "out");
    struct mp_stream_info *stream = mp_filter_find_stream_info(f);
    if (!stream || !stream->hwdec_devs)
        goto fail;
    struct hwdec_imgfmt_request req = {.imgfmt = IMGFMT_D3D11};
    hwdec_devices_request_for_img_fmt(stream->hwdec_devs, &req);
    struct mp_hwdec_ctx *hw = hwdec_devices_get_by_imgfmt_and_type(
        stream->hwdec_devs, IMGFMT_D3D11, AV_HWDEVICE_TYPE_D3D11VA);
    if (!hw || !hw->av_device_ref) {
        MP_ERR(f, "A D3D11 video output device is required.\n");
        goto fail;
    }
    p->d3d_device = av_buffer_ref(hw->av_device_ref);
    if (!p->d3d_device)
        goto fail;
    return f;
fail:
    talloc_free(f);
    return NULL;
}

const struct mp_user_filter_entry vf_cuda_d3d11 = {
    .desc = {
        .name = "cuda-d3d11",
        .description = "GPU-resident CUDA YUV444P16 to D3D11 RGB10A2 bridge",
    },
    .create = create,
};
