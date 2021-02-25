#ifndef BMUTILITY_IMAGE_H
#define BMUTILITY_IMAGE_H

#ifdef __cplusplus
extern "C" {
#include "libavutil/avutil.h"
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
}
#endif

#include "bmcv_api_ext.h"

namespace bm {
///////////////////////////////////////////////////////////////////////////
// BMCV_IMAGE
    struct bmcv {
        static inline bm_status_t bmimage_from_avframe(bm_handle_t &bm_handle,
                                               const AVFrame *pAVFrame,
                                               bm_image &out) {
            const AVFrame &in = *pAVFrame;
            if (in.format != AV_PIX_FMT_NV12) {
                std::cout << "format donot support" << std::endl;
                return BM_NOT_SUPPORTED;
            }

            if (in.channel_layout == 101) { /* COMPRESSED NV12 FORMAT */
                /* sanity check */
                if ((0 == in.height) || (0 == in.width) || \
    (0 == in.linesize[4]) || (0 == in.linesize[5]) || (0 == in.linesize[6]) || (0 == in.linesize[7]) || \
    (0 == in.data[4]) || (0 == in.data[5]) || (0 == in.data[6]) || (0 == in.data[7])) {
                    std::cout << "bm_image_from_frame: get yuv failed!!" << std::endl;
                    return BM_ERR_PARAM;
                }
                bm_image cmp_bmimg;
                bm_image_create(bm_handle,
                                in.height,
                                in.width,
                                FORMAT_COMPRESSED,
                                DATA_TYPE_EXT_1N_BYTE,
                                &cmp_bmimg);

                /* calculate physical address of avframe */
                bm_device_mem_t input_addr[4];
                int size = in.height * in.linesize[4];
                input_addr[0] = bm_mem_from_device((unsigned long long) in.data[6], size);
                size = (in.height / 2) * in.linesize[5];
                input_addr[1] = bm_mem_from_device((unsigned long long) in.data[4], size);
                size = in.linesize[6];
                input_addr[2] = bm_mem_from_device((unsigned long long) in.data[7], size);
                size = in.linesize[7];
                input_addr[3] = bm_mem_from_device((unsigned long long) in.data[5], size);
                bm_image_attach(cmp_bmimg, input_addr);

            } else { /* UNCOMPRESSED NV12 FORMAT */
                /* sanity check */
                if ((0 == in.height) || (0 == in.width) || \
    (0 == in.linesize[4]) || (0 == in.linesize[5]) || \
    (0 == in.data[4]) || (0 == in.data[5])) {
                    std::cout << "bm_image_from_frame: get yuv failed!!" << std::endl;
                    return BM_ERR_PARAM;
                }

                /* create bm_image with YUV-nv12 format */
                int stride[2];
                stride[0] = in.linesize[4];
                stride[1] = in.linesize[5];
                bm_image_create(bm_handle,
                                in.height,
                                in.width,
                                FORMAT_NV12,
                                DATA_TYPE_EXT_1N_BYTE,
                                &out,
                                stride);

                /* calculate physical address of yuv mat */
                bm_device_mem_t input_addr[2];
                int size = in.height * stride[0];
                input_addr[0] = bm_mem_from_device((unsigned long long) in.data[4], size);
                size = in.height * stride[1];
                input_addr[1] = bm_mem_from_device((unsigned long long) in.data[5], size);

                /* attach memory from mat to bm_image */
                bm_image_attach(out, input_addr);
            }

            return BM_SUCCESS;
        }

    };
}



#endif //!BMUTILITY_IMAGE_H