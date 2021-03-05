#ifndef BMUTILITY_IMAGE_H
#define BMUTILITY_IMAGE_H

#ifdef __cplusplus
extern "C" {
#include "libavutil/avutil.h"
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
}
#endif

#include "opencv2/opencv.hpp"
#include "bmcv_api_ext.h"

namespace bm {
///////////////////////////////////////////////////////////////////////////
// BMCV_IMAGE
    struct BMImage {
        static inline bm_status_t create_batch (bm_handle_t              handle,
                                                         int                      img_h,
                                                         int                      img_w,
                                                         bm_image_format_ext      img_format,
                                                         bm_image_data_format_ext data_type,
                                                         bm_image                 *image,
                                                         int                      batch_num,
                                                         int align=1) {

            // init images
            int stride[3]={0};
            if (FORMAT_RGB_PLANAR == img_format ||
                FORMAT_RGB_PACKED == img_format ||
                FORMAT_BGR_PLANAR == img_format ||
                FORMAT_BGR_PACKED == img_format) {
                stride[0] = FFALIGN(img_w, align);
            }else if (FORMAT_YUV420P == img_format) {
                stride[0] = FFALIGN(img_w, align);
                stride[1] = stride[2] = FFALIGN(img_w >> 1, align);
            }else if (FORMAT_NV12 == img_format || FORMAT_NV21 == img_format){
                stride[0] = FFALIGN(img_w, align);
                stride[1] = FFALIGN(img_w >> 1, align);
            }else{
                assert(0);
            }

            for (int i = 0; i < batch_num; i++) {
                bm_image_create(handle, img_h, img_w, img_format, data_type, &image[i], stride);
            }

            return BM_SUCCESS;
        }

        static inline bm_status_t destroy_batch (bm_image *image, int batch_num) {
            // deinit bm image
            for (int i = 0; i < batch_num; i++) {
                bm_image_destroy (image[i]);
            }

            return BM_SUCCESS;
        }

        static inline bm_status_t from_avframe(bm_handle_t &bm_handle,
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
                out = cmp_bmimg;

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

        static unsigned int face_align(unsigned int n, unsigned align)
        {
            return (n + (align - 1)) & (~(align - 1));
        }

        static void BGRPlanarToPacked(unsigned char *inout, int N, int H, int W)
        {
            unsigned char *temp = new unsigned char[H * W * 3];
            for (int n = 0; n < N; n++)
            {
                unsigned char *start = inout + 3 * H * W * n;
                for (int h = 0; h < H; h++)
                {
                    for (int w = 0; w < W; w++)
                    {
                        temp[3 * (h * W + w)] = start[(h * W + w)];
                        temp[3 * (h * W + w) + 1] = start[(h * W + w) + H * W];
                        temp[3 * (h * W + w) + 2] = start[(h * W + w) + 2 * H * W];
                    }
                }
                memcpy(start, temp, H * W * 3);
            }
            delete[] temp;
        }

        static void convert_4N_2_1N(unsigned char *inout, int N, int C, int H, int W)
        {
            unsigned char* temp_buf = new unsigned char[4 * C * H * W];
            for(int i = 0;i < face_align(N, 4) / 4; i++)
            {
                memcpy(temp_buf,inout + 4 * C * H * W * i, 4 * C * H * W);
                for(int loop = 0; loop < C * H * W; loop++)
                {
                    inout[i * 4 * C * H * W + loop] = temp_buf[4 * loop];
                    inout[i * 4 * C * H * W + 1 * C * H * W + loop] = temp_buf[4 * loop + 1];
                    inout[i * 4 * C * H * W + 2 * C * H * W + loop] = temp_buf[4 * loop + 2];
                    inout[i * 4 * C * H * W + 3 * C * H * W + loop] = temp_buf[4 * loop + 3];
                }
            }
            delete [] temp_buf;
        }

        static void interleave_fp32(float *inout, int N, int H, int W)
        {
            float *temp = new float[H * W * 3];
            for (int n = 0; n < N; n++)
            {
                float *start = inout + 3 * H * W * n;
                for (int h = 0; h < H; h++)
                {
                    for (int w = 0; w < W; w++)
                    {
                        temp[3 * (h * W + w)] = start[(h * W + w)];
                        temp[3 * (h * W + w) + 1] = start[(h * W + w) + H * W];
                        temp[3 * (h * W + w) + 2] = start[(h * W + w) + 2 * H * W];
                    }
                }
                memcpy(start, temp, H * W * 3 * sizeof(float));
            }
            delete[] temp;
        }

        static void dump_dev_memory(bm_handle_t bm_handle, bm_device_mem_t dev_mem, char *fn, int n, int h, int w, int b_fp32, int b_4N)
        {
            cv::Mat img;
            int c = 3;
            int tensor_size = face_align(n, 4) * c * h * w;
            int c_size = c * h * w;
            int element_size = 4;
            unsigned char *s = new unsigned char[tensor_size * element_size];
            if (bm_mem_get_type(dev_mem) == BM_MEM_TYPE_DEVICE) {
                bm_memcpy_d2s(bm_handle, (void *)s, dev_mem);
            }else {
                int element_size = b_fp32 ? 4 : 1;
                memcpy(s,
                       bm_mem_get_system_addr(dev_mem),
                       n * c * h * w * element_size);
            }
            if (b_4N) {
                convert_4N_2_1N(s, n, c, h, w);
            }
            if (b_fp32) {
                interleave_fp32((float *)s, n, h, w);
            }else {
                BGRPlanarToPacked(s, n, h, w);
            }
            for (int i = 0; i < n; i++) {
                char fname[256];
                sprintf(fname, "%s_%d.png", fn, i);
                if (b_fp32) {
                    img.create(h, w, CV_32FC3);
                    memcpy(img.data, (float *)s + c_size * i, c_size * 4);
                    cv::Mat img2;
                    img.convertTo(img2, CV_8UC3);
                    cv::imwrite(fn, img2);
                } else {
                    cv::Mat img(h, w, CV_8UC3);
                    memcpy(img.data, s + c_size * i, c_size);
                    cv::imwrite(fname, img);
                }
            }
            delete s;
        }
    };
}



#endif //!BMUTILITY_IMAGE_H