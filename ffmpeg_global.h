//
// Created by hsyuan on 2019-03-15.
//

#ifndef BM_UTILITY_FFMPEG_GLOBAL_H
#define BM_UTILITY_FFMPEG_GLOBAL_H

#ifdef __cplusplus
extern "C" {
#include "libavdevice/avdevice.h"
#include "libavcodec/avcodec.h"
#include "libavutil/avutil.h"
#include "libavformat/avformat.h"
#include "libavutil/time.h"
}
#endif //!cplusplus

namespace bm {

    class FfmpegGlobal {
    public:
        FfmpegGlobal() {
#if LIBAVCODEC_VERSION_MAJOR <= 56
            av_register_all();
#endif
            avformat_network_init();
            avdevice_register_all();
            av_log_set_level(AV_LOG_ERROR);
        }

        ~FfmpegGlobal() {
            std::cout << "~FfmpegGlobal() dtor.." << std::endl;
            avformat_network_deinit();
        }
    };
}

#endif //FACEDEMOSYSTEM_FFMPEG_GLOBAL_H
