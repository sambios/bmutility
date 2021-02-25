//
// Created by hsyuan on 2019-02-22.
//

#ifndef BM_UTILITY_STREAM_DEMUXER_H
#define BM_UTILITY_STREAM_DEMUXER_H

#include <iostream>
#include <thread>
#include <list>
#include "ffmpeg_global.h"

namespace bm {

    struct StreamDemuxerEvents {
        virtual void on_avformat_opened(AVFormatContext *ifmt_ctx) = 0;

        virtual void on_avformat_closed() = 0;

        virtual int on_read_frame(AVPacket *pkt) = 0;

        virtual void on_read_eof(AVPacket *pkt) = 0;
    };

    class StreamDemuxer : FfmpegGlobal {
        enum State : int8_t {
            Initialize = 0,
            Service,
            Down
        };

        AVFormatContext *m_ifmt_ctx;
        StreamDemuxerEvents *m_observer;
        State m_work_state;
        std::string m_inputUrl;
        std::thread *m_thread_reading;
        bool m_repeat;
        bool m_keep_running;
        int64_t m_last_frame_time{0};
        bool m_is_file_url{false};

    private:
        int do_initialize();

        int do_service();

        int do_down();

    public:
        StreamDemuxer();

        virtual ~StreamDemuxer();

        int open_stream(std::string url, StreamDemuxerEvents *observer, bool repeat = true, bool isSyncOpen=false);

        int close_stream(bool is_waiting);

        //int get_codec_parameters(int stream_index, AVCodecParameters **p_codecpar);
        //int get_codec_type(int stream_index, int *p_codec_type);
    };
}


#endif //TESTUV_STREAM_DEMUXER_H
