//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-PIPELINE is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef BMUTILITY_PIPELINE_H
#define BMUTILITY_PIPELINE_H

#include <memory>
#include "bmutility_thread_queue.h"

namespace bm {
    // declare before
    template<typename T> class BMInferencePipe;

    template<typename T1>
    class DetectorDelegate {
    protected:
        using DetectedFinishFunc = std::function<void(T1 &of)>;
        DetectedFinishFunc m_pfnDetectFinish = nullptr;
        BMInferencePipe<T1>* m_nextInferPipe = nullptr;
    public:
        virtual ~DetectorDelegate() {}

        virtual int preprocess(std::vector<T1> &frames) = 0;

        virtual int forward(std::vector<T1> &frames) = 0;

        virtual int postprocess(std::vector<T1> &frames) = 0;

        virtual int set_detected_callback(DetectedFinishFunc func) { m_pfnDetectFinish = func; return 0;};
        void set_next_inference_pipe(BMInferencePipe<T1> *nextPipe) { m_nextInferPipe = nextPipe; }
    };

    struct DetectorParam {
        DetectorParam() {
            preprocess_queue_size = 5;
            preprocess_thread_num = 4;

            inference_queue_size = 5;
            inference_thread_num = 1;

            postprocess_queue_size = 5;
            postprocess_thread_num = 2;
            batch_num=4;
        }

        int preprocess_queue_size;
        int preprocess_thread_num;

        int inference_queue_size;
        int inference_thread_num;

        int postprocess_queue_size;
        int postprocess_thread_num;
        int batch_num;

    };

    template<typename T1>
    class BMInferencePipe {
        DetectorParam m_param;
        std::shared_ptr<DetectorDelegate<T1>> m_detect_delegate;

        std::shared_ptr<BlockingQueue<T1>> m_preprocessQue;
        std::shared_ptr<BlockingQueue<T1>> m_postprocessQue;
        std::shared_ptr<BlockingQueue<T1>> m_forwardQue;

        WorkerPool<T1> m_preprocessWorkerPool;
        WorkerPool<T1> m_forwardWorkerPool;
        WorkerPool<T1> m_postprocessWorkerPool;


    public:
        BMInferencePipe() {

        }

        virtual ~BMInferencePipe() {

        }

        int init(const DetectorParam &param, std::shared_ptr<DetectorDelegate<T1>> delegate) {
            m_param = param;
            m_detect_delegate = delegate;

            const int underlying_type_std_queue = 0;
            m_preprocessQue = std::make_shared<BlockingQueue<T1>>(
                "preprocess", underlying_type_std_queue,
                param.preprocess_queue_size);
            m_postprocessQue = std::make_shared<BlockingQueue<T1>>(
                "postprocess", underlying_type_std_queue,
                param.postprocess_queue_size);
            m_forwardQue = std::make_shared<BlockingQueue<T1>>(
                "inference", underlying_type_std_queue,
                param.inference_queue_size);

            m_preprocessWorkerPool.init(m_preprocessQue.get(), param.preprocess_thread_num, param.batch_num, param.batch_num);
            m_preprocessWorkerPool.startWork([this, &param](std::vector<T1> &items) {
                m_detect_delegate->preprocess(items);
                this->m_forwardQue->push(items);
            });

            m_forwardWorkerPool.init(m_forwardQue.get(), param.inference_thread_num, 1, 8);
            m_forwardWorkerPool.startWork([this, &param](std::vector<T1> &items) {
                m_detect_delegate->forward(items);
                this->m_postprocessQue->push(items);
            });

            m_postprocessWorkerPool.init(m_postprocessQue.get(), param.postprocess_thread_num, 1, 8);
            m_postprocessWorkerPool.startWork([this, &param](std::vector<T1> &items) {
                m_detect_delegate->postprocess(items);
            });
            return 0;
        }

        int flush_frame() {
            m_preprocessWorkerPool.flush();
            return 0;
        }

        int push_frame(T1 *frame) {
            m_preprocessQue->push(*frame);
            return 0;
        }
    };
} // end namespace bm


#endif //SOPHON_PIPELINE_INFERENCE_H
