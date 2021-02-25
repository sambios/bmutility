//
// Created by yuan on 1/22/21.
//

#ifndef YOLOV5_DEMO_BMNN_UTILS_H
#define YOLOV5_DEMO_BMNN_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <assert.h>

#include <iostream>
#include <string>
#include <memory>
#include <unordered_map>
#include <vector>
#include <chrono>

#include <sys/time.h>

#include "bmruntime_interface.h"

#include "bmutility_image.h"

namespace bm {
    class NoCopyable {
    protected:
        NoCopyable() = default;

        ~NoCopyable() = default;

        NoCopyable(const NoCopyable &) = delete;

        NoCopyable &operator=(const NoCopyable &rhs) = delete;
    };

    class BMNNTensor {
        /**
         *  members from bm_tensor {
         *  bm_data_type_t dtype;
            bm_shape_t shape;
            bm_device_mem_t device_mem;
            bm_store_mode_t st_mode;
            }
         */
        bm_handle_t m_handle;

        std::string m_name;
        void *m_cpu_data;
        float m_scale;
        bm_tensor_t *m_tensor;

    public:
        BMNNTensor(bm_handle_t handle, const char *name, float scale,
                   bm_tensor_t *tensor) : m_handle(handle), m_name(name),
                                          m_cpu_data(nullptr), m_scale(scale), m_tensor(tensor) {
        }

        virtual ~BMNNTensor() {
            if (m_cpu_data != NULL) {
                if (m_tensor->dtype == BM_FLOAT32) {
                    float *dptr = static_cast<float *>(m_cpu_data);
                    delete[]dptr;
                } else {
                    int8_t *dptr = static_cast<int8_t *>(m_cpu_data);
                    delete[]dptr;
                }

                m_cpu_data = NULL;
            }
        }

        int set_device_mem(bm_device_mem_t *mem) {
            this->m_tensor->device_mem = *mem;
            return 0;
        }

        const bm_device_mem_t *get_device_mem() {
            return &this->m_tensor->device_mem;
        }

        void *get_cpu_data() {
            if (m_cpu_data == NULL) {
                int tensor_size = bmrt_tensor_bytesize(m_tensor);
                m_cpu_data = new int8_t[tensor_size];
                assert(NULL != m_cpu_data);
                bm_memcpy_d2s(m_handle, m_cpu_data, m_tensor->device_mem);
            }

            return m_cpu_data;
        }

        const bm_shape_t *get_shape() {
            return &m_tensor->shape;
        }

        bm_data_type_t get_dtype() {
            return m_tensor->dtype;
        }

        float get_scale() {
            return m_scale;
        }

        bm_tensor_t *bm_tensor() {
            return m_tensor;
        }

    };

    using BMNNTensorPtr = std::shared_ptr<BMNNTensor>;

    class BMNNNetwork : public NoCopyable {
        const bm_net_info_t *m_netinfo;
        bm_tensor_t *m_inputTensors;
        bm_tensor_t *m_outputTensors;
        bm_handle_t m_handle;
        void *m_bmrt;

        std::unordered_map<std::string, bm_tensor_t *> m_mapInputs;
        std::unordered_map<std::string, bm_tensor_t *> m_mapOutputs;

    public:
        BMNNNetwork(void *bmrt, const std::string &name) : m_bmrt(bmrt) {
            m_handle = static_cast<bm_handle_t>(bmrt_get_bm_handle(bmrt));
            m_netinfo = bmrt_get_network_info(bmrt, name.c_str());
            m_inputTensors = new bm_tensor_t[m_netinfo->input_num];
            m_outputTensors = new bm_tensor_t[m_netinfo->output_num];
            for (int i = 0; i < m_netinfo->input_num; ++i) {
                m_inputTensors[i].dtype = m_netinfo->input_dtypes[i];
                m_inputTensors[i].shape = m_netinfo->stages[0].input_shapes[i];
                m_inputTensors[i].st_mode = BM_STORE_1N;
                m_inputTensors[i].device_mem = bm_mem_null();
            }

            for (int i = 0; i < m_netinfo->output_num; ++i) {
                m_outputTensors[i].dtype = m_netinfo->output_dtypes[i];
                m_outputTensors[i].shape = m_netinfo->stages[0].output_shapes[i];
                m_outputTensors[i].st_mode = BM_STORE_1N;
                m_outputTensors[i].device_mem = bm_mem_null();
            }

            assert(m_netinfo->stage_num >= 1);
        }

        int inputTensorNum() {
            return m_netinfo->input_num;
        }

        std::shared_ptr<BMNNTensor> inputTensor(int index) {
            assert(index < m_netinfo->input_num);
            return std::make_shared<BMNNTensor>(m_handle, m_netinfo->input_names[index],
                                                m_netinfo->input_scales[index], &m_inputTensors[index]);
        }

        int outputTensorNum() {
            return m_netinfo->output_num;
        }

        std::shared_ptr<BMNNTensor> outputTensor(int index) {
            assert(index < m_netinfo->output_num);
            return std::make_shared<BMNNTensor>(m_handle, m_netinfo->output_names[index],
                                                m_netinfo->output_scales[index], &m_outputTensors[index]);
        }

        int forward() {

            bool ok = bmrt_launch_tensor(m_bmrt, m_netinfo->name, m_inputTensors, m_netinfo->input_num,
                                         m_outputTensors, m_netinfo->output_num);
            if (!ok) {
                std::cout << "bm_launch_tensor() failed=" << std::endl;
                return -1;
            }

#if 0
            for(int i = 0;i < m_netinfo->output_num; ++i) {
                auto tensor = m_outputTensors[i];
                // dump
                std::cout << "output_tensor [" << i << "] size=" << bmrt_tensor_device_size(&tensor) << std::endl;
            }
#endif

            return 0;
        }

        int forward(bm_tensor_t *input_tensors, int input_num, bm_tensor_t *output_tensors, int output_num)
        {
            bool ok = bmrt_launch_tensor_ex(m_bmrt, m_netinfo->name, input_tensors, input_num,
                    output_tensors, output_num, true, true);
            if (!ok) {
                std::cout << "bm_launch_tensor_ex() failed=" << std::endl;
                return -1;
            }

            return 0;
        }


    };

    using BMNNNetworkPtr=std::shared_ptr<BMNNNetwork>;

    class BMNNHandle : public NoCopyable {
        bm_handle_t m_handle;
        int m_dev_id;
    public:
        BMNNHandle(int dev_id = 0) : m_dev_id(dev_id) {
            int ret = bm_dev_request(&m_handle, dev_id);
            assert(BM_SUCCESS == ret);
        }

        ~BMNNHandle() {
            bm_dev_free(m_handle);
        }

        bm_handle_t handle() {
            return m_handle;
        }

        int dev_id() {
            return m_dev_id;
        }
    };

    using BMNNHandlePtr = std::shared_ptr<BMNNHandle>;

    class BMNNContext : public NoCopyable {
        BMNNHandlePtr m_handlePtr;
        void *m_bmrt;
        std::vector<std::string> m_network_names;

    public:
        BMNNContext(BMNNHandlePtr handle, const char *bmodel_file) : m_handlePtr(handle) {
            bm_handle_t hdev = m_handlePtr->handle();
            m_bmrt = bmrt_create(hdev);
            if (NULL == m_bmrt) {
                std::cout << "bmrt_create() failed!" << std::endl;
                exit(-1);
            }

            if (!bmrt_load_bmodel(m_bmrt, bmodel_file)) {
                std::cout << "load bmodel(" << bmodel_file << ") failed" << std::endl;
            }

            load_network_names();


        }

        ~BMNNContext() {
            if (m_bmrt != NULL) {
                bmrt_destroy(m_bmrt);
                m_bmrt = NULL;
            }
        }

        bm_handle_t handle() {
            return m_handlePtr->handle();
        }

        void *bmrt() {
            return m_bmrt;
        }

        void load_network_names() {
            const char **names;
            int num;
            num = bmrt_get_network_number(m_bmrt);
            bmrt_get_network_names(m_bmrt, &names);
            for (int i = 0; i < num; ++i) {
                m_network_names.push_back(names[i]);
            }

            free(names);
        }

        std::string network_name(int index) {
            if (index >= (int) m_network_names.size()) {
                return "Invalid index";
            }

            return m_network_names[index];
        }

        std::shared_ptr<BMNNNetwork> network(const std::string &net_name) {
            return std::make_shared<BMNNNetwork>(m_bmrt, net_name);
        }

        std::shared_ptr<BMNNNetwork> network(int net_index) {
            assert(net_index < (int) m_network_names.size());
            return std::make_shared<BMNNNetwork>(m_bmrt, m_network_names[net_index]);
        }


    };

    using BMNNContextPtr = std::shared_ptr<BMNNContext>;

    class BMPerf {
        int64_t start_us_;
        std::string tag_;
        int threshold_{50};
    public:
        BMPerf() {}

        ~BMPerf() {}

        void begin(const std::string &name, int threshold = 50) {
            tag_ = name;
            auto n = std::chrono::steady_clock::now();
            start_us_ = n.time_since_epoch().count();
            threshold_ = threshold;
        }

        void end() {
            auto n = std::chrono::steady_clock::now().time_since_epoch().count();
            auto delta = (n - start_us_) / 1000;
            if (delta < threshold_ * 1000) {
                //printf("%s used:%d us\n", tag_.c_str(), delta);
            } else {
                printf("WARN:%s used:%d us > %d\n", tag_.c_str(), delta, threshold_);
            }
        }
    };
} // end of namespace bm





#endif //YOLOV5_DEMO_BMNN_UTILS_H
