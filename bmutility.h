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
#include "bmutility_timer.h"
#include "bmutility_image.h"
#include "stream_decode.h"

namespace bm {
    static int bm_tensor_reshape_NCHW(bm_handle_t handle, bm_tensor_t *tensor, int n, int c, int h, int w) {
        tensor->shape.num_dims=4;
        tensor->shape.dims[0] = n;
        tensor->shape.dims[1] = c;
        tensor->shape.dims[2] = h;
        tensor->shape.dims[3] = w;

        int size = bmrt_tensor_bytesize(tensor);
        int ret = bm_malloc_device_byte(handle, &tensor->device_mem, size);
        assert(BM_SUCCESS == ret);
        return ret;
    }

    static int bm_tensor_reshape_NHWC(bm_handle_t handle, bm_tensor_t *tensor, int n, int h, int w, int c) {
        tensor->shape.num_dims=4;
        tensor->shape.dims[0] = n;
        tensor->shape.dims[1] = h;
        tensor->shape.dims[2] = w;
        tensor->shape.dims[3] = c;

        int size = bmrt_tensor_bytesize(tensor);
        int ret = bm_malloc_device_byte(handle, &tensor->device_mem, size);
        assert(BM_SUCCESS == ret);
        return ret;
    }

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
        BMNNTensor(bm_handle_t handle, const std::string& name, float scale,
                   bm_tensor_t *tensor) : m_handle(handle), m_name(name),
                                          m_cpu_data(nullptr), m_scale(scale), m_tensor(tensor)
                                          {
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
                float *pFP32 = nullptr;
                int count = bmrt_shape_count(&m_tensor->shape);
                if (m_tensor->dtype == BM_FLOAT32) {
                    pFP32 = new float[count];
                    bm_memcpy_d2s(m_handle, pFP32, m_tensor->device_mem);
                }else if (BM_INT8 == m_tensor->dtype) {
                    int tensor_size = bmrt_tensor_bytesize(m_tensor);
                    int8_t *pU8 = new int8_t[tensor_size];
                    pFP32 = new float[count];
                    bm_memcpy_d2s(m_handle, pU8, m_tensor->device_mem);
                    for(int i = 0;i < count; ++ i) {
                        pFP32[i] = pU8[i] * m_scale;
                    }
                    delete [] pU8;
                }else{
                    std::cout << "NOT support dtype=" << m_tensor->dtype << std::endl;
                }

                m_cpu_data = pFP32;
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

        int reshape_nchw(int n, int c, int h, int w) {
            return bm_tensor_reshape_NCHW(m_handle, m_tensor, n, c, h, w);
        }

        int reshape_nhwc(int n, int c, int h, int w) {
            return bm_tensor_reshape_NHWC(m_handle, m_tensor, n, c, h, w);
        }
    };

    using BMNNTensorPtr = std::shared_ptr<BMNNTensor>;


    class BMNNNetwork : public NoCopyable {
        const bm_net_info_t *m_netinfo;
        bm_tensor_t *m_inputTensors;
        bm_tensor_t *m_outputTensors;
        bm_handle_t m_handle;
        void *m_bmrt;

        std::unordered_map<std::string, int> m_mapInputName2Index;
        std::unordered_map<std::string, int> m_mapOutputName2Index;

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
                m_mapInputName2Index[m_netinfo->input_names[i]] = i;
            }

            for (int i = 0; i < m_netinfo->output_num; ++i) {
                m_outputTensors[i].dtype = m_netinfo->output_dtypes[i];
                m_outputTensors[i].shape = m_netinfo->stages[0].output_shapes[i];
                m_outputTensors[i].st_mode = BM_STORE_1N;
                m_outputTensors[i].device_mem = bm_mem_null();
                m_mapOutputName2Index[m_netinfo->output_names[i]] = i;
            }

            assert(m_netinfo->stage_num >= 1);
        }

        int inputTensorNum() {
            return m_netinfo->input_num;
        }

        int inputName2Index(const std::string& name) {
            if (m_mapInputName2Index.find(name) != m_mapInputName2Index.end()) {
                return m_mapInputName2Index[name];
            }

            return -1;
        }

        int outputName2Index(const std::string& name) {
            if (m_mapOutputName2Index.find(name) != m_mapOutputName2Index.end()) {
                return m_mapOutputName2Index[name];
            }

            return -1;
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

        int forward(const bm_tensor_t *input_tensors, int input_num, bm_tensor_t *output_tensors, int output_num)
        {
            bool ok = bmrt_launch_tensor_ex(m_bmrt, m_netinfo->name, input_tensors, input_num,
                    output_tensors, output_num, false, false);
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
        BMNNContext(BMNNHandlePtr handle, const std::string& bmodel_file) : m_handlePtr(handle) {
            bm_handle_t hdev = m_handlePtr->handle();
            m_bmrt = bmrt_create(hdev);
            if (NULL == m_bmrt) {
                std::cout << "bmrt_create() failed!" << std::endl;
                exit(-1);
            }

            if (!bmrt_load_bmodel(m_bmrt, bmodel_file.c_str())) {
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


} // end of namespace bm





#endif //YOLOV5_DEMO_BMNN_UTILS_H
