//
// Created by yuan on 2023/1/15.
//

#ifndef BMUTILITY_NN_H
#define BMUTILITY_NN_H

#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>
#include <numeric>

#include "bmruntime_interface.h"

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

    static bool operator == (const bm_shape_t& t1, const bm_shape_t& t2) {
        if (t1.num_dims != t2.num_dims) return false;
        for(int i = 0;i < t1.num_dims; ++i) {
            if (t1.dims[i] != t2.dims[i]) return false;
        }
        return true;
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
        float *m_cpu_data=nullptr;
        float m_scale;
        // user must free tensor's device memory by himself.
        bm_tensor_t *m_tensor = nullptr;
        int m_tensor_elem_count {0};
        int m_tensor_size {0};

        std::vector<int> m_shape;
        int m_dtype {-1};
        bool m_can_mmap {false};

        bool update_shape() {
            bool changed = false;
            if (m_tensor->shape.num_dims == m_shape.size()) {
                for (int i = 0; i < m_tensor->shape.num_dims; ++i) {
                    if (m_shape[i] != m_tensor->shape.dims[i]) {
                        changed = true;
                        break;
                    }
                }
            }else{
                changed = true;
            }

            if (changed) {
                m_shape.assign(m_tensor->shape.dims,
                               m_tensor->shape.dims + m_tensor->shape.num_dims);
                m_tensor_elem_count = std::accumulate(m_shape.begin(), m_shape.end(), 1,
                                                 std::multiplies<int>());
            }
            return changed;
        }

        void update_dtype() {
            if (m_dtype != m_tensor->dtype) {
                switch (m_tensor->dtype) {
                    case BM_FLOAT32:
                    case BM_UINT32:
                    case BM_INT32:
                        m_tensor_size = m_tensor_elem_count << 2;
                        break;
                    case BM_INT16:
                    case BM_UINT16:
                    case BM_FLOAT16:
                        m_tensor_size = m_tensor_elem_count << 1;
                        break;
                    default:
                        m_tensor_size = m_tensor_elem_count;
                }
            }
        }
    public:
        BMNNTensor(bm_handle_t handle, const std::string& name, float scale,
                   bm_tensor_t *tensor) : m_handle(handle), m_name(name),
                                          m_cpu_data(nullptr), m_scale(scale),
                                          m_tensor(tensor)
        {
            struct bm_misc_info misc_info;
            bm_status_t ret = bm_get_misc_info(handle, &misc_info);
            assert(BM_SUCCESS == ret);
            m_can_mmap = misc_info.pcie_soc_mode == 1;
        }

        virtual ~BMNNTensor() {
            if (m_cpu_data == nullptr) return;
            if(m_can_mmap && BM_FLOAT32 == m_tensor->dtype) {
                int tensor_size = bm_mem_get_device_size(m_tensor->device_mem);
                bm_status_t ret = bm_mem_unmap_device_mem(m_handle, m_cpu_data, tensor_size);
                assert(BM_SUCCESS == ret);
            } else {
                delete [] m_cpu_data;
            }
        }

        int set_device_mem(bm_device_mem_t *mem) {
            this->m_tensor->device_mem = *mem;
            return 0;
        }

        const bm_device_mem_t get_device_mem() {
            return this->m_tensor->device_mem;
        }

        int get_count() {
            //m_tensor maybe changed, so change
            update_shape();
            return m_tensor_elem_count;
        }

        int get_size() {
            update_shape();
            update_dtype();
            return m_tensor_size;
        }

        float *get_cpu_data() {
            if(m_cpu_data) return m_cpu_data;
            bm_status_t ret;
            float *pFP32 = nullptr;
            int count = bmrt_shape_count(&m_tensor->shape);
            // in SOC mode, device mem can be mapped to host memory, faster then using d2s
            if(m_can_mmap) {
                if (m_tensor->dtype == BM_FLOAT32) {
                    unsigned long long  addr;
                    ret = bm_mem_mmap_device_mem(m_handle, &m_tensor->device_mem, &addr);
                    assert(BM_SUCCESS == ret);
                    ret = bm_mem_invalidate_device_mem(m_handle, &m_tensor->device_mem);
                    assert(BM_SUCCESS == ret);
                    pFP32 = (float*)addr;
                } else if (BM_INT8 == m_tensor->dtype) {
                    int8_t * pI8 = nullptr;
                    unsigned long long  addr;
                    ret = bm_mem_mmap_device_mem(m_handle, &m_tensor->device_mem, &addr);
                    assert(BM_SUCCESS == ret);
                    ret = bm_mem_invalidate_device_mem(m_handle, &m_tensor->device_mem);
                    assert(BM_SUCCESS == ret);
                    pI8 = (int8_t*)addr;

                    // dtype convert
                    pFP32 = new float[count];
                    assert(pFP32 != nullptr);
                    for(int i = 0;i < count; ++ i) {
                    pFP32[i] = pI8[i] * m_scale;
                    }
                    ret = bm_mem_unmap_device_mem(m_handle, pI8, bm_mem_get_device_size(m_tensor->device_mem));
                    assert(BM_SUCCESS == ret);
                } else{
                    std::cout << "NOT support dtype=" << m_tensor->dtype << std::endl;
                }
            } else {
                // the common method using d2s
                if (m_tensor->dtype == BM_FLOAT32) {
                    pFP32 = new float[count];
                    assert(pFP32 != nullptr);
                    ret = bm_memcpy_d2s_partial(m_handle, pFP32, m_tensor->device_mem, count * sizeof(float));
                    assert(BM_SUCCESS ==ret);
                } else if (BM_INT8 == m_tensor->dtype) {
                    int8_t * pI8 = nullptr;
                    int tensor_size = bmrt_tensor_bytesize(m_tensor);
                    pI8 = new int8_t[tensor_size];
                    assert(pI8 != nullptr);

                    // dtype convert
                    pFP32 = new float[count];
                    assert(pFP32 != nullptr);
                    ret = bm_memcpy_d2s_partial(m_handle, pI8, m_tensor->device_mem, tensor_size);
                    assert(BM_SUCCESS ==ret);
                    for(int i = 0;i < count; ++ i) {
                    pFP32[i] = pI8[i] * m_scale;
                    }
                    delete [] pI8;
                } else{
                    std::cout << "NOT support dtype=" << m_tensor->dtype << std::endl;
                }
            }
            m_cpu_data = pFP32;
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

        size_t total() {
            return bmrt_shape_count(&m_tensor->shape);
        }

        int get_num() {
            return m_tensor->shape.dims[0];
        }

        const std::string& get_name() {
            return m_name;
        }
        bm_tensor_t *bm_tensor() {
            return m_tensor;
        }

        //int reshape_nchw(int n, int c, int h, int w) {
        //    return bm_tensor_reshape_NCHW(m_handle, m_tensor, n, c, h, w);
        //}

        //int reshape_nhwc(int n, int c, int h, int w) {
        //    return bm_tensor_reshape_NHWC(m_handle, m_tensor, n, c, h, w);
        //}
    };

    using BMNNTensorPtr = std::shared_ptr<BMNNTensor>;


    class BMNNNetwork : public NoCopyable {
        const bm_net_info_t *m_netinfo = nullptr;
        bm_tensor_t *m_inputTensors = nullptr;
        bm_tensor_t *m_outputTensors = nullptr;
        bm_handle_t m_handle = nullptr;
        void *m_bmrt = nullptr;
        //std::vector<std::vector<bm_shape_t>> m_input_shapes;

        std::unordered_map<std::string, int> m_mapInputName2Index;
        std::unordered_map<std::string, int> m_mapOutputName2Index;

        int m_stage_idx = 0;

        void init_io_tensors_by_stage(int stage_idx) {
            // free io tensor first
           release_io_tensors();

            // re-init io tensors
            m_inputTensors = new bm_tensor_t[m_netinfo->input_num];
            m_outputTensors = new bm_tensor_t[m_netinfo->output_num];
            for (int i = 0; i < m_netinfo->input_num; ++i) {
                m_inputTensors[i].dtype = m_netinfo->input_dtypes[i];
                m_inputTensors[i].shape = m_netinfo->stages[stage_idx].input_shapes[i];
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
        }

        void release_io_tensors() {
            // Free input tensors
            if (m_inputTensors != nullptr) {
                delete[] m_inputTensors;
            }

            // Free output tensors
            if (m_outputTensors != nullptr) {
                for (int i = 0; i < m_netinfo->output_num; ++i) {
                    if (m_outputTensors[i].device_mem.size != 0) {
                        bm_free_device(m_handle, m_outputTensors[i].device_mem);
                    }
                }
                delete[]m_outputTensors;
            }
        }
    public:
        static std::shared_ptr<BMNNNetwork> create(void *bmrt, const std::string& name)
        {
            return std::make_shared<BMNNNetwork>(bmrt, name);
        }
        BMNNNetwork(void *bmrt, const std::string &name) : m_bmrt(bmrt) {
            m_handle = static_cast<bm_handle_t>(bmrt_get_bm_handle(bmrt));
            m_netinfo = bmrt_get_network_info(bmrt, name.c_str());
            assert(m_netinfo->stage_num >= 1);
            init_io_tensors_by_stage(m_stage_idx);
        }

        bool is_dynamic() const
        {
            return m_netinfo->is_dynamic;
        }

        int set_stage_by_input_shape(int index, const bm_shape_t &shape) {
            int stage_idx = -1;
            for(int i = 0;i < m_netinfo->stage_num; ++i) {
                if (m_inputTensors[index].shape == shape) {
                    stage_idx = i;
                    break;
                }
            }

            if (-1 == stage_idx) return -1;

            // reset stage idx
            init_io_tensors_by_stage(stage_idx);
            return 0;
        }

        int set_stage_by_batch_size(int index, int batch_size) {
            int stage_idx = -1;
            for(int i = 0;i < m_netinfo->stage_num; ++i) {
                if (m_inputTensors[index].shape.dims[0] == batch_size) {
                    stage_idx = i;
                    break;
                }
            }

            if (-1 == stage_idx) return -1;

            // reset stage idx
            init_io_tensors_by_stage(stage_idx);
        }

        int get_stage_index() {
            return m_stage_idx;
        }

        // Get input shape by input index
        const bm_shape_t &get_input_shape(size_t index) const
        {
            assert(index < m_netinfo->input_num);
            return m_inputTensors[index].shape;
        }

        bm_data_type_t get_input_dtype(size_t index) const {
            assert(index < m_netinfo->input_num);
            return m_netinfo->input_dtypes[index];
        }

        float get_input_scale(size_t index) const {
            assert(index < m_netinfo->input_num);
            return m_netinfo->input_scales[index];
        }

        float get_output_scale(size_t index) const {
            assert(index < m_netinfo->output_num);
            return m_netinfo->output_scales[index];
        }

        ~BMNNNetwork() {
            release_io_tensors();
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

        std::shared_ptr<BMNNTensor> inputTensor(const std::string& name) {
            int index = outputName2Index(name);
            if (-1 == index) return nullptr;
            return std::make_shared<BMNNTensor>(m_handle, m_netinfo->input_names[index],
                                                m_netinfo->input_scales[index],
                                                &m_inputTensors[index]);
        }

        std::shared_ptr<BMNNTensor> inputTensor(int index) {
            assert(index < m_netinfo->input_num);
            return std::make_shared<BMNNTensor>(m_handle, m_netinfo->input_names[index],
                                                m_netinfo->input_scales[index],
                                                &m_inputTensors[index]);
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
            bool user_mem = false; // if false, bmrt will alloc mem every time.
            if (m_outputTensors->device_mem.size != 0) {
                // if true, bmrt don't alloc mem again.
                user_mem = true;
            }

            bool ok = bmrt_launch_tensor_ex(m_bmrt, m_netinfo->name, m_inputTensors, m_netinfo->input_num,
                                         m_outputTensors, m_netinfo->output_num, user_mem, true);
            if (!ok) {
                std::cout << "bm_launch_tensor() failed=" << std::endl;
                return -1;
            }

            /* wait for inference done */
            bm_status_t res = (bm_status_t)bm_thread_sync (m_handle);
            if (res != BM_SUCCESS) {
                std::cout << "bm_thread_sync: Failed to sync: " << m_netinfo->name << " inference" << std::endl;
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

            /* wait for inference done */
            bm_status_t res = (bm_status_t)bm_thread_sync (m_handle);
            if (res != BM_SUCCESS) {
                std::cout << "bm_thread_sync: Failed to sync: " << m_netinfo->name << " inference" << std::endl;
                return -1;
            }

            return 0;
        }

        int forward_user_mem(
            const bm_tensor_t *input_tensors,
            int input_num,
            bm_tensor_t *output_tensors,
            int output_num)
        {
            bool ok = bmrt_launch_tensor_ex(m_bmrt, m_netinfo->name, input_tensors, input_num,
                    output_tensors, output_num, true, false);
            if (!ok) {
                std::cout << "bm_launch_tensor_ex() failed=" << std::endl;
                return -1;
            }

            /* wait for inference done */
            bm_status_t res = (bm_status_t)bm_thread_sync (m_handle);
            if (res != BM_SUCCESS) {
                std::cout << "bm_thread_sync: Failed to sync: " << m_netinfo->name << " inference" << std::endl;
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
        static std::shared_ptr<BMNNHandle> create(int dev_id) {
            return std::make_shared<BMNNHandle>(dev_id);
        }
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
        static std::shared_ptr<BMNNContext> create(BMNNHandlePtr handle, const std::string& bmodel_file)
        {
            return std::make_shared<BMNNContext>(handle, bmodel_file);
        }

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

        int dev_id() {
            return m_handlePtr->dev_id();
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

#endif //MY_TRT_BMUTILITY_NN_H
