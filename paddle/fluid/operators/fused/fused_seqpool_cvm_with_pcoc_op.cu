//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>
#include "paddle/fluid/operators/fused/fused_seqpool_cvm_with_pcoc_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace operators {

using platform::PADDLE_CUDA_NUM_THREADS;

#define GET_BLOCK(N) \
  ((N + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS)

#define CUDA_KERNEL_LOOP(i, n)                                  \
  for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

// normal
template <typename T>
__global__ void FusedSeqpoolWithPCOCKernelNormal(
    const size_t N, T **input_values, T **seqpool_output_values,
    size_t **lods_values, const int batch_size, const int embedding_size,
    const float pad_value) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;
    int x = key / batch_size;  // slot id
    int y = key % batch_size;  // ins id
    auto &start = *(lods_values[x] + y);
    auto &end = *(lods_values[x] + y + 1);

    T val = static_cast<T>(pad_value);
    for (auto k = start; k < end; ++k) {
      val += *(input_values[x] + k * embedding_size + offset);
    }
    *(seqpool_output_values[x] + y * embedding_size + offset) = val;
  }
}
// not need filter quant
template <typename T>
__global__ void FusedSeqpoolWithPCOCKernelQuant(
    const size_t N, T **input_values, T **seqpool_output_values,
    size_t **lods_values, const int batch_size, const int embedding_size,
    const float pad_value, const int max_cvm_offset, const int quant_ratio) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;
    int x = key / batch_size;  // slot id
    int y = key % batch_size;  // ins id
    auto &start = *(lods_values[x] + y);
    auto &end = *(lods_values[x] + y + 1);

    T val = static_cast<T>(pad_value);
    // quant
    for (auto k = start; k < end; ++k) {
      if (offset < max_cvm_offset) {  // cvm
        val += *(input_values[x] + k * embedding_size + offset);
      } else {
        val += ((static_cast<int>(
                    *(input_values[x] + k * embedding_size + offset) *
                        quant_ratio +
                    0.5)) /
                static_cast<float>(quant_ratio));
      }
    }
    *(seqpool_output_values[x] + y * embedding_size + offset) = val;
  }
}
// quant filter
template <typename T>
__global__ void FusedSeqpoolWithPCOCKernelQuantFilter(
    const size_t N, T **input_values, T **seqpool_output_values,
    size_t **lods_values, const int batch_size, const int embedding_size,
    const float pad_value, const int max_cvm_offset, const float show_coeff,
    const float clk_coeff, const float threshold, const int quant_ratio) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;  // embedx id
    int x = key / batch_size;         // slot id
    int y = key % batch_size;         // ins id
    auto &start = *(lods_values[x] + y);
    auto &end = *(lods_values[x] + y + 1);

    T val = static_cast<T>(pad_value);
    for (auto k = start; k < end; ++k) {
      T &show = *(input_values[x] + k * embedding_size);
      T &click = *(input_values[x] + k * embedding_size + 1);
      if ((show - click) * show_coeff + click * clk_coeff < threshold) {
        continue;
      }
      if (offset < max_cvm_offset) {  // show & click
        val += *(input_values[x] + k * embedding_size + offset);
      } else {
        val += ((static_cast<int>(
                    *(input_values[x] + k * embedding_size + offset) *
                        quant_ratio +
                    0.5)) /
                static_cast<float>(quant_ratio));
      }
    }
    *(seqpool_output_values[x] + y * embedding_size + offset) = val;
  }
}
// join need cvm input
template <typename T>
__global__ void FusedCVMWithPCOCKernelWithCVM(const size_t output_N, T **output_values,
                                              T **seqpool_output_values,
                                              const int batch_size,
                                              const int input_embedding_size,
                                              const int ouput_embedding_size,
                                              const int pclk_num,
                                              const int embed_index_diff) {
  CUDA_KERNEL_LOOP(i, output_N) {
    int key = i / ouput_embedding_size;
    int offset = i % ouput_embedding_size;
    int x = key / batch_size;  // slot id
    int y = key % batch_size;  // ins id
    if (offset == 0) {  // show
      *(output_values[x] + y * ouput_embedding_size) =
          log(*(seqpool_output_values[x] + y * input_embedding_size) + 1);
    } else if (offset == 1) {  // ctr_smoth = log(click) - log(show)
      *(output_values[x] + y * ouput_embedding_size + offset) =
          log(*(seqpool_output_values[x] + y * input_embedding_size + 1) + 1) -
          log(*(seqpool_output_values[x] + y * input_embedding_size) + 1);
    } else if (offset < 2 + pclk_num) { // 2:(4-2)/3:(5-2)/4:(6-2)
      *(output_values[x] + y * ouput_embedding_size + offset) =
          log(*(seqpool_output_values[x] + y * input_embedding_size + (offset + 2)) + 1) -
          log(*(seqpool_output_values[x] + y * input_embedding_size + 2) + 1);
    } else if (offset < 2 + 2 * pclk_num) { // 5:(4-3)/6:(5-3)/7:(6-3)
      *(output_values[x] + y * ouput_embedding_size + offset) =
          log(*(seqpool_output_values[x] + y * input_embedding_size + (offset + 2 - pclk_num)) + 1) -
          log(*(seqpool_output_values[x] + y * input_embedding_size + 3) + 1);
    } else {
      *(output_values[x] + y * ouput_embedding_size + offset) =
          *(seqpool_output_values[x] + y * input_embedding_size + offset + embed_index_diff);
    }
  }
}

// update not need show click input
template <typename T>
__global__ void FusedCVMWithPCOCKernelNoCVM(const size_t output_N, T **output_values,
                                            T **seqpool_output_values,
                                            const int batch_size,
                                            const int no_cvm_embedding_size,
                                            const int max_cvm_offset) {
  CUDA_KERNEL_LOOP(i, output_N) {
    int key = i / no_cvm_embedding_size;
    int offset = i % no_cvm_embedding_size;
    int x = key / batch_size;  // slot id
    int y = key % batch_size;  // ins id
    // no cvm
    *(output_values[x] + y * no_cvm_embedding_size + offset) =
        *(seqpool_output_values[x] + y * (no_cvm_embedding_size + max_cvm_offset) +
          offset + max_cvm_offset);
  }
}

template <typename T>
void FusedSeqpoolCVMWithPCOC(const paddle::platform::Place &place,
                             const std::vector<const T *> &input_data,
                             const std::vector<T *> &output_data,
                             const std::vector<T *> &seqpool_output_data,
                             std::vector<const size_t *> lods, const int batch_size,
                             const int slot_num, const int embedding_size,
                             const float padding_value, const bool use_cvm,
                             const int used_cvm_offset, const int max_cvm_offset,
                             float need_filter, float show_coeff, float clk_coeff,
                             float threshold, const int quant_ratio) {
  auto stream = dynamic_cast<platform::CUDADeviceContext *>(
                    platform::DeviceContextPool::Instance().Get(
                        BOOST_GET_CONST(platform::CUDAPlace, place)))
                    ->stream();

  size_t total_ptr_len = input_data.size() + output_data.size() +
                         seqpool_output_data.size() + lods.size();
  auto temp_ptr = memory::AllocShared(place, total_ptr_len * sizeof(void *));
  void *ptr = temp_ptr->ptr();

  T **gpu_input_values = reinterpret_cast<T **>(temp_ptr->ptr());
  cudaMemcpyAsync(gpu_input_values, input_data.data(),
                  input_data.size() * sizeof(T *), cudaMemcpyHostToDevice,
                  stream);
  T **gpu_output_values =
      reinterpret_cast<T **>(&gpu_input_values[input_data.size()]);
  cudaMemcpyAsync(gpu_output_values, output_data.data(),
                  output_data.size() * sizeof(T *), cudaMemcpyHostToDevice,
                  stream);
  T **gpu_seqpool_output_values =
      reinterpret_cast<T **>(&gpu_output_values[output_data.size()]);
  cudaMemcpyAsync(gpu_seqpool_output_values, seqpool_output_data.data(),
                  seqpool_output_data.size() * sizeof(T *),
                  cudaMemcpyHostToDevice, stream);
  size_t **lods_values = reinterpret_cast<size_t **>(
      &gpu_seqpool_output_values[seqpool_output_data.size()]);
  cudaMemcpyAsync(lods_values, lods.data(), lods.size() * sizeof(size_t *),
                  cudaMemcpyHostToDevice, stream);

  size_t N = static_cast<size_t>(batch_size * slot_num * embedding_size);
    // first sum pool
  if (need_filter) {  // quant need filter
    FusedSeqpoolWithPCOCKernelQuantFilter<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0,
                                            stream>>>(
        N, gpu_input_values, gpu_seqpool_output_values, lods_values, batch_size,
        embedding_size, padding_value, max_cvm_offset, show_coeff, clk_coeff,
        threshold, quant_ratio);
  } else if (quant_ratio > 0) {  // quant not filter
    FusedSeqpoolWithPCOCKernelQuant<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0,
                                      stream>>>(
        N, gpu_input_values, gpu_seqpool_output_values, lods_values, batch_size,
        embedding_size, padding_value, max_cvm_offset, quant_ratio);
  } else {  // normal
    FusedSeqpoolWithPCOCKernelNormal<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0,
                                       stream>>>(
        N, gpu_input_values, gpu_seqpool_output_values, lods_values, batch_size,
        embedding_size, padding_value);
  }

  int pclk_num = used_cvm_offset - 4; // 4 : show/clk/show2/clk2
  int embed_index_diff = max_cvm_offset - 2 - 2 * pclk_num;
  int ouput_embedding_size = embedding_size - embed_index_diff;
  size_t output_N = static_cast<size_t>(batch_size * slot_num * ouput_embedding_size);
  // second log
  if (use_cvm) {
    FusedCVMWithPCOCKernelWithCVM<<<GET_BLOCK(output_N), PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
        output_N, gpu_output_values, gpu_seqpool_output_values, batch_size,
        embedding_size, ouput_embedding_size, pclk_num, embed_index_diff);
  } else {
    // not need cvm input
    output_N = static_cast<size_t>(batch_size * slot_num *
                            (embedding_size - max_cvm_offset));
    FusedCVMWithPCOCKernelNoCVM<<<GET_BLOCK(output_N), PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
        output_N, gpu_output_values, gpu_seqpool_output_values, batch_size,
        (embedding_size - max_cvm_offset), max_cvm_offset);
  }
}

// join grad
template <typename T>
__global__ void FusedSeqpoolCVMWithPCOCGradKernelWithCVM(
    const size_t N, T **out_grads_values, T **in_grads_values, T **cvm_values,
    size_t **lods_values, const int batch_size, const int embedding_size,
    const int used_cvm_offset, const int max_cvm_offset, const float *q_values) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;  // embedx offset
    int x = key / batch_size;         // slot id
    int y = key % batch_size;         // ins id
    T val = 0;
    int pclk_num = used_cvm_offset - 4; // 4 : show/clk/show2/clk2

    if (offset < max_cvm_offset) {
      if (offset < 4) { //show clk show2 clk2
        val = *(cvm_values[x] + y * used_cvm_offset + offset);
      } else if (offset < used_cvm_offset) { //pclk pclk2 pclk3...
        val = q_values[y * pclk_num + offset - 4];
      }
    } else {
      int embed_index_diff = max_cvm_offset - 2 - 2 * pclk_num;
      val = *(out_grads_values[x] + y * (embedding_size - embed_index_diff) +
            offset - embed_index_diff);
    }
    auto &start = *(lods_values[x] + y);
    auto &end = *(lods_values[x] + y + 1);
    for (auto k = start; k < end; ++k) {
      *(in_grads_values[x] + k * embedding_size + offset) = val;
    }
  }
}
// update grad
template <typename T>
__global__ void FusedSeqpoolCVMWithPCOCGradKernelNoCVM(
    const size_t N, T **out_grads_values, T **in_grads_values, T **cvm_values,
    size_t **lods_values, const int batch_size, const int embedding_size,
    const int used_cvm_offset, const int max_cvm_offset, const float *q_values) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;  // embedx offset
    int x = key / batch_size;         // slot id
    int y = key % batch_size;         // ins id
    T val = 0;
    int pclk_num = used_cvm_offset - 4; // 4 : show/clk/show2/clk2

    if (offset < max_cvm_offset) {
      if (offset < 4) { //show clk show2 clk2
        val = *(cvm_values[x] + y * used_cvm_offset + offset);
      } else if (offset < used_cvm_offset) { //pclk pclk2 pclk3
        val = q_values[y * pclk_num + offset - 4];
      }
    } else {
      val = *(out_grads_values[x] + y * (embedding_size - max_cvm_offset) +
              offset - max_cvm_offset);

    }
    auto &start = *(lods_values[x] + y);
    auto &end = *(lods_values[x] + y + 1);
    for (auto k = start; k < end; ++k) {
      *(in_grads_values[x] + k * embedding_size + offset) = val;
    }
  }
}

template <typename T>
void FusedSeqpoolCVMWithPCOCGrad(const paddle::platform::Place &place,
                                 const std::vector<const T *> &out_grads_data,
                                 const std::vector<T *> &in_grads_data,
                                 const std::vector<const T *> &cvm_data,
                                 const std::vector<const size_t *> &lods,
                                 const int batch_size, const int slot_num,
                                 const int embedding_size, const bool use_cvm,
                                 const int used_cvm_offset, const int max_cvm_offset,
                                 const float *q_values) {
  auto stream = dynamic_cast<platform::CUDADeviceContext *>(
                    platform::DeviceContextPool::Instance().Get(
                        BOOST_GET_CONST(platform::CUDAPlace, place)))
                    ->stream();
  size_t total_ptr_len = out_grads_data.size() + in_grads_data.size() +
                         cvm_data.size() + lods.size();
  auto temp_ptr = memory::AllocShared(place, total_ptr_len * sizeof(void *));
  T **gpu_out_grads_values = reinterpret_cast<T **>(temp_ptr->ptr());
  cudaMemcpyAsync(gpu_out_grads_values, out_grads_data.data(),
                  out_grads_data.size() * sizeof(T *), cudaMemcpyHostToDevice,
                  stream);

  T **gpu_in_grads_values =
      reinterpret_cast<T **>(&gpu_out_grads_values[out_grads_data.size()]);
  cudaMemcpyAsync(gpu_in_grads_values, in_grads_data.data(),
                  in_grads_data.size() * sizeof(T *), cudaMemcpyHostToDevice,
                  stream);

  T **gpu_cvm_values =
      reinterpret_cast<T **>(&gpu_in_grads_values[in_grads_data.size()]);
  cudaMemcpyAsync(gpu_cvm_values, cvm_data.data(),
                  cvm_data.size() * sizeof(T *), cudaMemcpyHostToDevice,
                  stream);

  size_t **lods_values =
      reinterpret_cast<size_t **>(&gpu_cvm_values[cvm_data.size()]);
  cudaMemcpyAsync(lods_values, lods.data(), lods.size() * sizeof(size_t *),
                  cudaMemcpyHostToDevice, stream);


  size_t N = static_cast<size_t>(batch_size * slot_num * embedding_size);
  if (use_cvm) {
    // join grad
    FusedSeqpoolCVMWithPCOCGradKernelWithCVM<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0,
                                               stream>>>(
        N, gpu_out_grads_values, gpu_in_grads_values, gpu_cvm_values,
        lods_values, batch_size, embedding_size, used_cvm_offset, max_cvm_offset, q_values);
  } else {
    // update grad
    FusedSeqpoolCVMWithPCOCGradKernelNoCVM<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0,
                                             stream>>>(
        N, gpu_out_grads_values, gpu_in_grads_values, gpu_cvm_values,
        lods_values, batch_size, embedding_size, used_cvm_offset, max_cvm_offset, q_values);
  }
}

template <typename T>
class FusedSeqpoolCVMWithPCOCCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto inputs = ctx.MultiInput<LoDTensor>("X");
    auto outputs = ctx.MultiOutput<framework::Tensor>("Out");

    const auto slot_size = inputs.size();
    std::vector<const float *> input_data(slot_size);
    std::vector<const size_t *> lods_data(slot_size);
    std::vector<T *> output_data(slot_size);

    std::vector<LoDTensor> seqpool_outputs(slot_size);
    std::vector<T *> seqpool_output_data(slot_size);

    auto padding_value = ctx.Attr<float>("pad_value");
    auto use_cvm = ctx.Attr<bool>("use_cvm");
    bool need_filter = ctx.Attr<bool>("need_filter");
    float show_coeff = ctx.Attr<float>("show_coeff");
    float clk_coeff = ctx.Attr<float>("clk_coeff");
    float threshold = ctx.Attr<float>("threshold");
    const int used_cvm_offset = ctx.Attr<int>("cvm_offset");
    const int max_cvm_offset = ctx.Attr<int>("max_cvm_offset");
    const int quant_ratio = ctx.Attr<int>("quant_ratio");
    int embed_index_diff = max_cvm_offset - 2 * used_cvm_offset + 6;

    int embedding_size = inputs[0]->numel() / inputs[0]->dims()[0];
    int batch_size = -1;
    for (size_t i = 0; i < slot_size; ++i) {
      const auto *input = inputs[i];
      auto dims = input->dims();

      auto lod = input->lod();
      auto lod_level = lod.size();
      int cur_batch = lod[lod_level - 1].size() - 1;
      if (batch_size == -1) {
        batch_size = cur_batch;
      } else {
        CHECK(batch_size == cur_batch) << "batch: " << batch_size
                                       << ", current: " << cur_batch;
      }

      input_data[i] = reinterpret_cast<const T *>(input->data<T>());
      auto *output = outputs[i];
      if (use_cvm) {
        output->Resize({batch_size, embedding_size - embed_index_diff});
      } else {
        output->Resize({batch_size, embedding_size - max_cvm_offset});
      }
      output_data[i] =
          reinterpret_cast<T *>(output->mutable_data<T>(ctx.GetPlace()));
      lods_data[i] = lod[lod_level - 1].CUDAData(ctx.GetPlace());

      seqpool_output_data[i] =
          reinterpret_cast<T *>(seqpool_outputs[i].mutable_data<T>(
              {batch_size, embedding_size}, ctx.GetPlace()));
    }

    FusedSeqpoolCVMWithPCOC(ctx.GetPlace(), input_data, output_data,
                    seqpool_output_data, lods_data, batch_size, slot_size,
                    embedding_size, padding_value, use_cvm, used_cvm_offset,
                    max_cvm_offset, need_filter, show_coeff, clk_coeff,
                    threshold, quant_ratio);

  }
};

template <typename T>
class FusedSeqpoolCVMWithPCOCGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto out_grads = ctx.MultiInput<LoDTensor>(framework::GradVarName("Out"));
    auto in_grads = ctx.MultiOutput<LoDTensor>(framework::GradVarName("X"));
    auto *cvm = ctx.Input<LoDTensor>("CVMWithPCOC");

    std::string pooltype = ctx.Attr<std::string>("pooltype");
    auto use_cvm = ctx.Attr<bool>("use_cvm");
    const int used_cvm_offset = ctx.Attr<int>("cvm_offset");
    const int max_cvm_offset = ctx.Attr<int>("max_cvm_offset");

    auto place = ctx.GetPlace();
    int device_id = boost::get<platform::CUDAPlace>(place).GetDeviceId();
    LoDTensor *qvalue_tensor = &(paddle::framework::BoxWrapper::GetInstance()->GetQTensor(device_id));
    const float *q_values = qvalue_tensor->data<float>();


    const auto slot_size = in_grads.size();
    std::vector<const T *> out_grads_data(slot_size);
    std::vector<T *> in_grads_data(slot_size);
    std::vector<const T *> cvm_data(slot_size);
    std::vector<const size_t *> lods_data(slot_size);

    int embedding_size = in_grads[0]->numel() / in_grads[0]->dims()[0];
    int batch_size = -1;
    for (size_t i = 0; i < slot_size; ++i) {
      auto *in_grad = in_grads[i];

      auto lod = in_grad->lod();
      auto lod_level = lod.size();
      int cur_batch = lod[lod_level - 1].size() - 1;
      if (batch_size == -1) {
        batch_size = cur_batch;
      } else {
        CHECK(batch_size == cur_batch) << "batch: " << batch_size
                                       << ", current: " << cur_batch;
      }

      auto *out_grad = out_grads[i];
      out_grads_data[i] = reinterpret_cast<const T *>(out_grad->data<T>());

      in_grads_data[i] =
          reinterpret_cast<T *>(in_grad->mutable_data<T>(ctx.GetPlace()));
      lods_data[i] = lod[lod_level - 1].CUDAData(ctx.GetPlace());
      cvm_data[i] = reinterpret_cast<const T *>(cvm->data<T>());

    }
    FusedSeqpoolCVMWithPCOCGrad(ctx.GetPlace(), out_grads_data, in_grads_data, cvm_data,
                                lods_data, batch_size, slot_size, embedding_size,
                                use_cvm, used_cvm_offset, max_cvm_offset, q_values);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(fused_seqpool_cvm_with_pcoc,
                        ops::FusedSeqpoolCVMWithPCOCCUDAKernel<float>);

REGISTER_OP_CUDA_KERNEL(fused_seqpool_cvm_with_pcoc_grad,
                        ops::FusedSeqpoolCVMWithPCOCGradCUDAKernel<float>);
