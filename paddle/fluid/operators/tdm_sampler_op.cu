/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License. */

#pragma once

#include <cuda.h>
#include <cub/cub.cuh>
#include "paddle/fluid/operators/tdm_sampler_op.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace operators {

__inline__ __device__ uint32_t CudaSRandom(const uint64_t seed,
                                           const uint32_t &max) {
  return (((214013u * seed + 2531011u) >> 16u) & 0x7fffu) % max;
}

__inline__ __device__ bool SetPositiveIdx(int *sample_positive_idxs,
                                          const int pos, const int layer_pos) {
  for (int k = 0; k < pos; ++k) {
    if (sample_positive_idxs[k] == layer_pos) {
      return false;
    }
  }
  sample_positive_idxs[pos] = layer_pos;
  return true;
}

template <typename T, typename TreeT>
__global__ void Kernel_TDMIndex(
    const size_t N, const T *input_data, const TreeT *travel_data,
    const TreeT *layer_data, int *positive_idxs, int *layer_idxs,
    const int sample_res_length, const int layer_nums,
    const int *layer_sample_offsets, const int *layer_offset_lod,
    const bool output_positive_flag, const int seed) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (; idx < N; idx += blockDim.x * gridDim.x) {
    const int ins_id = idx / layer_nums;
    const int layer_idx = idx % layer_nums;

    int sample_num =
        layer_sample_offsets[layer_idx + 1] - layer_sample_offsets[layer_idx];
    int start_offset =
        ins_id * sample_res_length + layer_sample_offsets[layer_idx];
    int positive_node_id = static_cast<int>(
        travel_data[static_cast<int>(input_data[ins_id] * layer_nums) +
                    layer_idx]);
    if (positive_node_id == 0) {
      for (int k = 0; k < sample_num; ++k) {
        layer_idxs[start_offset + k] = layer_idx;
      }
      continue;
    }
    layer_idxs[start_offset] = layer_idx;
    start_offset += static_cast<int>(output_positive_flag);
    sample_num = sample_num - static_cast<int>(output_positive_flag);

    const int &layer_node_pos = layer_offset_lod[layer_idx];
    uint32_t layer_data_num =
        static_cast<uint32_t>(layer_offset_lod[layer_idx + 1] - layer_node_pos);
    // skip positive_node_id
    int *sample_layer_idxs = &layer_idxs[start_offset];
    int *sample_positive_idxs = &positive_idxs[start_offset];
    assert(layer_data_num > sample_num);

    uint64_t x = static_cast<uint64_t>(clock64() + seed);
    uint64_t layer_data_pos = 0;
    uint64_t step =
        (static_cast<uint64_t>(layer_data_num / sample_num) >> 1) + 1;
    for (int k = 0; k < sample_num; ++k) {
      sample_layer_idxs[k] = layer_idx;
      do {
        layer_data_pos = layer_node_pos + CudaSRandom(x, layer_data_num);
        x += step;
      } while (positive_node_id == layer_data[layer_data_pos] ||
               !SetPositiveIdx(sample_positive_idxs, k, layer_data_pos));
    }
  }
}

template <typename T, typename TreeT = int, typename OutT = int>
__global__ void Kernel_TDMSampler(
    const size_t N, const T *input_data, const TreeT *travel_data,
    const TreeT *layer_data, OutT *output_data, OutT *label_data,
    OutT *mask_data, const bool output_positive_flag, const int layer_nums,
    const int sample_res_length, const int *layer_sample_offsets,
    const int *positive_idx, const int *layer_idxs_data) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (; idx < N; idx += blockDim.x * gridDim.x) {
    const int ins_id = idx / sample_res_length;
    const int layer_off = idx % sample_res_length;

    // find layer id
    const int &layer_idx = layer_idxs_data[idx];
    OutT positive_node_id = static_cast<OutT>(
        travel_data[static_cast<int>(input_data[ins_id] * layer_nums) +
                    layer_idx]);
    if (positive_node_id == 0) {
      // skip padding
      output_data[idx] = 0;
      label_data[idx] = 0;
      mask_data[idx] = 0;
      continue;
    }
    // If output positive, add itself
    if (output_positive_flag && layer_off == layer_sample_offsets[layer_idx]) {
      output_data[idx] = positive_node_id;
      label_data[idx] = 1;
      mask_data[idx] = 1;
      continue;
    }
    output_data[idx] = static_cast<OutT>(layer_data[positive_idx[idx]]);
    label_data[idx] = 0;
    mask_data[idx] = 1;
  }
}

template <typename T, typename TreeT = int, typename OutT = int>
void CudaTDMSamplerInner(const framework::ExecutionContext &context,
                         const LoDTensor &input_tensor,
                         const LoDTensor &travel_lod_tensor,
                         const LoDTensor &layer_lod_tensor,
                         LoDTensor *out_tensor, LoDTensor *label_tensor,
                         LoDTensor *mask_tensor) {
  auto neg_samples_num_vec =
      context.Attr<std::vector<int>>("neg_samples_num_list");
  auto layer_offset_lod = context.Attr<std::vector<int>>("layer_offset_lod");
  auto output_positive_flag = context.Attr<bool>("output_positive");

  // get dimension
  int input_ids_num = input_tensor.numel();
  auto layer_nums = neg_samples_num_vec.size();

  int sample_res_length = 0;
  std::vector<int> sample_offset_vec;
  sample_offset_vec.resize(layer_nums + 1);
  sample_offset_vec[0] = 0;
  for (size_t layer_idx = 0; layer_idx < layer_nums; ++layer_idx) {
    sample_res_length += (neg_samples_num_vec[layer_idx] +
                          static_cast<int>(output_positive_flag));
    sample_offset_vec[layer_idx + 1] = sample_res_length;
  }
  VLOG(3) << "TDM Cuda: input ids nums: " << input_ids_num
          << ", tree layer nums: " << layer_nums
          << ", sample res length: " << sample_res_length
          << ", layer_offset_lod size:[" << layer_offset_lod.size() << ","
          << layer_offset_lod[layer_nums] << "]";

  // get all data
  auto *input_data = input_tensor.data<T>();
  auto *travel_data = travel_lod_tensor.data<TreeT>();
  auto *layer_data = layer_lod_tensor.data<TreeT>();

  auto place = context.GetPlace();

  auto *output_data = out_tensor->mutable_data<OutT>(place);
  auto *label_data = label_tensor->mutable_data<OutT>(place);
  auto *mask_data = mask_tensor->mutable_data<OutT>(place);

  // generate uniform sampler
  auto seed = context.Attr<int>("seed");
  auto stream = context.cuda_device_context().stream();

  int total_out_len = input_ids_num * sample_res_length;
  LoDTensor data_tensor;
  int64_t data_total_len = static_cast<int64_t>(
      sample_offset_vec.size() + layer_offset_lod.size() + total_out_len * 2);
  int *layer_offset_lod_data =
      data_tensor.mutable_data<int>({data_total_len, 1}, place);
  int *sample_offset_data =
      &layer_offset_lod_data[static_cast<int>(layer_offset_lod.size())];

  cudaMemcpyAsync(layer_offset_lod_data, layer_offset_lod.data(),
                  layer_offset_lod.size() * sizeof(int), cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(sample_offset_data, sample_offset_vec.data(),
                  sample_offset_vec.size() * sizeof(int),
                  cudaMemcpyHostToDevice, stream);

  int *layer_idxs_data =
      &sample_offset_data[static_cast<int>(sample_offset_vec.size())];
  int *positive_idxs = &layer_idxs_data[total_out_len];

  int total_ins_layers_nums = input_ids_num * layer_nums;
  // build index
  Kernel_TDMIndex<<<(total_ins_layers_nums + 512 - 1) / 512, 512, 0, stream>>>(
      total_ins_layers_nums, input_data, travel_data, layer_data, positive_idxs,
      layer_idxs_data, sample_res_length, layer_nums, sample_offset_data,
      layer_offset_lod_data, output_positive_flag, seed);

  // fill tdm gpu sample data
  Kernel_TDMSampler<<<(total_out_len + 512 - 1) / 512, 512, 0, stream>>>(
      total_out_len, input_data, travel_data, layer_data, output_data,
      label_data, mask_data, output_positive_flag, layer_nums,
      sample_res_length, sample_offset_data, positive_idxs, layer_idxs_data);
  cudaStreamSynchronize(stream);
}

template <typename DeviceContext, typename T>
class TDMSamplerCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *input_var = context.InputVar("X");
    auto *travel_var = context.InputVar("Travel");
    auto *layer_var = context.InputVar("Layer");

    // get all tensor
    auto &input_tensor = input_var->Get<framework::LoDTensor>();
    auto &travel_lod_tensor = travel_var->Get<framework::LoDTensor>();
    auto &layer_lod_tensor = layer_var->Get<framework::LoDTensor>();

    const auto &input_type = input_tensor.type();
    bool input_type_match = input_type == framework::proto::VarType::INT32 ||
                            input_type == framework::proto::VarType::INT64;
    PADDLE_ENFORCE_EQ(input_type_match, true,
                      platform::errors::InvalidArgument(
                          "Input(X) holds the wrong type, it holds %s, but "
                          "desires to be %s or %s",
                          paddle::framework::DataTypeToString(input_type),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT32),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT64)));

    const auto &travel_type = travel_lod_tensor.type();
    bool travel_type_match = travel_type == framework::proto::VarType::INT32 ||
                             travel_type == framework::proto::VarType::INT64;
    PADDLE_ENFORCE_EQ(
        travel_type_match, true,
        platform::errors::InvalidArgument(
            "Input(Travel) holds the wrong type, it holds %s, but "
            "desires to be %s or %s",
            paddle::framework::DataTypeToString(travel_type),
            paddle::framework::DataTypeToString(
                framework::proto::VarType::INT32),
            paddle::framework::DataTypeToString(
                framework::proto::VarType::INT64)));

    const auto &layer_type = layer_lod_tensor.type();
    bool layer_type_match = layer_type == framework::proto::VarType::INT32 ||
                            layer_type == framework::proto::VarType::INT64;
    PADDLE_ENFORCE_EQ(layer_type_match, true,
                      platform::errors::InvalidArgument(
                          "Input(Layer) holds the wrong type, it holds %s, but "
                          "desires to be %s or %s",
                          paddle::framework::DataTypeToString(layer_type),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT32),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT64)));
    PADDLE_ENFORCE_EQ(
        travel_type, layer_type,
        platform::errors::InvalidArgument(
            "Input(Travel) must holds the same type with "
            "Input(Layer), but Travel holds %s, and Layer holds %s",
            paddle::framework::DataTypeToString(travel_type),
            paddle::framework::DataTypeToString(layer_type)));

    auto *out_tensor =
        context.OutputVar("Out")->GetMutable<framework::LoDTensor>();
    auto *label_tensor =
        context.OutputVar("Labels")->GetMutable<framework::LoDTensor>();
    auto *mask_tensor =
        context.OutputVar("Mask")->GetMutable<framework::LoDTensor>();

    auto output_type = static_cast<framework::proto::VarType::Type>(
        context.Attr<int>("dtype"));

    if (travel_type == framework::proto::VarType::INT32 &&
        output_type == framework::proto::VarType::INT32) {
      CudaTDMSamplerInner<T, int, int>(context, input_tensor, travel_lod_tensor,
                                       layer_lod_tensor, out_tensor,
                                       label_tensor, mask_tensor);
    } else if (travel_type == framework::proto::VarType::INT64 &&
               output_type == framework::proto::VarType::INT32) {
      CudaTDMSamplerInner<T, int64_t, int>(
          context, input_tensor, travel_lod_tensor, layer_lod_tensor,
          out_tensor, label_tensor, mask_tensor);
    } else if (travel_type == framework::proto::VarType::INT32 &&
               output_type == framework::proto::VarType::INT64) {
      CudaTDMSamplerInner<T, int, int64_t>(
          context, input_tensor, travel_lod_tensor, layer_lod_tensor,
          out_tensor, label_tensor, mask_tensor);
    } else if (travel_type == framework::proto::VarType::INT64 &&
               output_type == framework::proto::VarType::INT64) {
      CudaTDMSamplerInner<T, int64_t, int64_t>(
          context, input_tensor, travel_lod_tensor, layer_lod_tensor,
          out_tensor, label_tensor, mask_tensor);
    }
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    tdm_sampler,
    paddle::operators::TDMSamplerCUDAKernel<paddle::platform::CUDADeviceContext,
                                            float>,
    paddle::operators::TDMSamplerCUDAKernel<paddle::platform::CUDADeviceContext,
                                            double>,
    paddle::operators::TDMSamplerCUDAKernel<paddle::platform::CUDADeviceContext,
                                            int>,
    paddle::operators::TDMSamplerCUDAKernel<paddle::platform::CUDADeviceContext,
                                            int64_t>);
