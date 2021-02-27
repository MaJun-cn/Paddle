// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_BOX_PS
#include <algorithm>
#include <ctime>
#include <memory>
#include <numeric>
#include "paddle/fluid/framework/fleet/box_wrapper.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace framework {
#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

template <typename FEATURE_VALUE_GPU_TYPE>
__global__ void PullCopy(float** dest, const FEATURE_VALUE_GPU_TYPE* src,
                         const int hidden, const int expand_dim,
                         const int total_len, uint64_t** keys, int* total_dims,
                         const int64_t* slot_lens, const int slot_num,
                         const int* key2slot, const float scale,
                         const int cvm_offset) {
  assert(expand_dim == 0);
  // only process no expand data
  CUDA_KERNEL_LOOP(i, total_len) {
    int x = key2slot[i];
    int y = i - slot_lens[x];

    auto& src_val = src[i];
    //    *(dest[x] + y * hidden) = (src + i)->show;
    //    *(dest[x] + y * hidden + 1) = (src + i)->clk;
    //    *(dest[x] + y * hidden + 2) = (src + i)->embed_w;
    float* dest_ptr = dest[x] + y * hidden;
    const float* src_ptr = reinterpret_cast<const float*>(&src_val.show);
    for (int k = 0; k < cvm_offset; ++k) {
      dest_ptr[k] = src_ptr[k];
    }
    // embedx
    if (src_val.embedding_size > 0) {
      total_dims[i] = 0x01;
      for (int j = 0; j < hidden - cvm_offset; ++j) {
        dest_ptr[cvm_offset + j] = src_val.embedx[j] * scale;
      }
    } else {
      total_dims[i] = 0x00;
      for (int j = 0; j < hidden - cvm_offset; ++j) {
        dest_ptr[cvm_offset + j] = 0;
      }
    }
  }  // end kernel loop
}

template <typename FEATURE_VALUE_GPU_TYPE>
__global__ void PullCopyBase(float** dest, const FEATURE_VALUE_GPU_TYPE* src,
                             const int hidden, const int total_len,
                             uint64_t** keys, int* total_dims,
                             const int64_t* slot_lens, const int slot_num,
                             const int* key2slot, const int cvm_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int x = key2slot[i];
    int y = i - slot_lens[x];

    auto& src_val = src[i];
    //    *(dest[x] + y * hidden) = src_val.show;
    //    *(dest[x] + y * hidden + 1) = src_val.clk;
    //    *(dest[x] + y * hidden + 2) = src_val.embed_w;

    float* dest_ptr = dest[x] + y * hidden;
    const float* src_ptr = reinterpret_cast<const float*>(&src_val.show);
    for (int k = 0; k < cvm_offset; ++k) {
      dest_ptr[k] = src_ptr[k];
    }
    // && *(keys[x] + y) != 0
    total_dims[i] = static_cast<int>(src_val.embedding_size > 0);
  }  // end kernel loop
}
template <typename FEATURE_VALUE_GPU_TYPE>
__global__ void PullCopyExpand(float** dest, const FEATURE_VALUE_GPU_TYPE* src,
                               const int embedx_dim, const int total_len,
                               const int* total_dims, const int64_t* slot_lens,
                               const int slot_num, const int* key2slot,
                               const float scale, const int cvm_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int idx = i / embedx_dim;
    int col = i % embedx_dim;

    int x = key2slot[idx];
    int y = idx - slot_lens[x];

    auto& src_val = src[idx];
    int offset = y * (embedx_dim + cvm_offset) + cvm_offset + col;
    if (total_dims[idx] & 0x01) {
      *(dest[x] + offset) = src_val.embedx[col] * scale;
    } else {
      *(dest[x] + offset) = 0;
    }
  }  // end kernel loop
}

//================================== support nncross
//================================
template <typename FEATURE_VALUE_GPU_TYPE>
__global__ void PullCopyBaseNNCross(float** dest,
                                    const FEATURE_VALUE_GPU_TYPE* src,
                                    const int hidden, const int expand_dim,
                                    const int total_len, uint64_t** keys,
                                    int* total_dims, const int64_t* slot_lens,
                                    const int slot_num, const int* key2slot,
                                    const int cvm_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int x = key2slot[i];
    int y = i - slot_lens[x];

    auto& src_val = src[i];
    if (dest[x] != 0) {
      float* dest_ptr = dest[x] + y * hidden;
      const float* src_ptr = reinterpret_cast<const float*>(&src_val.show);
      for (int k = 0; k < cvm_offset; ++k) {
        dest_ptr[k] = src_ptr[k];
      }
    }
    // embedx flags + expand flags   && *(keys[x] + y) != 0  && *(keys[x] + y)
    // != 0
    total_dims[i] = static_cast<int>(src_val.embedding_size > 0) +
                    (static_cast<int>(src_val.embed_expand_size[0] > 0) << 1);
  }  // end kernel loop
}
template <typename FEATURE_VALUE_GPU_TYPE>
__global__ void PullCopyExpandNNCross(
    float** dest, const FEATURE_VALUE_GPU_TYPE* src, const int total_embedx_dim,
    const int embedx_dim, const int expand_dim, const int total_len,
    const int* total_dims, const int64_t* slot_lens, const int slot_num,
    const int* key2slot, const float scale, const int cvm_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int idx = i / total_embedx_dim;
    int col = i % total_embedx_dim;

    int x = key2slot[idx];
    int y = idx - slot_lens[x];

    auto& src_val = src[idx];
    if (col < embedx_dim) {  // embedx
      if (dest[x] == 0) {
        return;
      }
      int offset = y * (embedx_dim + cvm_offset) + cvm_offset + col;
      if (total_dims[idx] & 0x01) {
        *(dest[x] + offset) = src_val.embedx[col] * scale;
      } else {
        *(dest[x] + offset) = 0;
      }
    } else {  // expand
      if (dest[x + slot_num] == 0) {
        return;
      }
      int j = col - embedx_dim;
      if (total_dims[idx] & 0x02) {
        *(dest[x + slot_num] + y * expand_dim + j) =
            src_val.embed_expand[j] * scale;
      } else {
        *(dest[x + slot_num] + y * expand_dim + j) = 0;
      }
    }
  }  // end kernel loop
}

__global__ void FillKey2Slot(const int total_len, const int64_t* slot_lens,
                             const int slot_num, int* key2slots) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int low = 0;
    int high = slot_num - 1;
    while (low < high) {
      int mid = (low + high) / 2;
      if (i < slot_lens[mid + 1]) {
        high = mid;
      } else {
        low = mid + 1;
      }
    }
    key2slots[i] = low;
  }
}

__global__ void CopyKeysKernel(const int total_len, uint64_t** src_keys,
                               uint64_t* dest_total_keys,
                               const int64_t* slot_lens, const int* key2slot) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int x = key2slot[i];
    int y = i - slot_lens[x];
    dest_total_keys[i] = src_keys[x][y];
  }
}

template <typename FeaturePushValueGpuType>
__global__ void PushCopy(FeaturePushValueGpuType* dest, float** src,
                         const int hidden, const int expand_dim,
                         const int total_len, int bs, const int* slot_vector,
                         const int* total_dims, const int64_t* slot_lens,
                         const int slot_num, const int* key2slot,
                         const int cvm_offset) {
  assert(expand_dim == 0);
  // only process no expand
  CUDA_KERNEL_LOOP(i, total_len) {
    int x = key2slot[i];
    int y = i - slot_lens[x];

    auto& dest_val = dest[i];
    dest_val.slot = slot_vector[x];
    //    dest_val.show = *(src[x] + y * hidden);
    //    dest_val.clk = *(src[x] + y * hidden + 1);
    //    dest_val.embed_g = *(src[x] + y * hidden + 2) * -1. * bs;
    float* optr = reinterpret_cast<float*>(&dest_val.show);
    float* src_val = reinterpret_cast<float*>(src[x] + y * hidden);
    for (int k = 0; k < cvm_offset; ++k) {
      optr[k] = src_val[k];  // support variable length
    }
    dest_val.embed_g *= -1. * bs;

    if (total_dims[i] & 0x01) {
      for (int j = 0; j < hidden - cvm_offset; ++j) {
        dest_val.embedx_g[j] = src_val[cvm_offset + j] * -1. * bs;
      }
    } else {
      for (int j = 0; j < hidden - cvm_offset; ++j) {
        dest_val.embedx_g[j] = 0;
      }
    }
  }
}

template <typename FeaturePushValueGpuType>
__global__ void PushCopyBase(FeaturePushValueGpuType* dest, float** src,
                             const int hidden, const int total_len,
                             const int bs, const int* slot_vector,
                             const int* total_dims, const int64_t* slot_lens,
                             const int slot_num, const int* key2slot,
                             const int cvm_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int x = key2slot[i];
    int y = i - slot_lens[x];

    auto& dest_val = dest[i];
    dest_val.slot = slot_vector[x];
    //    dest_val.show = *(src[x] + y * hidden);
    //    dest_val.clk = *(src[x] + y * hidden + 1);
    //    dest_val.embed_g = *(src[x] + y * hidden + 2) * -1. * bs;
    float* optr = reinterpret_cast<float*>(&dest_val.show);
    float* src_val = reinterpret_cast<float*>(src[x] + y * hidden);
    for (int k = 0; k < cvm_offset; ++k) {
      optr[k] = src_val[k];  // support variable length
    }
    dest_val.embed_g *= -1. * bs;
  }
}
template <typename FeaturePushValueGpuType>
__global__ void PushCopyExpand(FeaturePushValueGpuType* dest, float** src,
                               const int embedx_dim, const int total_len,
                               const int bs, const int* slot_vector,
                               const int* total_dims, const int64_t* slot_lens,
                               const int slot_num, const int* key2slot,
                               const int cvm_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int idx = i / embedx_dim;
    int col = i % embedx_dim;

    int x = key2slot[idx];
    int y = idx - slot_lens[x];

    auto& dest_val = dest[idx];
    // embedx
    if ((total_dims[idx] & 0x01)) {
      dest_val.embedx_g[col] =
          *(src[x] + y * (embedx_dim + cvm_offset) + cvm_offset + col) * -1. *
          bs;
    } else {
      dest_val.embedx_g[col] = 0;
    }
  }
}

//============================== expand nncross ===============================
template <typename FeaturePushValueGpuType>
__global__ void PushCopyBaseNNCross(FeaturePushValueGpuType* dest, float** src,
                                    const int hidden, const int total_len,
                                    const int bs, const int* slot_vector,
                                    const int* total_dims,
                                    const int64_t* slot_lens,
                                    const int slot_num, const int* key2slot,
                                    const int cvm_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int x = key2slot[i];
    int y = i - slot_lens[x];

    auto& dest_val = dest[i];
    dest_val.slot = slot_vector[x];
    float* optr = reinterpret_cast<float*>(&dest_val.show);
    if (src[x] != 0) {
      float* src_val = reinterpret_cast<float*>(src[x] + y * hidden);
      for (int k = 0; k < cvm_offset; ++k) {
        optr[k] = src_val[k];  // support variable length
      }
      dest_val.embed_g *= -1. * bs;
    } else {
      for (int k = 1; k < cvm_offset; ++k) {
        optr[k] = 0;  // support variable length
      }
      dest_val.show = 1;
    }
  }
}
template <typename FeaturePushValueGpuType>
__global__ void PushCopyExpandNNCross(
    FeaturePushValueGpuType* dest, float** src, const int total_embedx_dim,
    const int embedx_dim, const int expand_dim, const int total_len,
    const int bs, const int* slot_vector, const int* total_dims,
    const int64_t* slot_lens, const int slot_num, const int* key2slot,
    const int cvm_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int idx = i / total_embedx_dim;
    int col = i % total_embedx_dim;

    int x = key2slot[idx];
    int y = idx - slot_lens[x];

    auto& dest_val = dest[idx];
    if (col < embedx_dim) {  // embedx
      if ((total_dims[idx] & 0x01) && src[x] != 0) {
        dest_val.embedx_g[col] =
            *(src[x] + y * (embedx_dim + cvm_offset) + cvm_offset + col) * -1. *
            bs;
      } else {
        dest_val.embedx_g[col] = 0;
      }
    } else {  // expand
      int j = col - embedx_dim;
      if ((total_dims[idx] & 0x02) && src[x + slot_num] != 0) {
        dest_val.embed_expand_g[j] =
            *(src[x + slot_num] + y * expand_dim + j) * -1. * bs;
      } else {
        dest_val.embed_expand_g[j] = 0;
      }
    }
  }
}

__device__ void add_calculator_value(const int table_size, const float pred,
                                     const int64_t label, const int idx,
                                     double* positive, double* negative,
                                     double* abs_error, double* sqr_error,
                                     double* local_pred) {
  int pos = static_cast<int>(pred * table_size);
  if (pos >= table_size) {
    pos = table_size - 1;
  }
  if (label == 0) {
    // atomicAdd(negative + pos, 1.0);
    paddle::platform::CudaAtomicAdd(negative + pos, 1.0);
  } else {
    // atomicAdd(positive + pos, 1.0);
    paddle::platform::CudaAtomicAdd(positive + pos, 1.0);
  }
  double err = pred - label;
  abs_error[idx] += fabs(err);
  sqr_error[idx] += err * err;
  local_pred[idx] += pred;
}

__global__ void AddBasicCalculator(const float* pred, const int64_t* label,
                                   double* positive, double* negative,
                                   double* abs_error, double* sqr_error,
                                   double* local_pred, int len,
                                   int table_size) {
  CUDA_KERNEL_LOOP(ins_idx, len) {
    add_calculator_value(table_size, pred[ins_idx], label[ins_idx], ins_idx,
                         positive, negative, abs_error, sqr_error, local_pred);
  }
}

__global__ void AddMaskCalculator(const float* pred, const int64_t* label,
                                  const int64_t* mask, double* positive,
                                  double* negative, double* abs_error,
                                  double* sqr_error, double* local_pred,
                                  int len, int table_size) {
  CUDA_KERNEL_LOOP(ins_idx, len) {
    if (mask[ins_idx] != 1) {
      continue;
    }
    add_calculator_value(table_size, pred[ins_idx], label[ins_idx], ins_idx,
                         positive, negative, abs_error, sqr_error, local_pred);
  }
}

void BoxWrapper::CopyForPull(const paddle::platform::Place& place,
                             uint64_t** gpu_keys, float** gpu_values,
                             void* total_values_gpu, const int64_t* slot_lens,
                             const int slot_num, const int* key2slot,
                             const int hidden_size, const int expand_embed_dim,
                             const int64_t total_length, int* total_dims) {
  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(
                        BOOST_GET_CONST(platform::CUDAPlace, place)))
                    ->stream();
#define EMBEDX_CASE(i, ...)                                                  \
  case i: {                                                                  \
    constexpr size_t EmbedxDim = i;                                          \
    switch (expand_embed_dim) {                                              \
      __VA_ARGS__                                                            \
      default:                                                               \
        PADDLE_THROW(platform::errors::InvalidArgument(                      \
            "Unsupport this expand embedding size [%d]", expand_embed_dim)); \
    }                                                                        \
  } break

#define EXPAND_EMBED_PULL_CASE(i, ...)                                         \
  case i: {                                                                    \
    constexpr size_t ExpandDim = i;                                            \
    if (feature_type_ == static_cast<int>(boxps::FEATURE_PCOC)) {              \
      PullCopy<boxps::FeaturePullValueGpuPCOC<EmbedxDim, ExpandDim>><<<        \
          (total_length + 512 - 1) / 512, 512, 0, stream>>>(                   \
          gpu_values,                                                          \
          reinterpret_cast<                                                    \
              boxps::FeaturePullValueGpuPCOC<EmbedxDim, ExpandDim>*>(          \
              total_values_gpu),                                               \
          hidden_size, expand_embed_dim, total_length, gpu_keys, total_dims,   \
          slot_lens, slot_num, key2slot, pull_embedx_scale_, cvm_offset_);     \
    } else if (feature_type_ == static_cast<int>(boxps::FEATURE_QUANT)) {      \
      PullCopy<boxps::FeaturePullValueGpuQuant<EmbedxDim, ExpandDim>><<<       \
          (total_length + 512 - 1) / 512, 512, 0, stream>>>(                   \
          gpu_values,                                                          \
          reinterpret_cast<                                                    \
              boxps::FeaturePullValueGpuQuant<EmbedxDim, ExpandDim>*>(         \
              total_values_gpu),                                               \
          hidden_size, expand_embed_dim, total_length, gpu_keys, total_dims,   \
          slot_lens, slot_num, key2slot, pull_embedx_scale_, cvm_offset_);     \
    } else {                                                                   \
      PullCopy<boxps::FeaturePullValueGpu<EmbedxDim, ExpandDim>><<<            \
          (total_length + 512 - 1) / 512, 512, 0, stream>>>(                   \
          gpu_values,                                                          \
          reinterpret_cast<boxps::FeaturePullValueGpu<EmbedxDim, ExpandDim>*>( \
              total_values_gpu),                                               \
          hidden_size, expand_embed_dim, total_length, gpu_keys, total_dims,   \
          slot_lens, slot_num, key2slot, 1.0, cvm_offset_);                    \
    }                                                                          \
  } break

#define EXPAND_EMBED_PULL_CASE2(i, ...)                                        \
  case i: {                                                                    \
    constexpr size_t ExpandDim = i;                                            \
    if (feature_type_ == static_cast<int>(boxps::FEATURE_PCOC)) {              \
      PullCopyBase<boxps::FeaturePullValueGpuPCOC<EmbedxDim, ExpandDim>><<<    \
          (total_length + 512 - 1) / 512, 512, 0, stream>>>(                   \
          gpu_values,                                                          \
          reinterpret_cast<                                                    \
              boxps::FeaturePullValueGpuPCOC<EmbedxDim, ExpandDim>*>(          \
              total_values_gpu),                                               \
          hidden_size, total_length, gpu_keys, total_dims, slot_lens,          \
          slot_num, key2slot, cvm_offset_);                                    \
      int embedx_total_length = total_length * EmbedxDim;                      \
      PullCopyExpand<boxps::FeaturePullValueGpuPCOC<EmbedxDim, ExpandDim>><<<  \
          (embedx_total_length + 512 - 1) / 512, 512, 0, stream>>>(            \
          gpu_values,                                                          \
          reinterpret_cast<                                                    \
              boxps::FeaturePullValueGpuPCOC<EmbedxDim, ExpandDim>*>(          \
              total_values_gpu),                                               \
          EmbedxDim, embedx_total_length, total_dims, slot_lens, slot_num,     \
          key2slot, pull_embedx_scale_, cvm_offset_);                          \
    } else if (feature_type_ == static_cast<int>(boxps::FEATURE_QUANT)) {      \
      PullCopyBase<boxps::FeaturePullValueGpuQuant<EmbedxDim, ExpandDim>><<<   \
          (total_length + 512 - 1) / 512, 512, 0, stream>>>(                   \
          gpu_values,                                                          \
          reinterpret_cast<                                                    \
              boxps::FeaturePullValueGpuQuant<EmbedxDim, ExpandDim>*>(         \
              total_values_gpu),                                               \
          hidden_size, total_length, gpu_keys, total_dims, slot_lens,          \
          slot_num, key2slot, cvm_offset_);                                    \
      int embedx_total_length = total_length * EmbedxDim;                      \
      PullCopyExpand<boxps::FeaturePullValueGpuQuant<EmbedxDim, ExpandDim>><<< \
          (embedx_total_length + 512 - 1) / 512, 512, 0, stream>>>(            \
          gpu_values,                                                          \
          reinterpret_cast<                                                    \
              boxps::FeaturePullValueGpuQuant<EmbedxDim, ExpandDim>*>(         \
              total_values_gpu),                                               \
          EmbedxDim, embedx_total_length, total_dims, slot_lens, slot_num,     \
          key2slot, pull_embedx_scale_, cvm_offset_);                          \
    } else {                                                                   \
      PullCopyBase<boxps::FeaturePullValueGpu<EmbedxDim, ExpandDim>><<<        \
          (total_length + 512 - 1) / 512, 512, 0, stream>>>(                   \
          gpu_values,                                                          \
          reinterpret_cast<boxps::FeaturePullValueGpu<EmbedxDim, ExpandDim>*>( \
              total_values_gpu),                                               \
          hidden_size, total_length, gpu_keys, total_dims, slot_lens,          \
          slot_num, key2slot, cvm_offset_);                                    \
      int embedx_total_length = total_length * EmbedxDim;                      \
      PullCopyExpand<boxps::FeaturePullValueGpu<EmbedxDim, ExpandDim>><<<      \
          (embedx_total_length + 512 - 1) / 512, 512, 0, stream>>>(            \
          gpu_values,                                                          \
          reinterpret_cast<boxps::FeaturePullValueGpu<EmbedxDim, ExpandDim>*>( \
              total_values_gpu),                                               \
          EmbedxDim, embedx_total_length, total_dims, slot_lens, slot_num,     \
          key2slot, 1.0, cvm_offset_);                                         \
    }                                                                          \
  } break

#define EXPAND_EMBED_PULL_NNCROSS(i, ...)                                      \
  case i: {                                                                    \
    constexpr size_t ExpandDim = i;                                            \
    if (feature_type_ == static_cast<int>(boxps::FEATURE_PCOC)) {              \
      PullCopyBaseNNCross<boxps::FeaturePullValueGpuPCOC<                      \
          EmbedxDim,                                                           \
          ExpandDim>><<<(total_length + 512 - 1) / 512, 512, 0, stream>>>(     \
          gpu_values,                                                          \
          reinterpret_cast<                                                    \
              boxps::FeaturePullValueGpuPCOC<EmbedxDim, ExpandDim>*>(          \
              total_values_gpu),                                               \
          hidden_size, expand_embed_dim, total_length, gpu_keys, total_dims,   \
          slot_lens, slot_num, key2slot, cvm_offset_);                         \
      int embedx_total_length = total_length * (EmbedxDim + ExpandDim);        \
      PullCopyExpandNNCross<                                                   \
          boxps::FeaturePullValueGpuPCOC<EmbedxDim, ExpandDim>><<<             \
          (embedx_total_length + 512 - 1) / 512, 512, 0, stream>>>(            \
          gpu_values,                                                          \
          reinterpret_cast<                                                    \
              boxps::FeaturePullValueGpuPCOC<EmbedxDim, ExpandDim>*>(          \
              total_values_gpu),                                               \
          (EmbedxDim + ExpandDim), EmbedxDim, ExpandDim, embedx_total_length,  \
          total_dims, slot_lens, slot_num, key2slot, pull_embedx_scale_,       \
          cvm_offset_);                                                        \
    } else if (feature_type_ == static_cast<int>(boxps::FEATURE_QUANT)) {      \
      PullCopyBaseNNCross<boxps::FeaturePullValueGpuQuant<                     \
          EmbedxDim,                                                           \
          ExpandDim>><<<(total_length + 512 - 1) / 512, 512, 0, stream>>>(     \
          gpu_values,                                                          \
          reinterpret_cast<                                                    \
              boxps::FeaturePullValueGpuQuant<EmbedxDim, ExpandDim>*>(         \
              total_values_gpu),                                               \
          hidden_size, expand_embed_dim, total_length, gpu_keys, total_dims,   \
          slot_lens, slot_num, key2slot, cvm_offset_);                         \
      int embedx_total_length = total_length * (EmbedxDim + ExpandDim);        \
      PullCopyExpandNNCross<                                                   \
          boxps::FeaturePullValueGpuQuant<EmbedxDim, ExpandDim>><<<            \
          (embedx_total_length + 512 - 1) / 512, 512, 0, stream>>>(            \
          gpu_values,                                                          \
          reinterpret_cast<                                                    \
              boxps::FeaturePullValueGpuQuant<EmbedxDim, ExpandDim>*>(         \
              total_values_gpu),                                               \
          (EmbedxDim + ExpandDim), EmbedxDim, ExpandDim, embedx_total_length,  \
          total_dims, slot_lens, slot_num, key2slot, pull_embedx_scale_,       \
          cvm_offset_);                                                        \
    } else {                                                                   \
      PullCopyBaseNNCross<boxps::FeaturePullValueGpu<EmbedxDim, ExpandDim>><<< \
          (total_length + 512 - 1) / 512, 512, 0, stream>>>(                   \
          gpu_values,                                                          \
          reinterpret_cast<boxps::FeaturePullValueGpu<EmbedxDim, ExpandDim>*>( \
              total_values_gpu),                                               \
          hidden_size, expand_embed_dim, total_length, gpu_keys, total_dims,   \
          slot_lens, slot_num, key2slot, cvm_offset_);                         \
      int embedx_total_length = total_length * (EmbedxDim + ExpandDim);        \
      PullCopyExpandNNCross<                                                   \
          boxps::FeaturePullValueGpu<EmbedxDim, ExpandDim>><<<                 \
          (embedx_total_length + 512 - 1) / 512, 512, 0, stream>>>(            \
          gpu_values,                                                          \
          reinterpret_cast<boxps::FeaturePullValueGpu<EmbedxDim, ExpandDim>*>( \
              total_values_gpu),                                               \
          (EmbedxDim + ExpandDim), EmbedxDim, ExpandDim, embedx_total_length,  \
          total_dims, slot_lens, slot_num, key2slot, 1.0, cvm_offset_);        \
    }                                                                          \
  } break

  switch (hidden_size - cvm_offset_) {
    EMBEDX_CASE(8, EXPAND_EMBED_PULL_CASE(0); EXPAND_EMBED_PULL_NNCROSS(8);
                EXPAND_EMBED_PULL_NNCROSS(64););
    EMBEDX_CASE(16, EXPAND_EMBED_PULL_CASE2(0); EXPAND_EMBED_PULL_NNCROSS(64););
    EMBEDX_CASE(32, EXPAND_EMBED_PULL_CASE2(0););
    EMBEDX_CASE(64, EXPAND_EMBED_PULL_CASE2(0););
    EMBEDX_CASE(256, EXPAND_EMBED_PULL_CASE2(0););
    EMBEDX_CASE(128, EXPAND_EMBED_PULL_CASE2(0););
    EMBEDX_CASE(280, EXPAND_EMBED_PULL_CASE2(0););
    default:
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unsupport this embedding size [%d]", hidden_size - cvm_offset_));
  }
  cudaStreamSynchronize(stream);
#undef EXPAND_EMBED_PULL_CASE
#undef EMBEDX_CASE
}

void BoxWrapper::CopyKeys(const paddle::platform::Place& place,
                          uint64_t** origin_keys, uint64_t* total_keys,
                          const int64_t* slot_lens, int slot_num, int total_len,
                          int* key2slot) {
  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(
                        BOOST_GET_CONST(platform::CUDAPlace, place)))
                    ->stream();
  FillKey2Slot<<<(total_len + 512 - 1) / 512, 512, 0, stream>>>(
      total_len, slot_lens, slot_num, key2slot);
  CopyKeysKernel<<<(total_len + 512 - 1) / 512, 512, 0, stream>>>(
      total_len, origin_keys, total_keys, slot_lens, key2slot);
  cudaStreamSynchronize(stream);
}

void BoxWrapper::CopyForPush(const paddle::platform::Place& place,
                             float** grad_values, void* total_grad_values_gpu,
                             const int* d_slot_vector, const int64_t* slot_lens,
                             const int slot_num, const int hidden_size,
                             const int expand_embed_dim,
                             const int64_t total_length, const int batch_size,
                             const int* total_dims, const int* key2slot) {
  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(
                        BOOST_GET_CONST(platform::CUDAPlace, place)))
                    ->stream();
#define EMBEDX_CASE(i, ...)                                                  \
  case i: {                                                                  \
    constexpr size_t EmbedxDim = i;                                          \
    switch (expand_embed_dim) {                                              \
      __VA_ARGS__                                                            \
      default:                                                               \
        PADDLE_THROW(platform::errors::InvalidArgument(                      \
            "Unsupport this expand embedding size [%d]", expand_embed_dim)); \
    }                                                                        \
  } break
#define EXPAND_EMBED_PUSH_CASE(i, ...)                                         \
  case i: {                                                                    \
    constexpr size_t ExpandDim = i;                                            \
    if (feature_type_ == static_cast<int>(boxps::FEATURE_PCOC)) {              \
      PushCopy<boxps::FeaturePushValueGpuPCOC<EmbedxDim, ExpandDim>><<<        \
          (total_length + 512 - 1) / 512, 512, 0, stream>>>(                   \
          reinterpret_cast<                                                    \
              boxps::FeaturePushValueGpuPCOC<EmbedxDim, ExpandDim>*>(          \
              total_grad_values_gpu),                                          \
          grad_values, hidden_size, expand_embed_dim, total_length,            \
          batch_size, d_slot_vector, total_dims, slot_lens, slot_num,          \
          key2slot, cvm_offset_);                                              \
    } else {                                                                   \
      PushCopy<boxps::FeaturePushValueGpu<EmbedxDim, ExpandDim>><<<            \
          (total_length + 512 - 1) / 512, 512, 0, stream>>>(                   \
          reinterpret_cast<boxps::FeaturePushValueGpu<EmbedxDim, ExpandDim>*>( \
              total_grad_values_gpu),                                          \
          grad_values, hidden_size, expand_embed_dim, total_length,            \
          batch_size, d_slot_vector, total_dims, slot_lens, slot_num,          \
          key2slot, cvm_offset_);                                              \
    }                                                                          \
  } break

#define EXPAND_EMBED_PUSH_CASE2(i, ...)                                        \
  case i: {                                                                    \
    constexpr size_t ExpandDim = i;                                            \
    if (feature_type_ == static_cast<int>(boxps::FEATURE_PCOC)) {              \
      PushCopyBase<boxps::FeaturePushValueGpuPCOC<EmbedxDim, ExpandDim>><<<    \
          (total_length + 512 - 1) / 512, 512, 0, stream>>>(                   \
          reinterpret_cast<                                                    \
              boxps::FeaturePushValueGpuPCOC<EmbedxDim, ExpandDim>*>(          \
              total_grad_values_gpu),                                          \
          grad_values, hidden_size, total_length, batch_size, d_slot_vector,   \
          total_dims, slot_lens, slot_num, key2slot, cvm_offset_);             \
      int embedx_total_length = total_length * EmbedxDim;                      \
      PushCopyExpand<boxps::FeaturePushValueGpuPCOC<EmbedxDim, ExpandDim>><<<  \
          (embedx_total_length + 512 - 1) / 512, 512, 0, stream>>>(            \
          reinterpret_cast<                                                    \
              boxps::FeaturePushValueGpuPCOC<EmbedxDim, ExpandDim>*>(          \
              total_grad_values_gpu),                                          \
          grad_values, EmbedxDim, embedx_total_length, batch_size,             \
          d_slot_vector, total_dims, slot_lens, slot_num, key2slot,            \
          cvm_offset_);                                                        \
    } else {                                                                   \
      PushCopyBase<boxps::FeaturePushValueGpu<EmbedxDim, ExpandDim>><<<        \
          (total_length + 512 - 1) / 512, 512, 0, stream>>>(                   \
          reinterpret_cast<boxps::FeaturePushValueGpu<EmbedxDim, ExpandDim>*>( \
              total_grad_values_gpu),                                          \
          grad_values, hidden_size, total_length, batch_size, d_slot_vector,   \
          total_dims, slot_lens, slot_num, key2slot, cvm_offset_);             \
      int embedx_total_length = total_length * EmbedxDim;                      \
      PushCopyExpand<boxps::FeaturePushValueGpu<EmbedxDim, ExpandDim>><<<      \
          (embedx_total_length + 512 - 1) / 512, 512, 0, stream>>>(            \
          reinterpret_cast<boxps::FeaturePushValueGpu<EmbedxDim, ExpandDim>*>( \
              total_grad_values_gpu),                                          \
          grad_values, EmbedxDim, embedx_total_length, batch_size,             \
          d_slot_vector, total_dims, slot_lens, slot_num, key2slot,            \
          cvm_offset_);                                                        \
    }                                                                          \
  } break

#define EXPAND_EMBED_PUSH_NNCROSS(i, ...)                                      \
  case i: {                                                                    \
    constexpr size_t ExpandDim = i;                                            \
    if (feature_type_ == static_cast<int>(boxps::FEATURE_PCOC)) {              \
      PushCopyBaseNNCross<boxps::FeaturePushValueGpuPCOC<                      \
          EmbedxDim,                                                           \
          ExpandDim>><<<(total_length + 512 - 1) / 512, 512, 0, stream>>>(     \
          reinterpret_cast<                                                    \
              boxps::FeaturePushValueGpuPCOC<EmbedxDim, ExpandDim>*>(          \
              total_grad_values_gpu),                                          \
          grad_values, hidden_size, total_length, batch_size, d_slot_vector,   \
          total_dims, slot_lens, slot_num, key2slot, cvm_offset_);             \
      int embedx_total_length = total_length * (EmbedxDim + ExpandDim);        \
      PushCopyExpandNNCross<                                                   \
          boxps::FeaturePushValueGpuPCOC<EmbedxDim, ExpandDim>><<<             \
          (embedx_total_length + 512 - 1) / 512, 512, 0, stream>>>(            \
          reinterpret_cast<                                                    \
              boxps::FeaturePushValueGpuPCOC<EmbedxDim, ExpandDim>*>(          \
              total_grad_values_gpu),                                          \
          grad_values, (EmbedxDim + ExpandDim), EmbedxDim, ExpandDim,          \
          embedx_total_length, batch_size, d_slot_vector, total_dims,          \
          slot_lens, slot_num, key2slot, cvm_offset_);                         \
    } else {                                                                   \
      PushCopyBaseNNCross<boxps::FeaturePushValueGpu<EmbedxDim, ExpandDim>><<< \
          (total_length + 512 - 1) / 512, 512, 0, stream>>>(                   \
          reinterpret_cast<boxps::FeaturePushValueGpu<EmbedxDim, ExpandDim>*>( \
              total_grad_values_gpu),                                          \
          grad_values, hidden_size, total_length, batch_size, d_slot_vector,   \
          total_dims, slot_lens, slot_num, key2slot, cvm_offset_);             \
      int embedx_total_length = total_length * (EmbedxDim + ExpandDim);        \
      PushCopyExpandNNCross<                                                   \
          boxps::FeaturePushValueGpu<EmbedxDim, ExpandDim>><<<                 \
          (embedx_total_length + 512 - 1) / 512, 512, 0, stream>>>(            \
          reinterpret_cast<boxps::FeaturePushValueGpu<EmbedxDim, ExpandDim>*>( \
              total_grad_values_gpu),                                          \
          grad_values, (EmbedxDim + ExpandDim), EmbedxDim, ExpandDim,          \
          embedx_total_length, batch_size, d_slot_vector, total_dims,          \
          slot_lens, slot_num, key2slot, cvm_offset_);                         \
    }                                                                          \
  } break
  switch (hidden_size - cvm_offset_) {
    EMBEDX_CASE(8, EXPAND_EMBED_PUSH_CASE(0); EXPAND_EMBED_PUSH_NNCROSS(8);
                EXPAND_EMBED_PUSH_NNCROSS(64););
    EMBEDX_CASE(16, EXPAND_EMBED_PUSH_CASE2(0); EXPAND_EMBED_PUSH_NNCROSS(64););
    EMBEDX_CASE(32, EXPAND_EMBED_PUSH_CASE2(0););
    EMBEDX_CASE(64, EXPAND_EMBED_PUSH_CASE2(0););
    EMBEDX_CASE(256, EXPAND_EMBED_PUSH_CASE2(0););
    EMBEDX_CASE(128, EXPAND_EMBED_PUSH_CASE2(0););
    EMBEDX_CASE(280, EXPAND_EMBED_PUSH_CASE2(0););
    default:
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unsupport this embedding size [%d]", hidden_size - cvm_offset_));
  }

  cudaStreamSynchronize(stream);
#undef EXPAND_EMBED_PUSH_CASE
#undef EMBEDX_CASE
}

void BasicAucCalculator::cuda_add_data(const paddle::platform::Place& place,
                                       const int64_t* label, const float* pred,
                                       int len) {
  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(
                        BOOST_GET_CONST(platform::CUDAPlace, place)))
                    ->stream();

  int i = BOOST_GET_CONST(platform::CUDAPlace, place).GetDeviceId();

  cudaSetDevice(i);

  AddBasicCalculator<<<(len + 512 - 1) / 512, 512, 0, stream>>>(
      pred, label, reinterpret_cast<double*>(_d_positive[i]->ptr()),
      reinterpret_cast<double*>(_d_negative[i]->ptr()),
      reinterpret_cast<double*>(_d_abserr[i]->ptr()),
      reinterpret_cast<double*>(_d_sqrerr[i]->ptr()),
      reinterpret_cast<double*>(_d_pred[i]->ptr()), len, _table_size);
}

void BasicAucCalculator::cuda_add_mask_data(
    const paddle::platform::Place& place, const int64_t* label,
    const float* pred, const int64_t* mask, int len) {
  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(
                        BOOST_GET_CONST(platform::CUDAPlace, place)))
                    ->stream();
  int i = BOOST_GET_CONST(platform::CUDAPlace, place).GetDeviceId();

  cudaSetDevice(i);

  AddMaskCalculator<<<(len + 512 - 1) / 512, 512, 0, stream>>>(
      pred, label, mask, reinterpret_cast<double*>(_d_positive[i]->ptr()),
      reinterpret_cast<double*>(_d_negative[i]->ptr()),
      reinterpret_cast<double*>(_d_abserr[i]->ptr()),
      reinterpret_cast<double*>(_d_sqrerr[i]->ptr()),
      reinterpret_cast<double*>(_d_pred[i]->ptr()), len, _table_size);
}

__global__ void pull_cache_value_kernel(int len, int dim, uint64_t* key,
                                        float* val, float* table) {
  CUDA_KERNEL_LOOP(i, len) { val[i] = table[key[i / dim] * dim + i % dim]; }
}

void GpuReplicaCache::PullCacheValue(uint64_t* d_keys, float* d_vals, int num,
                                     int gpu_id) {
  auto place = platform::CUDAPlace(gpu_id);
  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(place))
                    ->stream();
  int len = emb_dim_ * num;
  const int BLOCK_SIZE_ = 256;
  pull_cache_value_kernel<<<(len + BLOCK_SIZE_ - 1) / BLOCK_SIZE_, BLOCK_SIZE_,
                            0, stream>>>(len, emb_dim_, d_keys, d_vals,
                                         d_embs_[gpu_id]);
}

}  // end namespace framework
}  // end namespace paddle
#endif
