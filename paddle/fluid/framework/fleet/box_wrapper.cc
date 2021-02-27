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
#include "paddle/fluid/framework/fleet/box_wrapper.h"

#include <algorithm>
#include <ctime>
#include <memory>
#include <numeric>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/gpu_info.h"

DECLARE_bool(use_gpu_replica_cache);
DECLARE_int32(gpu_replica_cache_dim);
namespace paddle {
namespace framework {

std::shared_ptr<BoxWrapper> BoxWrapper::s_instance_ = nullptr;
std::shared_ptr<boxps::PaddleShuffler> BoxWrapper::data_shuffle_ = nullptr;
cudaStream_t BoxWrapper::stream_list_[8];
int BoxWrapper::embedx_dim_ = 8;
int BoxWrapper::expand_embed_dim_ = 0;
int BoxWrapper::feature_type_ = 0;
float BoxWrapper::pull_embedx_scale_ = 1.0;

void BasicAucCalculator::add_unlock_data(double pred, int label) {
  PADDLE_ENFORCE_GE(pred, 0.0, platform::errors::PreconditionNotMet(
                                   "pred should be greater than 0"));
  PADDLE_ENFORCE_LE(pred, 1.0, platform::errors::PreconditionNotMet(
                                   "pred should be lower than 1"));
  PADDLE_ENFORCE_EQ(
      label * label, label,
      platform::errors::PreconditionNotMet(
          "label must be equal to 0 or 1, but its value is: %d", label));
  int pos = std::min(static_cast<int>(pred * _table_size), _table_size - 1);
  PADDLE_ENFORCE_GE(
      pos, 0,
      platform::errors::PreconditionNotMet(
          "pos must be equal or greater than 0, but its value is: %d", pos));
  PADDLE_ENFORCE_LT(
      pos, _table_size,
      platform::errors::PreconditionNotMet(
          "pos must be less than table_size, but its value is: %d", pos));
  _local_abserr += fabs(pred - label);
  _local_sqrerr += (pred - label) * (pred - label);
  _local_pred += pred;
  ++_table[label][pos];
}

void BasicAucCalculator::add_data(const float* d_pred, const int64_t* d_label,
                                  int batch_size,
                                  const paddle::platform::Place& place) {
  if (_mode_collect_in_gpu) {
    cuda_add_data(place, d_label, d_pred, batch_size);
  } else {
    thread_local std::vector<float> h_pred;
    thread_local std::vector<int64_t> h_label;
    h_pred.resize(batch_size);
    h_label.resize(batch_size);
    cudaMemcpy(h_pred.data(), d_pred, sizeof(float) * batch_size,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_label.data(), d_label, sizeof(int64_t) * batch_size,
               cudaMemcpyDeviceToHost);

    std::lock_guard<std::mutex> lock(_table_mutex);
    for (int i = 0; i < batch_size; ++i) {
      add_unlock_data(h_pred[i], h_label[i]);
    }
  }
}
// add mask data
void BasicAucCalculator::add_mask_data(const float* d_pred,
                                       const int64_t* d_label,
                                       const int64_t* d_mask, int batch_size,
                                       const paddle::platform::Place& place) {
  if (_mode_collect_in_gpu) {
    cuda_add_mask_data(place, d_label, d_pred, d_mask, batch_size);
  } else {
    thread_local std::vector<float> h_pred;
    thread_local std::vector<int64_t> h_label;
    thread_local std::vector<int64_t> h_mask;
    h_pred.resize(batch_size);
    h_label.resize(batch_size);
    h_mask.resize(batch_size);

    cudaMemcpy(h_pred.data(), d_pred, sizeof(float) * batch_size,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_label.data(), d_label, sizeof(int64_t) * batch_size,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mask.data(), d_mask, sizeof(int64_t) * batch_size,
               cudaMemcpyDeviceToHost);

    std::lock_guard<std::mutex> lock(_table_mutex);
    for (int i = 0; i < batch_size; ++i) {
      if (h_mask[i]) {
        add_unlock_data(h_pred[i], h_label[i]);
      }
    }
  }
}

void BasicAucCalculator::init(int table_size, int max_batch_size) {
  if (_mode_collect_in_gpu) {
    PADDLE_ENFORCE_GE(
        max_batch_size, 0,
        platform::errors::PreconditionNotMet(
            "max_batch_size should be greater than 0 in mode_collect_in_gpu"));
  }
  set_table_size(table_size);
  set_max_batch_size(max_batch_size);
  // init CPU memory
  for (int i = 0; i < 2; i++) {
    _table[i] = std::vector<double>();
  }
  // init GPU memory
  if (_mode_collect_in_gpu) {
    for (int i = 0; i < platform::GetCUDADeviceCount(); ++i) {
      auto place = platform::CUDAPlace(i);
      _d_positive.emplace_back(
          memory::AllocShared(place, _table_size * sizeof(double)));
      _d_negative.emplace_back(
          memory::AllocShared(place, _table_size * sizeof(double)));
      _d_abserr.emplace_back(
          memory::AllocShared(place, _max_batch_size * sizeof(double)));
      _d_sqrerr.emplace_back(
          memory::AllocShared(place, _max_batch_size * sizeof(double)));
      _d_pred.emplace_back(
          memory::AllocShared(place, _max_batch_size * sizeof(double)));
    }
  }
  // reset
  reset();
}

void BasicAucCalculator::reset() {
  // reset CPU counter
  for (int i = 0; i < 2; i++) {
    _table[i].assign(_table_size, 0.0);
  }
  _local_abserr = 0;
  _local_sqrerr = 0;
  _local_pred = 0;
  // reset GPU counter
  if (_mode_collect_in_gpu) {
    // backup orginal device
    int ori_device;
    cudaGetDevice(&ori_device);
    for (int i = 0; i < platform::GetCUDADeviceCount(); ++i) {
      cudaSetDevice(i);
      auto place = platform::CUDAPlace(i);
      platform::CUDADeviceContext* context =
          dynamic_cast<platform::CUDADeviceContext*>(
              platform::DeviceContextPool::Instance().Get(place));
      auto stream = context->stream();
      cudaMemsetAsync(_d_positive[i]->ptr(), 0, sizeof(double) * _table_size,
                      stream);
      cudaMemsetAsync(_d_negative[i]->ptr(), 0, sizeof(double) * _table_size,
                      stream);
      cudaMemsetAsync(_d_abserr[i]->ptr(), 0, sizeof(double) * _max_batch_size,
                      stream);
      cudaMemsetAsync(_d_sqrerr[i]->ptr(), 0, sizeof(double) * _max_batch_size,
                      stream);
      cudaMemsetAsync(_d_pred[i]->ptr(), 0, sizeof(double) * _max_batch_size,
                      stream);
    }
    // restore device
    cudaSetDevice(ori_device);
  }
}

void BasicAucCalculator::collect_data_nccl() {
  // backup orginal device
  int ori_device;
  cudaGetDevice(&ori_device);
  // transfer to CPU
  platform::dynload::ncclGroupStart();
  // nccl allreduce sum
  for (int i = 0; i < platform::GetCUDADeviceCount(); ++i) {
    cudaSetDevice(i);
    auto place = platform::CUDAPlace(i);
    auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                      platform::DeviceContextPool::Instance().Get(place))
                      ->stream();
    auto comm = platform::NCCLCommContext::Instance().Get(0, place);
    platform::dynload::ncclAllReduce(
        _d_positive[i]->ptr(), _d_positive[i]->ptr(), _table_size, ncclFloat64,
        ncclSum, comm->comm(), stream);
    platform::dynload::ncclAllReduce(
        _d_negative[i]->ptr(), _d_negative[i]->ptr(), _table_size, ncclFloat64,
        ncclSum, comm->comm(), stream);
    platform::dynload::ncclAllReduce(_d_abserr[i]->ptr(), _d_abserr[i]->ptr(),
                                     _max_batch_size, ncclFloat64, ncclSum,
                                     comm->comm(), stream);
    platform::dynload::ncclAllReduce(_d_sqrerr[i]->ptr(), _d_sqrerr[i]->ptr(),
                                     _max_batch_size, ncclFloat64, ncclSum,
                                     comm->comm(), stream);
    platform::dynload::ncclAllReduce(_d_pred[i]->ptr(), _d_pred[i]->ptr(),
                                     _max_batch_size, ncclFloat64, ncclSum,
                                     comm->comm(), stream);
  }
  platform::dynload::ncclGroupEnd();
  // sync
  for (int i = 0; i < platform::GetCUDADeviceCount(); ++i) {
    cudaSetDevice(i);
    auto place = platform::CUDAPlace(i);
    auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                      platform::DeviceContextPool::Instance().Get(place))
                      ->stream();
    cudaStreamSynchronize(stream);
  }
  // restore device
  cudaSetDevice(ori_device);
}

void BasicAucCalculator::copy_data_d2h(int device) {
  // backup orginal device
  int ori_device;
  cudaGetDevice(&ori_device);
  cudaSetDevice(device);
  auto place = platform::CUDAPlace(device);
  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(place))
                    ->stream();

  std::vector<double> h_abserr;
  std::vector<double> h_sqrerr;
  std::vector<double> h_pred;
  h_abserr.resize(_max_batch_size);
  h_sqrerr.resize(_max_batch_size);
  h_pred.resize(_max_batch_size);
  cudaMemcpyAsync(&_table[0][0], _d_negative[device]->ptr(),
                  _table_size * sizeof(double), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(&_table[1][0], _d_positive[device]->ptr(),
                  _table_size * sizeof(double), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(h_abserr.data(), _d_abserr[device]->ptr(),
                  _max_batch_size * sizeof(double), cudaMemcpyDeviceToHost,
                  stream);
  cudaMemcpyAsync(h_sqrerr.data(), _d_sqrerr[device]->ptr(),
                  _max_batch_size * sizeof(double), cudaMemcpyDeviceToHost,
                  stream);
  cudaMemcpyAsync(h_pred.data(), _d_pred[device]->ptr(),
                  _max_batch_size * sizeof(double), cudaMemcpyDeviceToHost,
                  stream);
  cudaStreamSynchronize(stream);

  cudaSetDevice(ori_device);

  _local_abserr = 0;
  _local_sqrerr = 0;
  _local_pred = 0;

  // sum in CPU
  for (int i = 0; i < _max_batch_size; ++i) {
    _local_abserr += h_abserr[i];
    _local_sqrerr += h_sqrerr[i];
    _local_pred += h_pred[i];
  }
  // restore device
  cudaSetDevice(ori_device);
}

void BasicAucCalculator::compute() {
  if (_mode_collect_in_gpu) {
    // collect data
    collect_data_nccl();
    // copy from GPU0
    copy_data_d2h(0);
  }

  double* table[2] = {&_table[0][0], &_table[1][0]};
  if (boxps::MPICluster::Ins().size() > 1) {
    boxps::MPICluster::Ins().allreduce_sum(table[0], _table_size);
    boxps::MPICluster::Ins().allreduce_sum(table[1], _table_size);
  }

  double area = 0;
  double fp = 0;
  double tp = 0;

  for (int i = _table_size - 1; i >= 0; i--) {
    double newfp = fp + table[0][i];
    double newtp = tp + table[1][i];
    area += (newfp - fp) * (tp + newtp) / 2;
    fp = newfp;
    tp = newtp;
  }

  if (fp < 1e-3 || tp < 1e-3) {
    _auc = -0.5;  // which means all nonclick or click
  } else {
    _auc = area / (fp * tp);
  }

  if (boxps::MPICluster::Ins().size() > 1) {
    // allreduce sum
    double local_err[3] = {_local_abserr, _local_sqrerr, _local_pred};
    boxps::MPICluster::Ins().allreduce_sum(local_err, 3);

    _mae = local_err[0] / (fp + tp);
    _rmse = sqrt(local_err[1] / (fp + tp));
    _predicted_ctr = local_err[2] / (fp + tp);
  } else {
    _mae = _local_abserr / (fp + tp);
    _rmse = sqrt(_local_sqrerr / (fp + tp));
    _predicted_ctr = _local_pred / (fp + tp);
  }
  _actual_ctr = tp / (fp + tp);

  _size = fp + tp;

  calculate_bucket_error();
}

void BoxWrapper::CheckEmbedSizeIsValid(int embedx_dim, int expand_embed_dim) {
  PADDLE_ENFORCE_EQ(
      embedx_dim_, embedx_dim,
      platform::errors::InvalidArgument("SetInstance(): invalid embedx_dim. "
                                        "When embedx_dim = %d, but got %d.",
                                        embedx_dim_, embedx_dim));
  PADDLE_ENFORCE_EQ(expand_embed_dim_, expand_embed_dim,
                    platform::errors::InvalidArgument(
                        "SetInstance(): invalid expand_embed_dim. When "
                        "expand_embed_dim = %d, but got %d.",
                        expand_embed_dim_, expand_embed_dim));
}

void BoxWrapper::PullSparse(const paddle::platform::Place& place,
                            const std::vector<const uint64_t*>& keys,
                            const std::vector<float*>& values,
                            const std::vector<int64_t>& slot_lengths,
                            const int hidden_size, const int expand_embed_dim) {
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

#define PULLSPARSE_CASE(i, ...)                                              \
  case i: {                                                                  \
    constexpr size_t ExpandDim = i;                                          \
    if (feature_type_ == static_cast<int>(boxps::FEATURE_PCOC)) {            \
      PullSparseCase<boxps::FeatureValueGpuPCOC<EmbedxDim, ExpandDim>>(      \
          place, keys, values, slot_lengths, hidden_size, expand_embed_dim); \
    } else if (feature_type_ == static_cast<int>(boxps::FEATURE_QUANT)) {    \
      PullSparseCase<boxps::FeatureValueGpuQuant<EmbedxDim, ExpandDim>>(     \
          place, keys, values, slot_lengths, hidden_size, expand_embed_dim); \
    } else {                                                                 \
      PullSparseCase<boxps::FeaturePullValueGpu<EmbedxDim, ExpandDim>>(      \
          place, keys, values, slot_lengths, hidden_size, expand_embed_dim); \
    }                                                                        \
  } break

  CheckEmbedSizeIsValid(hidden_size - cvm_offset_, expand_embed_dim);
  switch (hidden_size - cvm_offset_) {
    EMBEDX_CASE(8, PULLSPARSE_CASE(0); PULLSPARSE_CASE(8);
                PULLSPARSE_CASE(64););
    EMBEDX_CASE(16, PULLSPARSE_CASE(0); PULLSPARSE_CASE(64););
    EMBEDX_CASE(32, PULLSPARSE_CASE(0););
    EMBEDX_CASE(64, PULLSPARSE_CASE(0););
    EMBEDX_CASE(256, PULLSPARSE_CASE(0););
    EMBEDX_CASE(128, PULLSPARSE_CASE(0););
    EMBEDX_CASE(280, PULLSPARSE_CASE(0););
    default:
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unsupport this embedding size [%d]", hidden_size - cvm_offset_));
  }
#undef PULLSPARSE_CASE
#undef EMBEDX_CASE
}

void BoxWrapper::PushSparseGrad(const paddle::platform::Place& place,
                                const std::vector<const uint64_t*>& keys,
                                const std::vector<const float*>& grad_values,
                                const std::vector<int64_t>& slot_lengths,
                                const int hidden_size,
                                const int expand_embed_dim,
                                const int batch_size) {
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

#define PUSHSPARSE_CASE(i, ...)                                             \
  case i: {                                                                 \
    constexpr size_t ExpandDim = i;                                         \
    if (feature_type_ == static_cast<int>(boxps::FEATURE_PCOC)) {           \
      PushSparseGradCase<                                                   \
          boxps::FeaturePushValueGpuPCOC<EmbedxDim, ExpandDim>>(            \
          place, keys, grad_values, slot_lengths, hidden_size,              \
          expand_embed_dim, batch_size);                                    \
    } else {                                                                \
      PushSparseGradCase<boxps::FeaturePushValueGpu<EmbedxDim, ExpandDim>>( \
          place, keys, grad_values, slot_lengths, hidden_size,              \
          expand_embed_dim, batch_size);                                    \
    }                                                                       \
  } break

  CheckEmbedSizeIsValid(hidden_size - cvm_offset_, expand_embed_dim);
  switch (hidden_size - cvm_offset_) {
    EMBEDX_CASE(8, PUSHSPARSE_CASE(0); PUSHSPARSE_CASE(8);
                PUSHSPARSE_CASE(64););
    EMBEDX_CASE(16, PUSHSPARSE_CASE(0); PUSHSPARSE_CASE(64););
    EMBEDX_CASE(32, PUSHSPARSE_CASE(0););
    EMBEDX_CASE(64, PUSHSPARSE_CASE(0););
    EMBEDX_CASE(256, PUSHSPARSE_CASE(0););
    EMBEDX_CASE(128, PUSHSPARSE_CASE(0););
    EMBEDX_CASE(280, PUSHSPARSE_CASE(0););
    default:
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unsupport this embedding size [%d]", hidden_size - cvm_offset_));
  }
#undef PUSHSPARSE_CASE
#undef EMBEDX_CASE
}

void BasicAucCalculator::calculate_bucket_error() {
  double last_ctr = -1;
  double impression_sum = 0;
  double ctr_sum = 0.0;
  double click_sum = 0.0;
  double error_sum = 0.0;
  double error_count = 0;
  double* table[2] = {&_table[0][0], &_table[1][0]};
  for (int i = 0; i < _table_size; i++) {
    double click = table[1][i];
    double show = table[0][i] + table[1][i];
    double ctr = static_cast<double>(i) / _table_size;
    if (fabs(ctr - last_ctr) > kMaxSpan) {
      last_ctr = ctr;
      impression_sum = 0.0;
      ctr_sum = 0.0;
      click_sum = 0.0;
    }
    impression_sum += show;
    ctr_sum += ctr * show;
    click_sum += click;
    double adjust_ctr = ctr_sum / impression_sum;
    double relative_error =
        sqrt((1 - adjust_ctr) / (adjust_ctr * impression_sum));
    if (relative_error < kRelativeErrorBound) {
      double actual_ctr = click_sum / impression_sum;
      double relative_ctr_error = fabs(actual_ctr / adjust_ctr - 1);
      error_sum += relative_ctr_error * impression_sum;
      error_count += impression_sum;
      last_ctr = -1;
    }
  }
  _bucket_error = error_count > 0 ? error_sum / error_count : 0.0;
}

// Deprecated: should use BeginFeedPass & EndFeedPass
void BoxWrapper::FeedPass(int date,
                          const std::vector<uint64_t>& feasgin_to_box) {
  int ret = boxps_ptr_->FeedPass(date, feasgin_to_box);
  PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
                                "FeedPass failed in BoxPS."));
}

void BoxWrapper::BeginFeedPass(int date, boxps::PSAgentBase** agent) {
  int ret = boxps_ptr_->BeginFeedPass(date, *agent);
  if (FLAGS_use_gpu_replica_cache) {
    int dim = FLAGS_gpu_replica_cache_dim;
    VLOG(3) << "gpu cache dim:" << dim;
    gpu_replica_cache.emplace_back(dim);
  }
  if (dataset_name_ == "InputTableDataset") {
    VLOG(3) << "lookup input dim: " << input_table_dim_;
    input_table_deque_.emplace_back(input_table_dim_);
  }
  PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
                                "BeginFeedPass failed in BoxPS."));
}

void BoxWrapper::EndFeedPass(boxps::PSAgentBase* agent) {
  if (FLAGS_use_gpu_replica_cache) {
    auto& t = gpu_replica_cache.back();
    t.ToHBM();
    VLOG(0) << "gpu cache memory: " << t.GpuMemUsed() << "MB";
  }
  if (dataset_name_ == "InputTableDataset") {
    auto& t = input_table_deque_.back();
    VLOG(0) << "input table size: " << t.size() << " miss: " << t.miss()
            << ", cpu memory: " << t.CpuMemUsed() << "MB";
  }
  int ret = boxps_ptr_->EndFeedPass(agent);
  PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
                                "EndFeedPass failed in BoxPS."));
}

void BoxWrapper::BeginPass() {
  int gpu_num = platform::GetCUDADeviceCount();
  for (int i = 0; i < gpu_num; ++i) {
    DeviceBoxData& dev = device_caches_[i];
    dev.ResetTimer();
  }
  int ret = boxps_ptr_->BeginPass();
  PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
                                "BeginPass failed in BoxPS."));
}

void BoxWrapper::SetTestMode(bool is_test) const {
  boxps_ptr_->SetTestMode(is_test);
}

void BoxWrapper::EndPass(bool need_save_delta) {
  if (FLAGS_use_gpu_replica_cache) {
    gpu_replica_cache.pop_front();
  }
  if (dataset_name_ == "InputTableDataset") {
    input_table_deque_.pop_front();
  }
  int ret = boxps_ptr_->EndPass(need_save_delta);
  PADDLE_ENFORCE_EQ(
      ret, 0, platform::errors::PreconditionNotMet("EndPass failed in BoxPS."));
  int gpu_num = platform::GetCUDADeviceCount();
  for (int i = 0; i < gpu_num; ++i) {
    auto& dev = device_caches_[i];
    LOG(WARNING) << "gpu[" << i
                 << "] sparse pull span: " << dev.all_pull_timer.ElapsedSec()
                 << ", boxps span: " << dev.boxps_pull_timer.ElapsedSec()
                 << ", push span: " << dev.all_push_timer.ElapsedSec()
                 << ", boxps span:" << dev.boxps_push_timer.ElapsedSec()
                 << ", dense nccl:" << dev.dense_nccl_timer.ElapsedSec()
                 << ", sync stream:" << dev.dense_sync_timer.ElapsedSec()
                 << ", wrapper gpu memory:" << dev.GpuMemUsed() << "MB";
  }
}

void BoxWrapper::RecordReplace(std::vector<SlotRecord>* records,
                               const std::set<uint16_t>& slots) {
  VLOG(0) << "Begin RecordReplace";

  platform::Timer timer;
  timer.Start();

  std::vector<std::thread> threads;
  std::valarray<int> del_num(0, auc_runner_thread_num_);
  std::valarray<int> add_num(0, auc_runner_thread_num_);
  for (int tid = 0; tid < auc_runner_thread_num_; ++tid) {
    threads.push_back(std::thread([this, records, tid, &slots, &del_num,
                                   &add_num]() {
      size_t ins_num = records->size();
      int start = tid * ins_num / auc_runner_thread_num_;
      int end = (tid + 1) * ins_num / auc_runner_thread_num_;
      VLOG(3) << "ReplaceRecord begin for thread[" << tid << "], and process ["
              << start << ", " << end << "), total ins: " << ins_num;
      for (int j = start; j < end; ++j) {
        auto record = records->at(j);
        auto info = get_auc_runner_info(record);
        auto& random_pool = random_ins_pool_list[info->pool_id_];
        FeasignValuesCandidate& candidate =
            random_pool.GetUseReplaceId(info->replaced_id_);
        record_replacers_[info->record_id_].replace(
            &record->slot_uint64_feasigns_, candidate.feasign_values_, slots,
            &del_num[tid], &add_num[tid]);
      }
      VLOG(3) << "thread[" << tid << "]: erase feasign num: " << del_num[tid]
              << " repush feasign num: " << add_num[tid];
    }));
  }
  for (int tid = 0; tid < auc_runner_thread_num_; ++tid) {
    threads[tid].join();
  }

  timer.Pause();
  VLOG(0) << "End RecordReplace: " << timer.ElapsedMS() << std::endl
          << "del feasign num: " << del_num.sum() << std::endl
          << "add feasign num: " << add_num.sum();
}

void BoxWrapper::RecordReplaceBack(std::vector<SlotRecord>* records,
                                   const std::set<uint16_t>& slots) {
  VLOG(0) << "Begin RecordReplaceBack";

  platform::Timer timer;
  timer.Start();

  std::vector<std::thread> threads;
  std::valarray<int> del_num(0, auc_runner_thread_num_);
  std::valarray<int> add_num(0, auc_runner_thread_num_);
  for (int tid = 0; tid < auc_runner_thread_num_; ++tid) {
    threads.push_back(
        std::thread([this, records, tid, &slots, &del_num, &add_num]() {
          size_t ins_num = records->size();
          int start = tid * ins_num / auc_runner_thread_num_;
          int end = (tid + 1) * ins_num / auc_runner_thread_num_;
          VLOG(3) << "RecordReplaceBack begin for thread[" << tid
                  << "], and process [" << start << ", " << end
                  << "), total ins: " << ins_num;
          for (int j = start; j < end; ++j) {
            auto record = records->at(j);
            auto info = get_auc_runner_info(record);
            record_replacers_[info->record_id_].replace_back(
                &record->slot_uint64_feasigns_, slots, &del_num[tid],
                &add_num[tid]);
          }
          VLOG(3) << "thread[" << tid
                  << "]: erase feasign num: " << del_num[tid]
                  << " repush feasign num: " << add_num[tid];
        }));
  }
  for (int tid = 0; tid < auc_runner_thread_num_; ++tid) {
    threads[tid].join();
  }
  timer.Pause();
  VLOG(0) << "End RecordReplaceBack: " << timer.ElapsedMS() << std::endl
          << "del feasign num: " << del_num.sum() << std::endl
          << "add feasign num: " << add_num.sum();
}

void BoxWrapper::GetRandomReplace(std::vector<SlotRecord>* records) {
  VLOG(0) << "Begin GetRandomReplace";
  platform::Timer timer;
  timer.Start();

  std::lock_guard<std::mutex> lock(mutex4random_pool_);
  size_t ins_num = records->size();
  std::vector<std::thread> threads;
  for (int tid = 0; tid < auc_runner_thread_num_; ++tid) {
    threads.push_back(std::thread([this, records, tid, ins_num]() {
      int start = tid * ins_num / auc_runner_thread_num_;
      int end = (tid + 1) * ins_num / auc_runner_thread_num_;
      VLOG(3) << "GetRandomReplace begin for thread[" << tid
              << "], and process [" << start << ", " << end
              << "), total ins: " << ins_num;
      auto& random_pool = random_ins_pool_list[tid];
      for (int i = start; i < end; ++i) {
        auto record = records->at(i);
        auto info = get_auc_runner_info(record);
        info->record_id_ = i;
        info->pool_id_ = tid;
        info->replaced_id_ =
            random_pool.AddAndGet(record->slot_uint64_feasigns_);
      }
    }));
  }

  for (int tid = 0; tid < auc_runner_thread_num_; ++tid) {
    threads[tid].join();
  }
  timer.Pause();

  VLOG(0) << "End GetRandomReplace: " << timer.ElapsedMS();
}

void BoxWrapper::AddReplaceFeasign(boxps::PSAgentBase* p_agent,
                                   int feed_pass_thread_num) {
  VLOG(0) << "Begin AddReplaceFeasign";
  platform::Timer timer;
  timer.Start();

  std::lock_guard<std::mutex> lock(mutex4random_pool_);
  std::vector<std::thread> threads;
  for (int tid = 0; tid < feed_pass_thread_num; ++tid) {
    threads.push_back(std::thread([this, tid, p_agent, feed_pass_thread_num]() {
      VLOG(3) << "AddReplaceFeasign begin for thread[" << tid << "]";
      for (size_t pool_id = tid; pool_id < random_ins_pool_list.size();
           pool_id += feed_pass_thread_num) {
        auto& random_pool = random_ins_pool_list[pool_id];
        for (size_t i = 0; i < random_pool.Size(); ++i) {
          auto& candidate = random_pool.GetUseId(i);
          for (const auto& pair : candidate.feasign_values_) {
            for (const auto feasign : pair.second) {
              p_agent->AddKey(feasign, tid);
            }
          }
        }
      }
    }));
  }

  for (int tid = 0; tid < feed_pass_thread_num; ++tid) {
    threads[tid].join();
  }
  timer.Pause();

  VLOG(0) << "End AddReplaceFeasign: " << timer.ElapsedMS();
}

}  // end namespace framework
}  // end namespace paddle
#endif
