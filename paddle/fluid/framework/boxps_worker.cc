/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#ifdef PADDLE_WITH_BOX_PS
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"

#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/framework/garbage_collector.h"

#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/framework/fleet/box_wrapper.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/trainer_desc.pb.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/lodtensor_printer.h"

namespace paddle {
namespace framework {

BoxPSAsynDenseTable::BoxPSAsynDenseTable(const int device_num)
    : device_num_(device_num) {
  device_grads_.resize(device_num);
}
BoxPSAsynDenseTable::~BoxPSAsynDenseTable() {}

void BoxPSAsynDenseTable::Init(
    const Scope& root_scope, const std::vector<std::string>& param_need_sync,
    const std::vector<std::string>& persistable_vars) {
  root_scope_ = const_cast<paddle::framework::Scope*>(&root_scope);
  VLOG(0) << "Begin Init For Aysnc Optimize";
  for (const auto& e : param_need_sync) {
    if (e.find("param") != std::string::npos &&
        e.find("pow_acc") == std::string::npos) {
      VLOG(0) << "async mode choose " << e << " to update";
      async_param_list_.push_back(e);
      async_param_list_.push_back(e + "_moment1_0");
      async_param_list_.push_back(e + "_moment2_0");
    }
  }
  ps_.resize(async_param_list_.size());
  VLOG(0) << "async_param_list_.size(): " << async_param_list_.size();
  std::sort(
      async_param_list_.begin(),
      async_param_list_
          .end());  // xx_param.b_0, xx_param_moment1_0, xx_param_moment2_0
  for (size_t i = 0; i < async_param_list_.size(); ++i) {
    VLOG(0) << "begin to copy " << async_param_list_[i];
    const LoDTensor& root_tensor =
        root_scope.FindVar(async_param_list_[i])->Get<LoDTensor>();
    VLOG(0) << "its size is " << root_tensor.numel();
    async_param_size_.push_back(root_tensor.numel());
    ps_[i].mutable_data<float>({root_tensor.numel(), 1}, platform::CPUPlace());
    TensorCopy(*static_cast<const Tensor*>(&root_tensor), platform::CPUPlace(),
               static_cast<Tensor*>(&(ps_[i])));
  }

  // Copy global lr for async mode
  for (const auto& e : persistable_vars) {
    if (e.find("learning_rate_") != std::string::npos) {
      PADDLE_ENFORCE_LE(
          base_lr_, 0,
          platform::errors::PreconditionNotMet(
              "lr have been set, previous value: %f, current var is %s",
              base_lr_, e.c_str()));
      VLOG(0) << "begin to copy global learning rate: " << e;
      const LoDTensor& root_tensor = root_scope.FindVar(e)->Get<LoDTensor>();
      const float* gpu_lr = root_tensor.data<float>();
      if (platform::is_gpu_place(root_tensor.place())) {
        cudaMemcpy(&base_lr_, gpu_lr, sizeof(float), cudaMemcpyDeviceToHost);
      } else {
        base_lr_ = *gpu_lr;
      }
    }
  }
  VLOG(0) << "base lr is " << base_lr_;
  ps_buffer_.reset(new PSBufferQueue(8 * 3));  // magic number

  update_thread_ = new std::thread(&BoxPSAsynDenseTable::AsyncUpdate, this);
}

void BoxPSAsynDenseTable::Finalize(void) {
  if (update_thread_ == nullptr) {
    return;
  }
  ps_buffer_->Close();
  update_thread_->join();

  for (size_t i = 0; i < async_param_list_.size(); ++i) {
    VLOG(0) << "begin to copy back" << async_param_list_[i];
    auto* root_tensor =
        root_scope_->Var(async_param_list_[i])->GetMutable<LoDTensor>();
    TensorCopySync(*static_cast<const Tensor*>(&ps_[i]), platform::CPUPlace(),
                   root_tensor);
  }

  ps_buffer_ = nullptr;
  delete update_thread_;
  update_thread_ = nullptr;
}

void BoxPSAsynDenseTable::AsyncUpdate() {
  VLOG(0) << "Begin AsyncUpdate";
  std::vector<std::vector<LoDTensor>*> grad(4, nullptr);  // max package

  auto box_ptr = BoxWrapper::GetInstance();
  std::map<std::string, float> lr_map = box_ptr->GetLRMap();

  while (ps_buffer_->Receive(&grad[0])) {
    size_t merge_num = ps_buffer_->Size() + 1;
    if (merge_num > 4) {
      merge_num = 4;
    }
    for (size_t i = 1; i < merge_num; ++i) {
      ps_buffer_->Receive(&grad[i]);
    }
    AutoWRLock ps_lock(&ps_lock_);
    // VLOG(0) << "AsyncUpdate recevie grads, and begin to update param, merge "
    // << merge_num;
    for (size_t i = 0; i < async_param_list_.size() / 3; ++i) {
      LoDTensor* param_tensor = &ps_[i * 3];
      LoDTensor* mom1_tensor = &ps_[i * 3 + 1];
      LoDTensor* mom2_tensor = &ps_[i * 3 + 2];
      LoDTensor* grad_tensor = &(*grad[0])[i];
      auto len = async_param_size_[i * 3];
      float* grad_data = grad_tensor->mutable_data<float>(platform::CPUPlace());
      float* param_data =
          param_tensor->mutable_data<float>(platform::CPUPlace());
      float* mom1_data = mom1_tensor->mutable_data<float>(platform::CPUPlace());
      float* mom2_data = mom2_tensor->mutable_data<float>(platform::CPUPlace());

      // merge grad
      for (size_t k = 1; k < merge_num; ++k) {
        LoDTensor* other_grad_tensor = &(*grad[k])[i];
        float* other_grad_data =
            other_grad_tensor->mutable_data<float>(platform::CPUPlace());
        for (size_t j = 0; j < len; ++j) {
          grad_data[j] += other_grad_data[j];
        }
      }
      if (merge_num > 1) {
        for (size_t j = 0; j < len; ++j) {
          grad_data[j] /= merge_num;
        }
      }
      // float tmp = param_data[0];
      float learning_rate = base_lr_;
      if (lr_map.find(async_param_list_[i * 3]) != lr_map.end()) {
        learning_rate = lr_map[async_param_list_[i * 3]];
      }
      // VLOG(0) << "learning rate for " << async_param_list_[i * 3] << " is "
      // << learning_rate;
      for (size_t j = 0; j < len; ++j) {
        mom1_data[j] = 0.99 * mom1_data[j] +
                       0.01 * grad_data[j];  // magic beta and episilon
        mom2_data[j] =
            0.9999 * mom2_data[j] + 0.0001 * grad_data[j] * grad_data[j];
        param_data[j] -=
            learning_rate * (mom1_data[j] / (sqrt(mom2_data[j]) + 1e-8));
      }
      // VLOG(0) << "update dense for " << async_param_list_[i*3] << ", param["
      // << tmp << "] - 0.000005 * [" << mom1_data[0] << "] / [" << mom1_data[1]
      // << "] = [" << param_data[0]  << "]";
    }
  }
  VLOG(0) << "Quit AsyncUpdate";
}

void BoxPSAsynDenseTable::ReShape(const platform::Place& place) {
  int device_id = boost::get<platform::CUDAPlace>(place).GetDeviceId();
  auto& grad = device_grads_[device_id];
  grad.resize(async_param_size_.size() / 3);
  for (size_t i = 0; i < async_param_size_.size(); ++i) {
    if (i % 3 != 0) {
      continue;
    }
    grad[i / 3].mutable_data<float>(
        {static_cast<int64_t>(async_param_size_[i]), 1}, place);
  }
}

// async
void BoxPSAsynDenseTable::PullDense(const platform::Place& place,
                                    const Scope& scope) {
  // while(ps_buffer_->Size() != 0) {//Size have lock, may have perf problem.
  // And will hang when the lock was removed
  //   ;
  // }
  AutoRDLock ps_lock(&ps_lock_);
  for (size_t i = 0; i < async_param_list_.size(); ++i) {
    if (i % 3 != 0) {
      continue;
    }
    const std::string& param_name = async_param_list_[i];
    Variable* var = scope.FindVar(param_name);
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    TensorCopy(*static_cast<const Tensor*>(&ps_[i]), place,
               static_cast<Tensor*>(tensor));

    // float *p = (*ps_)[i].mutable_data<float>(platform::CPUPlace());
    // VLOG(0) << "pull dense for " << (*async_param_list_)[i] << ", and the
    // first ele is " << p[0];
  }
}
void BoxPSAsynDenseTable::PushDense(const platform::Place& place,
                                    const Scope& scope) {
  int device_id = boost::get<platform::CUDAPlace>(place).GetDeviceId();
  auto& grad = device_grads_[device_id];
  for (size_t i = 0; i < async_param_list_.size(); ++i) {
    if (i % 3 != 0) {
      continue;
    }
    // VLOG(0) << "push dense for " << (*async_param_list_)[i] << "@GRAD";
    std::string grad_name = async_param_list_[i] + "@GRAD";
    Variable* var = scope.FindVar(grad_name);
    CHECK(var != nullptr) << "var[" << grad_name << "] not found";
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    // VLOG(0) << "the first element of grad_name is: " << tmp;
    TensorCopy(*static_cast<const Tensor*>(tensor), platform::CPUPlace(),
               static_cast<Tensor*>(&grad[i / 3]));
  }
  ps_buffer_->Send(&grad);
}

void BoxPSWorker::Initialize(const TrainerDesc& desc) {
  dev_ctx_ = platform::DeviceContextPool::Instance().Get(place_);
}

void BoxPSWorker::SetDenseTable(BoxPSAsynDenseTable* dense) {
  dense_table_ = dense;
  dense_table_->ReShape(place_);
}

void BoxPSWorker::AutoSetCPUAffinity(bool reuse) {
  std::vector<int>& train_cores = boxps::get_train_cores();
  if (train_cores.empty()) {
    LOG(WARNING) << "not found binding train cores";
    return;
  }

  int cpuid = train_cores[device_id_];
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(cpuid, &mask);
  pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
}

void BoxPSWorker::CreateDeviceResource(const ProgramDesc& main_prog) {
  program_.reset(new ProgramDesc(main_prog));
  for (auto& op_desc : program_->Block(0).AllOps()) {
    ops_.push_back(OpRegistry::CreateOp(*op_desc));
  }
  auto& block = program_->Block(0);
  thread_scope_ = &(root_scope_->NewScope());
  // init var and copy persistable
  for (auto& var : block.AllVars()) {
    std::string name = var->Name();
    if (!var->Persistable()) {
      auto* ptr = thread_scope_->Var(name);
      //      printf("init tensor var name: %s\n", var->Name().c_str());
      InitializeVariable(ptr, var->GetType());
    } else {
      const LoDTensor& root_tensor =
          root_scope_->FindVar(name)->Get<LoDTensor>();
      LoDTensor* gpu_tensor = thread_scope_->Var(name)->GetMutable<LoDTensor>();
      TensorCopy(*static_cast<const Tensor*>(&root_tensor), place_,
                 static_cast<Tensor*>(gpu_tensor));
    }
  }
}

int BoxPSWorker::PackBatchTask(void) {
  device_reader_->AssignFeedVar(*thread_scope_);
  return device_reader_->Next();
}

/**
 * @brief add auc monitor
 */
inline void AddAucMonitor(const Scope* scope, const platform::Place& place) {
  auto box_ptr = BoxWrapper::GetInstance();
  auto& metric_list = box_ptr->GetMetricList();
  for (auto iter = metric_list.begin(); iter != metric_list.end(); iter++) {
    auto* metric_msg = iter->second;
    if (box_ptr->Phase() != metric_msg->MetricPhase()) {
      continue;
    }
    metric_msg->add_data(scope, place);
  }
}

void BoxPSWorker::TrainFiles() {
  VLOG(3) << "begin gpubox_worker TrainFiles";
  AutoSetCPUAffinity(true);

  int64_t accum_num = 0;
  int batch_size = 0;
  if (device_reader_ != nullptr) {
    device_reader_->Start();
  }

  while ((batch_size = PackBatchTask()) > 0) {
    VLOG(3) << "begin running ops, batch size:" << batch_size;
    if (dense_table_) {
      dense_table_->PullDense(place_, *thread_scope_);
    }
    for (auto& op : ops_) {
      op->Run(*thread_scope_, place_);
    }
    if (dense_table_) {
      dense_table_->PushDense(place_, *thread_scope_);
    }
    AddAucMonitor(thread_scope_, place_);

    accum_num += batch_size;
    thread_scope_->DropKids();
  }
  dev_ctx_->Wait();
  thread_scope_->DropKids();
}
/**
static
void print_hbm_mem(const int gpu_id, const char *name = "") {
    size_t hbm_free = 0;
    size_t hbm_total = 0;
    platform::GpuMemoryUsage(&hbm_free, &hbm_total);

    VLOG(0) << "hbm usage:("
            << name << "), device_id: "
            << gpu_id << " total_size: "
            << (hbm_total >> 20) << "MB, "
            << "free: " << (hbm_free >> 20) << " MB, "
            << "used: " << ((hbm_total - hbm_free) >> 20) << "MB";
}

class GPUOpMemStat {
public:
    GPUOpMemStat(int device_id) :
        device_id_(device_id),
        start_mem_used_(0),
        end_mem_used_(0) {

    }

    void start(void) {
        start_mem_used_ = get_used_mem();
    }

    void stop(void) {
        end_mem_used_ = get_used_mem();
    }

    void print(const std::string &name) {
        size_t used_mem = ((end_mem_used_ - start_mem_used_) >> 20);
        if (used_mem == 0) {
            return;
        }
        VLOG(0) << "hbm usage:(" << name << "), "
                << "device_id: " << device_id_
                << " before: " << (start_mem_used_ >> 20) << "MB, "
                << "after: " << (end_mem_used_ >> 20) << " MB, "
                << "used: " << used_mem << "MB";
    }

private:
    size_t get_used_mem(void) {
        size_t hbm_free = 0;
        size_t hbm_total = 0;
        platform::GpuMemoryUsage(&hbm_free, &hbm_total);
        return (hbm_total - hbm_free);
    }

private:
    int device_id_;
    size_t start_mem_used_;
    size_t end_mem_used_;
};
*/
void BoxPSWorker::TrainFilesWithProfiler() {
  VLOG(3) << "begin section_worker TrainFiles with profiler";
  AutoSetCPUAffinity(true);

  int64_t step_cnt = 0;
  int64_t accum_num = 0;
  int batch_size = 0;

  platform::Timer reader_timer;
  platform::Timer cal_timer;
  platform::Timer trans_timer;
  platform::Timer sync_timer;
  platform::Timer main_timer;
  platform::Timer outer_timer;

  std::vector<double> op_total_time;
  std::vector<std::string> op_name;
  for (auto& op : ops_) {
    op_name.push_back(op->Type());
  }
  op_total_time.resize(ops_.size());
  for (size_t i = 0; i < op_total_time.size(); ++i) {
    op_total_time[i] = 0.0;
  }
  platform::Timer timeline;
  device_reader_->Start();

  //  print_hbm_mem(device_id_, "BoxPSWorker");
  //
  //  GPUOpMemStat op_mem(device_id_);

  outer_timer.Start();
  while (true) {
    main_timer.Resume();

    reader_timer.Resume();
    batch_size = PackBatchTask();
    reader_timer.Pause();
    if (batch_size <= 0) {
      break;
    }
    VLOG(3) << "begin running ops, read batch size: " << batch_size;

    cal_timer.Resume();
    int op_id = 0;
    dev_ctx_->Wait();
    for (auto& op : ops_) {
      //      op_mem.start();
      timeline.Start();
      op->Run(*thread_scope_, place_);
      dev_ctx_->Wait();
      timeline.Pause();
      op_total_time[op_id++] += timeline.ElapsedUS();
      //      op_mem.stop();
      //      op_mem.print(op->Type());
    }
    dev_ctx_->Wait();
    cal_timer.Pause();

    AddAucMonitor(thread_scope_, place_);

    if (need_dump_field_) {
      DumpField(*thread_scope_, dump_mode_, dump_interval_);
    }
    if (need_dump_param_ && device_id_ == 0) {
      DumpParam(*thread_scope_, step_cnt);
    }
    thread_scope_->DropKids();
    ++step_cnt;
    accum_num += batch_size;
    main_timer.Pause();
  }
  if (need_dump_field_ || need_dump_param_) {
    writer_.Flush();
  }

  thread_scope_->DropKids();
  dev_ctx_->Wait();
  outer_timer.Pause();

  LOG(ERROR) << "log_for_profile"
             << " card:" << device_id_ << " thread:" << thread_id_
             << " step_count:" << step_cnt << " batch_count:" << accum_num
             << " read_time:" << reader_timer.ElapsedUS()
             << " trans_time:" << trans_timer.ElapsedUS()
             << " cal_time:" << cal_timer.ElapsedUS()
             << " sync_time:" << sync_timer.ElapsedUS()
             << " main_time:" << main_timer.ElapsedUS()
             << " outer_time:" << outer_timer.ElapsedUS();
  for (size_t i = 0; i < ops_.size(); ++i) {
    LOG(ERROR) << "card:" << device_id_ << ", op: " << op_name[i]
               << ", mean time: " << op_total_time[i] / accum_num
               << "us, sum:" << op_total_time[i] / 1000000.0 << "sec";
  }
}
}  // namespace framework
}  // namespace paddle
#endif
