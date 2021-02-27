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

#include "paddle/fluid/operators/fused/fused_seqpool_cvm_with_pcoc_op.h"
#include <string>
namespace paddle {
namespace operators {

class FusedSeqpoolCVMWithPCOCOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_GE(ctx->Inputs("X").size(), 1UL,
                      "Inputs(X) of FusedSeqpoolCVMWithPCOCOp should not be empty.");
    PADDLE_ENFORCE_GE(ctx->Outputs("Out").size(), 1UL,
                      "Outputs(Out) of FusedSeqpoolCVMWithPCOCOp should not be empty.");

    auto cvmwithpcoc_dims = ctx->GetInputDim("CVMWithPCOC");
    const int used_cvm_offset = ctx->Attrs().Get<int>("cvm_offset");
    const int max_cvm_offset = ctx->Attrs().Get<int>("max_cvm_offset");
    PADDLE_ENFORCE_EQ(
        cvmwithpcoc_dims.size(), 2UL,
        platform::errors::InvalidArgument("Input(CVMWithPCOC)'s rank should be 2."));
    PADDLE_ENFORCE_EQ(cvmwithpcoc_dims[1], used_cvm_offset, platform::errors::InvalidArgument(
                                            "The 2nd dimension of "
                                            "Input(CVMWithPCOC) should be equal to used_cvm_offset."));

    auto ins_dims = ctx->GetInputsDim("X");
    const size_t num_inputs = ins_dims.size();
    std::vector<framework::DDim> outs_dims;
    outs_dims.resize(num_inputs);

    // need filter quant_ratio more than zero
    if (ctx->Attrs().Get<bool>("need_filter")) {
      const int quant_ratio = ctx->Attrs().Get<int>("quant_ratio");
      PADDLE_ENFORCE_GT(
          quant_ratio, 0,
          platform::errors::InvalidArgument(
              "Input need filter quant_ratio should be greater than 0"));
    }

    PADDLE_ENFORCE_GT(num_inputs, 0UL,
                      platform::errors::InvalidArgument(
                          "Input tensors count should be greater than 0, "
                          "but received value is %d.",
                          num_inputs));

    // The output height should be confirmed in Compute,
    // since input lod is not accessible here.
    PADDLE_ENFORCE_EQ(ins_dims[0].size(), 2,
                      platform::errors::InvalidArgument(
                          "The dims size of first input should be equal to 2, "
                          "but received value is %d.",
                          ins_dims[0].size()));

    for (size_t i = 0; i < num_inputs; ++i) {
      const auto dims = ins_dims[i];
      int rank = dims.size();
      if (ctx->Attrs().Get<bool>("use_cvm")) {
        PADDLE_ENFORCE_GT(
            dims[rank - 1], 2,
            "Shape error in %lu id, the last dimension(embedding) of the "
            "'X' tensor must be larger than 2.",
            i);
      }
      // input lod is not accessible here
      std::vector<int64_t> out_dim;
      int embed_index_diff = max_cvm_offset - 2 * used_cvm_offset + 6;
      if (ctx->Attrs().Get<bool>("use_cvm")) {
        out_dim = {-1, dims[rank - 1] - embed_index_diff};
      } else {
        out_dim = {-1, dims[rank - 1] - max_cvm_offset};
      }
      outs_dims[i] = framework::make_ddim(out_dim);
    }
    ctx->SetOutputsDim("Out", outs_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(framework::proto::VarType::FP32,
                                   ctx.device_context());
  }
};

class FusedSeqpoolCVMWithPCOCOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(vector<LoDTensor>) The input tensors of"
             " operator.")
        .AsDuplicable();
    AddInput("CVMWithPCOC",
             "(Tensor),  a 2-D Tensor with shape [N x used_cvm_offset], where N is the batch "
             "size, used_cvm_offset is show, click, show2, click2, pclk, pclk2, pclk3....");
    AddOutput("Out",
              "(vector<Tensor>) The output of Op does not contain LoD "
              "information.")
        .AsDuplicable();
    AddAttr<std::string>("pooltype",
                         "(string, default 'SUM') the pooling pooltype of "
                         "SequencePoolOp, only support SUM now.")
        .SetDefault("SUM")
        .InEnum({"SUM"});
    AddAttr<float>("pad_value",
                   "(float, default 0.0) The value to pad for empty sequence.")
        .SetDefault(0.0);
    AddAttr<bool>("use_cvm", "bool, use cvm or not").SetDefault(true);
    AddAttr<bool>("need_filter", "(bool, default false)").SetDefault(false);
    AddAttr<float>("show_coeff", "(float, default 0.2)").SetDefault(0.2);
    AddAttr<float>("clk_coeff", "(float, default 1)").SetDefault(1);
    AddAttr<float>("threshold", "(float, default 0.96)").SetDefault(0.96);
    AddAttr<int>("cvm_offset", "(int, default 7)").SetDefault(7);
    AddAttr<int>("max_cvm_offset", "(int, default 7)").SetDefault(7);
    AddAttr<int>("quant_ratio", "(int, default 128)").SetDefault(0);

    AddComment(R"DOC(
Fuse multiple pairs of Sequence Pool and CVMWithPCOC Operator.

)DOC");
  }
};

class FusedSeqpoolCVMWithPCOCGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto og_dims = ctx->GetInputsDim(framework::GradVarName("Out"));
    auto x_dims = ctx->GetInputsDim("X");
    auto cvm_dims = ctx->GetInputDim("CVMWithPCOC");
    const int used_cvm_offset = ctx->Attrs().Get<int>("cvm_offset");
    const int max_cvm_offset = ctx->Attrs().Get<int>("max_cvm_offset");
    int embed_index_diff = max_cvm_offset - 2 * used_cvm_offset + 6;

    PADDLE_ENFORCE_EQ(
        cvm_dims.size(), 2,
        platform::errors::InvalidArgument("Input(CVMWithPCOC)'s rank should be 2."));

    for (size_t i = 0; i < og_dims.size(); i++) {
      PADDLE_ENFORCE_EQ(
          og_dims[i].size(), x_dims[i].size(),
          platform::errors::InvalidArgument(
              "The rank of output grad must equal to Input(X). But "
              "received: input rank %u, input shape [%s].",
              og_dims[i].size(), og_dims[i]));
      if (ctx->Attrs().Get<bool>("use_cvm")) {
        PADDLE_ENFORCE_EQ(
            og_dims[i][og_dims[i].size() - 1],
            x_dims[i][og_dims[i].size() - 1] - embed_index_diff,
            platform::errors::InvalidArgument(
                "The dimension mismatch between Input(OUT@GRAD) and "
                "Input(X). Received Input(OUT@GRAD): input rank %u, "
                "input shape [%s]; received Input(X): input rank %u, "
                "input shape [%s].",
                og_dims[i].size(), og_dims[i], x_dims[i].size(), x_dims[i]));
      } else {
        PADDLE_ENFORCE_EQ(
            og_dims[i][og_dims[i].size() - 1],
            x_dims[i][og_dims[i].size() - 1] - max_cvm_offset,
            platform::errors::InvalidArgument(
                "The dimension mismatch between Input(OUT@GRAD) and "
                "Input(X). Received Input(OUT@GRAD): input rank %u, "
                "input shape [%s]; received Input(X): input rank %u, "
                "input shape [%s].",
                og_dims[i].size(), og_dims[i], x_dims[i].size(), x_dims[i]));
      }
    }
    for (size_t i = 0; i < x_dims.size(); ++i) {
      ctx->ShareLoD("X", framework::GradVarName("X"), i, i);
      ctx->ShareDim("X", framework::GradVarName("X"), i, i);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

template <typename T>
class FusedSeqpoolCVMWithPCOCGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op_desc_ptr) const override {
    op_desc_ptr->SetType("fused_seqpool_cvm_with_pcoc_grad");
    op_desc_ptr->SetInput("X", this->Input("X"));
    op_desc_ptr->SetInput("CVMWithPCOC", this->Input("CVMWithPCOC"));

    op_desc_ptr->SetInput(framework::GradVarName("Out"),
                          this->OutputGrad("Out"));
    op_desc_ptr->SetOutput(framework::GradVarName("X"),
                           this->InputGrad("X", false));
    op_desc_ptr->SetOutput(framework::GradVarName("CVMWithPCOC"),
                           this->InputGrad("CVMWithPCOC"));
    op_desc_ptr->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(fused_seqpool_cvm_with_pcoc, ops::FusedSeqpoolCVMWithPCOCOp,
                  ops::FusedSeqpoolCVMWithPCOCOpMaker,
                  ops::FusedSeqpoolCVMWithPCOCGradOpMaker<paddle::framework::OpDesc>,
                  ops::FusedSeqpoolCVMWithPCOCGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(fused_seqpool_cvm_with_pcoc_grad, ops::FusedSeqpoolCVMWithPCOCGradOp)

REGISTER_OP_CPU_KERNEL(fused_seqpool_cvm_with_pcoc,
                       ops::FusedSeqpoolCVMWithPCOCOpCPUKernel<float>)
REGISTER_OP_CPU_KERNEL(fused_seqpool_cvm_with_pcoc_grad,
                       ops::FusedSeqpoolCVMWithPCOCGradOpCPUKernel<float>)
