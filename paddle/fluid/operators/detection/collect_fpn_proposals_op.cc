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

#include "paddle/fluid/operators/detection/collect_fpn_proposals_op.h"

namespace paddle {
namespace operators {

class CollectFpnProposalsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* context) const override {
    PADDLE_ENFORCE(context->HasInputs("MultiLayerRois"),
                   "Input(MultiLayerRois) shouldn't be null");
    PADDLE_ENFORCE(context->HasInputs("MultiLayerScores"),
                      "Input(MultiLayerScores) shouldn't be null");
    PADDLE_ENFORCE(context->HasOutput("FpnRois"),
                      "Outputs(MultiFpnRois) of DistributeOp should not be empty");
  }
protected:
  framework::OpKernelType GetExpectedKernelType(
          const framework::ExecutionContext& ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(ctx.MultiInputVar("MultiLayerRois")[0]);
    return framework::OpKernelType(data_type, platform::CPUPlace());
  }
};

class CollectFpnProposalsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("MultiLayerRois", "(LoDTensor) Multi roi LoDTensors"
                               " from each levels in shape (-1, 4)").AsDuplicable();
    AddInput("MultiLayerScores", "(LoDTensor) Multi score LoDTensors"
                                 " from each levels in shape (-1, 1)").AsDuplicable();
    AddOutput("FpnRois", "(LoDTensor) All selected rois with highest scores");
    AddAttr<int>("post_nms_topN", "Select post_nms_topN rois from"
                                  " all images and all fpn layers");
    AddComment(R"DOC(
This operator collect all proposals from different images
 and different FPN levels. Then sort all of those proposals
by objectness confidence. Select the post_nms_topN rois in
 total. Then we select 
)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(collect_fpn_proposals, ops::CollectFpnProposalsOp,
                ops::CollectFpnProposalsOpMaker,
                paddle::framework::EmptyGradOpMaker);
REGISTER_OP_CPU_KERNEL(collect_fpn_proposals,
                ops::CollectFpnProposalsOpKernel<float>,
                ops::CollectFpnProposalsOpKernel<double>);


