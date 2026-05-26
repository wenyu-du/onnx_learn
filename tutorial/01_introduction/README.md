# 01 Introduction

本章节介绍 ONNX 的基础知识。

- **ONNX (Open Neural Network Exchange)**: 开放神经网络交换格式。
- **存储方式**: Protobuf。
- **组成部分**: 
    - `ModelProto`: 包含版本信息、图结构。
    - `GraphProto`: 包含节点、权重、输入输出。
    - `NodeProto`: 具体算子（Conv, Relu 等）。
