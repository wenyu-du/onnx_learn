# ONNX 基础与实战 Tutorial

本教程根据知乎文章 [ONNX 基础与实战](https://zhuanlan.zhihu.com/p/582974246) 整理而成，涵盖了从 ONNX 基础概念到 PyTorch 模型导出与推理的全过程。

## 目录结构

- `01_introduction/`: ONNX 核心概念与模型结构介绍。
- `02_pytorch_to_onnx/`: PyTorch 模型导出为 ONNX 的实战代码。
- `03_model_validation/`: ONNX 模型验证与可视化。
- `04_inference_runtime/`: 使用 ONNX Runtime 进行推理。
- `05_lerobot_to_onnx/`: 实战：LeRobot 机器人模型转换与仿真推理。
- `scripts/`: 环境搭建脚本。

## 环境准备

本教程建议使用 Python 3.8+ 环境。

### 构建虚拟环境

你可以运行以下脚本自动创建并配置虚拟环境：

```bash
bash scripts/setup_env.sh
# 如果cuda相关的包安装太慢，可以手动安装
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

或者手动执行以下步骤：

1. 创建虚拟环境：
   ```bash
   python3 -m venv onnx_env
   ```
2. 激活环境：
   ```bash
   source onnx_env/bin/activate  # Linux/macOS
   # 或
   onnx_env\Scripts\activate     # Windows
   ```
3. 安装依赖：
   ```bash
   pip install -r tutorial/requirements.txt
   ```

## 教程内容概要

### 1. ONNX 核心概念
了解 ONNX 作为中间表示（IR）的角色，以及它如何使用 Protobuf 进行存储。

### 2. 模型结构
学习 `ModelProto`, `GraphProto`, `NodeProto` 等协议层级。

### 3. PyTorch 导出实战
使用 `torch.onnx.export()` 将训练好的模型转换为 `.onnx` 格式。

### 4. 推理与验证
使用 `onnx` 库检查模型合法性，并使用 `onnxruntime` 进行高性能推理。

## 推荐工具
- **Netron**: 用于可视化 ONNX 模型结构。 [https://netron.app/](https://netron.app/)
