# 03 Model Validation

本章节介绍如何验证 ONNX 模型。

可以使用 `onnx` 库进行基础校验：

```python
import onnx
model = onnx.load("model.onnx")
onnx.checker.check_model(model)
print("Model check passed!")
```

此外，推荐使用 **Netron** 进行可视化分析。
