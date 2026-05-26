import onnxruntime
import numpy as np

def run_inference(model_path):
    # 1. 创建 InferenceSession
    ort_session = onnxruntime.InferenceSession(model_path)

    # 2. 准备输入数据 (随机数据)
    # 注意：输入形状应与导出时的一致，或者符合 dynamic_axes 定义
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # 模拟输入
    x = np.random.randn(1, 1, 224, 224).astype(np.float32)

    # 3. 运行推理
    ort_inputs = {ort_session.get_inputs()[0].name: x}
    ort_outs = ort_session.run(None, ort_inputs)

    # 4. 获取输出
    img_out_y = ort_outs[0]
    print(f"Inference completed. Output shape: {img_out_y.shape}")

if __name__ == "__main__":
    # 假设 super_resolution.onnx 已生成并放在当前目录或指定路径
    import os
    model_file = "super_resolution.onnx"
    if os.path.exists(model_file):
        run_inference(model_file)
    else:
        print(f"Model file {model_file} not found. Please run the export script first.")
