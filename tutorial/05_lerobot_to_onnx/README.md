# 实战：LeRobot 模型转换与仿真推理

本章将介绍如何将 Hugging Face [LeRobot](https://github.com/huggingface/lerobot) 框架下的机器人策略模型（如 ACT）转换为 ONNX 格式，并在仿真环境（或使用 ONNX Runtime）中实现推理。

## 1. 背景介绍

LeRobot 是 Hugging Face 推出的机器人学习框架，旨在降低机器人 AI 的门槛。`act_aloha_sim_transfer_cube_human` 是一个基于 ACT (Action Chunking with Transformers) 算法的模型，用于在仿真环境（ALOHA）中完成抓取任务。

将此类模型转换为 ONNX 的意义在于：
- **脱离深度学习框架**：推理时不需要安装复杂的 PyTorch 及其庞大的依赖。
- **高性能推理**：利用 ONNX Runtime 进行硬件加速。
- **跨平台部署**：方便部署到嵌入式设备或不同操作系统的机器人控制器上。

## 2. 模型转换流程

由于 LeRobot 的模型输入通常是复杂的嵌套字典（包含图像和机器人状态），我们需要编写一个包装器（Wrapper）来简化 ONNX 的输入接口。

### 导出脚本示例

下面是一个简化的导出脚本 `export_lerobot.py` 的核心逻辑：

```python
import torch
from lerobot.policies.act.modeling_act import ACTPolicy

# 1. 加载预训练模型
model_id = "lerobot/act_aloha_sim_transfer_cube_human"
policy = ACTPolicy.from_pretrained(model_id)
policy.eval()

# 2. 定义包装器，将字典输入转换为张量输入
class ACTWrapper(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
    
    def forward(self, state, image_top):
        # 重新构建 LeRobot 期望的 batch 字典
        batch = {
            "observation.state": state,
            "observation.images.top": image_top,
        }
        return self.policy.predict_action_chunk(batch)

wrapper = ACTWrapper(policy)

# 3. 准备 Dummy Input
dummy_state = torch.randn(1, 14)
dummy_image_top = torch.randn(1, 3, 480, 640)

# 4. 执行导出
torch.onnx.export(
    wrapper,
    (dummy_state, dummy_image_top),
    "act_aloha.onnx",
    input_names=['state', 'image_top'],
    output_names=['action_chunk'],
    opset_version=17
)
```

## 3. 在仿真环境中使用 ONNX 推理

在仿真环境（如 MuJoCo）中，我们需要获取机器人状态和图像，并通过 ONNX Runtime 进行推理。

### 环境准备

你需要安装仿真相关的依赖：
```bash
pip install mujoco gym-aloha imageio[ffmpeg] pyopengl
```

### 完整推理与可视化脚本

下面是 `inference_sim.py` 的实现，它会运行 200 个步骤并将结果保存为视频：

```python
import onnxruntime as ort
import numpy as np
import imageio
from lerobot.envs.factory import make_env
from lerobot.envs.configs import AlohaEnv
import os

# 设置无头模式渲染
os.environ["MUJOCO_GL"] = "egl" 

def run_inference(model_path, output_video, num_steps=200):
    print(f"加载模型: {model_path}")
    session = ort.InferenceSession(model_path)

    print("设置 ALOHA 仿真环境...")
    cfg = AlohaEnv(task="AlohaTransferCube-v0")
    envs_dict = make_env(cfg)
    vec_env = envs_dict['aloha'][0]

    obs, _ = vec_env.reset()
    frames = []

    print(f"运行仿真 {num_steps} 步...")
    for i in range(num_steps):
        # 1. 准备输入
        state = obs['agent_pos'].astype(np.float32)
        image = obs['pixels']['top'].astype(np.float32) / 255.0
        image = np.transpose(image, (0, 3, 1, 2))
        
        ort_inputs = {"state": state, "image_top": image}
        
        # 2. 推理获取动作序列 (Action Chunk)
        action_chunk = session.run(None, ort_inputs)[0]
        action = action_chunk[:, 0, :] # 取第一步动作
        
        # 3. 执行动作
        obs, reward, terminated, truncated, info = vec_env.step(action)
        
        # 4. 渲染并保存帧
        frame = vec_env.render()
        if frame is not None:
            if isinstance(frame, (list, tuple)): frames.append(frame[0])
            elif frame.ndim == 4: frames.append(frame[0])
            elif frame.ndim == 3: frames.append(frame)
        
        if terminated.any() or truncated.any(): break

    print(f"保存视频到 {output_video}...")
    imageio.mimsave(output_video, [f.astype(np.uint8) for f in frames], fps=cfg.fps)
    print("完成!")

if __name__ == "__main__":
    run_inference("tutorial/05_lerobot_to_onnx/act_aloha.onnx", "tutorial/05_lerobot_to_onnx/simulation_result.mp4")
```

## 4. 总结

通过本章，你学习了：
1. 如何使用包装器处理复杂的机器人模型输入。
2. 如何导出包含多模态输入的 ACT 模型到 ONNX。
3. 如何在没有 PyTorch 的情况下，仅使用 ONNX Runtime 驱动仿真机器人完成任务。
