import imageio
import numpy as np
from lerobot.envs.factory import make_env
from lerobot.envs.configs import AlohaEnv
import os

os.environ["MUJOCO_GL"] = "egl"

cfg = AlohaEnv(task="AlohaTransferCube-v0")
envs_dict = make_env(cfg)
vec_env = envs_dict['aloha'][0]

obs, _ = vec_env.reset()
frame = vec_env.render()
print(f"Frame shape: {frame.shape if hasattr(frame, 'shape') else type(frame)}")

if frame is not None:
    # Handle VectorEnv return (tuple or ndarray)
    if isinstance(frame, (tuple, list)):
        img = frame[0]
    else:
        img = frame[0] if frame.ndim == 4 else frame
    
    print(f"Saving test_frame.png, mean value: {np.mean(img)}")
    imageio.imwrite("tutorial/05_lerobot_to_onnx/test_frame.png", img.astype(np.uint8))

vec_env.close()
