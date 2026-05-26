import os
print("Setting GL...")
os.environ["MUJOCO_GL"] = "egl"
import onnxruntime as ort
print("ORT Loaded")
from lerobot.envs.factory import make_env
from lerobot.envs.configs import AlohaEnv
print("LeRobot Envs Loaded")
cfg = AlohaEnv(task="AlohaTransferCube-v0")
print("Making Env...")
envs_dict = make_env(cfg)
print("Env Created")
