import onnxruntime as ort
import numpy as np
import imageio
import json
from lerobot.envs.factory import make_env
from lerobot.envs.configs import AlohaEnv
import os
import sys

PROVIDERS = ['CPUExecutionProvider']
os.environ["MUJOCO_GL"] = "egl" 

def load_stats(stats_path):
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    for k in stats:
        stats[k] = np.array(stats[k], dtype=np.float32)
    return stats

def run_inference(model_path, stats_path, output_video, num_steps=1000):
    print(f"Loading ONNX model...")
    session = ort.InferenceSession(model_path, providers=PROVIDERS)
    stats = load_stats(stats_path)

    print("Setting up environment...")
    cfg = AlohaEnv(task="AlohaTransferCube-v0", episode_length=num_steps)
    envs_dict = make_env(cfg)
    vec_env = envs_dict['aloha'][0]

    obs, _ = vec_env.reset()
    frames = []
    action_queue = []
    
    print(f"Running simulation...")
    try:
        for t in range(num_steps):
            if len(action_queue) == 0:
                state = obs['agent_pos'].astype(np.float32)
                norm_state = (state - stats['normalize_inputs.buffer_observation_state.mean']) / stats['normalize_inputs.buffer_observation_state.std']
                
                image = obs['pixels']['top'].astype(np.float32) / 255.0
                image = np.transpose(image, (0, 3, 1, 2))
                norm_image = (image - stats['normalize_inputs.buffer_observation_images_top.mean']) / stats['normalize_inputs.buffer_observation_images_top.std']
                
                ort_inputs = {"state": norm_state, "image_top": norm_image}
                action_chunk_norm = session.run(None, ort_inputs)[0]
                
                unnorm_chunk = action_chunk_norm[0] * stats['unnormalize_outputs.buffer_action.std'] + stats['unnormalize_outputs.buffer_action.mean']
                action_queue = list(unnorm_chunk[:80])

            action = action_queue.pop(0)[None, :]
            obs, reward, terminated, truncated, info = vec_env.step(action)
            
            frame = vec_env.render()
            if frame is not None:
                if isinstance(frame, (list, tuple)): frames.append(frame[0])
                elif frame.ndim == 4: frames.append(frame[0])
                elif frame.ndim == 3: frames.append(frame)
            
            if (t + 1) % 100 == 0:
                print(f"  Step {t+1}, Reward: {reward[0]:.4f}")

            if terminated.any() or truncated.any():
                print(f"Terminated/Truncated at step {t}. Terminated: {terminated}, Truncated: {truncated}")
                # Log success if available in info
                if 'is_success' in info: print(f"Success: {info['is_success']}")
                break
    except Exception as e:
        print(f"Error: {e}")
    finally:
        vec_env.close()

    if frames:
        imageio.mimsave(output_video, [f.astype(np.uint8) for f in frames], fps=cfg.fps)
        print(f"Video saved: {output_video}")

if __name__ == "__main__":
    run_inference("tutorial/05_lerobot_to_onnx/act_aloha.onnx", 
                  "tutorial/05_lerobot_to_onnx/stats.json", 
                  "tutorial/05_lerobot_to_onnx/simulation_result.mp4")
