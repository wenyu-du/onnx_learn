import torch
from lerobot.policies.act.modeling_act import ACTPolicy
import os

def export_to_onnx(model_id, output_path):
    print(f"Loading model: {model_id}")
    # Load the policy
    try:
        policy = ACTPolicy.from_pretrained(model_id)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    policy.eval()
    policy.to("cpu")

    # Let's try to infer shapes from the config
    config = policy.config
    print(f"Model Config: {config}")

    # For ACT, common inputs are:
    # "observation.state": (batch, state_dim)
    # "observation.images.top": (batch, C, H, W)
    
    # We'll create a wrapper to handle the dict input for ONNX export
    class ACTWrapper(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy
        
        def forward(self, state, image_top):
            # Reconstruct the batch dict
            batch = {
                "observation.state": state,
                "observation.images.top": image_top,
            }
            # Use predict_action_chunk for inference only path
            return self.policy.predict_action_chunk(batch)

    wrapper = ACTWrapper(policy)
    
    # Dummy inputs
    batch_size = 1
    dummy_state = torch.randn(batch_size, 14)
    dummy_image_top = torch.randn(batch_size, 3, 480, 640)

    print("Exporting to ONNX...")
    # Using opset 17 for better support of transformer ops
    torch.onnx.export(
        wrapper,
        (dummy_state, dummy_image_top),
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['state', 'image_top'],
        output_names=['action_chunk'],
        dynamic_axes={
            'state': {0: 'batch_size'},
            'image_top': {0: 'batch_size'},
            'action_chunk': {0: 'batch_size'}
        }
    )
    print(f"Exported to {output_path}")

if __name__ == "__main__":
    model_id = "lerobot/act_aloha_sim_transfer_cube_human"
    output_path = "tutorial/05_lerobot_to_onnx/act_aloha.onnx"
    export_to_onnx(model_id, output_path)
