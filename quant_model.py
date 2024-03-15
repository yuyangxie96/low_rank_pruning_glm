import torch
import safetensors
import os
# Get the list of all files in the current directory with .safetensors extension
bin_files = [f for f in os.listdir('.') if f.endswith('.safetensors')]
for file in bin_files:
    tensor_dict = {}
    with safetensors.safe_open(file, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            print(f"Key: {key}, Type: {tensor.dtype}, Shape: {tensor.shape}")
            if key.endswith("q_proj.weight") or key.endswith("k_proj.weight") or key.endswith("v_proj.weight") or key.endswith("o_proj.weight") or key.endswith("gate_proj.weight") or key.endswith("up_proj.weight") or key.endswith("down_proj.weight"):
                weight = tensor.to(torch.float32)
                weight_scale = weight.abs().max(dim=-1).values / (2 ** (8 - 1) - 1)
                weight = torch.round(weight / weight_scale.unsqueeze(-1)).to(torch.int8).view(weight.shape[0], -1).contiguous().cpu()
                # for checking result
                weight = weight.to(torch.float16) * weight_scale.unsqueeze(-1)
                tensor_dict[key] = weight
                # tensor_dict[key.replace("weight", "scales")] = weight_scale                
        safetensors.torch.save_file(tensor_dict, file, metadata={"format": "pt"})








