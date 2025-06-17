import os
import time
import torch
import numpy as np
from PIL import Image
from refine_plugin.models.dncnn_arch import DnCNN

def run_denoiser(input_path, output_path, model_name="25", samples=None):
    start_time = time.time()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_model = torch.cuda.get_device_name(device)
    else:
        device = torch.device("cpu")
        gpu_model = None

    model = DnCNN(channels=3, num_of_layers=17).to(device)
    weights_filename = f"DnCNN_sigma{model_name}.pth"
    weights_path = os.path.join(os.path.dirname(__file__), "models", weights_filename)

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    image = Image.open(input_path).convert("RGB")
    width, height = image.size
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        out_tensor = model(image_tensor).clamp(0, 1)

    out_np = (out_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(out_np).save(output_path)

    elapsed = time.time() - start_time
    return {
        "width": width,
        "height": height,
        "samples": samples,
        "checkpoint": weights_filename,
        "elapsed": elapsed,
        "device": str(gpu_model)
    }
