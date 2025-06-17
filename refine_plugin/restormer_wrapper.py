import os, time
import torch, numpy as np
from PIL import Image
from .models.restormer_arch import Restormer


def run_denoiser(input_path, output_path, model_name="sigma25"):

    start_time = time.time()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_model = torch.cuda.get_device_name(device)
    else:
        device = torch.device("cpu")
        gpu_model = None

    model = Restormer(LayerNorm_type='BiasFree')

    model_map = {
        "sigma15": "gaussian_color_denoising_sigma15.pth",
        "sigma25": "gaussian_color_denoising_sigma25.pth",
        "sigma50": "gaussian_color_denoising_sigma50.pth",
        "blind":   "gaussian_color_denoising_blind.pth"
    }

    weights_filename = model_map.get(model_name, "gaussian_color_denoising_sigma25.pth")
    weights_path = os.path.join(os.path.dirname(__file__), "models", weights_filename)

    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['params'] if 'params' in checkpoint else checkpoint)
    model.to(device).eval()

    image = Image.open(input_path).convert("RGB")
    width, height = image.size
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        out_tensor = model(image_tensor)[0].clamp(0, 1)

    out_np = (out_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(out_np).save(output_path)

    elapsed = time.time() - start_time
    if device.type == 'cuda':
        del model
        del image_tensor
        del out_tensor
        torch.cuda.empty_cache()

    return {
        "elapsed": elapsed,
        "device": str(gpu_model),
        "checkpoint": weights_filename,
        "width": width,
        "height": height
    }
