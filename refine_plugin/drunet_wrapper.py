def run_denoiser(input_path, output_path, model_name="color", noise_sigma=25.0):
    import os, time
    import torch, numpy as np
    from PIL import Image
    from .models.drunet_arch import UNetRes as DRUNet

    start_time = time.time()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_model = torch.cuda.get_device_name(device)
    else:
        device = torch.device("cpu")
        gpu_model = None

    model_map = {
        "gray": "drunet_gray.pth",
        "color": "drunet_color.pth"
    }
    weights_file = model_map.get(model_name, "drunet_color.pth")

    image = Image.open(input_path).convert("RGB")
    width, height = image.size
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)

    noise_tensor = torch.full((1, 1, height, width), noise_sigma / 255.0, device=device)
    input_tensor = torch.cat((image_tensor, noise_tensor), dim=1)

    model = DRUNet(in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R')
    weights_path = os.path.join(os.path.dirname(__file__), "models", weights_file)
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=True)
    model.eval().to(device)

    with torch.no_grad():
        out_tensor = model(input_tensor).clamp(0, 1)

    output_np = (out_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(output_np).save(output_path)

    elapsed = time.time() - start_time
    if device.type == 'cuda':
        del model
        del image_tensor
        del out_tensor
        torch.cuda.empty_cache()

    return {
        "elapsed": elapsed,
        "device": str(gpu_model),
        "checkpoint": weights_file,
        "width": width,
        "height": height
    }
