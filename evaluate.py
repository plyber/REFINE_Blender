import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import lpips
import torch
from PIL import Image

# init LPIPS once
loss_fn_alex = lpips.LPIPS(net='alex')

def compute_metrics(gt, denoised):
    # resize if necesary
    if gt.size != denoised.size:
        denoised = denoised.resize(gt.size, Image.BICUBIC)

    gt_np = np.array(gt).astype(np.float32) / 255.0
    denoised_np = np.array(denoised).astype(np.float32) / 255.0

    # shape
    if gt_np.ndim == 2:
        gt_np = np.expand_dims(gt_np, axis=-1)
    if denoised_np.ndim == 2:
        denoised_np = np.expand_dims(denoised_np, axis=-1)

    #ssim & psnr
    ssim_val = ssim(gt_np, denoised_np, channel_axis=-1, data_range=1.0)
    psnr_val = psnr(gt_np, denoised_np, data_range=1.0)

    # lpips
    gt_tensor = torch.tensor(gt_np).permute(2, 0, 1).unsqueeze(0).float()
    denoised_tensor = torch.tensor(denoised_np).permute(2, 0, 1).unsqueeze(0).float()

    if torch.cuda.is_available():
        gt_tensor = gt_tensor.cuda()
        denoised_tensor = denoised_tensor.cuda()
        loss_fn_alex.cuda()

    lpips_val = loss_fn_alex(gt_tensor, denoised_tensor).item()

    return ssim_val, psnr_val, lpips_val

def parse_filename(name):
    # denoised_<scene>_<model>_s<samples>_<timestamp>.png
    parts = name.replace('.png', '').split('_')
    if len(parts) < 5:
        return None
    return {
        'scene': parts[1],
        'model': parts[2],
        'samples': parts[3].lstrip('s')
    }

def evaluate_images(folder):
    all_files = os.listdir(folder)
    ground_truths = {}
    denoised_images = []

    for file in all_files:
        if file.startswith('ground_') and file.endswith('.png'):
            parts = file.replace('.png', '').split('_')
            if len(parts) >= 3:
                scene = parts[1]
                ground_truths[scene] = file

        if file.startswith('denoised_') and file.endswith('.png'):
            denoised_images.append(file)

    rows = []

    for denoised_file in denoised_images:
        info = parse_filename(denoised_file)
        if not info:
            continue

        scene = info['scene']
        model = info['model']
        samples = info['samples']

        ground_file = ground_truths.get(scene)
        if not ground_file:
            print(f"[!] No ground truth found for scene: {scene}")
            continue

        ground_path = os.path.join(folder, ground_file)
        denoised_path = os.path.join(folder, denoised_file)

        gt_img = Image.open(ground_path).convert("RGB")
        denoised_img = Image.open(denoised_path).convert("RGB")

        try:
            ssim_val, psnr_val, lpips_val = compute_metrics(gt_img, denoised_img)
            rows.append({
                'Scene': scene,
                'Model': model,
                'Samples': samples,
                'SSIM': ssim_val,
                'PSNR': psnr_val,
                'LPIPS': lpips_val,
                'Denoised File': denoised_file,
                'Ground Truth': ground_file
            })
            print(f"[✓] Evaluated {scene} with {model} at {samples} samples.")
        except Exception as e:
            print(f"[!] Error evaluating {denoised_file}: {e}")

    # save
    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(folder, 'evaluation_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"[✔] Evaluation complete. Results saved to {csv_path}")
    else:
        print("[!] No evaluations completed.")

def browse_folder():
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        evaluate_images(folder_selected)

if __name__ == "__main__":
    browse_folder()
