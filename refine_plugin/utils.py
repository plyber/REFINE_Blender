from datetime import datetime

def write_denoise_report(input_path, output_path, width, height, samples, weights_filename, elapsed, device, report_path):
    with open(report_path, "a") as f:
        f.write(f"=== REFINE Denoising Report ===\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Noisy Image: {input_path}\n")
        f.write(f"Denoised Image: {output_path}\n")
        f.write(f"Resolution: {width}x{height}\n")
        f.write(f"Samples Used: {samples}\n")
        f.write(f"Checkpoint Used: {weights_filename}\n")
        f.write(f"Denoising Time: {elapsed:.2f} seconds\n")
        f.write(f"Device: {device}\n")
        f.write("-" * 40 + "\n")
