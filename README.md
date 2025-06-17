# REFINE: AI-Enhanced Denoising Plugin for Blender

**REFINE** (Render Enhancement through Fast Intelligent Neural Engines) is a Blender plugin that enhances low-sample Cycles renders using AI-based denoising models like DRUNet and Restormer. It enables high-quality renders at a fraction of the time required for traditional high-sample rendering.

---

## Features

- AI-enhanced denoising for low-sample renders
- Supports DRUNet and Restormer models
- Compatible with Blender’s Cycles renderer
- GPU acceleration via PyTorch and CUDA
- Local dependency bundling for Blender's isolated Python environment
- Reproducible performance logging

---

## Project Structure

```
REFINE_Blender/
├── refine_plugin/
│   ├── models/                # Pretrained models (not included in repo)
│   ├── deps/                  # Site-packages to be bundled (ignored)
│   ├── libs/                  # Bundled site-packages (ignored)
│   ├── panel.py               # Blender UI definitions
│   ├── operators.py           # Execution logic
│   ├── restormer_wrapper.py
│   ├── drunet_wrapper.py
│   ├── utils.py
│   └── ...
├── bundle_deps.py            # Bundles site-packages for Blender
├── package_plugin.py         # Creates plugin zip for installation
├── README.md
└── .gitignore
```

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/plyber/REFINE_Blender.git
   cd REFINE_Blender
   ```

2. (Optional) Create a virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate       # Windows
   source venv/bin/activate    # macOS/Linux
   ```

3. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```

---

## Plugin Packaging Workflow

### Step 1: Bundle Python dependencies

To make the plugin self-contained for Blender's internal Python:

```bash
python bundle_deps.py
```

### Step 2: Package the plugin as a zip

```bash
python package_plugin.py
```

This generates a zip archive such as:

```
refine_plugin-v1.2.2.zip
```

---

## Installing in Blender

1. Open Blender.
2. Go to **Edit → Preferences → Add-ons → Install...**
3. Select the zip file generated in the previous step.
4. Enable the REFINE plugin.
5. In the **Render Properties** tab:
   - Choose sample count (e.g., 16)
   - Select the AI model (e.g., `drunet_color`)
   - Set noise sigma if needed (DRUNet only)
   - Click **Render & Denoise**

Denoised output and logs will appear in the `renders/` folder.

---

## Required AI Models

Download and place the following pretrained models in:

```
refine_plugin/models/
```

Models:

- `drunet_color.pth`
- `drunet_gray.pth`
- `DnCNN_sigma25.pth`
- `gaussian_color_denoising_sigma25.pth`
- `RealESRGAN_x4plus.pth`
- `restormer.pth`

These are excluded from the repo due to size and license constraints. Download them from their official sources:

- DRUNet, DnCNN: [https://github.com/cszn/DPIR](https://github.com/cszn/DPIR)
- Real-ESRGAN: [https://github.com/xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- Restormer: [https://github.com/swz30/Restormer](https://github.com/swz30/Restormer)

---

## Output Logging

Each denoise operation creates a log like the following:

```
=== Denoise Report ===
Timestamp     : 2025-06-17 14:21:30
Noisy Image   : noisy_example_16spp.png
Denoised Image: denoised_example_16spp.png
Resolution    : 1920x1080
Samples Used  : 16
Checkpoint    : drunet_color.pth
Denoising Time: 1.45 seconds
Device        : NVIDIA RTX 4080
```

---

## Files and Folders Ignored by Git

```
refine_plugin/models/*.pth
refine_plugin/libs/
refine_plugin/deps/
refine_plugin*.zip
.idea/
__pycache__/
*.pyc
```

---

## Academic Context

This plugin was developed as part of a Master's thesis at the West University of Timișoara, Faculty of Mathematics and Computer Science, MSc in Software Engineering (2025).

---

## License

This repository is intended for academic and research use. Ensure compliance with licenses for all third-party models and dependencies.

---

## Author

**O. Dienes**\
West University of Timișoara\
2025\
[https://github.com/plyber](https://github.com/plyber)

