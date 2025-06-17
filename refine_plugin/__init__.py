import bpy
from bpy.props import EnumProperty
import os
import sys
import traceback

# patch sys.path include bundled dependencies
addon_dir = os.path.dirname(__file__)
libs_path = os.path.join(addon_dir, "libs")
if libs_path not in sys.path:
    sys.path.insert(0, libs_path)

bl_info = {
    "name": "REFINE Denoiser",
    "author": "Oliviu Dienes",
    "version": (1, 2, 0),
    "blender": (4, 3, 2),
    "location": "Properties > Render",
    "description": "AI denoising with DRUNet. Patch includes the addition of OptiX report system",
    "category": "Render",
}

# dependency check

def check_dependencies():
    try:
        import torch
        import torch
        import numpy as np
        from PIL import Image
        from .models.restormer_arch import Restormer
        return True
    except Exception as e:
        print("[REFINE] Dependency check failed:")
        traceback.print_exc()
        return False

#settings

class RefineRenderSettings(bpy.types.PropertyGroup):
    samples: bpy.props.IntProperty(
        name="Sample Count",
        description="Cycles samples before AI denoising",
        default=32,
        min=1,
        max=4096
    )
    model_choice: EnumProperty(
        name="Model Checkpoint",
        description="Select pre-trained weights",
        items=[
            ('restormer_sigma15', "Restormer σ=15", ""),
            ('restormer_sigma25', "Restormer σ=25", ""),
            ('restormer_sigma50', "Restormer σ=50", ""),
            ('restormer_blind', "Restormer Blind", ""),
            ('drunet_gray', "DRUNet Grayscale", ""),
            ('drunet_color', "DRUNet Color", ""),
            ('dncnn_25', "DnCNN σ=25", "")
        ],
        default='restormer_sigma25'
    )

    noise_sigma: bpy.props.FloatProperty(
        name="Noise Sigma",
        description="Adjust the noise level for DRUNet",
        default=25.0,
        min=0.0,
        max=100.0
    )


classes = []

def register():
    from .panel import RENDER_PT_RefinePanel
    from .operators import RENDER_OT_RefineDenoise

    bpy.utils.register_class(RefineRenderSettings)
    bpy.types.Scene.refine_settings = bpy.props.PointerProperty(type=RefineRenderSettings)

    bpy.utils.register_class(RENDER_PT_RefinePanel)
    classes.append(RENDER_PT_RefinePanel)

    if check_dependencies():
        bpy.utils.register_class(RENDER_OT_RefineDenoise)
        classes.append(RENDER_OT_RefineDenoise)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    if hasattr(bpy.types.Scene, "refine_settings"):
        del bpy.types.Scene.refine_settings
    bpy.utils.unregister_class(RefineRenderSettings)