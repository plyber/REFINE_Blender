import bpy
from bpy.types import Operator
import os
import time
from datetime import datetime
from refine_plugin.utils import write_denoise_report

class RENDER_OT_RefineDenoise(Operator):
    bl_idname = "render.refine_denoise"
    bl_label = "Render with AI Denoising"
    bl_description = "Render the image with Cycles and apply deep-learning-based denoising and OptiX comparison"

    def execute(self, context):
        scene = context.scene
        settings = scene.refine_settings
        sample_count = settings.samples
        model_choice = settings.model_choice

        if model_choice.startswith("restormer"):
            from refine_plugin.restormer_wrapper import run_denoiser
            model_name = model_choice.replace("restormer_", "")
            model_used = "restormer"
        elif model_choice.startswith("drunet"):
            from refine_plugin.drunet_wrapper import run_denoiser
            model_name = model_choice.replace("drunet_", "")
            model_used = "drunet"
        elif model_choice.startswith("dncnn"):
            from refine_plugin.dncnn_wrapper import run_denoiser
            model_name = model_choice.replace("dncnn_", "")
            model_used = "dncnn"
        else:
            self.report({'ERROR'}, f"Unknown model selection: {model_choice}")
            return {'CANCELLED'}

        noise_sigma = getattr(settings, "noise_sigma", None) if model_choice == "drunet_color" else None

        scene.cycles.samples = sample_count
        scene.cycles.use_denoising = False

        base_dir = bpy.path.abspath("//")
        blend_name = bpy.path.display_name_from_filepath(bpy.data.filepath)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_dir, "renders")
        os.makedirs(output_dir, exist_ok=True)

        temp_path = os.path.join(output_dir, f"noisy_{blend_name}_{model_used}_s{sample_count}_{timestamp}.png")
        output_path = os.path.join(output_dir, f"denoised_{blend_name}_{model_used}_s{sample_count}_{timestamp}.png")
        optix_path = os.path.join(output_dir, f"denoised_{blend_name}_optix_s{sample_count}_{timestamp}.png")
        report_filename = f"refine_report_{blend_name}.txt"
        report_path = os.path.join(output_dir, report_filename)

        try:
            start_custom = time.time()
            scene.render.filepath = temp_path
            bpy.ops.render.render(write_still=True)

            denoise_meta = run_denoiser(
                input_path=temp_path,
                output_path=output_path,
                model_name=model_name,
                noise_sigma=noise_sigma
            )
            elapsed_custom = time.time() - start_custom
            denoise_meta["elapsed"] = elapsed_custom

            write_denoise_report(
                input_path=os.path.basename(temp_path),
                output_path=os.path.basename(output_path),
                width=denoise_meta["width"],
                height=denoise_meta["height"],
                samples=sample_count,
                weights_filename=denoise_meta["checkpoint"],
                elapsed=denoise_meta["elapsed"],
                device=denoise_meta["device"],
                report_path=report_path
            )
        except Exception as e:
            self.report({'ERROR'}, f"Custom Denoising failed: {e}")
            return {'CANCELLED'}

        try:
            scene.cycles.use_denoising = True
            scene.render.filepath = optix_path
            start_optix = time.time()
            bpy.ops.render.render(write_still=True)
            elapsed_optix = time.time() - start_optix

            write_denoise_report(
                input_path=os.path.basename(temp_path),
                output_path=os.path.basename(optix_path),
                width=denoise_meta["width"],
                height=denoise_meta["height"],
                samples=sample_count,
                weights_filename="OptiX Built-in",
                elapsed=elapsed_optix,
                device="cuda",
                report_path=report_path
            )
        except Exception as e:
            self.report({'WARNING'}, f"OptiX Denoising failed: {e}")
        finally:
            scene.cycles.use_denoising = False

        self.report({'INFO'}, f"Finished rendering and denoising. Report at {report_path}")
        return {'FINISHED'}
