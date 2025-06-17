import bpy
from . import check_dependencies


class RENDER_PT_RefinePanel(bpy.types.Panel):
    bl_label = "REFINE Denoiser"
    bl_idname = "RENDER_PT_refine_denoiser"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"


    @classmethod
    def poll(cls, context):
        return context.scene.render.engine == 'CYCLES'

    def update_noise_visibility(self, context):
        bpy.context.area.tag_redraw()

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        settings = scene.refine_settings

        if not check_dependencies():
            layout.label(text="Dependencies failed to load", icon="ERROR")
            layout.label(text="See Console for details (Window > Toggle System Console or switch to Scripting Layout)")
        else:
            layout.prop(scene.refine_settings, "samples")
            layout.prop(scene.refine_settings, "model_choice", text="Checkpoint")

            if settings.model_choice == 'drunet_color':
                layout.prop(settings, "noise_sigma")

            layout.operator("render.refine_denoise", icon="RENDER_STILL", text="Render & Denoise")
            layout.prop(scene.refine_settings, "denoiser_backend", text="Denoiser")
