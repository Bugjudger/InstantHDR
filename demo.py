"""Interactive InstantHDR reconstruction with exposure control."""

import argparse
import json
import math
from functools import partial
from pathlib import Path
from uuid import uuid4


EXAMPLES_ROOT = Path(__file__).resolve().parent / "examples"
DEMO_CSS = """
.gradio-container { max-width: 1440px !important; margin: 0 auto !important; }
#project-header { text-align: center; padding: 44px 20px 36px; }
#project-header h1 {
    margin: 0 0 28px; font-size: clamp(18px, 2.2vw, 30px) !important;
    font-weight: 700; line-height: 1.4; white-space: nowrap;
    color: var(--body-text-color);
}
#project-header { overflow-x: auto; }
#project-header .authors, #project-header .affiliations {
    display: flex; flex-wrap: wrap; justify-content: center;
    gap: 10px 24px; margin: 0 auto 16px; line-height: 1.6;
    color: var(--body-text-color);
}
#project-header .authors { font-size: 21px; }
#project-header .affiliations { font-size: 18px; }
#project-header .authors span, #project-header .affiliations span { white-space: nowrap; }
#project-header sup { font-size: 0.65em; margin-left: 3px; }
#demo-instructions {
    border-top: 1px solid var(--border-color-primary);
    padding-top: 28px; margin-bottom: 20px;
}
#demo-instructions h2 { font-size: 30px !important; margin-bottom: 16px; }
#demo-instructions li { margin-bottom: 8px; line-height: 1.65; }
#demo-workspace { gap: 28px; align-items: flex-start; }
#render-exposure { width: 100%; }
#render-exposure input {
    box-sizing: border-box !important;
    width: 100% !important;
    min-width: 0 !important;
    height: 76px !important;
    padding: 12px 20px !important;
    font-size: 32px !important;
    line-height: 1.4 !important;
    font-weight: 700 !important;
}
#exposure-readout p { font-size: 24px !important; font-weight: 700; line-height: 1.5; }
"""


# Author order and affiliations: https://arxiv.org/html/2603.11298v1
PROJECT_HEADER = """
<header>
  <h1>InstantHDR: Single-forward Gaussian Splatting Initialization for HDR 3D Reconstruction</h1>
  <div class="authors">
    <span>Dingqiang Ye<sup>1</sup></span>
    <span>Jiacong Xu<sup>1</sup></span>
    <span>Jianglu Ping<sup>1</sup></span>
    <span>Yuxiang Guo<sup>1</sup></span>
    <span>Chao Fan<sup>2</sup></span>
    <span>Vishal M. Patel<sup>1</sup></span>
  </div>
  <div class="affiliations">
    <span><sup>1</sup>Johns Hopkins University, USA</span>
    <span><sup>2</sup>Shenzhen University, China</span>
  </div>
</header>
"""


def load_example(name):
    directory = EXAMPLES_ROOT / name
    exposure_map = json.loads((directory / "exposure.json").read_text())
    paths = sorted(str(directory / "images" / filename) for filename in exposure_map)
    table = [[Path(p).name, math.log2(float(exposure_map[Path(p).name]))] for p in paths]
    return paths, table


def exposure_readout(ev):
    return f"Selected log2 exposure: **{float(ev):+.2f}** · Linear exposure: **{2.0 ** float(ev):.5g}**"


def build_demo(model, output_root):
    import gradio as gr
    from inference import reconstruct

    def upload(paths):
        paths = sorted(paths or [])
        names = [Path(p).name for p in paths]
        if len(names) != len(set(names)):
            raise gr.Error("Please give uploaded images unique filenames.")
        return paths, [[name, 0.0] for name in names], *invalidate()

    def invalidate():
        return None, gr.Button(interactive=False), "Ready for **Step 1: Reconstruct**.", None, None, None, None

    def select_example(name):
        paths, rows = load_example(name)
        return paths, paths, rows, *invalidate()

    def run(paths, table, render_ev):
        paths = sorted(paths or [])
        try:
            exposure_map = {str(row[0]): 2.0 ** float(row[1]) for row in table}
            exposures = [exposure_map[Path(p).name] for p in paths]
            directory = Path(output_root) / uuid4().hex
            scene, video, depth, preview = reconstruct(model, paths, exposures, directory, render_ev)
            return (video, depth, scene, preview, scene, gr.Button(interactive=True),
                    "**Step 1 complete.** Choose an exposure below, then click **2. Render at selected exposure**.")
        except (ValueError, KeyError, OverflowError) as exc:
            raise gr.Error(f"Check the images and exposure table: {exc}") from exc

    def rerender(scene_path, ev):
        import torch
        from src.model.types import Gaussians, tonemapping
        from src.misc.image_io import save_interpolated_video

        if not scene_path:
            raise gr.Error("Reconstruct a scene first.")
        if not math.isfinite(float(ev)):
            raise gr.Error("Render exposure must be finite.")
        device = next(model.parameters()).device
        data = torch.load(scene_path, map_location=device, weights_only=True)
        gaussians = Gaussians(**data["gaussians"], tone_mapper=tonemapping)
        cameras = data["cameras"]
        h, w = data["image_shape"]
        # A unique path makes browsers refresh the video after exposure changes.
        directory = Path(scene_path).parent / uuid4().hex
        with torch.no_grad():
            video, _, _ = save_interpolated_video(
                cameras["extrinsic"], cameras["intrinsic"], 1, h, w, gaussians,
                str(directory), model.decoder, target_exposure=2.0 ** float(ev),
            )
        return video

    with gr.Blocks(title="InstantHDR") as demo:
        gr.HTML(PROJECT_HEADER, elem_id="project-header")
        gr.Markdown(
            "## Interactive Demo\n"
            "### Instructions\n"
            "1. **Choose your inputs:** click a preset below to load images and their exposure times, "
            "or upload at least two overlapping views of one static scene. For your own images, "
            "fill in each image's **log2 exposure time** (0.25 → −2, 1 → 0, 4 → 2).\n"
            "2. **Reconstruct:** click **1. Reconstruct** and wait for the initial RGB/depth videos.\n"
            "3. **Render:** choose the desired output exposure, then click **2. Render at selected exposure**. "
            "Repeat this step to explore brightness without reconstructing again. "
            "Changing the input images or their exposure times requires a new reconstruction.",
            elem_id="demo-instructions",
        )
        initial_paths, initial_table = load_example("bear")
        state = gr.State(None)
        with gr.Row(elem_id="demo-workspace"):
            with gr.Column(scale=1, min_width=360):
                gr.Markdown("## 1. Reconstruction\nChoose a preset to get started. Bear is already loaded.")
                with gr.Row():
                    presets = [(gr.Button(label, size="sm"), name) for label, name in
                               [("Bear · 4 views", "bear"),
                                ("Chair · 4 views", "chair"), ("Dog · 4 views", "dog")]]
                files = gr.File(label="LDR images", file_count="multiple", file_types=["image"],
                                type="filepath", value=initial_paths)
                gallery = gr.Gallery(label="Input views", columns=3, value=initial_paths)
                table = gr.Dataframe(headers=["Image", "log2 exposure time"],
                                     datatype=["str", "number"], type="array", interactive=True,
                                     value=initial_table)
                submit = gr.Button("1. Reconstruct", variant="primary")
                status = gr.Markdown("Ready for **Step 1: Reconstruct**.")
                gr.Markdown("## 2. Exposure rendering\nAfter reconstruction, adjust the output exposure and click Render. "
                            "Higher values make the result brighter; lower values make it darker.")
                ev = gr.Number(value=0, minimum=-12, maximum=12, step=0.25,
                               label="Render log2 exposure time", precision=2,
                               info="Enter a value from −12 to 12 (for example, −2.25 or 4.00).",
                               elem_id="render-exposure")
                readout = gr.Markdown(exposure_readout(0), elem_id="exposure-readout")
                render = gr.Button("2. Render at selected exposure", variant="primary", interactive=False)
            with gr.Column(scale=1, min_width=360):
                gr.Markdown("## Results\nReconstructed views and scene downloads appear here after Step 1. "
                            "Step 2 updates the novel-view video at your selected exposure.")
                video = gr.Video(label="Novel-view video")
                depth = gr.Video(label="Depth video")
                scene = gr.File(label="HDR scene (complete tensors and cameras)")
                preview = gr.File(label="PLY preview")
        reset_outputs = [state, render, status, video, depth, scene, preview]
        # User-only file events keep programmatic preset updates from resetting exposures.
        for event in (files.upload, files.delete, files.clear):
            event(upload, [files], [gallery, table, *reset_outputs], concurrency_id="gpu")
        table.input(invalidate, [], reset_outputs, concurrency_id="gpu")
        for button, name in presets:
            button.click(partial(select_example, name), [], [files, gallery, table, *reset_outputs],
                         concurrency_id="gpu")
        ev.change(exposure_readout, [ev], [readout], queue=False)
        submit.click(run, [files, table, ev], [video, depth, scene, preview, state, render, status], concurrency_id="gpu")
        render.click(rerender, [state, ev], [video], concurrency_id="gpu")
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/gradio"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    from src.runtime import load_model
    demo = build_demo(load_model(args.checkpoint), args.output)
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, share=args.share, css=DEMO_CSS,
    )


if __name__ == "__main__":
    main()
