"""Reconstruct an HDR scene and render an interpolated camera path."""

import argparse
import json
import math
from pathlib import Path


def read_exposures(image_paths, exposure_file):
    """JSON maps image basenames to positive, linear exposure times."""
    data = json.loads(Path(exposure_file).read_text())
    values = []
    for path in image_paths:
        value = float(data[Path(path).name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Exposure must be finite and positive: {path}")
        values.append(value)
    return values


def reconstruct(model, image_paths, exposures, output_dir, render_ev=0.0):
    import torch
    from dataclasses import fields
    from src.misc.image_io import save_interpolated_video
    from src.model.ply_export import export_ply
    from src.utils.image import process_image

    if len(image_paths) < 2:
        raise ValueError("Provide at least two overlapping views of the same static scene.")
    if len(image_paths) != len(exposures):
        raise ValueError("Provide one exposure time for each image.")
    if any(not math.isfinite(float(t)) or t <= 0 for t in exposures):
        raise ValueError("Exposure times must be finite and positive.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    images = torch.stack([process_image(str(p)) for p in image_paths])[None].to(device)
    times = [torch.tensor([t], dtype=torch.float32, device=device) for t in exposures]
    with torch.no_grad():
        gaussians, cameras = model.inference((images + 1) * 0.5, times)
        b, _, _, h, w = images.shape
        video, depth, _ = save_interpolated_video(
            cameras["extrinsic"], cameras["intrinsic"], b, h, w,
            gaussians, str(output_dir), model.decoder, target_exposure=2.0 ** render_ev,
        )
        # Preserve the complete HDR representation; a PLY cannot store the tone mapper.
        state = {f.name: getattr(gaussians, f.name).detach().cpu()
                 for f in fields(gaussians) if isinstance(getattr(gaussians, f.name), torch.Tensor)}
        torch.save({"gaussians": state,
                    "cameras": {k: v.detach().cpu() for k, v in cameras.items()},
                    "image_names": [Path(p).name for p in image_paths],
                    "exposures": list(exposures), "image_shape": [h, w]}, output_dir / "scene.pt")
        export_ply(gaussians.means[0], gaussians.scales[0], gaussians.rotations[0],
                   gaussians.harmonics[0], gaussians.opacities[0],
                   output_dir / "preview.ply", save_sh_dc_only=True, is_hdr=True)
    return str(output_dir / "scene.pt"), video, depth, str(output_dir / "preview.ply")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Directory of overlapping LDR images")
    parser.add_argument("--exposures", type=Path, required=True, help="JSON mapping basenames to linear exposure times")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--render-ev", type=float, default=0.0, help="Target log2 exposure time")
    args = parser.parse_args()
    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    exposures = read_exposures(paths, args.exposures)
    from src.runtime import load_model
    outputs = reconstruct(load_model(args.checkpoint), paths, exposures, args.output, args.render_ev)
    print("\n".join(outputs))


if __name__ == "__main__":
    main()
