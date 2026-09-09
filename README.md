# InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction

Dingqiang Ye, Jiacong Xu, Jianglu Ping, Yuxiang Guo, Chao Fan, and Vishal M. Patel

**ECCV 2026 · Official Implementation**

[Paper](https://arxiv.org/abs/2603.11298) | [Checkpoint](https://huggingface.co/datasets/Bugjudger/HDR_Pretrain/resolve/main/checkpoints/InstantHDR.ckpt) | [Dataset](https://huggingface.co/datasets/Bugjudger/HDR_Pretrain)

## Overview

InstantHDR reconstructs an HDR Gaussian scene from multi-exposure LDR images in a single forward pass, enabling novel-view rendering at different exposures.

<p align="center"><img src="https://raw.githubusercontent.com/Bugjudger/InstantHDR/main/assets/figure1.png" width="100%" alt="Figure 1: Reconstruction and exposure-controlled rendering comparisons of GaussianHDR, AnySplat, and InstantHDR"></p>

## Installation

Requires Linux, Python 3.10, CUDA 12.1, and an NVIDIA Ampere or newer GPU. Run all commands from the repository root.

```bash
conda create -n instanthdr python=3.10 -y
conda activate instanthdr
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install xformers==0.0.24
pip install torch-scatter==2.1.2+pt22cu121 -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install https://github.com/nerfstudio-project/gsplat/releases/download/v1.4.0/gsplat-1.4.0%2Bpt22cu121-cp310-cp310-linux_x86_64.whl
```

## Checkpoint

Download the [InstantHDR checkpoint](https://huggingface.co/datasets/Bugjudger/HDR_Pretrain/resolve/main/checkpoints/InstantHDR.ckpt) from Hugging Face:

```bash
mkdir -p checkpoints
curl -L --fail https://huggingface.co/datasets/Bugjudger/HDR_Pretrain/resolve/main/checkpoints/InstantHDR.ckpt \
  -o checkpoints/InstantHDR.ckpt
export INSTANTHDR_CHECKPOINT="$PWD/checkpoints/InstantHDR.ckpt"
```

## Demo

```bash
python demo.py
```

Open http://127.0.0.1:7860. Try **Bear**, **Chair**, or **Dog** with prefilled exposure values, or upload your own overlapping images and enter their **log2 exposure times**. Bear is loaded by default.

1. **Reconstruct** — Use at least two input views and click **1. Reconstruct** to create the HDR scene.
2. **Render** — Adjust the output exposure and click **2. Render at selected exposure** to explore the scene at different brightness levels.

<details open>
<summary><strong>Preview the demo interface</strong></summary>

<p align="center"><img src="https://raw.githubusercontent.com/Bugjudger/InstantHDR/main/assets/demo.png" width="800" alt="InstantHDR demo interface showing preset inputs, reconstruction results, and exposure rendering controls"></p>

</details>

## Quick Start

Run `inference.py` on the four included **Bear** views. After completing Installation and downloading the checkpoint, run the following from the repository root:

```bash
conda activate instanthdr
python inference.py \
  --checkpoint checkpoints/InstantHDR.ckpt \
  --images examples/bear/images \
  --exposures examples/bear/exposure.json \
  --output outputs/bear \
  --render-ev 0
```

The checkpoint is passed explicitly; `INSTANTHDR_CHECKPOINT` is not needed for this command. The exposure JSON maps filenames to **positive linear exposure times**. Bear uses `2`, `0.125`, `32`, and `0.125` for its four input views. `--render-ev 0` selects an output log2 exposure of 0 (linear exposure 1).

Outputs are saved to:

```text
outputs/bear/
├── rgb.mp4       # Novel-view RGB video
├── depth.mp4     # Depth visualization video
├── scene.pt      # HDR Gaussian tensors and cameras
└── preview.ply   # Gaussian PLY preview
```

To try another included scene, replace both `examples/bear` paths with `examples/chair` or `examples/dog`, and choose a matching output directory.

## Training

Download [HDR-Pretrain](https://huggingface.co/datasets/Bugjudger/HDR_Pretrain) and follow the [data preparation instructions](docs/data.md). Run from the repository root, in the environment prepared during Installation.

Set `DATASET_ROOT` to the directory containing `scenes/`, `train_index.json`, and `test_index.json`, not to `scenes/` itself. The training index includes all 168 scenes; the validation index selects five of them and uses their `test` views.

```bash
DATASET_ROOT=/path/to/HDR-Pretrain
export TMPDIR=/tmp  # Keep data-loader temporary files on local storage.
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=hdr \
  "dataset.hdr.roots=[$DATASET_ROOT]" \
  trainer.max_steps=10000 \
  model.encoder.pretrained_weights=checkpoints/anysplat.safetensors  # Download: https://huggingface.co/lhjiang/anysplat/resolve/main/model.safetensors
```

Choose an available GPU with `CUDA_VISIBLE_DEVICES`. Setting `TMPDIR=/tmp` avoids worker cleanup errors when the default temporary directory is on NFS.

### Configuration

Training uses Hydra to combine the following files. Command-line overrides take precedence over the YAML settings.

| File | Settings |
| --- | --- |
| [config/main.yaml](config/main.yaml) | Entry config, data-loader workers, logging, and trainer defaults |
| [config/experiment/hdr.yaml](config/experiment/hdr.yaml) | HDR experiment selected by `+experiment=hdr`: initialization, losses, learning rate, input resolution, and training schedule |
| [config/dataset/hdr.yaml](config/dataset/hdr.yaml) | Dataset root and camera/data processing settings |
| [config/dataset/view_sampler/arbitrary.yaml](config/dataset/view_sampler/arbitrary.yaml) | View sampling defaults; the HDR experiment overrides the maximum context-view count to 8 |

Edit `config/experiment/hdr.yaml` to adjust training settings. Results are saved to `output/exp_hdr/`. To resume, add `checkpointing.load=/path/to/training.ckpt` with `trainer.max_steps` greater than the saved step.

## Post Optimization

Refine an InstantHDR scene using multi-exposure **LDR images**. The pipeline first refines camera poses for 100 iterations, then optimizes the Gaussians, tone-mapping parameters, and training-camera poses.

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.post_opt.simple_trainer_hdr default \
  --checkpoint checkpoints/InstantHDR.ckpt \
  --data-dir /path/to/scene/images \
  --result-dir outputs/post_opt/example \
  --eval-mode ldr \
  --max-steps 1000
```

Provide `train*`/`test*` images and `exposure.json`; see [data format](docs/data.md). Post optimization requires at least two training views and one test view; the demo presets contain only training views. Use one GPU per scene. Results include checkpoints, renders, and metrics.

Both training and test images participate in camera estimation and initial pose refinement; only training images supervise Gaussian optimization. `hdr` in the script name refers to the scene representation. To evaluate HDR output, use `--eval-mode hdr` with matching EXR references; optimization still uses LDR supervision.

## Dataset

**HDR-Pretrain** is our synthetic dataset for pretraining feed-forward HDR reconstruction models. It contains **168 indoor scenes**, built from HSSD assets and rendered with Blender Cycles. Each scene provides:

- **35 viewpoints** sampled on a 5 × 7 grid, rendered at **448 × 448** resolution.
- **Five exposure levels per view**, paired with **32-bit HDR ground truth**.
- **Depth and normal maps**, with one of **AgX, Filmic, or Standard** tone-mapping operators selected per scene.

<p align="center"><img src="https://raw.githubusercontent.com/Bugjudger/InstantHDR/main/assets/hdr_pretrain.png" width="100%" alt="HDR-Pretrain examples showing multi-view, multi-exposure LDR images, HDR ground truth, depth and normal maps, and different tone-mapping operators"></p>

Download the dataset from [Hugging Face](https://huggingface.co/datasets/Bugjudger/HDR_Pretrain). See the [data preparation instructions](docs/data.md#training) for the directory layout, exposure metadata, and training/validation index format, then follow [Training](#training) to use it.

### Dataset Generation

The Blender scene files and dataset generation code are also available on Hugging Face: [rendering script (`instanthdr_render.py`)](https://huggingface.co/datasets/Bugjudger/HDR_Pretrain/blob/main/instanthdr_render.py) and [batch launcher (`instanthdr.sh`)](https://huggingface.co/datasets/Bugjudger/HDR_Pretrain/blob/main/instanthdr.sh).

The script loads each `.blend` scene, samples a 5 × 7 grid of camera views, and uses Blender Cycles to render multi-exposure LDR images, HDR references, depth, and normals. It also exports camera parameters in COLMAP format and writes `exposure.json`. To generate data, download the source scenes into `blender_scenes/` and keep the scripts at the dataset root. Install Blender 4.5.6, then run `BLENDER=/path/to/blender CUDA_VISIBLE_DEVICES=0 bash instanthdr.sh`; outputs go to `scenes/<scene_id>/`. Exposure levels, resolution, and tone-mapping choices can be adjusted in the rendering script's `CONFIG` dictionary.

## Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{ye2026instanthdr,
  title={InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction},
  author={Ye, Dingqiang and Xu, Jiacong and Ping, Jianglu and Guo, Yuxiang and Fan, Chao and Patel, Vishal M.},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

## Acknowledgements

Built on [AnySplat](https://github.com/InternRobotics/AnySplat), [VGGT](https://github.com/facebookresearch/vggt), [DUSt3R](https://github.com/naver/dust3r), [CroCo](https://github.com/naver/croco), and [gsplat](https://github.com/nerfstudio-project/gsplat). See [third-party licenses](THIRD_PARTY.md).
