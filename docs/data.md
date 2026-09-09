# Data format

All exposure values in JSON files are positive **linear** times. Use consistent units across a scene. Only the interactive demo table and `--render-ev` use log2 values.

## Reconstruction and post-optimization

```text
scene/
├── exposure.json
├── images/
│   ├── train_ldr_000_0.png
│   ├── train_ldr_001_4.png
│   └── test_ldr_000_2.png
└── images_hdr/                 # optional, for --eval-mode hdr
    └── test_hdr_000.exr
```

`exposure.json` maps image basenames to exposure times. PNG, JPG, and JPEG images are supported. Post-optimization uses `train*` images to reconstruct and optimize the scene and `test*` images for evaluation. Provide at least two input views and one evaluation view. Standard inference simply uses every image in its input directory.

HDR references follow the mapping `test_ldr_000_2.png` → `test_hdr_000.exr` (also supported: `test_r_0_2.png` → `test_hdr_000.exr`). The evaluator resizes HDR references to 448 × 448, normalizes by the 99th percentile, and applies μ-law tone mapping with μ = 5000. For aligned comparisons, prepare square images; reconstruction center-crops LDR inputs to a square.

## Training

```text
HDR-Pretrain/
├── train_index.json
├── test_index.json
├── scenes/
│   └── 102343992/
│       ├── exposure.json
│       ├── images/
│       │   ├── train_ldr_000_0.png
│       │   ├── train_ldr_000_1.png
│       │   ├── train_ldr_000_2.png
│       │   ├── train_ldr_000_3.png
│       │   ├── train_ldr_000_4.png
│       │   └── ...
│       ├── sparse/0/
│       │   ├── cameras.bin    # or cameras.txt
│       │   └── images.bin     # or images.txt
│       └── images_hdr/        # optional HDR diagnostics
│           └── train_hdr_000.exr
├── blender_scenes/            # optional: source assets for regeneration
│   └── 102343992.blend
├── instanthdr_render.py
└── instanthdr.sh
```

Each index is a JSON array of scene paths relative to the dataset root:

```json
["scenes/102343992", "scenes/102344022"]
```

The loader reads `train_index.json` for training and `test_index.json` for validation/test. It selects COLMAP image records containing `train` or `test`, respectively. Every indexed scene must have frames for the relevant split.

The packaged `train_index.json` includes all 168 scenes. `test_index.json` selects five of those scenes for validation using their `test` views; these are shared scenes with separate view subsets, not held-out scenes. Each scene has 18 `train` views and 17 `test` views, with five LDR exposures per view. Keep the provided scene IDs when downloading; no renaming is required.

The dataset also includes `depth/`, `normal/`, and `transforms_train.json` / `transforms_test.json` inside each scene. The current training loader does not require these extra files. The `.blend` source scenes are stored in `blender_scenes/`; `instanthdr.sh` and `instanthdr_render.py` remain at the dataset root. The launcher writes rendered data into `scenes/<scene_id>/`.

For each pose, provide five exposure variants, indexed `0` through `4` at the end of the filename, and all five entries in `exposure.json`. COLMAP should contain **one representative image record per pose**, not five duplicate pose records. The loader substitutes the final exposure digit when sampling an image. Filenames encode the numerical view ID in the penultimate underscore-separated field, for example `train_ldr_012_2.png`.

COLMAP cameras must use `PINHOLE` or `SIMPLE_PINHOLE`. Intrinsics are normalized by the camera's stored width and height, and COLMAP world-to-camera poses are converted to camera-to-world transforms. Training resizes images to 448 × 448 and retains the original augmentation and view sampling behavior.

When `images_hdr/` exists, provide the corresponding EXR reference for every sampled pose, named `train_hdr_NNN.exr` or `test_hdr_NNN.exr`. Otherwise omit this directory entirely. The released training loss uses LDR supervision; EXRs support HDR diagnostics.
