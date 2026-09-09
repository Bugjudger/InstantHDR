"""CPU regression checks for release I/O; CUDA integration is checked separately."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from inference import read_exposures
from src.misc.image_io import save_interpolated_video
from src.model.ply_export import export_ply
from src.post_opt.datasets.colmap import Dataset


class ReleaseIOTests(unittest.TestCase):
    def test_exposure_filename_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exposure.json"
            path.write_text(json.dumps({"b.png": 4, "a.png": 0.25}))
            self.assertEqual(read_exposures(["a.png", "b.png"], path), [0.25, 4])
            path.write_text(json.dumps({"a.png": 0}))
            with self.assertRaises(ValueError):
                read_exposures(["a.png"], path)

    def test_interpolated_path_keeps_endpoints_and_exposure(self):
        class Decoder:
            def forward(self, gaussians, cameras, intrinsics, near, far, shape, **kwargs):
                self.cameras = cameras
                self.exposures = kwargs["target_exposure"]
                count = cameras.shape[1]
                return SimpleNamespace(color=torch.ones(1, count, 3, 4, 4),
                                       depth=torch.ones(1, count, 4, 4))

        cameras = torch.eye(4).repeat(1, 2, 1, 1)
        cameras[0, 1, 0, 3] = 2
        intrinsics = torch.eye(3).repeat(1, 2, 1, 1)
        decoder = Decoder()
        with patch("src.misc.image_io.save_video") as save:
            save_interpolated_video(cameras, intrinsics, 1, 4, 4, None, ".", decoder,
                                    target_exposure=4, t=2)
            self.assertTrue(torch.equal(decoder.cameras[:, 0], cameras[:, 0]))
            self.assertTrue(torch.equal(decoder.cameras[:, -1], cameras[:, -1]))
            self.assertEqual(decoder.cameras.shape[1], 4)
            self.assertTrue(all(t.item() == 4 for t in decoder.exposures))
            self.assertTrue(torch.isfinite(save.call_args_list[0].args[0]).all())

    def test_dataset_intrinsics_do_not_modify_shared_input(self):
        base = np.eye(3, dtype=np.float32)[None]
        intrinsics = np.broadcast_to(base, (2, 3, 3))
        kwargs = dict(images=np.zeros((2, 3, 4, 4), dtype=np.float32),
                      camtoworlds=np.repeat(np.eye(4)[None], 2, axis=0),
                      Ks=intrinsics, exposure=np.ones(2))
        first = Dataset(**kwargs)
        second = Dataset(**kwargs)
        self.assertEqual(base[0, 0, 0], 1)
        np.testing.assert_array_equal(first.Ks, second.Ks)
        self.assertEqual(first.Ks[0, 0, 0], 4)

    def test_ply_opacity_is_encoded_as_logit(self):
        from plyfile import PlyData
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preview.ply"
            export_ply(torch.zeros(2, 3), torch.ones(2, 3),
                       torch.tensor([[0., 0., 0., 1.]]).repeat(2, 1),
                       torch.zeros(2, 3, 25), torch.tensor([0.25, 0.75]), path, is_hdr=True)
            encoded = torch.from_numpy(PlyData.read(path)["vertex"]["opacity"].copy())
            torch.testing.assert_close(encoded.sigmoid(), torch.tensor([0.25, 0.75]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA rasterization check")
    def test_post_opt_preserves_gaussian_rotation(self):
        from src.post_opt.simple_trainer_hdr import rasterization
        from src.model.encoder.common.gaussians import build_covariance

        means = torch.tensor([[0., 0., 3.]], device="cuda")
        scales = torch.tensor([[0.1, 0.3, 0.2]], device="cuda")
        quats = torch.tensor([[0., 0., 0.38268343, 0.92387953]], device="cuda")
        opacity = torch.tensor([0.8], device="cuda")
        colors = torch.tensor([[0.8, 0.1, 0.2]], device="cuda")
        camera = torch.eye(4, device="cuda")[None]
        intrinsics = torch.tensor([[[60., 0., 32.], [0., 60., 32.], [0., 0., 1.]]], device="cuda")
        args = (means, quats, scales, opacity, colors, camera, intrinsics, 64, 64)
        direct = rasterization(*args, sh_degree=None, packed=False)[0]
        explicit = rasterization(*args, sh_degree=None, packed=False,
                                 covars=build_covariance(scales, quats))[0]
        torch.testing.assert_close(direct, explicit, atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
