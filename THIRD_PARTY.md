# Third-party code

This release derives from the authors' AnySplat-based research implementation. Source headers identify additional upstream components. The root [LICENSE](LICENSE) preserves AnySplat's original MIT notice; it does not replace the licenses of other incorporated components.

| Component | Included code | License notice |
| --- | --- | --- |
| [AnySplat](https://github.com/OpenRobotLab/AnySplat) | Training, model scaffolding, data utilities | [MIT](LICENSE) |
| [VGGT](https://github.com/facebookresearch/vggt) | `src/model/encoder/vggt/` | [VGGT License](licenses/VGGT.txt) |
| DINOv2 and other Apache-licensed VGGT components | Files with Apache-2.0 headers under `src/model/encoder/vggt/layers/` | [Apache-2.0](licenses/Apache-2.0.txt) |
| [CroCo](https://github.com/naver/croco) | `src/model/encoder/backbone/croco/` and related helpers | [CroCo notice](licenses/CroCo.txt), CC BY-NC-SA 4.0 |
| [DUSt3R](https://github.com/naver/dust3r) | DPT heads, geometry, losses, and image utilities bearing Naver headers | [DUSt3R notice](licenses/DUSt3R.txt), CC BY-NC-SA 4.0 |
| [gsplat](https://github.com/nerfstudio-project/gsplat) | Post-optimization trainer, viewer, and helpers | [Apache-2.0](licenses/Apache-2.0.txt) |

The HDR modules and release adaptations modify these components. The release changes explicit weight loading, configurable data paths, demo I/O, trajectory/export handling, and evaluation plumbing while retaining the research model and optimization code. Preserve the source copyright headers and these notices when redistributing derived code. Dependencies and pretrained weights also retain their upstream terms.
