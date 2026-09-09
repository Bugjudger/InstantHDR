from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, Literal, TypeVar

from jaxtyping import Float
from torch import Tensor, nn

from ..types import Gaussians

DepthRenderingMode = Literal[
    "depth",
    "log",
    "disparity",
    "relative_disparity",
]

from typing import Optional, Dict
ExposureRenderings = Dict[float, Float[Tensor, "batch view 3 height width"]]

@dataclass
class DecoderOutput:
    color: Float[Tensor, "batch view 3 height width"] 
    depth: Float[Tensor, "batch view height width"] | None
    alpha: Float[Tensor, "batch view height width"] | None
    lod_rendering: dict | None
    
    hdr_color: Optional[Float[Tensor, "batch view 3 height width"]] = None # radiance
    linear_rad: Optional[Float[Tensor, "batch view 3 height width"]] = None # radiance
    extra_features: Optional[Float[Tensor, "batch view 4 height width"]] = None # radiance
    # exposure_rendering: ExposureRenderings = field(default_factory=dict)  # 新增：不同曝光的 color

T = TypeVar("T")


class Decoder(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg
    
    @abstractmethod
    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
    ) -> DecoderOutput:
        pass
