from .loss import Loss
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_depth_consis import LossDepthConsis, LossDepthConsisCfgWrapper

LOSSES = {
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
    LossDepthConsisCfgWrapper: LossDepthConsis,
}
LossCfgWrapper = LossLpipsCfgWrapper | LossMseCfgWrapper | LossDepthConsisCfgWrapper


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
