import diffusers
import packaging

if packaging.version.parse(diffusers.__version__) < packaging.version.parse("0.18.0"):
    from .legacy_controlnet import ControlNet2dModel, ControlNetOutput
else:
    from .controlnet import ControlNet2dModel, ControlNetOutput  # type: ignore

from .unet_2d import UNet2dModel

__all__ = ["ControlNet2dModel", "ControlNetOutput", "UNet2dModel"]
