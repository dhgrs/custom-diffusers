import diffusers
import packaging

if packaging.version.parse(diffusers.__version__) < packaging.version.parse("0.18.0"):
    from .legacy_controlnet import (
        ControlNet2dModel,
        ControlNetOutput,
        UNet2dModel,
        UNet2dModelArgs,
        UNet2DOutput,
    )
else:
    from .controlnet import (
        ControlNet2dModel,
        ControlNetOutput,
        UNet2dModel,
        UNet2dModelArgs,
        UNet2DOutput,
    )


__all__ = [
    "ControlNet2dModel",
    "ControlNetOutput",
    "UNet2dModel",
    "UNet2DOutput",
    "UNet2dModelArgs",
]
