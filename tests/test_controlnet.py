import dataclasses
import math
from typing import Tuple

import diffusers
import pytest
import torch
from custom_diffusers import (
    ControlNet2dModel,
    ControlNetOutput,
    UNet2dModel,
    UNet2dModelArgs,
)


@pytest.mark.parametrize(
    "unet_2d_model_args",
    [
        UNet2dModelArgs(),
        UNet2dModelArgs(
            block_out_channels=(32, 64),
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
            attention_head_dim=3,
            out_channels=3,
            in_channels=3,
            layers_per_block=2,
            sample_size=32,
        ),
        UNet2dModelArgs(
            sample_size=32,
            in_channels=4,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(32, 64),
            attention_head_dim=32,
            down_block_types=("DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D"),
        ),
        UNet2dModelArgs(
            block_out_channels=(32, 64, 64, 64),
            in_channels=3,
            layers_per_block=1,
            out_channels=3,
            time_embedding_type="fourier",
            norm_eps=1e-6,
            mid_block_scale_factor=math.sqrt(2.0),
            norm_num_groups=None,
            down_block_types=(
                "SkipDownBlock2D",
                "AttnSkipDownBlock2D",
                "SkipDownBlock2D",
                "SkipDownBlock2D",
            ),
            up_block_types=(
                "SkipUpBlock2D",
                "SkipUpBlock2D",
                "AttnSkipUpBlock2D",
                "SkipUpBlock2D",
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "conditioning_embedding_out_channels", [(16,), (16, 32), (16, 32, 64)]
)
def test_from_unet(
    unet_2d_model_args: UNet2dModelArgs,
    conditioning_embedding_out_channels: Tuple[int],
    conditioning_channels: int = 32,
) -> None:
    condition_scale = 2 ** (len(conditioning_embedding_out_channels) - 1)
    if unet_2d_model_args.sample_size is None:
        sample = torch.randn((4, unet_2d_model_args.in_channels, 32, 32))
        condition = torch.randn(
            (4, conditioning_channels, 32 * condition_scale, 32 * condition_scale)
        )
    elif isinstance(unet_2d_model_args.sample_size, int):
        sample = torch.randn(
            (
                4,
                unet_2d_model_args.in_channels,
                unet_2d_model_args.sample_size,
                unet_2d_model_args.sample_size,
            )
        )
        condition = torch.randn(
            (
                4,
                conditioning_channels,
                unet_2d_model_args.sample_size * condition_scale,
                unet_2d_model_args.sample_size * condition_scale,
            )
        )
    else:
        sample = torch.randn(
            (4, unet_2d_model_args.in_channels, *unet_2d_model_args.sample_size)
        )
        condition = torch.randn(
            (
                4,
                conditioning_channels,
                *(
                    sample_size * condition_scale
                    for sample_size in unet_2d_model_args.sample_size
                ),
            )
        )

    timestep = 1

    diffusers_unet = diffusers.UNet2DModel(**dataclasses.asdict(unet_2d_model_args))

    unet = UNet2dModel.from_unet(diffusers_unet, load_weights_from_diffusers_unet=True)

    controlnet = ControlNet2dModel.from_unet(
        unet,
        conditioning_channels=conditioning_channels,
        conditioning_embedding_out_channels=conditioning_embedding_out_channels,
        load_weights_from_unet=True,
    )
    controlnet_output: ControlNetOutput = controlnet(
        sample=sample, timestep=timestep, controlnet_cond=condition
    )

    expected = diffusers_unet(sample=sample, timestep=timestep).sample
    torch.testing.assert_close(
        expected,
        unet(
            sample=sample,
            timestep=timestep,
            down_block_additional_residuals=controlnet_output.down_block_res_samples,
            mid_block_additional_residual=controlnet_output.mid_block_res_sample,
        ).sample,
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        expected, unet(sample=sample, timestep=timestep).sample, atol=1e-6, rtol=1e-6
    )
