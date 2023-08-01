import dataclasses
import math
from typing import Optional, Tuple, Union

import diffusers
import packaging
import pytest
import torch
from custom_diffusers import UNet2dModel

@dataclasses.dataclass
class UNet2dModelArgs:  # type: ignore
    sample_size: Optional[Union[int, Tuple[int, int]]] = None
    in_channels: int = 3
    out_channels: int = 3
    center_input_sample: bool = False
    time_embedding_type: str = "positional"
    freq_shift: int = 0
    flip_sin_to_cos: bool = True
    down_block_types: Tuple[str, ...] = (
        "DownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    )
    up_block_types: Tuple[str, ...] = (
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
    )
    block_out_channels: Tuple[int, ...] = (224, 448, 672, 896)
    layers_per_block: int = 2
    mid_block_scale_factor: float = 1
    downsample_padding: int = 1
    downsample_type: str = "conv"
    upsample_type: str = "conv"
    act_fn: str = "silu"
    attention_head_dim: Optional[int] = 8
    norm_num_groups: Optional[int] = 32
    norm_eps: float = 1e-5
    resnet_time_scale_shift: str = "default"
    add_attention: bool = True
    class_embed_type: Optional[str] = None
    num_class_embeds: Optional[int] = None


@pytest.mark.skipif(packaging.version.parse(diffusers.__version__) < packaging.version.parse("0.19.0"))
@pytest.mark.parametrize(
    "unet_2d_model_args",
    [
UNet2dModelArgs(),
        UNet2dModelArgs(
            block_out_channels=(32, 64),
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
            attention_head_dim=None,
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
def test_from_unet(
    unet_2d_model_args: UNet2dModelArgs,
    conditioning_channels: int = 32
) -> None:
    t2i_adapter = diffusers.T2IAdapter(
        conditioning_channels,
        channels=unet_2d_model_args.block_out_channels[:1]+unet_2d_model_args.block_out_channels[:-1],
        num_res_blocks=unet_2d_model_args.layers_per_block,
        downscale_factor=1,
        adapter_type="full_adapter"
    )
    if unet_2d_model_args.sample_size is None:
        sample = torch.randn((4, unet_2d_model_args.in_channels, 32, 32))
        condition = torch.randn(
            (4, conditioning_channels, 32, 32)
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
                unet_2d_model_args.sample_size,
                unet_2d_model_args.sample_size,
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
                    sample_size
                    for sample_size in unet_2d_model_args.sample_size
                ),
            )
        )

    timestep = 1

    diffusers_unet = diffusers.UNet2DModel(**dataclasses.asdict(unet_2d_model_args))

    unet = UNet2dModel.from_unet(diffusers_unet, load_weights_from_diffusers_unet=True)

    t2i_adapter_outputs = t2i_adapter(condition)

    expected = diffusers_unet(sample=sample, timestep=timestep).sample
    torch.testing.assert_close(
        expected,
        unet(
            sample=sample,
            timestep=timestep,
            down_block_additional_residuals=[torch.zeros_like(t2i_adapter_output) for t2i_adapter_output in t2i_adapter_outputs],
            mid_block_additional_residual=None,
        ).sample,
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        expected, unet(sample=sample, timestep=timestep).sample, atol=1e-6, rtol=1e-6
    )
