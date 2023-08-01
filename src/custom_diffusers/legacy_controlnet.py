from typing import Optional, Tuple, Type, Union

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.controlnet import (
    ControlNetConditioningEmbedding,
    ControlNetOutput,
    zero_module,
)
from diffusers.models.embeddings import (
    GaussianFourierProjection,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unet_2d_blocks import UNetMidBlock2D, get_down_block

from .unet_2d import UNet2dModel


class ControlNet2dModel(ModelMixin, ConfigMixin):  # type: ignore
    @register_to_config
    def __init__(
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 3,
        conditioning_channels: int = 3,
        center_input_sample: bool = False,
        time_embedding_type: str = "positional",
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        down_block_types: Tuple[str, ...] = (
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types: Tuple[str, ...] = (
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ),
        block_out_channels: Tuple[int, ...] = (224, 448, 672, 896),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        act_fn: str = "silu",
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        add_attention: bool = True,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        conditioning_embedding_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ) -> None:
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                "Must provide the same number of `down_block_types` as "
                f"`up_block_types`. `down_block_types`: {down_block_types}. "
                f"`up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                "Must provide the same number of `block_out_channels` as "
                f"`down_block_types`. `block_out_channels`: {block_out_channels}. "
                f"`down_block_types`: {down_block_types}."
            )

        # input
        self.conv_in = torch.nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1)
        )

        # time
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(
                embedding_size=block_out_channels[0], scale=16
            )
            timestep_input_dim = 2 * block_out_channels[0]
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(
                block_out_channels[0], flip_sin_to_cos, freq_shift
            )
            timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # class embedding
        self.class_embedding: Optional[torch.nn.Module]
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = torch.nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = torch.nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.class_embedding = None

        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=block_out_channels[0],
            block_out_channels=conditioning_embedding_out_channels,
            conditioning_channels=conditioning_channels,
        )
        self.down_blocks = torch.nn.ModuleList()
        self.controlnet_down_blocks = torch.nn.ModuleList()

        # down
        output_channel = block_out_channels[0]

        controlnet_block = torch.nn.Conv2d(
            output_channel, output_channel, kernel_size=1
        )
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_down_blocks.append(controlnet_block)

        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=attention_head_dim
                if attention_head_dim is not None
                else output_channel,
                downsample_padding=downsample_padding,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
            self.down_blocks.append(down_block)

            for _ in range(layers_per_block):
                controlnet_block = torch.nn.Conv2d(
                    output_channel, output_channel, kernel_size=1
                )
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)

            if not is_final_block:
                controlnet_block = torch.nn.Conv2d(
                    output_channel, output_channel, kernel_size=1
                )
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)

        # mid
        mid_block_channel = block_out_channels[-1]

        controlnet_block = torch.nn.Conv2d(
            mid_block_channel, mid_block_channel, kernel_size=1
        )
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_mid_block = controlnet_block

        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attn_num_head_channels=attention_head_dim
            if attention_head_dim is not None
            else block_out_channels[-1],
            resnet_groups=norm_num_groups,
            add_attention=add_attention,
        )

    @classmethod
    def from_unet(
        cls: Type["ControlNet2dModel"],
        unet: UNet2dModel,
        conditioning_channels: int,
        conditioning_embedding_out_channels: Tuple[int],
        load_weights_from_unet: bool,
    ) -> "ControlNet2dModel":
        controlnet = cls(
            sample_size=unet.config.sample_size,
            in_channels=unet.config.in_channels,
            conditioning_channels=conditioning_channels,
            center_input_sample=unet.config.center_input_sample,
            time_embedding_type=unet.config.time_embedding_type,
            freq_shift=unet.config.freq_shift,
            flip_sin_to_cos=unet.config.flip_sin_to_cos,
            down_block_types=unet.config.down_block_types,
            up_block_types=unet.config.up_block_types,
            block_out_channels=unet.config.block_out_channels,
            layers_per_block=unet.config.layers_per_block,
            mid_block_scale_factor=unet.config.mid_block_scale_factor,
            downsample_padding=unet.config.downsample_padding,
            act_fn=unet.config.act_fn,
            attention_head_dim=unet.config.attention_head_dim,
            norm_num_groups=unet.config.norm_num_groups,
            norm_eps=unet.config.norm_eps,
            resnet_time_scale_shift=unet.config.resnet_time_scale_shift,
            add_attention=unet.config.add_attention,
            class_embed_type=unet.config.class_embed_type,
            num_class_embeds=unet.config.num_class_embeds,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
        )

        if load_weights_from_unet:
            controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())
            controlnet.time_proj.load_state_dict(unet.time_proj.state_dict())
            controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())

            if controlnet.class_embedding is not None:
                assert unet.class_embedding is not None
                controlnet.class_embedding.load_state_dict(
                    unet.class_embedding.state_dict()
                )

            controlnet.down_blocks.load_state_dict(unet.down_blocks.state_dict())
            controlnet.mid_block.load_state_dict(unet.mid_block.state_dict())

        return controlnet  # type: ignore

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[ControlNetOutput, Tuple]:
        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):  # type: ignore
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:  # type: ignore
            assert isinstance(timesteps, torch.Tensor)
            timesteps = timesteps[None].to(sample.device)
        assert isinstance(timesteps, torch.Tensor)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(
            sample.shape[0], dtype=timesteps.dtype, device=timesteps.device
        )

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError(
                    "class_labels should be provided when doing class conditioning"
                )

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # 2. pre-process
        skip_sample = sample
        sample = self.conv_in(sample)

        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
        sample = sample + controlnet_cond

        # 3. down
        down_block_res_samples: Tuple[torch.Tensor, ...] = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(sample, emb)

        # 5. Control net blocks

        controlnet_down_block_res_samples: Tuple[torch.Tensor, ...] = ()

        for down_block_res_sample, controlnet_block in zip(
            down_block_res_samples, self.controlnet_down_blocks
        ):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples = controlnet_down_block_res_samples + (
                down_block_res_sample,
            )

        down_block_res_samples = controlnet_down_block_res_samples

        mid_block_res_sample = self.controlnet_mid_block(sample)

        if not return_dict:
            return (down_block_res_samples, mid_block_res_sample)

        return ControlNetOutput(
            down_block_res_samples=down_block_res_samples,
            mid_block_res_sample=mid_block_res_sample,
        )
