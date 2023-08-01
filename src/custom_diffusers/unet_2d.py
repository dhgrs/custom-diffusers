from typing import List, Optional, Tuple, Union

import diffusers
import torch


class UNet2dModel(diffusers.UNet2DModel):  # type: ignore
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
        down_block_additional_residuals: Optional[List[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[diffusers.models.unet_2d.UNet2DOutput, Tuple]:
        r"""
        The [`UNet2DModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape
                `(batch, channel, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`):
                The number of timesteps to denoise an input.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed
                with the timestep embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d.UNet2DOutput`] instead of
                a plain tuple.

        Returns:
            [`~models.unet_2d.UNet2DOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_2d.UNet2DOutput`] is
                returned, otherwise a `tuple` is returned where the first element is
                the sample tensor.
        """
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
        skip_sample: Optional[torch.Tensor] = sample
        sample = self.conv_in(sample)

        # 3. down
        is_controlnet = (
            mid_block_additional_residual is not None
            and down_block_additional_residuals is not None
        )
        is_adapter = (
            mid_block_additional_residual is None
            and down_block_additional_residuals is not None
        )

        down_block_res_samples: Tuple[torch.Tensor, ...] = (sample,)
        for downsample_block in self.down_blocks:
            if is_adapter and len(down_block_additional_residuals) > 0:  # type: ignore
                sample += down_block_additional_residuals.pop(0)  # type: ignore
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples: Tuple[torch.Tensor, ...] = ()
            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals  # type: ignore
            ):
                down_block_res_sample = (
                    down_block_res_sample + down_block_additional_residual
                )
                new_down_block_res_samples = new_down_block_res_samples + (
                    down_block_res_sample,
                )

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        sample = self.mid_block(sample, emb)

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        # 5. up
        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(
                    sample, res_samples, emb, skip_sample
                )
            else:
                sample = upsample_block(sample, res_samples, emb)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample

        if self.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape(
                (sample.shape[0], *([1] * len(sample.shape[1:])))
            )
            sample = sample / timesteps

        if not return_dict:
            return (sample,)

        return diffusers.models.unet_2d.UNet2DOutput(sample=sample)

    @classmethod
    def from_unet(
        cls,
        diffusers_unet: diffusers.UNet2DModel,
        load_weights_from_diffusers_unet: bool,
    ) -> "UNet2dModel":
        unet = cls(**diffusers_unet.config)

        if load_weights_from_diffusers_unet:
            unet.load_state_dict(diffusers_unet.state_dict())

        return unet
