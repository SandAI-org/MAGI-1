# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import math
import os
from typing import Tuple

import torch
import torch.distributed
import torch.nn as nn
from einops import rearrange

from inference.common import (
    InferenceParams,
    MagiConfig,
    ModelMetaArgs,
    PackedCoreAttnParams,
    PackedCrossAttnParams,
    env_is_true,
    print_per_rank,
    print_rank_0,
)
from inference.infra.checkpoint import load_checkpoint
from inference.infra.distributed import parallel_state as mpu
from inference.infra.parallelism import cp_post_process, cp_pre_process, pp_scheduler

from .dit_module import CaptionEmbedder, FinalLinear, LearnableRotaryEmbeddingCat, TimestepEmbedder, TransformerBlock


class VideoDiTModel(torch.nn.Module):
    """VideoDiT model for video diffusion.

    Args:
        config (MagiConfig): Transformer config
        pre_process (bool, optional): Include embedding layer (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional): Include an output layer (used with pipeline parallelism). Defaults to True.
    """

    def __init__(self, config: MagiConfig, pre_process: bool = True, post_process: bool = True) -> None:
        super().__init__()

        self.model_config = config.model_config
        self.runtime_config = config.runtime_config
        self.engine_config = config.engine_config

        self.pre_process = pre_process
        self.post_process = post_process
        self.in_channels = self.model_config.in_channels
        self.out_channels = self.model_config.out_channels
        self.patch_size = self.model_config.patch_size
        self.t_patch_size = self.model_config.t_patch_size
        self.caption_max_length = self.model_config.caption_max_length
        self.num_heads = self.model_config.num_attention_heads

        self.x_embedder = nn.Conv3d(
            self.model_config.in_channels,
            self.model_config.hidden_size,
            kernel_size=(self.model_config.t_patch_size, self.model_config.patch_size, self.model_config.patch_size),
            stride=(self.model_config.t_patch_size, self.model_config.patch_size, self.model_config.patch_size),
            bias=False,
        )
        self.t_embedder = TimestepEmbedder(model_config=self.model_config)
        self.y_embedder = CaptionEmbedder(model_config=self.model_config)
        self.rope = LearnableRotaryEmbeddingCat(
            self.model_config.hidden_size // self.model_config.num_attention_heads, in_pixels=False
        )

        # trm block
        self.videodit_blocks = TransformerBlock(
            model_config=self.model_config,
            engine_config=self.engine_config,
            pre_process=pre_process,
            post_process=post_process,
        )

        self.final_linear = FinalLinear(
            self.model_config.hidden_size, self.model_config.patch_size, self.model_config.t_patch_size, self.out_channels
        )

    def generate_kv_range_for_uncondition(self, uncond_x) -> torch.Tensor:
        device = f"cuda:{torch.cuda.current_device()}"
        B, C, T, H, W = uncond_x.shape
        chunk_token_nums = (
            (T // self.model_config.t_patch_size) * (H // self.model_config.patch_size) * (W // self.model_config.patch_size)
        )

        k_chunk_start = torch.linspace(0, (B - 1) * chunk_token_nums, steps=B).reshape((B, 1))
        k_chunk_end = torch.linspace(chunk_token_nums, B * chunk_token_nums, steps=B).reshape((B, 1))
        return torch.concat([k_chunk_start, k_chunk_end], dim=1).to(torch.int32).to(device)

    def unpatchify(self, x, H, W):
        return rearrange(
            x,
            "(T H W) N (pT pH pW C) -> N C (T pT) (H pH) (W pW)",
            H=H,
            W=W,
            pT=self.t_patch_size,
            pH=self.patch_size,
            pW=self.patch_size,
        ).contiguous()

    @torch.no_grad()
    def get_embedding_and_meta(self, x, t, y, caption_dropout_mask, xattn_mask, kv_range, **kwargs):
        """
        Forward embedding and meta for VideoDiT.
        NOTE: This function should only handle single card behavior.

        Input:
            x: (N, C, T, H, W). torch.Tensor of spatial inputs (images or latent representations of images)
            t: (N, denoising_range_num). torch.Tensor of diffusion timesteps
            y: (N * denoising_range_num, 1, L, C). torch.Tensor of class labels
            caption_dropout_mask: (N). torch.Tensor of whether to drop caption
            xattn_mask: (N * denoising_range_num, 1, L). torch.Tensor of xattn mask
            kv_range: (N * denoising_range_num, 2). torch.Tensor of kv range

        Output:
            x: (S, N, D). torch.Tensor of inputs embedding (images or latent representations of images)
            condition: (N, denoising_range_num, D). torch.Tensor of condition embedding
            condition_map: (S, N). torch.Tensor determine which condition to use for each token
            rope: (S, 96). torch.Tensor of rope
            y_xattn_flat: (total_token, D). torch.Tensor of y_xattn_flat
            cuda_graph_inputs: (y_xattn_flat, xattn_mask) or None. None means no cuda graph
                NOTE: y_xattn_flat and xattn_mask with static shape
            H: int. Height of the input
            W: int. Width of the input
            ardf_meta: dict. Meta information for ardf
            cross_attn_params: PackedCrossAttnParams. Packed sequence parameters for cross_atten
        """

        ###################################
        #          Part1: Embed x         #
        ###################################
        x = self.x_embedder(x)  # [N, C, T, H, W]
        batch_size, _, T, H, W = x.shape

        # Prepare necessary variables
        range_num = kwargs["range_num"]
        denoising_range_num = kwargs["denoising_range_num"]
        slice_point = kwargs.get("slice_point", 0)
        frame_in_range = T // denoising_range_num
        prev_clean_T = frame_in_range * slice_point
        T_total = T + prev_clean_T

        ###################################
        #          Part2: rope            #
        ###################################
        # caculate rescale_factor for multi-resolution & multi aspect-ratio training
        # the base_size [16*16] is A predefined size based on data:(256x256)  vae: (8,8,4) patch size: (1,1,2)
        # This definition do not have any relationship with the actual input/model/setting.
        # ref_feat_shape is used to calculate innner rescale factor, so it can be float.
        rescale_factor = math.sqrt((H * W) / (16 * 16))
        rope = self.rope.get_embed(shape=[T_total, H, W], ref_feat_shape=[T_total, H / rescale_factor, W / rescale_factor])
        # the shape of rope is (T*H*W, -1) aka (seq_length, head_dim), as T is the first dimension, we can directly cut it.
        rope = rope[-(T * H * W) :]

        ###################################
        #          Part3: Embed t         #
        ###################################
        assert t.shape[0] == batch_size, f"Invalid t shape, got {t.shape[0]} != {batch_size}"  # nolint
        assert t.shape[1] == denoising_range_num, f"Invalid t shape, got {t.shape[1]} != {denoising_range_num}"  # nolint
        t_flat = t.flatten()  # (N * denoising_range_num,)
        t = self.t_embedder(t_flat)  # (N, D)

        if self.engine_config.distill:
            distill_dt_scalar = 2
            if kwargs["num_steps"] == 12:
                base_chunk_step = 4
                distill_dt_factor = base_chunk_step / kwargs["distill_interval"] * distill_dt_scalar
            else:
                distill_dt_factor = kwargs["num_steps"] / 4 * distill_dt_scalar
            distill_dt = torch.ones_like(t_flat) * distill_dt_factor
            distill_dt_embed = self.t_embedder(distill_dt)
            t = t + distill_dt_embed
        t = t.reshape(batch_size, denoising_range_num, -1)  # (N, range_num, D)

        ######################################################
        # Part4: Embed y, prepare condition and y_xattn_flat #
        ######################################################
        # (N * denoising_range_num, 1, L, D)
        y_xattn, y_adaln = self.y_embedder(y, self.training, caption_dropout_mask)

        assert xattn_mask is not None
        xattn_mask = xattn_mask.squeeze(1).squeeze(1)

        # condition: (N, range_num, D)
        y_adaln = y_adaln.squeeze(1)  # (N, D)
        condition = t + y_adaln.unsqueeze(1)

        assert condition.shape[0] == batch_size
        assert condition.shape[1] == denoising_range_num
        seqlen_per_chunk = (T * H * W) // denoising_range_num
        condition_map = torch.arange(batch_size * denoising_range_num, device=x.device)
        condition_map = torch.repeat_interleave(condition_map, seqlen_per_chunk)
        condition_map = condition_map.reshape(batch_size, -1).transpose(0, 1).contiguous()

        # y_xattn_flat: (total_token, D)
        y_xattn_flat = torch.masked_select(y_xattn.squeeze(1), xattn_mask.unsqueeze(-1).bool()).reshape(-1, y_xattn.shape[-1])
        xattn_mask_for_cuda_graph = None

        ######################################################
        # Part5: Prepare cross_attn_params for cross_atten   #
        ######################################################
        # (N * denoising_range_num, L)
        xattn_mask = xattn_mask.reshape(xattn_mask.shape[0], -1)
        y_index = torch.sum(xattn_mask, dim=-1)
        clip_token_nums = H * W * frame_in_range

        cu_seqlens_q = torch.Tensor([0] + ([clip_token_nums] * denoising_range_num * batch_size)).to(torch.int64).to(x.device)
        cu_seqlens_k = torch.cat([y_index.new_tensor([0]), y_index]).to(torch.int64).to(x.device)
        cu_seqlens_q = cu_seqlens_q.cumsum(-1).to(torch.int32)
        cu_seqlens_k = cu_seqlens_k.cumsum(-1).to(torch.int32)
        assert (
            cu_seqlens_q.shape == cu_seqlens_k.shape
        ), f"cu_seqlens_q.shape: {cu_seqlens_q.shape}, cu_seqlens_k.shape: {cu_seqlens_k.shape}"

        xattn_q_ranges = torch.cat([cu_seqlens_q[:-1].unsqueeze(1), cu_seqlens_q[1:].unsqueeze(1)], dim=1)
        xattn_k_ranges = torch.cat([cu_seqlens_k[:-1].unsqueeze(1), cu_seqlens_k[1:].unsqueeze(1)], dim=1)
        assert (
            xattn_q_ranges.shape == xattn_k_ranges.shape
        ), f"xattn_q_ranges.shape: {xattn_q_ranges.shape}, xattn_k_ranges.shape: {xattn_k_ranges.shape}"

        cross_attn_params = PackedCrossAttnParams(
            q_ranges=xattn_q_ranges,
            kv_ranges=xattn_k_ranges,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_k,
            max_seqlen_q=clip_token_nums,
            max_seqlen_kv=self.caption_max_length,
        )

        ##################################################
        #  Part6: Prepare core_atten related q/kv range  #
        ##################################################
        q_range = torch.cat([cu_seqlens_q[:-1].unsqueeze(1), cu_seqlens_q[1:].unsqueeze(1)], dim=1)
        flat_kv = torch.unique(kv_range, sorted=True)
        max_seqlen_k = (flat_kv[-1] - flat_kv[0]).cpu().item()

        ardf_meta = dict(
            clip_token_nums=clip_token_nums,
            slice_point=slice_point,
            range_num=range_num,
            denoising_range_num=denoising_range_num,
            q_range=q_range,
            k_range=kv_range,
            max_seqlen_q=clip_token_nums,
            max_seqlen_k=max_seqlen_k,
        )

        return (x, condition, condition_map, rope, y_xattn_flat, xattn_mask_for_cuda_graph, H, W, ardf_meta, cross_attn_params)

    @torch.no_grad()
    def forward_pre_process(
        self, x, t, y, caption_dropout_mask=None, xattn_mask=None, kv_range=None, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ModelMetaArgs]:
        assert kv_range is not None, "Please ensure kv_range is provided"

        x = x * self.model_config.x_rescale_factor

        if self.model_config.half_channel_vae:
            assert x.shape[1] == 16
            x = torch.cat([x, x], dim=1)

        x = x.float()
        t = t.float()
        y = y.float()
        # embedder context will ensure that the processing is in high precision even if the embedder params is in bfloat16 mode
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            (
                x,
                condition,
                condition_map,
                rope,
                y_xattn_flat,
                xattn_mask_for_cuda_graph,
                H,
                W,
                ardf_meta,
                cross_attn_params,
            ) = self.get_embedding_and_meta(x, t, y, caption_dropout_mask, xattn_mask, kv_range, **kwargs)

        # Downcast x and rearrange x
        x = x.to(self.model_config.params_dtype)
        x = rearrange(x, "N C T H W -> (T H W) N C").contiguous()  # (thw, N, D)

        # condition and y_xattn_flat will be downcast to bfloat16 in transformer block.
        condition = condition.to(self.model_config.params_dtype)
        y_xattn_flat = y_xattn_flat.to(self.model_config.params_dtype)

        core_attn_params = PackedCoreAttnParams(
            q_range=ardf_meta["q_range"],
            k_range=ardf_meta["k_range"],
            np_q_range=ardf_meta["q_range"].cpu().numpy(),
            np_k_range=ardf_meta["k_range"].cpu().numpy(),
            max_seqlen_q=ardf_meta["max_seqlen_q"],
            max_seqlen_k=ardf_meta["max_seqlen_k"],
        )

        (x, condition_map, rope, cp_pad_size, cp_split_sizes, core_attn_params, cross_attn_params) = cp_pre_process(
            self.engine_config.cp_size,
            self.engine_config.cp_strategy,
            x,
            condition_map,
            rope,
            xattn_mask_for_cuda_graph,
            ardf_meta,
            core_attn_params,
            cross_attn_params,
        )

        meta_args = ModelMetaArgs(
            H=H,
            W=W,
            cp_pad_size=cp_pad_size,
            cp_split_sizes=cp_split_sizes,
            slice_point=ardf_meta["slice_point"],
            denoising_range_num=ardf_meta["denoising_range_num"],
            range_num=ardf_meta["range_num"],
            extract_prefix_video_feature=kwargs.get("extract_prefix_video_feature", False),
            fwd_extra_1st_chunk=kwargs["fwd_extra_1st_chunk"],
            distill_nearly_clean_chunk=kwargs.get("distill_nearly_clean_chunk", False),
            clip_token_nums=ardf_meta["clip_token_nums"],
            enable_cuda_graph=xattn_mask_for_cuda_graph is not None,
            core_attn_params=core_attn_params,
            cross_attn_params=cross_attn_params,
        )

        return (x, condition, condition_map, y_xattn_flat, rope, meta_args)

    @torch.no_grad()
    def forward_post_process(self, x, meta_args: ModelMetaArgs) -> torch.Tensor:
        x = x.float()
        # embedder context will ensure that the processing is in high precision even if the embedder params is in bfloat16 mode
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            x = self.final_linear(x)  # (thw/cp, N, patch_size ** 2 * out_channels)

        # leave context parallel region
        x = cp_post_process(self.engine_config.cp_size, self.engine_config.cp_strategy, x, meta_args)

        # N C T H W
        x = self.unpatchify(x, meta_args.H, meta_args.W)

        if self.model_config.half_channel_vae:
            assert x.shape[1] == 32
            x = x[:, :16]

        x = x / self.model_config.x_rescale_factor

        return x

    @torch.no_grad()
    def forward(
        self,
        x,
        t,
        y,
        caption_dropout_mask=None,
        xattn_mask=None,
        kv_range=None,
        inference_params: InferenceParams = None,
        **kwargs,
    ) -> torch.Tensor:
        (x, condition, condition_map, y_xattn_flat, rope, meta_args) = self.forward_pre_process(
            x, t, y, caption_dropout_mask, xattn_mask, kv_range, **kwargs
        )

        if not self.pre_process:
            x = pp_scheduler().recv_prev_data(x.shape, x.dtype)
            self.videodit_blocks.set_input_tensor(x)
        else:
            # clone a new tensor to ensure x is not a view of other tensor
            x = x.clone()

        x = self.videodit_blocks.forward(
            hidden_states=x,
            condition=condition,
            condition_map=condition_map,
            y_xattn_flat=y_xattn_flat,
            rotary_pos_emb=rope,
            inference_params=inference_params,
            meta_args=meta_args,
        )

        if not self.post_process:
            pp_scheduler().isend_next(x)

        return self.forward_post_process(x, meta_args)

    def forward_3cfg(
        self, x, timestep, y, mask, kv_range, inference_params, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Forward pass of PixArt, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb

        assert x.shape[0] == 2
        assert mask.shape[0] % 2 == 0  # mask should be a multiple of 2
        x = torch.cat([x[0:1], x[0:1]], dim=0)
        caption_dropout_mask = torch.tensor([False, True], dtype=torch.bool, device=x.device)

        inference_params.update_kv_cache = False
        out_cond_pre_and_text = self.forward(
            x[0:1],
            timestep[0:1],
            y[0 : y.shape[0] // 2],
            caption_dropout_mask=caption_dropout_mask[0:1],
            xattn_mask=mask[0 : y.shape[0] // 2],
            kv_range=kv_range,
            inference_params=inference_params,
            **kwargs,
        )

        inference_params.update_kv_cache = True
        out_cond_pre = self.forward(
            x[1:2],
            timestep[1:2],
            y[y.shape[0] // 2 : y.shape[0]],
            caption_dropout_mask=caption_dropout_mask[1:2],
            xattn_mask=mask[y.shape[0] // 2 : y.shape[0]],
            kv_range=kv_range,
            inference_params=inference_params,
            **kwargs,
        )

        def chunk_to_batch(input, denoising_range_num):
            input = input.squeeze(0)
            input = input.reshape(-1, denoising_range_num, kwargs["chunk_width"], *input.shape[2:])
            return input.transpose(0, 1)  # (denoising_range_num, chn, chunk_width, h, w)

        def batch_to_chunk(input, denoising_range_num):
            input = input.transpose(0, 1)
            input = input.reshape(1, -1, denoising_range_num * kwargs["chunk_width"], *input.shape[3:])
            return input

        class UnconditionGuard:
            def __init__(self, kwargs):
                self.kwargs = kwargs
                self.prev_state = {
                    "range_num": kwargs["range_num"],
                    "denoising_range_num": kwargs["denoising_range_num"],
                    "slice_point": kwargs["slice_point"],
                    "fwd_extra_1st_chunk": kwargs["fwd_extra_1st_chunk"],
                }

            def __enter__(self):
                if self.kwargs.get("fwd_extra_1st_chunk", False):
                    self.kwargs["denoising_range_num"] -= 1
                    self.kwargs["slice_point"] += 1
                    self.kwargs["fwd_extra_1st_chunk"] = False

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.kwargs["range_num"] = self.prev_state["range_num"]
                self.kwargs["denoising_range_num"] = self.prev_state["denoising_range_num"]
                self.kwargs["slice_point"] = self.prev_state["slice_point"]
                self.kwargs["fwd_extra_1st_chunk"] = self.prev_state["fwd_extra_1st_chunk"]

        with UnconditionGuard(kwargs):
            denoising_range_num = kwargs["denoising_range_num"]
            denoise_width = kwargs["chunk_width"] * denoising_range_num
            uncond_x = chunk_to_batch(x[0:1, :, -denoise_width:], denoising_range_num)
            timestep = timestep[0:1, -denoising_range_num:].transpose(0, 1)
            uncond_y = y[y.shape[0] // 2 : y.shape[0]][-denoising_range_num:]
            caption_dropout_mask = torch.tensor([True], dtype=torch.bool, device=x.device)
            uncond_mask = mask[y.shape[0] // 2 : y.shape[0]][-denoising_range_num:]
            uncond_kv_range = self.generate_kv_range_for_uncondition(uncond_x)

            kwargs["range_num"] = 1
            kwargs["denoising_range_num"] = 1
            kwargs["slice_point"] = 0
            out_uncond = self.forward(
                uncond_x,
                timestep,
                uncond_y,
                caption_dropout_mask=caption_dropout_mask,
                xattn_mask=uncond_mask,
                kv_range=uncond_kv_range,
                inference_params=None,
                **kwargs,
            )
            out_uncond = batch_to_chunk(out_uncond, denoising_range_num)

        return out_cond_pre_and_text, out_cond_pre, out_uncond, denoise_width

    def get_cfg_scale(self, t, cfg_t_range, prev_chunk_scale_s, text_scale_s):
        indices = torch.searchsorted(cfg_t_range - 1e-7, t) - 1
        assert indices.min() >= 0 and indices.max() < len(prev_chunk_scale_s)
        return prev_chunk_scale_s[indices], text_scale_s[indices]

    def forward_dispatcher(self, x, timestep, y, mask, kv_range, inference_params, **kwargs):
        if self.runtime_config.cfg_number == 3:
            (out_cond_pre_and_text, out_cond_pre, out_uncond, denoise_width) = self.forward_3cfg(
                x, timestep, y, mask, kv_range, inference_params, **kwargs
            )

            prev_chunk_scale_s = torch.tensor(self.runtime_config.prev_chunk_scales).cuda()
            text_scale_s = torch.tensor(self.runtime_config.text_scales).cuda()
            cfg_t_range = torch.tensor(self.runtime_config.cfg_t_range).cuda()
            applied_cfg_range_num, chunk_width = (kwargs["denoising_range_num"], kwargs["chunk_width"])
            if kwargs["fwd_extra_1st_chunk"]:
                applied_cfg_range_num -= 1
            cfg_timestep = timestep[0, -applied_cfg_range_num:]

            assert len(prev_chunk_scale_s) == len(cfg_t_range), "prev_chunks_scale and t_range should have the same length"
            assert len(text_scale_s) == len(cfg_t_range), "text_scale and t_range should have the same length"

            cfg_output_list = []

            for chunk_idx in range(applied_cfg_range_num):
                prev_chunk_scale, text_scale = self.get_cfg_scale(
                    cfg_timestep[chunk_idx], cfg_t_range, prev_chunk_scale_s, text_scale_s
                )
                l = chunk_idx * chunk_width
                r = (chunk_idx + 1) * chunk_width
                cfg_output = (
                    (1 - prev_chunk_scale) * out_uncond[:, :, l:r]
                    + (prev_chunk_scale - text_scale) * out_cond_pre[:, :, -denoise_width:][:, :, l:r]
                    + text_scale * out_cond_pre_and_text[:, :, -denoise_width:][:, :, l:r]
                )
                cfg_output_list.append(cfg_output)

            cfg_output = torch.cat(cfg_output_list, dim=2)

            x = torch.cat([x[0:1, :, :-denoise_width], cfg_output], dim=2)
            x = torch.cat([x, x], dim=0)
            return x
        elif self.runtime_config.cfg_number == 1:
            assert x.shape[0] == 2
            x = torch.cat([x[0:1], x[0:1]], dim=0)

            kwargs["caption_dropout_mask"] = torch.tensor([False], dtype=torch.bool, device=x.device)
            inference_params.update_kv_cache = True
            if kwargs.get("distill_nearly_clean_chunk", False):
                prev_chunks_scale = float(os.getenv("prev_chunks_scale", 0.7))
                slice_start = 1 if kwargs["fwd_extra_1st_chunk"] else 0
                cond_pre_and_text_channel = x.shape[2]
                new_x_chunk = x[0:1, :, slice_start * kwargs["chunk_width"] : (slice_start + 1) * kwargs["chunk_width"]]
                new_kvrange = self.generate_kv_range_for_uncondition(new_x_chunk)
                kwargs["denoising_range_num"] += 1
                cat_x_chunk = torch.cat([x[0:1], new_x_chunk], dim=2)
                new_kvrange = new_kvrange + kv_range.max()
                cat_kvrange = torch.cat([kv_range, new_kvrange], dim=0)
                cat_t = torch.cat([timestep[0:1], timestep[0:1, slice_start : slice_start + 1]], dim=1)
                cat_y = torch.cat([y[0 : y.shape[0] // 2], y[slice_start : slice_start + 1]], dim=0)
                cat_xattn_mask = torch.cat([mask[0 : y.shape[0] // 2], mask[slice_start : slice_start + 1]], dim=0)

                cat_out = self.forward(
                    cat_x_chunk,
                    cat_t,
                    cat_y,
                    xattn_mask=cat_xattn_mask,
                    kv_range=cat_kvrange,
                    inference_params=inference_params,
                    **kwargs,
                )
                near_clean_out_cond_pre_and_text = cat_out[
                    :, :, slice_start * kwargs["chunk_width"] : (slice_start + 1) * kwargs["chunk_width"]
                ]
                near_clean_out_cond_text = cat_out[:, :, cond_pre_and_text_channel:]
                near_out_cond_pre_and_text = (
                    near_clean_out_cond_pre_and_text * prev_chunks_scale + near_clean_out_cond_text * (1 - prev_chunks_scale)
                )
                cat_out[
                    :, :, slice_start * kwargs["chunk_width"] : (slice_start + 1) * kwargs["chunk_width"]
                ] = near_out_cond_pre_and_text
                out_cond_pre_and_text = cat_out[:, :, :cond_pre_and_text_channel]
            else:
                out_cond_pre_and_text = self.forward(
                    x[0:1],
                    timestep[0:1],
                    y[0 : y.shape[0] // 2],
                    xattn_mask=mask[0 : y.shape[0] // 2],
                    kv_range=kv_range,
                    inference_params=inference_params,
                    **kwargs,
                )

            denoise_width = kwargs["chunk_width"] * kwargs["denoising_range_num"]
            if kwargs["fwd_extra_1st_chunk"]:
                denoise_width -= kwargs["chunk_width"]

            x = torch.cat([x[0:1, :, :-denoise_width], out_cond_pre_and_text[:, :, -denoise_width:]], dim=2)
            x = torch.cat([x[0:1], x[0:1]], dim=0)
            return x
        else:
            raise NotImplementedError


def _build_dit_model(config: MagiConfig):
    """Builds the model"""
    device = "cuda" if env_is_true("SKIP_LOAD_MODEL") else "meta"
    with torch.device(device):
        model = VideoDiTModel(
            config=config, pre_process=mpu.is_pipeline_first_stage(), post_process=mpu.is_pipeline_last_stage()
        )
    print_rank_0(model)

    # Print number of parameters.
    param_count = sum([p.nelement() for p in model.parameters()])
    model_size_gb = sum([p.nelement() * p.element_size() for p in model.parameters()]) / (1024**3)
    print_per_rank(
        f"(cp, pp) rank ({mpu.get_cp_rank()}, {mpu.get_pp_rank()}): param count {param_count}, model size {model_size_gb:.2f} GB".format(
            mpu.get_cp_rank(), mpu.get_pp_rank(), param_count, model_size_gb
        )
    )

    return model


def _high_precision_promoter(module: VideoDiTModel):
    module.x_embedder.float()
    module.y_embedder.float()
    module.t_embedder.float()
    module.final_linear.float()
    module.rope.float()
    for name, sub_module in module.named_modules():
        # skip qk_layernorm_xattn
        if "_xattn" in name:
            continue
        # high precision qk_layernorm by default
        if "q_layernorm" in name or "k_layernorm" in name:
            sub_module.float()
        if "self_attn_post_norm" in name or "mlp_post_norm" in name:
            sub_module.float()
        if "final_layernorm" in name:
            sub_module.float()
    return module


def get_dit(config: MagiConfig):
    """Build and load VideoDiT model"""
    model = _build_dit_model(config)
    print_rank_0("Build DiTModel successfully")

    mem_allocated_gb = torch.cuda.memory_allocated() / 1024**3
    mem_reserved_gb = torch.cuda.memory_reserved() / 1024**3
    print_rank_0(
        f"After build_dit_model, memory allocated: {mem_allocated_gb:.2f} GB, memory reserved: {mem_reserved_gb:.2f} GB"
    )

    # To avoid Error in debug mode, set default iteration to 0
    if not env_is_true("SKIP_LOAD_MODEL"):
        model = load_checkpoint(model)
        mem_allocated_gb = torch.cuda.memory_allocated() / 1024**3
        mem_reserved_gb = torch.cuda.memory_reserved() / 1024**3
        print_rank_0(
            f"After load_checkpoint, memory allocated: {mem_allocated_gb:.2f} GB, memory reserved: {mem_reserved_gb:.2f} GB"
        )

    model = _high_precision_promoter(model)
    mem_allocated_gb = torch.cuda.memory_allocated() / 1024**3
    mem_reserved_gb = torch.cuda.memory_reserved() / 1024**3
    print_rank_0(
        f"After high_precision_promoter, memory allocated: {mem_allocated_gb:.2f} GB, memory reserved: {mem_reserved_gb:.2f} GB"
    )

    model.eval()
    gc.collect()
    torch.cuda.empty_cache()

    print_rank_0("Load checkpoint successfully")
    return model
