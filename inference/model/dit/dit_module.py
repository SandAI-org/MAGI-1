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

import math
import numbers
from functools import partial
from typing import Callable, List, Optional, Tuple

import flashinfer
import torch
import torch.distributed
import torch.nn as nn
import triton
import triton.language as tl
from einops import rearrange
from flash_attn import flash_attn_varlen_func
from flash_attn.flash_attn_interface import flash_attn_func
from flash_attn.layers.rotary import apply_rotary_emb as flash_apply_rotary_emb
from flashinfer.gemm import bmm_fp8
from magi_attention.functional import flex_flash_attn_func as flex_attention

# from dffa.functional import flex_flash_attn_func as flex_attention
from torch import Tensor
from torch.nn import Parameter

from inference.common import EngineConfig, InferenceParams, ModelConfig, ModelMetaArgs, PackedCrossAttnParams, divide
from inference.infra.distributed import parallel_state
from inference.infra.parallelism import CSOHelper, UlyssesScheduler, cso_communication


##########################################################
# TimestepEmbedder
##########################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, model_config: ModelConfig, frequency_embedding_size=256):
        super().__init__()

        self.data_type = model_config.params_dtype
        hidden_size = model_config.hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, int(hidden_size * model_config.cond_hidden_ratio), bias=True),
            nn.SiLU(),
            nn.Linear(
                int(hidden_size * model_config.cond_hidden_ratio), int(hidden_size * model_config.cond_hidden_ratio), bias=True
            ),
        )
        self.frequency_embedding_size = frequency_embedding_size

        # rescale the timestep for the general transport model
        self.timestep_rescale_factor = 1000

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000, timestep_rescale_factor=1):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None] * timestep_rescale_factor
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t = t.to(torch.float32)
        t_freq = self.timestep_embedding(
            t, self.frequency_embedding_size, timestep_rescale_factor=self.timestep_rescale_factor
        )
        t_emb = self.mlp(t_freq.to(self.data_type))
        return t_emb


##########################################################
# CaptionEmbedder
##########################################################
class CaptionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__()

        in_channels = model_config.caption_channels
        hidden_size = model_config.hidden_size
        caption_max_length = model_config.caption_max_length

        self.y_proj_xattn = nn.Sequential(
            nn.Linear(in_channels, int(hidden_size * model_config.xattn_cond_hidden_ratio), bias=True), nn.SiLU()
        )

        self.y_proj_adaln = nn.Sequential(nn.Linear(in_channels, int(hidden_size * model_config.cond_hidden_ratio), bias=True))

        self.null_caption_embedding = Parameter(torch.empty(caption_max_length, in_channels))

    def caption_drop(self, caption, caption_dropout_mask):
        """
        Drops labels to enable classifier-free guidance.
        caption.shape = (N, 1, cap_len, C)
        """
        dropped_caption = torch.where(
            caption_dropout_mask[:, None, None, None],  # (N, 1, 1, 1)
            self.null_caption_embedding[None, None, :],  # (1, 1, cap_len, C)
            caption,  # (N, 1, cap_len, C)
        )
        return dropped_caption

    def caption_drop_single_token(self, caption_dropout_mask):
        dropped_caption = torch.where(
            caption_dropout_mask[:, None, None],  # (N, 1, 1)
            self.null_caption_embedding[None, -1, :],  # (1, 1, C)
            self.null_caption_embedding[None, -2, :],  # (1, 1, C)
        )
        return dropped_caption  # (N, 1, C)

    def forward(self, caption, train, caption_dropout_mask=None):
        if train and caption_dropout_mask is not None:
            caption = self.caption_drop(caption, caption_dropout_mask)
        caption_xattn = self.y_proj_xattn(caption)
        if caption_dropout_mask is not None:
            caption = self.caption_drop_single_token(caption_dropout_mask)

        caption_adaln = self.y_proj_adaln(caption)
        return caption_xattn, caption_adaln


##########################################################
# FinalLinear
##########################################################
class FinalLinear(nn.Module):
    """
    The final linear layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, t_patch_size, out_channels):
        super().__init__()
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * t_patch_size * out_channels, bias=False)

    def forward(self, x):
        x = self.linear(x)
        return x


##########################################################
# AdaModulateLayer
##########################################################
class AdaModulateLayer(torch.nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config

        self.gate_num_chunks = 2
        self.act = nn.SiLU()
        self.proj = nn.Sequential(
            nn.Linear(
                int(self.model_config.hidden_size * self.model_config.cond_hidden_ratio),
                int(self.model_config.hidden_size * self.model_config.cond_gating_ratio * self.gate_num_chunks),
                bias=True,
                dtype=self.model_config.params_dtype,
            )
        )

    def forward(self, c):
        c = self.act(c)
        return self.proj(c)


##########################################################
# bias_modulate_add
##########################################################
@triton.jit
def range_mod_kernel_fwd(
    X,  # pointer to the input
    MAP,  # map x index to gating index
    GATINGS,  # pointer to the gatings
    Y,  # pointer to the output
    M,  # number of rows in X, unused
    N,  # number of columns in X
    stride_xm,  # how much to increase the pointer when moving by 1 row in X
    stride_xn,  # how much to increase the pointer when moving by 1 column in X
    stride_gm,  # how much to increase the pointer when moving by 1 row in GATINGS
    stride_gn,  # how much to increase the pointer when moving by 1 column in GATINGS
    stride_ym,  # how much to increase the pointer when moving by 1 row in Y
    stride_yn,  # how much to increase the pointer when moving by 1 column in Y
    BLOCK_SIZE: tl.constexpr,  # number of columns in a block
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)

    cur_X = X + row * stride_xm
    x_cols = tl.arange(0, BLOCK_SIZE) * stride_xn
    x_mask = x_cols < N * stride_xn
    x = tl.load(cur_X + x_cols, mask=x_mask, other=0.0)

    cur_MAP = MAP + row
    gating_index = tl.load(cur_MAP)
    cur_GATING = GATINGS + gating_index * stride_gm
    gating_cols = tl.arange(0, BLOCK_SIZE) * stride_gn
    gating_mask = gating_cols < N * stride_gn
    gating = tl.load(cur_GATING + gating_cols, mask=gating_mask, other=0.0)

    cur_Y = Y + row * stride_ym
    y_cols = tl.arange(0, BLOCK_SIZE) * stride_yn
    y_mask = y_cols < N * stride_yn
    tl.store(cur_Y + y_cols, x * gating, mask=y_mask)


def range_mod_triton(x, c_mapping, gatings):
    """
    Inputs:
        x: (s, b, h). Tensor of inputs embedding (images or latent representations of images)
        c_mapping: (s, b). Tensor of condition map
        gatings: (b, denoising_range_num, h). Tensor of condition embedding
    """

    assert x.is_cuda, "x is not on cuda"
    assert c_mapping.is_cuda, "c_mapping is not on cuda"
    assert gatings.is_cuda, "gatings is not on cuda"

    # TODO: use 3D tensor for x, c_mapping, and gatings
    s, b, h = x.shape
    x = x.transpose(0, 1).flatten(0, 1)
    c_mapping = c_mapping.transpose(0, 1).flatten(0, 1)
    gatings = gatings.flatten(0, 1)

    assert x.dim() == 2, f"x must be a 2D tensor but got {x.dim()}D"
    assert c_mapping.dim() == 1, f"c_mapping must be a 1D tensor but got {c_mapping.dim()}D"
    assert gatings.dim() == 2, f"gatings must be a 2D tensor but got {gatings.dim()}D"

    M, N = x.shape
    assert c_mapping.size(0) == M, "c_mapping must have the same number of rows as x"

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE:
        raise RuntimeError("range_mod_triton doesn't support feature dim >= 64KB.")

    MAP = c_mapping
    y = torch.empty_like(x)

    range_mod_kernel_fwd[(M,)](
        x,
        MAP,
        gatings,
        y,
        M,
        N,
        x.stride(0),
        x.stride(1),
        gatings.stride(0),
        gatings.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    y = y.reshape(b, s, h).transpose(0, 1)

    return y


def bias_modulate_add(
    x: torch.Tensor, residual: torch.Tensor, condition_map: torch.Tensor, gate: torch.Tensor, post_norm: torch.nn.Module
):
    assert gate.shape[-1] == x.shape[-1]

    original_dtype = x.dtype
    x = x.float()
    residual = residual.float()
    gate = gate.float()

    x = range_mod_triton(x, condition_map, gate)
    x = post_norm(x)
    x = x + residual
    x = x.to(original_dtype)

    return x


##########################################################
# FusedLayerNorm
##########################################################
def make_viewless_tensor(inp, requires_grad):
    # return tensor as-is, if not a 'view'
    if inp._base is None:
        return inp

    out = torch.empty((1,), dtype=inp.dtype, device=inp.device, requires_grad=requires_grad)
    out.data = inp.data
    return out


class FusedLayerNorm(torch.nn.Module):

    """
    Layer Norm, fused into a single CUDA kernel.
    Borrow from: https://github.com/NVIDIA/Megatron-LM/blob/6501752396e9cc360ce894cda4b2217a58c1c09d/megatron/core/fusions/fused_layer_norm.py#L30

    Args:
      hidden_size (int): Transformer hidden dimension.

      eps (float): Epsilon added to denominator, for numerical stability.

      zero_centered_gamma (bool): Adjust LayerNorm weights such that they are
      centered around zero. This improves numerical stability.

      model_config (ModelConfig): Transformer config. Include to match custom
      layer norm interfaces.

      normalization (str): Normalization type, used for Transformer Engine.
      Must equal 'LayerNorm' here.
    """

    def __init__(self, model_config: ModelConfig, hidden_size: int):
        super().__init__()

        self.zero_centered_gamma = model_config.apply_layernorm_1p
        if isinstance(hidden_size, numbers.Integral):
            hidden_size = (hidden_size,)
        self.hidden_size = torch.Size(hidden_size)
        self.eps = model_config.layernorm_epsilon
        self.weight = Parameter(torch.empty(*hidden_size, dtype=model_config.params_dtype))
        self.bias = Parameter(torch.empty(*hidden_size, dtype=model_config.params_dtype))

    def forward(self, input: Tensor) -> Tensor:
        weight = self.weight + 1 if self.zero_centered_gamma else self.weight
        return torch.nn.functional.layer_norm(input, self.hidden_size, weight, self.bias, self.eps)


def softcap(x: torch.Tensor, cap: int):
    return (cap * torch.tanh(x.float() / cap)).to(x.dtype)


def div_clamp_to(x: torch.Tensor, scale: torch.Tensor):
    fp8_min = torch.finfo(torch.float8_e4m3fn).min
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    prefix_shape = x.shape[:-1]
    last_shape = x.shape[-1]
    x = x.flatten().reshape(-1, last_shape)
    part_size = 256 * 1024 * 1024 // last_shape
    part_num = (x.shape[0] + part_size - 1) // part_size
    return torch.cat([torch.clamp(x[i * part_size:(i+1) * part_size].float() / scale.float(), fp8_min, fp8_max).bfloat16() for i in range(part_num)], dim=0).to(torch.float8_e4m3fn).reshape(*prefix_shape, last_shape).contiguous()


##########################################################
# CustomLayerNormLinear
##########################################################
class CustomLayerNormLinear(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size_q: int,
        output_size_kv: int,
        layer_number: int,
        model_config: ModelConfig,
        engine_config: EngineConfig,
    ):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(input_size, eps=model_config.layernorm_epsilon, dtype=model_config.params_dtype)

        self.layer_number = layer_number
        layers = {"q": output_size_q, "qx": output_size_q, "k": output_size_kv, "v": output_size_kv}

        for name, output_size in layers.items():
            if not engine_config.fp8_quant or self.layer_number == 0 or self.layer_number == model_config.num_layers - 1:
                setattr(self, name, torch.nn.Linear(input_size, output_size, bias=False, dtype=model_config.params_dtype))
            else:
                setattr(self, name, PerTensorQuantizedFp8Linear(input_size, output_size))

    def forward_ln(self, hidden_states):
        return self.layer_norm(hidden_states)

    def forward_q(self, hidden_states):
        return self.q(hidden_states)

    def forward_qx(self, hidden_states):
        return self.qx(hidden_states)

    def forward_k(self, hidden_states):
        return self.k(hidden_states)

    def forward_v(self, hidden_states):
        return self.v(hidden_states)


##########################################################
# PerTensorQuantizedFp8Linear
##########################################################
class PerTensorQuantizedFp8Linear(torch.nn.Module):
    # The bias and device parameter is not used; it is included for compatibility with Linear's parameters.
    def __init__(self, in_features: int, out_features: int, bias=False, dtype=torch.bfloat16, device=None) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.finfo = torch.finfo(torch.float8_e4m3fn)
        self.output_dtype = dtype

        self.weight = Parameter(torch.empty((1, out_features, in_features), dtype=torch.float8_e4m3fn))
        self.weight_scale = Parameter(torch.empty(1, dtype=torch.float32))
        self.input_scale = Parameter(torch.empty(in_features, dtype=torch.float32))

    def forward(self, input: torch.Tensor):
        input = div_clamp_to(input, self.input_scale)

        prefix_shape = input.shape[:-1]
        # column major weight
        return bmm_fp8(
            input.reshape(1, -1, self.in_features),
            self.weight.transpose(-2, -1),
            self.input_scale,
            self.weight_scale,
            dtype=self.output_dtype,
        ).reshape(prefix_shape + (self.out_features,))


##########################################################
# PerChannelQuantizedFp8Linear
##########################################################
class PerChannelQuantizedFp8Linear(torch.nn.Module):
    # The bias and device parameter is not used; it is included for compatibility with Linear's parameters.
    def __init__(self, in_features: int, out_features: int, bias=False, dtype=torch.bfloat16, device=None) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.output_dtype = dtype
        self.finfo = torch.finfo(torch.float8_e4m3fn)

        self.weight = Parameter(torch.empty((1, out_features, in_features), dtype=torch.float8_e4m3fn))
        self.weight_scale = Parameter(torch.empty(1, dtype=torch.float32))
        self.input_scale = Parameter(torch.empty(1, dtype=torch.float32))
        self.smooth_scale = Parameter(torch.empty(1, in_features, dtype=torch.float32))

    def forward(self, x):
        x = div_clamp_to(x, self.smooth_scale.to(torch.float32))

        prefix_shape = x.shape[:-1]
        return bmm_fp8(
            x.reshape(1, -1, self.in_features),
            self.weight.transpose(-2, -1),
            self.input_scale,
            self.weight_scale,
            dtype=self.output_dtype,
        ).reshape(prefix_shape + (self.out_features,))


##########################################################
# CustomMLP
##########################################################
class CustomMLP(torch.nn.Module):
    """
    CustomMLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.


    Returns an output and a bias to be added to the output.

    We use the following notation:
     h: hidden size
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(self, model_config: ModelConfig, engine_config: EngineConfig, layer_number: int, input_size: int = None):
        super().__init__()

        self.model_config: ModelConfig = model_config
        self.engine_config: EngineConfig = engine_config
        self.layer_number = layer_number

        self.input_size = input_size if input_size != None else self.model_config.hidden_size
        self.layer_norm = torch.nn.LayerNorm(
            self.input_size, eps=self.model_config.layernorm_epsilon, dtype=self.model_config.params_dtype
        )

        submodules_linear_fc1 = torch.nn.Linear
        if self.engine_config.fp8_quant and self.layer_number != 0 and self.layer_number != model_config.num_layers - 1:
            submodules_linear_fc1 = PerTensorQuantizedFp8Linear

        if self.model_config.gated_linear_unit:
            self.linear_fc1 = submodules_linear_fc1(
                self.input_size, 2 * self.model_config.ffn_hidden_size, bias=False, dtype=self.model_config.params_dtype
            )
        else:
            self.linear_fc1 = submodules_linear_fc1(
                self.input_size, self.model_config.ffn_hidden_size, bias=False, dtype=self.model_config.params_dtype
            )

        submodules_linear_fc2 = torch.nn.Linear
        if engine_config.fp8_quant and self.layer_number != 0 and self.layer_number != model_config.num_layers - 1:
            submodules_linear_fc2 = PerChannelQuantizedFp8Linear

        self.linear_fc2 = submodules_linear_fc2(
            self.model_config.ffn_hidden_size, self.model_config.hidden_size, bias=False, dtype=self.model_config.params_dtype
        )

    def forward(self, hidden_states):
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.linear_fc1(hidden_states)
        if self.model_config.gated_linear_unit:
            hidden_states = flashinfer.activation.silu_and_mul(hidden_states)
        else:
            hidden_states = torch.nn.functional.gelu(hidden_states)
        hidden_states = self.linear_fc2(hidden_states)

        return hidden_states


##########################################################
# LearnableRotaryEmbeddingCat
##########################################################
def ndgrid(*tensors) -> Tuple[torch.Tensor, ...]:
    """generate N-D grid in dimension order.

    The ndgrid function is like meshgrid except that the order of the first two input arguments are switched.

    That is, the statement
    [X1,X2,X3] = ndgrid(x1,x2,x3)

    produces the same result as

    [X2,X1,X3] = meshgrid(x2,x1,x3)

    This naming is based on MATLAB, the purpose is to avoid confusion due to torch's change to make
    torch.meshgrid behaviour move from matching ndgrid ('ij') indexing to numpy meshgrid defaults of ('xy').

    """
    try:
        return torch.meshgrid(*tensors, indexing="ij")
    except TypeError:
        # old PyTorch < 1.10 will follow this path as it does not have indexing arg,
        # the old behaviour of meshgrid was 'ij'
        return torch.meshgrid(*tensors)


def pixel_freq_bands(
    num_bands: int, max_freq: float = 224.0, linear_bands: bool = True, device: Optional[torch.device] = None
):
    if linear_bands:
        bands = torch.linspace(1.0, max_freq / 2, num_bands, dtype=torch.float32, device=device)
    else:
        bands = 2 ** torch.linspace(0, math.log(max_freq, 2) - 1, num_bands, dtype=torch.float32, device=device)
    return bands * torch.pi


def freq_bands(
    num_bands: int, temperature: float = 10000.0, step: int = 2, device: Optional[torch.device] = None
) -> torch.Tensor:
    exp = torch.arange(0, num_bands, step, dtype=torch.int64, device=device).to(torch.float32) / num_bands
    bands = 1.0 / (temperature**exp)
    return bands


def build_fourier_pos_embed(
    feat_shape: List[int],
    bands: Optional[torch.Tensor] = None,
    num_bands: int = 64,
    max_res: int = 224,
    temperature: float = 10000.0,
    linear_bands: bool = False,
    include_grid: bool = False,
    in_pixels: bool = True,
    ref_feat_shape: Optional[List[int]] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> List[torch.Tensor]:
    """

    Args:
        feat_shape: Feature shape for embedding.
        bands: Pre-calculated frequency bands.
        num_bands: Number of frequency bands (determines output dim).
        max_res: Maximum resolution for pixel based freq.
        temperature: Temperature for non-pixel freq.
        linear_bands: Linear band spacing for pixel based freq.
        include_grid: Include the spatial grid in output.
        in_pixels: Output in pixel freq.
        ref_feat_shape: Reference feature shape for resize / fine-tune.
        dtype: Output dtype.
        device: Output device.

    Returns:

    """
    if bands is None:
        if in_pixels:
            bands = pixel_freq_bands(num_bands, float(max_res), linear_bands=linear_bands, device=device)
        else:
            bands = freq_bands(num_bands, temperature=temperature, step=1, device=device)
    else:
        if device is None:
            device = bands.device
        if dtype is None:
            dtype = bands.dtype

    if in_pixels:
        t = [torch.linspace(-1.0, 1.0, steps=s, device=device, dtype=torch.float32) for s in feat_shape]
    else:
        t = [torch.arange(s, device=device, dtype=torch.int64).to(torch.float32) for s in feat_shape]
        # align spatial center (H/2,W/2) to (0,0)
        t[1] = t[1] - (feat_shape[1] - 1) / 2
        t[2] = t[2] - (feat_shape[2] - 1) / 2
    if ref_feat_shape is not None:
        # eva's scheme for resizing rope embeddings (ref shape = pretrain)
        # aligning to the endpoint e.g [0,1,2] -> [0, 0.4, 0.8, 1.2, 1.6, 2]
        t_rescaled = []
        for x, f, r in zip(t, feat_shape, ref_feat_shape):
            # deal with image input
            if f == 1:
                assert r == 1, "ref_feat_shape must be 1 when feat_shape is 1"
                t_rescaled.append(x)
            else:
                t_rescaled.append(x / (f - 1) * (r - 1))
        t = t_rescaled

    grid = torch.stack(ndgrid(t), dim=-1)
    grid = grid.unsqueeze(-1)
    pos = grid * bands

    pos_sin, pos_cos = pos.sin().to(dtype=dtype), pos.cos().to(dtype)
    out = [grid, pos_sin, pos_cos] if include_grid else [pos_sin, pos_cos]
    return out


def build_rotary_pos_embed(
    feat_shape: List[int],
    bands: Optional[torch.Tensor] = None,
    dim: int = 64,
    max_res: int = 224,
    temperature: float = 10000.0,
    linear_bands: bool = False,
    in_pixels: bool = True,
    ref_feat_shape: Optional[List[int]] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
):
    """

    Args:
        feat_shape: Spatial shape of the target tensor for embedding.
        bands: Optional pre-generated frequency bands
        dim: Output dimension of embedding tensor.
        max_res: Maximum resolution for pixel mode.
        temperature: Temperature (inv freq) for non-pixel mode
        linear_bands: Linearly (instead of log) spaced bands for pixel mode
        in_pixels: Pixel vs language (inv freq) mode.
        dtype: Output dtype.
        device: Output device.

    Returns:

    """
    sin_emb, cos_emb = build_fourier_pos_embed(
        feat_shape,
        bands=bands,
        num_bands=dim // 8,
        max_res=max_res,
        temperature=temperature,
        linear_bands=linear_bands,
        in_pixels=in_pixels,
        ref_feat_shape=ref_feat_shape,
        device=device,
        dtype=dtype,
    )
    num_spatial_dim = 1
    # this would be much nicer as a .numel() call to torch.Size(), but torchscript sucks
    for x in feat_shape:
        num_spatial_dim *= x

    sin_emb = sin_emb.reshape(num_spatial_dim, -1)
    cos_emb = cos_emb.reshape(num_spatial_dim, -1)
    return sin_emb, cos_emb


class LearnableRotaryEmbeddingCat(nn.Module):
    """Rotary position embedding w/ concatenatd sin & cos

    The following impl/resources were referenced for this impl:
    * https://github.com/lucidrains/vit-pytorch/blob/6f3a5fcf0bca1c5ec33a35ef48d97213709df4ba/vit_pytorch/rvt.py
    * https://blog.eleuther.ai/rotary-embeddings/
    """

    def __init__(
        self,
        dim,
        max_res=224,
        temperature=10000,
        in_pixels=True,
        linear_bands: bool = False,
        feat_shape: Optional[List[int]] = None,
        ref_feat_shape: Optional[List[int]] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_res = max_res
        self.temperature = temperature
        self.in_pixels = in_pixels
        self.linear_bands = linear_bands
        self.feat_shape = feat_shape
        self.ref_feat_shape = ref_feat_shape
        self.bands = nn.Parameter(self.get_default_bands())

    def get_default_bands(self):
        if self.in_pixels:
            bands = pixel_freq_bands(
                self.dim // 8, float(self.max_res), linear_bands=self.linear_bands, devicse=torch.cuda.current_device()
            )
        else:
            bands = freq_bands(self.dim // 8, temperature=self.temperature, step=1, device=torch.cuda.current_device())
        return bands

    def get_embed(self, shape: Optional[List[int]], ref_feat_shape: Optional[List[int]] = None):
        # rebuild bands and embeddings every call, use if target shape changes
        embeds = build_rotary_pos_embed(
            feat_shape=shape,
            bands=self.bands,  # use learned bands
            dim=self.dim,
            max_res=self.max_res,
            linear_bands=self.linear_bands,
            in_pixels=self.in_pixels,
            ref_feat_shape=ref_feat_shape if ref_feat_shape else self.ref_feat_shape,
            temperature=self.temperature,
            device=torch.cuda.current_device(),
        )
        return torch.cat(embeds, -1)


##########################################################
# Attention
##########################################################
class Attention(torch.nn.Module):
    """
    Attention layer abstract class.
    """

    def __init__(self, model_config: ModelConfig, engine_config: EngineConfig, layer_number: int):
        super().__init__()

        self.model_config: ModelConfig = model_config
        self.engine_config: EngineConfig = engine_config
        self.layer_number = layer_number

        self.hidden_size_per_attention_head = self.model_config.kv_channels
        # num_query_groups and num_attention_heads are different for GQA
        self.query_projection_size = self.model_config.kv_channels * self.model_config.num_attention_heads
        self.kv_projection_size = self.model_config.kv_channels * self.model_config.num_query_groups

        # Per attention head and per partition values.
        world_size = parallel_state.get_tp_world_size(with_context_parallel=True)
        if world_size > self.model_config.num_query_groups and world_size % self.model_config.num_query_groups == 0:
            self.num_query_groups_per_partition = 1
        else:
            self.num_query_groups_per_partition = divide(self.model_config.num_query_groups, world_size)

    def _allocate_key_and_value_memory(self, sequence_length, batch_size, dtype):
        """Allocate memory to store kv cache during inference."""

        if self.engine_config.kv_offload:
            return torch.empty(
                sequence_length * batch_size,
                self.num_query_groups_per_partition,
                self.hidden_size_per_attention_head * 2,
                dtype=dtype,
                device=torch.cpu.current_device(),
                pin_memory=True,
            )
        else:
            return torch.empty(
                sequence_length * batch_size,
                self.num_query_groups_per_partition,
                self.hidden_size_per_attention_head * 2,
                dtype=dtype,
                device=torch.cuda.current_device(),
            )


##########################################################
# FullyParallelAttention
##########################################################
def split_tensor_along_last_dim(
    tensor: torch.Tensor, num_partitions: int, contiguous_split_chunks: bool = False
) -> List[torch.Tensor]:
    """Split a tensor along its last dimension.

    Args:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


class FullyParallelAttention(Attention):
    def __init__(self, model_config: ModelConfig, engine_config: EngineConfig, layer_number: int):
        super().__init__(model_config=model_config, engine_config=engine_config, layer_number=layer_number)

        # output 2x query, one for self-attn, one for cross-attn with condition
        self.linear_qkv = CustomLayerNormLinear(
            input_size=self.model_config.hidden_size,
            output_size_q=self.query_projection_size,
            output_size_kv=self.kv_projection_size,
            layer_number=self.layer_number,
            model_config=self.model_config,
            engine_config=self.engine_config,
        )

        # kv from condition, e.g., caption
        self.linear_kv_xattn = torch.nn.Linear(
            int(self.model_config.hidden_size * self.model_config.xattn_cond_hidden_ratio),  # 6144
            2 * self.kv_projection_size,  # 2048
            dtype=self.model_config.params_dtype,
            bias=False,
        )

        # Output.
        self.adapt_linear_quant = (
            self.engine_config.fp8_quant and self.layer_number != 0 and self.layer_number != model_config.num_layers - 1
        )
        submodules_linear_proj = PerChannelQuantizedFp8Linear if self.adapt_linear_quant else torch.nn.Linear
        self.linear_proj = submodules_linear_proj(
            2 * self.query_projection_size, self.model_config.hidden_size, dtype=self.model_config.params_dtype, bias=False
        )

        self.q_layernorm = FusedLayerNorm(model_config=self.model_config, hidden_size=self.hidden_size_per_attention_head)
        self.q_layernorm_xattn = FusedLayerNorm(
            model_config=self.model_config, hidden_size=self.hidden_size_per_attention_head
        )
        self.k_layernorm = FusedLayerNorm(model_config=self.model_config, hidden_size=self.hidden_size_per_attention_head)
        self.k_layernorm_xattn = FusedLayerNorm(
            model_config=self.model_config, hidden_size=self.hidden_size_per_attention_head
        )

    def _full_adjust_key_and_value(
        self, inference_params: InferenceParams, key_and_value: torch.Tensor, meta_args: ModelMetaArgs
    ):
        """
        Saves the generated key and value tensors to the end of the buffers in inference_params.
        Returns the full size keys and values from the provided inference_params

        Returns a tuple: (key, value)
        """
        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        inf_max_seq_length = inference_params.max_sequence_length
        inf_max_batch_size = inference_params.max_batch_size
        if self.layer_number not in inference_params.key_value_memory_dict:
            inference_key_and_value_memory = self._allocate_key_and_value_memory(
                inf_max_seq_length, inf_max_batch_size, key_and_value.dtype
            )
            inference_params.key_value_memory_dict[self.layer_number] = inference_key_and_value_memory
        else:
            # Get the pre-allocated buffers for this layer
            inference_key_and_value_memory = inference_params.key_value_memory_dict[self.layer_number]

        sequence_start = meta_args.slice_point * meta_args.clip_token_nums * inf_max_batch_size
        get_key_and_value = inference_key_and_value_memory[:sequence_start, ...].cuda()

        # Copy key and values.
        if inference_params.update_kv_cache:
            key_and_value_total = key_and_value

            clip_size = (
                key_and_value_total.size(0) - meta_args.clip_token_nums * inf_max_batch_size
                if meta_args.distill_nearly_clean_chunk
                else key_and_value_total.size(0)
            )
            sequence_end = sequence_start + clip_size
            assert sequence_end <= inference_key_and_value_memory.size(0)
            # update kv cache
            inference_key_and_value_memory[sequence_start:sequence_end, ...] = key_and_value_total[:clip_size]

        return torch.cat([get_key_and_value, key_and_value], dim=0)

    def adjust_key_and_value_for_inference(
        self, key_and_value: torch.Tensor, inference_params: InferenceParams, meta_args: ModelMetaArgs
    ):
        if inference_params is None:
            return torch.chunk(key_and_value, 2, dim=-1)

        # Only update kvcache when necessary, include 3 conditions:
        # 1. extract prefix video clean feature
        # 2. the first chunk of current kv is clean, we need to save their feature
        # 3. previous chunk is clean and we need to save/load their feature
        if meta_args.extract_prefix_video_feature or meta_args.fwd_extra_1st_chunk or meta_args.slice_point > 0:
            key_and_value = self._full_adjust_key_and_value(inference_params, key_and_value, meta_args)
        key, value = torch.chunk(key_and_value, 2, dim=-1)
        return key.contiguous(), value.contiguous()

    # =====================
    # Get Query for core attn
    # [sq, b, (hn hd)] -> [(sq b), hn, hd]
    # =====================

    def get_q(self, mixed_qqkv: torch.Tensor, cos_emb: torch.Tensor, sin_emb: torch.Tensor):
        query = self.linear_qkv.forward_q(mixed_qqkv)
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        assert self.q_layernorm is not None
        original_dtype = query.dtype
        query = query.float()
        query = self.q_layernorm(query)
        query = query.transpose(0, 1).contiguous()
        query = flash_apply_rotary_emb(query, cos_emb, sin_emb)
        query = query.to(original_dtype)
        return rearrange(query, "b sq hn hd -> (sq b) hn hd").contiguous()

    # =====================
    # Get Key for core attn
    # [sq, b, (hn hd)] -> [(sq b), hn, hd]
    # =====================

    def get_k(self, mixed_qqkv: torch.Tensor, cos_emb: torch.Tensor, sin_emb: torch.Tensor):
        key = self.linear_qkv.forward_k(mixed_qqkv)
        key = key.reshape(key.size(0), key.size(1), -1, self.hidden_size_per_attention_head)
        assert self.k_layernorm is not None
        original_dtype = key.dtype
        key = key.float()
        key = self.k_layernorm(key)
        key = key.transpose(0, 1).contiguous()
        key = flash_apply_rotary_emb(key, cos_emb, sin_emb)
        key = key.to(original_dtype)
        return rearrange(key, "b sq hn hd -> (sq b) hn hd").contiguous()

    # =====================
    # Get Value for core attn
    # [sq, b, (hn hd)] -> [(sq b), hn, hd]
    # =====================

    def get_v(self, mixed_qqkv: torch.Tensor):
        value = self.linear_qkv.forward_v(mixed_qqkv)
        return rearrange(value, "sq b (hn hd) -> (sq b) hn hd", hd=self.hidden_size_per_attention_head).contiguous()

    def get_kv(self, mixed_qqkv: torch.Tensor, cos_emb: torch.Tensor, sin_emb: torch.Tensor):
        # Get KV together for better performance when encoutering cpu-bound, mainly used by cuda graph
        key = self.get_k(mixed_qqkv, cos_emb, sin_emb)
        value = self.get_v(mixed_qqkv)
        # [(sq b), hn, hd] -> [(sq b), hn, 2 * hd]
        return torch.cat([key, value], dim=-1)

    def get_qkv(self, mixed_qqkv: torch.Tensor, cos_emb: torch.Tensor, sin_emb: torch.Tensor):
        # Get QKV together for better performance when encoutering cpu-bound, mainly used by cuda graph
        q = self.get_q(mixed_qqkv, cos_emb, sin_emb)
        k = self.get_k(mixed_qqkv, cos_emb, sin_emb)
        v = self.get_v(mixed_qqkv)
        return q, k, v

    def get_xqkv(self, mixed_qqkv: torch.Tensor, key_value_states: torch.Tensor):
        query_xattn = self.linear_qkv.forward_qx(mixed_qqkv)
        query_xattn = rearrange(query_xattn, "sq b (hn hd) -> (b sq) hn hd", hd=self.hidden_size_per_attention_head)
        query_xattn = self.q_layernorm_xattn(query_xattn)

        # [y_total_token, h] --> [y_total_token, 2*hp]
        mixed_kv_xattn = torch.concat(
            [torch.matmul(key_value_states, w.t()) for w in torch.chunk(self.linear_kv_xattn.weight, 8, axis=0)], axis=1
        )
        # [y_total_token, 2*hn*hd] --> [y_total_token, hn, 2*hd]
        mixed_kv_xattn = mixed_kv_xattn.view(key_value_states.shape[0], -1, 2 * self.hidden_size_per_attention_head)

        # [y_total_token, hn, 2*hd] --> 2 [y_total_token, hn, hd]
        (key_xattn, value_xattn) = split_tensor_along_last_dim(mixed_kv_xattn, 2)

        key_xattn = self.k_layernorm_xattn(key_xattn)
        return query_xattn, key_xattn, value_xattn

    def core_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, bs: int, meta_args: ModelMetaArgs):
        # (sq b) hn hd -> b sq hn hd
        query = query.reshape(-1, bs, query.shape[1], query.shape[2]).transpose(0, 1).contiguous()
        # (sq b) hn hd -> b sq hn hd
        key = key.reshape(-1, bs, key.shape[1], key.shape[2]).transpose(0, 1).contiguous()
        # (sq b) hn hd -> b sq hn hd
        value = value.reshape(-1, bs, value.shape[1], value.shape[2]).transpose(0, 1).contiguous()

        if torch.cuda.get_device_capability()[0] >= 9:
            core_attn_out, _ = flex_attention(
                query.flatten(0, 1),
                key.flatten(0, 1),
                value.flatten(0, 1),
                meta_args.core_attn_params.q_range,
                meta_args.core_attn_params.k_range,
                max_seqlen_q=meta_args.core_attn_params.max_seqlen_q,
                max_seqlen_k=meta_args.core_attn_params.max_seqlen_k,
                softmax_scale=None,
                deterministic=torch.are_deterministic_algorithms_enabled(),
                disable_fwd_atomic_reduction=True,
            )
            # (b sq) hn hd -> (sq b) hn hd
            core_attn_out = rearrange(core_attn_out, "(b sq) h d -> (sq b) h d", b=bs)
        else:
            # NOTE(lml): We convert multi denoising_range_num input into multi batch_size input at third time forward under 3_cfg mode, thus could not support normal multi batch_size input. We use an assert statement to ensure that it is still in this situation, thereby guaranteeing the correct use of q_range and k_range later on.
            assert not (bs > 1 and meta_args.denoising_range_num > 1)
            q_range = meta_args.core_attn_params.np_q_range
            k_range = meta_args.core_attn_params.np_k_range
            core_attn_outs = []
            for i in range(meta_args.denoising_range_num):
                if bs == 1:
                    q = query[:, q_range[i, 0] : q_range[i, 1]]
                    k = key[:, k_range[i, 0] : k_range[i, 1]]
                    v = value[:, k_range[i, 0] : k_range[i, 1]]
                else:
                    assert i == 0
                    q = query[:, q_range[0, 0] : q_range[0, 1]]
                    k = key[:, k_range[0, 0] : k_range[0, 1]]
                    v = value[:, k_range[0, 0] : k_range[0, 1]]
                o = flash_attn_func(q=q, k=k, v=v, deterministic=torch.are_deterministic_algorithms_enabled())
                o = rearrange(o, "b sq h d -> (sq b) h d", b=bs)
                core_attn_outs.append(o)
            core_attn_out = torch.cat(core_attn_outs, dim=0)
        return core_attn_out

    def full_attention(self, bs: int, meta_args: ModelMetaArgs, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, i: int):
        # NOTE(lml): full_attention is used under cp_shuffle_overlap strategy. We further limit it to the case of bs=1, so that we do not need to pay attention to the arrangement of sq and bs dimensions.
        assert bs == 1
        if torch.cuda.get_device_capability()[0] >= 9:
            q_range = meta_args.core_attn_params.q_range[i : i + 1] - meta_args.core_attn_params.q_range[i, 0]
            k_range = meta_args.core_attn_params.k_range[i : i + 1]
            o, _ = flex_attention(
                q,
                k,
                v,
                q_ranges=q_range,
                k_ranges=k_range,
                max_seqlen_q=meta_args.core_attn_params.max_seqlen_q,
                max_seqlen_k=meta_args.core_attn_params.max_seqlen_k,
                softmax_scale=None,
                deterministic=torch.are_deterministic_algorithms_enabled(),
                disable_fwd_atomic_reduction=True,
            )
        else:
            k_range = meta_args.core_attn_params.np_k_range[i : i + 1]
            k = k[k_range[0, 0] : k_range[0, 1]]
            v = v[k_range[0, 0] : k_range[0, 1]]
            o = flash_attn_func(
                q=q.unsqueeze(0),
                k=k.unsqueeze(0),
                v=v.unsqueeze(0),
                deterministic=torch.are_deterministic_algorithms_enabled(),
            ).flatten(0, 1)
        return o

    def cross_attention(
        self,
        mixed_qqkv: torch.Tensor,
        key_value_states: torch.Tensor,
        cross_attn_params: PackedCrossAttnParams,
        get_xqkv_func: Callable,
    ):
        # =================
        # cross-attn for aggragating caption / condition
        # =================
        query_xattn, key_xattn, value_xattn = get_xqkv_func(mixed_qqkv, key_value_states)

        if torch.cuda.get_device_capability()[0] >= 9:
            xattn_out, _ = flex_attention(
                query_xattn,
                key_xattn,
                value_xattn,
                cross_attn_params.q_ranges,
                cross_attn_params.kv_ranges,
                max_seqlen_q=cross_attn_params.max_seqlen_q,
                max_seqlen_k=cross_attn_params.max_seqlen_kv,
                softmax_scale=None,
                deterministic=False,
                disable_fwd_atomic_reduction=True,
            )
        else:
            xattn_out = flash_attn_varlen_func(
                query_xattn,  # [b*sq, hn, hd]
                key_xattn,  # [y_total_token, hn, hd]
                value_xattn,  # [y_total_token, hn, hd]
                cu_seqlens_q=cross_attn_params.cu_seqlens_q,
                cu_seqlens_k=cross_attn_params.cu_seqlens_kv,
                max_seqlen_q=cross_attn_params.max_seqlen_q,
                max_seqlen_k=cross_attn_params.max_seqlen_kv,
                deterministic=torch.are_deterministic_algorithms_enabled(),
            )
        batch_size = mixed_qqkv.shape[1]
        xattn_out = rearrange(xattn_out, "(b sq) hn hd -> sq b (hn hd)", b=batch_size).contiguous()
        return xattn_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor,
        inference_params: InferenceParams,
        rotary_pos_emb: torch.Tensor,
        meta_args: ModelMetaArgs,
    ):
        assert rotary_pos_emb is not None, "FullyParallelAttention needs rotary_pos_emb"
        sin_emb, cos_emb = rotary_pos_emb.tensor_split(2, -1)
        batch_size = hidden_states.shape[1]
        # All comminications operate on dimensions shaped as (cp * sq * b)
        batch_cp_split_sizes = None if meta_args.cp_split_sizes is None else [x * batch_size for x in meta_args.cp_split_sizes]

        # Attention heads [sq, b, h] --> [sq, b, q + qx + k + v]
        mixed_qqkv = self.linear_qkv.forward_ln(hidden_states)

        # =====================
        # Function wrapper
        # =====================
        get_kv_func = self.get_kv
        get_q_func = self.get_q
        get_qkv_func = self.get_qkv
        get_xqkv_func = self.get_xqkv

        # =====================
        # Parallel Strategy
        # =====================
        if self.engine_config.cp_strategy == "none":
            assert self.engine_config.cp_size == 1
            key_and_value = get_kv_func(mixed_qqkv, cos_emb, sin_emb)
            query = get_q_func(mixed_qqkv, cos_emb, sin_emb)
            key, value = self.adjust_key_and_value_for_inference(key_and_value, inference_params, meta_args)
            core_attn_out = self.core_attention(query, key, value, batch_size, meta_args)
            core_attn_out = rearrange(core_attn_out, "(sq b) hn hd -> sq b (hn hd)", b=batch_size)
            xattn_out = self.cross_attention(mixed_qqkv, key_value_states, meta_args.cross_attn_params, get_xqkv_func)

        elif self.engine_config.cp_strategy == "cp_ulysses":
            get_kv_func = partial(get_kv_func, mixed_qqkv, cos_emb, sin_emb)
            get_q_func = partial(get_q_func, mixed_qqkv, cos_emb, sin_emb)
            get_qkv_func = partial(get_qkv_func, mixed_qqkv, cos_emb, sin_emb)
            kv_cache_func = partial(
                self.adjust_key_and_value_for_inference, inference_params=inference_params, meta_args=meta_args
            )
            if meta_args.enable_cuda_graph and meta_args.denoising_range_num <= 3:
                # Temporal solution for first chunk opt
                core_attn_out, xattn_out = UlyssesScheduler.get_attn_and_xattn_with_fused_qkv_comm(
                    get_qkv_func,
                    kv_cache_func,
                    partial(self.core_attention, bs=batch_size, meta_args=meta_args),
                    partial(self.cross_attention, mixed_qqkv, key_value_states, meta_args.cross_attn_params, get_xqkv_func),
                    self.engine_config.ulysses_overlap_degree,
                    batch_size,
                    self.engine_config.cp_size,
                    batch_cp_split_sizes,
                )
            else:
                core_attn_out, xattn_out = UlyssesScheduler.get_attn_and_xattn_with_fused_kv_comm(
                    get_q_func,
                    get_kv_func,
                    kv_cache_func,
                    partial(self.core_attention, bs=batch_size, meta_args=meta_args),
                    partial(self.cross_attention, mixed_qqkv, key_value_states, meta_args.cross_attn_params, get_xqkv_func),
                    self.engine_config.ulysses_overlap_degree,
                    batch_size,
                    self.engine_config.cp_size,
                    batch_cp_split_sizes,
                )

        elif self.engine_config.cp_strategy == "cp_shuffle_overlap":
            key_and_value = self.get_kv(mixed_qqkv, cos_emb, sin_emb)
            key_and_value, handle_kv = cso_communication(key_and_value, self.engine_config.cp_size, batch_cp_split_sizes, "kv")

            query = get_q_func(mixed_qqkv, cos_emb, sin_emb)
            cso_helper = CSOHelper(meta_args.denoising_range_num, self.engine_config.cp_size, batch_cp_split_sizes)
            query, handle_q = cso_helper.split_query_for_overlap(query)

            handle_kv.wait()
            # NOTE(lml): rearrange and unpad key_and_value for later attention compute under cp_shuffle_overlap strategy, and we should split sqb into sq and b when support multi batch_size input.
            key_and_value = (
                rearrange(
                    key_and_value,
                    "(cp dn sqb) hn nhd -> dn (cp sqb) hn nhd",
                    dn=meta_args.denoising_range_num,
                    cp=self.engine_config.cp_size,
                )[:, : meta_args.clip_token_nums]
                .flatten(0, 1)
                .contiguous()
            )
            key, value = self.adjust_key_and_value_for_inference(key_and_value, inference_params, meta_args)

            handle_q.wait()
            core_attn_out, handle_attn = cso_helper.overlap(
                partial(self.full_attention, hidden_states.shape[1], meta_args), query, key, value
            )
            xattn_out = self.cross_attention(mixed_qqkv, key_value_states, meta_args.cross_attn_params, get_xqkv_func)

            handle_attn.wait()
            core_attn_out = rearrange(
                torch.concat(core_attn_out, dim=0),
                "(dn cp sq b)  hn hd -> (dn sq) b (cp hn hd)",
                cp=self.engine_config.cp_size,
                b=hidden_states.shape[1],
                dn=meta_args.denoising_range_num,
            )
        else:
            raise ValueError(f"Unsupported cp_strategy: {self.engine_config.cp_strategy}")

        return core_attn_out, xattn_out


##########################################################
# TransformerLayer
##########################################################
class TransformerLayer(torch.nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, model_config: ModelConfig, engine_config: EngineConfig, layer_number: int = 1):
        super().__init__()
        self.model_config = model_config
        self.engine_config = engine_config
        self.layer_number = layer_number + self._get_layer_offset()
        ## [Module 1: ada_modulate_layer
        self.ada_modulate_layer = AdaModulateLayer(model_config=self.model_config)

        ## [Module 2: SelfAttention]
        self.self_attention = FullyParallelAttention(
            model_config=self.model_config, engine_config=self.engine_config, layer_number=self.layer_number
        )

        ## [Module 3: SelfAttention PostNorm]
        self.self_attn_post_norm = FusedLayerNorm(model_config=self.model_config, hidden_size=self.model_config.hidden_size)

        ## [Module 4: MLP block]
        self.mlp = CustomMLP(model_config=self.model_config, engine_config=self.engine_config, layer_number=self.layer_number)

        ## [Module 5: MLP PostNorm]
        self.mlp_post_norm = FusedLayerNorm(model_config=self.model_config, hidden_size=self.model_config.hidden_size)

    def _get_layer_offset(self):
        pipeline_rank = parallel_state.get_pp_rank()

        num_layers_per_pipeline_rank = self.model_config.num_layers // parallel_state.get_pp_world_size()

        # Each stage gets a contiguous set of layers.
        if parallel_state.get_pp_world_size() > 1:
            offset = pipeline_rank * num_layers_per_pipeline_rank
        else:
            offset = 0

        return offset

    def forward(
        self,
        hidden_states: torch.Tensor,
        condition: torch.Tensor,
        condition_map: torch.Tensor,
        y_xattn_flat: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        inference_params: InferenceParams,
        meta_args: ModelMetaArgs,
    ):
        # hidden_states: [s/cp/sp, b, h]
        residual = hidden_states

        # Self attention.
        core_attn_out, cross_attn_out = self.self_attention(
            hidden_states,
            key_value_states=y_xattn_flat,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            meta_args=meta_args,
        )

        hidden_states = self.attn_post_process(core_attn_out, cross_attn_out, residual, condition, condition_map)

        return hidden_states

    def attn_post_process(
        self,
        core_attn_out: torch.Tensor,
        cross_attn_out: torch.Tensor,
        residual: torch.Tensor,
        condition: torch.Tensor,
        condition_map: torch.Tensor,
    ):
        hidden_states = self.attn_linear_proj(core_attn_out, cross_attn_out)
        hidden_states = self.gating_and_mlp(hidden_states, residual, condition, condition_map)
        return hidden_states

    def attn_linear_proj(self, core_attn_out: torch.Tensor, cross_attn_out: torch.Tensor):
        # ============================================
        # attention post-process , output. [sq, b, h]
        # ============================================
        attn_out = torch.concat([core_attn_out, cross_attn_out], dim=2)
        # NOTE: hn=8 is hardcoded to align with TP8 traning and TP1 inference
        attn_out = rearrange(attn_out, "sq b (n hn hd) -> sq b (hn n hd)", n=2, hn=8)
        if self.self_attention.adapt_linear_quant:
            attn_out = self.self_attention.linear_proj(attn_out)
        else:
            # Use high-precision for non-quantized linear projection
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                attn_out = self.self_attention.linear_proj(attn_out)

        return attn_out

    def gating_and_mlp(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, condition: torch.Tensor, condition_map: torch.Tensor
    ):
        gate_output = self.ada_modulate_layer(condition)
        softcap_gate_cap = 1.0
        gate_output = softcap(gate_output, softcap_gate_cap)
        gate_msa, gate_mlp = gate_output.chunk(2, dim=-1)

        # Residual connection for self-attention.
        hidden_states = bias_modulate_add(hidden_states, residual, condition_map, gate_msa, self.self_attn_post_norm).to(
            self.model_config.params_dtype
        )

        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        # Residual connection for MLP.
        hidden_states = bias_modulate_add(hidden_states, residual, condition_map, gate_mlp, self.mlp_post_norm).to(
            self.model_config.params_dtype
        )
        return hidden_states


##########################################################
# TransformerBlock
##########################################################
class TransformerBlock(torch.nn.Module):
    """Transformer class."""

    def __init__(
        self, model_config: ModelConfig, engine_config: EngineConfig, pre_process: bool = True, post_process: bool = True
    ):
        super().__init__()

        self.model_config = model_config
        self.engine_config = engine_config
        self.pre_process = pre_process
        self.post_process = post_process

        # required for pipeline parallel schedules
        self.input_tensor = None

        layer_number = self.model_config.num_layers // parallel_state.get_pp_world_size()
        # offset is implicit in TransformerLayer
        self.layers = torch.nn.ModuleList(
            [
                TransformerLayer(model_config=self.model_config, engine_config=self.engine_config, layer_number=i)
                for i in range(layer_number)
            ]
        )
        if self.post_process:
            # Final layer norm before output.
            self.final_layernorm = FusedLayerNorm(model_config=self.model_config, hidden_size=self.model_config.hidden_size)

    def set_input_tensor(self, input_tensor: Tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    @torch.no_grad()
    def forward(
        self,
        hidden_states: Tensor,
        condition: Tensor,
        condition_map: Tensor,
        y_xattn_flat: Tensor,
        rotary_pos_emb: Tensor,
        inference_params: InferenceParams,
        meta_args: ModelMetaArgs,
    ) -> torch.Tensor:
        if not self.pre_process:
            assert self.input_tensor is not None, "please call set_input_tensor for pp"
            hidden_states = self.input_tensor

        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                condition=condition,
                condition_map=condition_map,
                y_xattn_flat=y_xattn_flat,
                rotary_pos_emb=rotary_pos_emb,
                inference_params=inference_params,
                meta_args=meta_args,
            )

        # Final layer norm.
        if self.post_process:
            hidden_states = self.final_layernorm(hidden_states.float())

        return hidden_states
