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

import json
import os
from abc import ABC, abstractmethod
from typing import Literal

import torch
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config

from inference.infra.parallelism import TileProcessor

from .vae_module import DiagonalGaussianDistribution, ViTDecoder, ViTEncoder


class VideoTokenizerABC(ABC):
    """
    Abstract base class for video tokenizers.

    This class defines the interface for video tokenizers and provides common methods and properties.
    """

    @property
    @abstractmethod
    def spatial_downsample_factor(self):
        """
        Property representing the spatial downsample factor.

        Returns:
            int: The spatial downsample factor.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def temporal_downsample_factor(self):
        """
        Property representing the temporal downsample factor.

        Returns:
            int: The temporal downsample factor.
        """
        raise NotImplementedError

    @property
    def first_frame_as_image(self):
        """
        Property representing the first frame as image.
        For tokenizer like CausalVAE, Omnitokenizer, the first frame is treated as image.
        in this case if the temporal downsample factor is 4, the input should be 4*x+1, and encoded tensor would be x+1.
        for example encode 65 frames to 17 frames. and decode 17 frames to 65 frames.

        Returns:
            bool: The first frame as image.
        """
        return False

    @property
    def allow_spatial_tiling(self):
        """
        Determines whether spatial tiling is allowed or not.

        Returns:
            bool: True if spatial tiling is allowed, False otherwise.
        """
        return True

    @abstractmethod
    def encode(self, x) -> torch.Tensor:
        """
        Abstract method for encoding the input tensor.

        Args:
            x (torch.Tensor [N C T H W] range[-1, 1]): The input tensor to be encoded.

        Returns:
            torch.Tensor: The encoded tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, x) -> torch.Tensor:
        """
        Abstract method for decoding the input tensor.

        Args:
            x (torch.Tensor [N C T H W]): The input tensor to be decoded.

        Returns:
            torch.Tensor [N C T H W] range[-1, 1]: The decoded tensor.
        """
        raise NotImplementedError

    def tile_processor(
        self,
        tile_sample_min_height=256,
        tile_sample_min_width=256,
        tile_sample_min_length=16,
        spatial_tile_overlap_factor: float = 0.25,
        temporal_tile_overlap_factor: float = 0,
        parallel_group: torch.distributed.ProcessGroup = None,
    ) -> TileProcessor:
        """
        Property representing the tiled encoder or decoder.

        Returns:
            TileProcessor: The tiled encoder or decoder.
        """
        return TileProcessor(
            encode_fn=self.encode,
            decode_fn=self.decode,
            tile_sample_min_height=tile_sample_min_height,
            tile_sample_min_width=tile_sample_min_width,
            tile_sample_min_length=tile_sample_min_length,
            spatial_tile_overlap_factor=spatial_tile_overlap_factor,
            temporal_tile_overlap_factor=temporal_tile_overlap_factor,
            sr_ratio=getattr(self, 'sr_ratio', 1),
            spatial_downsample_factor=self.spatial_downsample_factor,
            temporal_downsample_factor=self.temporal_downsample_factor,
            first_frame_as_image=self.first_frame_as_image,
            parallel_group=parallel_group,
        )

    @torch.inference_mode()
    def tiled_encode_3d(
        self,
        x,
        tile_sample_min_height=256,
        tile_sample_min_width=256,
        tile_sample_min_length: int = 16,
        spatial_tile_overlap_factor: float = 0.25,
        temporal_tile_overlap_factor: float = 0,
        allow_spatial_tiling: bool = None,
        verbose: bool = False,
        parallel_group: torch.distributed.ProcessGroup = None,
    ) -> torch.Tensor:
        """
        Encodes the input tensor `x` using tiled encoding.

        Args:
            x (torch.Tensor shape:[N C T H W]): The input tensor to be encoded.
            tile_sample_min_height (int, optional): The minimum height of each tile sample. Defaults to 256.
            tile_sample_min_width (int, optional): The minimum width of each tile sample. Defaults to 256.
            tile_sample_min_length (int, optional): The minimum length of each tile sample. Defaults to 16.
            spatial_tile_overlap_factor (float, optional): Overlap factor for spatial tiles. Defaults to 0.25.
            temporal_tile_overlap_factor (float, optional): Overlap factor for temporal tiles. Defaults to 0.
            allow_spatial_tiling (bool, optional): Whether spatial tiling is allowed. Defaults to None.
            verbose (bool, optional): Whether to print verbose information. Defaults to False.
            parallel_group (torch.distributed.ProcessGroup, optional): Distributed encoding group. Defaults to None.
        Returns:
            torch.Tensor: The encoded tensor.
        """
        allow_spatial_tiling = allow_spatial_tiling if allow_spatial_tiling is not None else self.allow_spatial_tiling
        if not allow_spatial_tiling:
            tile_sample_min_height = 100000
            tile_sample_min_width = 100000
        return self.tile_processor(
            tile_sample_min_height=tile_sample_min_height,
            tile_sample_min_width=tile_sample_min_width,
            tile_sample_min_length=tile_sample_min_length,
            spatial_tile_overlap_factor=spatial_tile_overlap_factor,
            temporal_tile_overlap_factor=temporal_tile_overlap_factor,
            parallel_group=parallel_group,
        ).tiled_encode(x, verbose)

    @torch.inference_mode()
    def tiled_decode_3d(
        self,
        x,
        tile_sample_min_height=256,
        tile_sample_min_width=256,
        tile_sample_min_length: int = 16,
        spatial_tile_overlap_factor: float = 0.25,
        temporal_tile_overlap_factor: float = 0,
        allow_spatial_tiling: bool = None,
        verbose: bool = False,
        parallel_group: torch.distributed.ProcessGroup = None,
    ) -> torch.Tensor:
        """
        Decodes the input tensor using the tile autoencoder.

        Args:
            x (torch.Tensor): The input tensor to be decoded.
            tile_sample_min_height (int, optional): The minimum height of each tile sample. Defaults to 256.
            tile_sample_min_width (int, optional): The minimum width of each tile sample. Defaults to 256.
            tile_sample_min_length (int, optional): The minimum length of each tile sample. Defaults to 16.
            spatial_tile_overlap_factor (float, optional): Overlap factor for spatial tiles. Defaults to 0.25.
            temporal_tile_overlap_factor (float, optional): Overlap factor for temporal tiles. Defaults to 0.
            allow_spatial_tiling (bool, optional): Whether spatial tiling is allowed. Defaults to None.
            verbose (bool, optional): Whether to print verbose information. Defaults to False.
            parallel_group (torch.distributed.ProcessGroup, optional): Distributed decoding group. Defaults to None.
        Returns:
            torch.Tensor shape:[N C T H W]: The decoded tensor.
        """
        allow_spatial_tiling = allow_spatial_tiling if allow_spatial_tiling is not None else self.allow_spatial_tiling
        if not allow_spatial_tiling:
            tile_sample_min_height = 100000
            tile_sample_min_width = 100000
        return self.tile_processor(
            tile_sample_min_height=tile_sample_min_height,
            tile_sample_min_width=tile_sample_min_width,
            tile_sample_min_length=tile_sample_min_length,
            spatial_tile_overlap_factor=spatial_tile_overlap_factor,
            temporal_tile_overlap_factor=temporal_tile_overlap_factor,
            parallel_group=parallel_group,
        ).tiled_decode(x, verbose)


class ViTVAE(ModelMixin, ConfigMixin, VideoTokenizerABC):
    @register_to_config
    def __init__(self, ddconfig: dict, model_type: Literal['vit', 'vit_ncthw'] = 'vit'):
        super().__init__()

        if model_type == 'vit':
            self.encoder = ViTEncoder(**ddconfig)
            self.decoder = ViTDecoder(**ddconfig)
        elif model_type == 'vit_ncthw':
            from videotokenizer.modules.vit_ncthw import ViTDecoderNCTHW, ViTEncoderNCTHW

            self.encoder = ViTEncoderNCTHW(**ddconfig)
            self.decoder = ViTDecoderNCTHW(**ddconfig)
        else:
            raise ValueError(f"model_type {model_type} not supported")

        if 'patch_length' in ddconfig:
            self._temporal_downsample_factor = ddconfig['patch_length']
        else:
            self._temporal_downsample_factor = 1

        if 'patch_size' in ddconfig:
            self._spatial_downsample_factor = ddconfig['patch_size']
        else:
            self._spatial_downsample_factor = 8

    @property
    def spatial_downsample_factor(self):
        return self._spatial_downsample_factor

    @property
    def temporal_downsample_factor(self):
        return self._temporal_downsample_factor

    def init_from_ckpt(self, path, ignore_keys=list()):
        raise NotImplementedError

    def encode(self, x, sample_posterior=True):
        """
        Encode the input video.

        Args:
            x (torch.Tensor): Input video tensor has shape N C T H W

        Returns:
            tuple: Tuple containing the quantized tensor, embedding loss, and additional information.
        """
        N, C, T, H, W = x.shape
        if T == 1 and self._temporal_downsample_factor > 1:
            x = x.expand(-1, -1, 4, -1, -1)
            x = self.encoder(x)
            posterior = DiagonalGaussianDistribution(x)
            if sample_posterior:
                z = posterior.sample()
            else:
                z = posterior.mode()

            return z[:, :, :1, :, :].type(x.dtype)
        else:
            x = self.encoder(x)
            posterior = DiagonalGaussianDistribution(x)
            if sample_posterior:
                z = posterior.sample()
            else:
                z = posterior.mode()

            return z.type(x.dtype)

    def decode(self, x):
        """
        Decode the quantized tensor.

        Args:
            quant (torch.Tensor): Quantized tensor.

        Returns:
            torch.Tensor: Decoded tensor.
        """
        N, C, T, H, W = x.shape
        if T == 1:
            x = x.expand(-1, -1, 1, -1, -1)
            x = self.decoder(x)
            x = x[:, :, :1, :, :]
            return x
        else:
            x = self.decoder(x)
            return x

    def forward(self, x, sample_posterior=True):
        x = self.encoder(x)
        posterior = DiagonalGaussianDistribution(x)

        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        dec = self.decoder(z)
        return dec, posterior

    def get_last_layer(self):
        """
        Get the last layer of the decoder.

        Returns:
            torch.Tensor: Last layer of the decoder.
        """
        return self.decoder.last_layer.weight

    @property
    def allow_spatial_tiling(self):
        return False


class AutoModel:
    r"""
    :class:`~models.AutoModel` is a generic model class
    that will be instantiated as one of the base model classes of the library
    when created with the `AutoModel.from_pretrained(pretrained_model_name_or_path)`


    This class cannot be instantiated using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated "
            "using the `AutoModel.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs) -> VideoTokenizerABC:
        config = os.path.join(pretrained_model_name_or_path, 'config.json')
        if not os.path.exists(config):
            raise ValueError("Can't find a model config file at {}.".format(config))
        # Load config
        with open(config, 'r') as json_file:
            config_dict = json.load(json_file)
        assert config_dict['_class_name'] == 'ViTVAE'
        return ViTVAE.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
