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

import torch

from inference.infra.parallelism.tile_parallel import TileProcessor


def test_latent_tile_uses_ceiling_division():
    processor = TileProcessor(
        encode_fn=lambda x: x,
        decode_fn=lambda x: x,
        tile_sample_min_height=7,
        tile_sample_min_width=7,
        tile_sample_min_length=1,
        spatial_downsample_factor=8,
        temporal_downsample_factor=4,
    )

    assert processor.tile_latent_min_height == 1
    assert processor.tile_latent_min_width == 1
    assert processor.tile_latent_min_length == 1


def test_single_latent_frame_decode_has_positive_stride():
    def decode(tile):
        return tile.expand(-1, -1, 4, 8, 8)

    processor = TileProcessor(
        encode_fn=lambda x: x,
        decode_fn=decode,
        tile_sample_min_height=8,
        tile_sample_min_width=8,
        tile_sample_min_length=1,
        spatial_downsample_factor=8,
        temporal_downsample_factor=4,
        spatial_tile_overlap_factor=0.0,
        temporal_tile_overlap_factor=0.0,
    )

    latent = torch.ones((1, 1, 1, 1, 1))
    decoded = processor.tiled_decode(latent)

    assert decoded.shape == (1, 1, 4, 8, 8)
    assert torch.isfinite(decoded).all()


def test_full_overlap_factor_still_makes_progress():
    assert TileProcessor._tile_stride(1, 1.0) == 1
    assert TileProcessor._tile_stride(16, 2.0) == 1
