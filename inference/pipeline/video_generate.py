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
from collections import Counter
from dataclasses import dataclass, field
from queue import Queue
from typing import Dict, Generator, List, Tuple, Union

import torch
import torch.distributed as dist
from tqdm import tqdm

import inference.infra.distributed.parallel_state as mpu
from inference.common import InferenceParams, event_path_timer, print_rank_0
from inference.infra.parallelism import pp_scheduler

from .prompt_process import get_negative_special_token_keys, get_special_token_keys, pad_special_token


@dataclass(frozen=True)
class InferenceInput:
    caption_embs: torch.Tensor
    emb_masks: torch.Tensor
    y: torch.Tensor
    prefix_video: Union[torch.Tensor, None]
    latent_size: Tuple[int]
    t_schedule_config: Dict = field(default_factory=dict)
    num_steps: int = None
    vae_ckpt: str = None
    task_idx_list: List[int] = None
    report_chunk_num_list: List[int] = None
    chunk_num: int = None


def _process_txt_embeddings(
    caption_embs: torch.Tensor, emb_masks: torch.Tensor, null_emb: torch.Tensor, infer_chunk_num: int, clean_chunk_num: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    special_token_keys = get_special_token_keys()
    print_rank_0(f"special_token = {list(special_token_keys)}")

    # denoise chunk with caption_embs
    caption_embs = caption_embs.repeat(1, infer_chunk_num - clean_chunk_num, 1, 1)
    emb_masks = emb_masks.unsqueeze(1).repeat(1, infer_chunk_num - clean_chunk_num, 1)
    caption_embs, emb_masks = pad_special_token(special_token_keys, caption_embs, emb_masks)

    # clean chunk with null_emb
    caption_embs = torch.cat([null_emb.repeat(1, clean_chunk_num, 1, 1), caption_embs], dim=1)
    emb_masks = torch.cat(
        [torch.zeros(1, clean_chunk_num, emb_masks.size(2), dtype=emb_masks.dtype, device=emb_masks.device), emb_masks], dim=1
    )
    return caption_embs, emb_masks


def _process_null_embeddings(
    null_caption_embedding: torch.Tensor, null_emb_masks: torch.Tensor, infer_chunk_num: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    null_embs = null_caption_embedding.repeat(1, infer_chunk_num, 1, 1)
    negative_special_token_keys = get_negative_special_token_keys()
    if negative_special_token_keys:
        null_embs, _ = pad_special_token(negative_special_token_keys, null_embs, None)

    null_token_length = 50
    null_emb_masks[:, :, :null_token_length] = 1
    null_emb_masks[:, :, null_token_length:] = 0

    return null_embs, null_emb_masks


@torch.inference_mode()
def extract_feature_for_inference(
    model: torch.nn.Module, prompt: str, prefix_video: torch.Tensor, caption_embs: torch.Tensor, emb_masks: torch.Tensor
) -> InferenceInput:
    model_config = model.model_config
    runtime_config = model.runtime_config
    ### Prepare prefix video feature
    clean_chunk_num = 0
    if prefix_video is not None:
        clean_chunk_num = prefix_video.size(2) // runtime_config.chunk_width
        infer_chunk_num = math.ceil(
            (runtime_config.num_frames // runtime_config.temporal_downsample_factor * 1.0 + prefix_video.size(2))
            / runtime_config.chunk_width
        )
    else:
        infer_chunk_num = math.ceil(
            (runtime_config.num_frames // runtime_config.temporal_downsample_factor * 1.0) / runtime_config.chunk_width
        )

    ### Prepare text feature
    # [1, caption_max_length (800), hidden_size(4096)]
    null_caption_embedding = model.y_embedder.null_caption_embedding.unsqueeze(0)
    caption_embs, caption_emb_masks = _process_txt_embeddings(
        caption_embs, emb_masks, null_caption_embedding, infer_chunk_num, clean_chunk_num
    )
    null_emb_masks = torch.zeros_like(caption_emb_masks)
    null_embs, null_emb_masks = _process_null_embeddings(null_caption_embedding, null_emb_masks, infer_chunk_num)

    if len(prompt) == 0:
        emb_masks = torch.cat([null_emb_masks, null_emb_masks], dim=0)
        y = torch.cat([null_embs, null_embs])
    else:
        emb_masks = torch.cat([caption_emb_masks, null_emb_masks], dim=0)
        y = torch.cat([caption_embs, null_embs])

    ### Prepare latent feature dims
    in_channels = model_config.in_channels
    if model_config.half_channel_vae:
        in_channels = 16
    latent_size_t = infer_chunk_num * runtime_config.chunk_width
    latent_size_h = runtime_config.video_size_h // 8
    latent_size_w = runtime_config.video_size_w // 8

    return InferenceInput(
        caption_embs=caption_embs,
        emb_masks=emb_masks,
        y=y,
        prefix_video=prefix_video,
        latent_size=(1, in_channels, latent_size_t, latent_size_h, latent_size_w),  # NCTHW
        t_schedule_config={},
        num_steps=runtime_config.num_steps,
        task_idx_list=[0],
        report_chunk_num_list=[infer_chunk_num - clean_chunk_num],
        chunk_num=latent_size_t // runtime_config.chunk_width,
    )


# Example1: when chunk_num=8, window_size=8
# clip_start: [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7]
# clip_end  : [1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8]
# t_start   : [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7]
# t_end     : [1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8]

# Example2: when chunk_num=8, window_size=4
# clip_start: [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7]
# clip_end  : [1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8]
# t_start   : [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3]
# t_end     : [1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4]

# Example3: when chunk_num=8, window_size=4, chunk_offset=2
# clip_start: [2, 2, 2, 2, 3, 4, 5, 6, 7]
# clip_end  : [3, 4, 5, 6, 7, 8, 8, 8, 8]
# t_start   : [0, 0, 0, 0, 0, 0, 1, 2, 3]
# t_end     : [1, 2, 3, 4, 4, 4, 4, 4, 4]

# Example4: when chunk_num=8, window_size=1
# clip_start: [0, 1, 2, 3, 4, 5, 6, 7]
# clip_end  : [1, 2, 3, 4, 5, 6, 7, 8]
# t_start   : [0, 0, 0, 0, 0, 0, 0, 0]
# t_end     : [1, 1, 1, 1, 1, 1, 1, 1]


def generate_sequences(chunk_num, window_size, chunk_offset):
    # Adjust range to include the offset
    start_index = chunk_offset
    end_index = chunk_num + window_size - 1

    # Generate clip_start and clip_end
    clip_start = [max(chunk_offset, i - window_size + 1) for i in range(start_index, end_index)]
    clip_end = [min(chunk_num, i + 1) for i in range(start_index, end_index)]

    # Generate t_start and t_end
    t_start = [max(0, i - chunk_num + 1) for i in range(start_index, end_index)]
    t_end = [
        min(window_size, i - chunk_offset + 1) if i - chunk_offset < window_size else window_size
        for i in range(start_index, end_index)
    ]

    return clip_start, clip_end, t_start, t_end


def init_t(t_schedule_config: Union[Dict, None], num_steps: int, device: torch.device, shortcut_mode: str = ""):
    """Init Timestep and Transform t"""
    if num_steps == 12:
        base_t = torch.linspace(0, 1, 4 + 1, device=device) / 4
        accu_num = torch.linspace(0, 1, 4 + 1, device=device)
        if shortcut_mode == "16,16,8":
            base_t = base_t[:3]
        else:
            base_t = torch.cat([base_t[:1], base_t[2:4]], dim=0)
        t = torch.cat([base_t + accu for accu in accu_num], dim=0)[: (num_steps + 1)]
    else:
        t = torch.linspace(0, 1, num_steps + 1, device=device)
    t_schedule_func = t_schedule_config.get("tSchedulerFunc", "sd3")
    if t_schedule_func == "sd3":

        def t_resolution_transform(x, shift=3.0):
            # sd3: with a **reverse** time-schedule (0: clean, 1: noise)
            # ours (0: noise, 1: clean)
            # https://github.com/Stability-AI/sd3-ref/blob/master/sd3_impls.py#L33
            assert shift >= 1.0, "shift should >=1"
            shift_inv = 1.0 / shift
            return shift_inv * x / (1 + (shift_inv - 1) * x)

        t = t**2
        shift = t_schedule_config.get("shift", 3.0)
        t = t_resolution_transform(t, shift)
    elif t_schedule_func == "square":
        t = t**2
    elif t_schedule_func == "piecewise":

        def t_transform(x):
            mask = x < 0.875
            x[mask] = x[mask] * (0.5 / 0.875)
            x[~mask] = 0.5 + (x[~mask] - 0.875) * (0.5 / (1 - 0.875))
            return x

        t = t_transform(t)
    else:  # identity
        pass
    return t


def init_intervel(num_steps: int, device: torch.device, shortcut_mode: str = ""):
    """Init intervel"""
    base_intervel = torch.ones(num_steps, device=device)
    if num_steps % 3 == 0:
        repeat_times = num_steps // 3
        if shortcut_mode == "16,16,8":
            base_intervel = torch.tensor([1, 1, 2] * repeat_times, device=device)
        else:
            base_intervel = torch.tensor([2, 1, 1] * repeat_times, device=device)
    return base_intervel


@dataclass
class WorkStatus:
    infer_idx: int
    cur_denoise_step: int


def find_dit_model(model):
    if hasattr(model, "y_embedder"):
        return model
    if hasattr(model, "module"):
        return find_dit_model(model.module)
    raise ValueError("Cannot find the real model")


class SampleTransport:
    def __init__(self, model: torch.nn.Module, transport_inputs: List[InferenceInput], device: torch.device):
        # ========= Input Tensor =========
        self.model = model
        self.transport_inputs = transport_inputs
        self.device = device

        # ========= Init Global Members =========
        self.model_config = model.model_config
        self.runtime_config = model.runtime_config
        self.engine_config = model.engine_config
        self.chunk_width = self.runtime_config.chunk_width
        self.window_size = self.runtime_config.window_size

        # ========= Init Batched Inputs and Work Queue =========
        self.work_queue = Queue()
        self.chunk_denoise_count: List[Counter] = []
        self.ts: List[torch.Tensor] = []
        self.time_interval: List[torch.Tensor] = []
        self.xs: List[torch.Tensor] = []
        self.x_chunks: List[torch.Tensor] = []
        self.velocities: List[torch.Tensor] = []
        self.time_record: List[tqdm] = []
        self.inference_params: List[InferenceParams] = []
        self.init_work_queue()

    def init_work_queue(self) -> None:
        shortcut_mode = self.engine_config.shortcut_mode
        if mpu.get_pp_world_size() > 1:
            if len(self.transport_inputs) == 1:
                print_rank_0("Warning: For better performance, please use multiple inputs for PP>1")
        else:
            assert len(self.transport_inputs) == 1, "Only support single input for PP=1"

        for idx, tran_input in enumerate(self.transport_inputs):
            self.work_queue.put(WorkStatus(infer_idx=idx, cur_denoise_step=0))

            self.chunk_denoise_count.append(Counter())
            self.ts.append(
                init_t(tran_input.t_schedule_config, tran_input.num_steps, self.device, shortcut_mode=shortcut_mode)
            )
            self.time_interval.append(init_intervel(tran_input.num_steps, self.device, shortcut_mode=shortcut_mode))
            self.x_chunks.append(None)
            self.velocities.append(None)

            if torch.distributed.get_rank() == 0:
                report_chunk_num = sum(
                    dict(
                        zip(self.transport_inputs[idx].task_idx_list, self.transport_inputs[idx].report_chunk_num_list)
                    ).values()
                )

                progress_bar = tqdm(total=report_chunk_num, desc=f"InferBatch {idx}")
                self.time_record.append(progress_bar)

            print_rank_0(f"transport_inputs len: {len(self.transport_inputs)}")
            x = torch.randn(*tran_input.latent_size, device=self.device)  # NCTHW
            x = torch.cat([x, x], 0)  # [2 * N, C, T, H, W]
            self.xs.append(x)

            max_sequence_length = (
                x.shape[2] * (x.shape[3] // self.model_config.patch_size) * (x.shape[4] // self.model_config.patch_size)
            )
            self.inference_params.append(InferenceParams(max_batch_size=1, max_sequence_length=max_sequence_length))

    def append_dims(self, x, target_dims):
        """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
        return x[(...,) + (None,) * dims_to_append]

    def get_timestep(
        self,
        t_total: torch.Tensor,
        denoise_step_per_stage: int,
        start: int,
        end: int,
        denoise_idx: int,
        has_clean_t: bool = False,
    ) -> torch.Tensor:
        """Const Method"""
        t_index = []
        for i in range(start, end):
            t_index.append(i * denoise_step_per_stage + denoise_idx)
        t_index.reverse()
        timestep = t_total[t_index]
        if has_clean_t:
            ones = torch.ones(1, device=self.device) * self.runtime_config.clean_t
            timestep = torch.cat([ones, timestep], 0)
        return timestep

    def get_denoise_step_of_each_chunk(
        self,
        infer_idx: int,
        denoise_step_per_stage: int,
        t_start: int,
        t_end: int,
        denoise_idx: int,
        has_clean_t: bool = False,
    ):
        denoise_step_of_each_chunk = []
        for i in range(t_start, t_end):
            denoise_step_of_each_chunk.append(i * denoise_step_per_stage + denoise_idx)
        denoise_step_of_each_chunk.reverse()
        if has_clean_t:
            denoise_step_of_each_chunk = [self.transport_inputs[infer_idx].num_steps] + denoise_step_of_each_chunk
        return denoise_step_of_each_chunk

    def get_batch_size_and_chunk_token_nums(self, infer_idx: int):
        """Const Method"""
        batch_size = 1
        # T H W
        chunk_token_nums = (
            self.chunk_width
            * (self.transport_inputs[infer_idx].latent_size[3] // self.model_config.patch_size)
            * (self.transport_inputs[infer_idx].latent_size[4] // self.model_config.patch_size)
        )
        return batch_size, chunk_token_nums

    def generate_kvrange_for_prefix_video(self, infer_idx: int, range_num: int):
        """Const Method"""
        batch_size, chunk_token_nums = self.get_batch_size_and_chunk_token_nums(infer_idx)
        if self.runtime_config.clean_chunk_kvrange != -1:
            prev_chunk_num = self.runtime_config.clean_chunk_kvrange
        elif len(self.runtime_config.noise2clean_kvrange) > 0:
            prev_chunk_num = self.runtime_config.noise2clean_kvrange[-1]
        else:
            prev_chunk_num = 8

        k_chunk_end = torch.linspace(1, range_num, steps=range_num).reshape((range_num, 1))
        k_chunk_start = torch.clamp(k_chunk_end - prev_chunk_num, min=0).reshape((range_num, 1))
        k_chunk_range = torch.concat([k_chunk_start, k_chunk_end], dim=1)
        k_batch_range = (
            torch.concat([k_chunk_range + i * range_num for i in range(batch_size)], dim=0).to(torch.int32).to(self.device)
        )
        return k_batch_range * chunk_token_nums

    def extract_prefix_video_feature(
        self, infer_idx: int, prefix_video: torch.Tensor, y: torch.Tensor, chunk_offset: int, model_kwargs: dict
    ):
        """Non-Const Method"""
        print_rank_0(f"extract clean feature for prefix video, chunk_offset: {chunk_offset}")

        x_chunk = prefix_video[:, :, : chunk_offset * self.chunk_width]
        x_chunk = torch.cat([x_chunk, x_chunk], 0)  # [2 * N, C, T, H, W]

        # clean feature without y embedding
        null_y_chunk = self.transport_inputs[infer_idx].y[1:2, :chunk_offset]
        null_y_chunk = torch.cat([null_y_chunk, null_y_chunk], 0)
        mask_chunk = self.transport_inputs[infer_idx].emb_masks[1:2, :chunk_offset]
        mask_chunk = torch.cat([mask_chunk, mask_chunk], 0)

        null_y_chunk_flatten = null_y_chunk.flatten(start_dim=0, end_dim=1).unsqueeze(1)
        mask_chunk_flatten = mask_chunk.flatten(start_dim=0, end_dim=1).unsqueeze(1)

        t = torch.ones(chunk_offset, device=self.device) * self.runtime_config.clean_t
        t = t.unsqueeze(0).repeat(x_chunk.size(0), 1)

        fwd_model_kwargs = model_kwargs.copy()
        fwd_model_kwargs.update(
            {
                "slice_point": 0,
                "range_num": chunk_offset,
                "denoising_range_num": chunk_offset,
                "fwd_extra_1st_chunk": False,
                "extract_prefix_video_feature": True,
            }
        )

        kv_range = self.generate_kvrange_for_prefix_video(infer_idx, chunk_offset)

        forward_fn = find_dit_model(self.model).forward_dispatcher
        fwd_model_kwargs["distill_interval"] = self.time_interval[infer_idx][0]
        forward_fn(
            x=x_chunk,
            timestep=t,
            y=null_y_chunk_flatten,
            mask=mask_chunk_flatten,
            kv_range=kv_range,
            inference_params=self.inference_params[infer_idx],
            **fwd_model_kwargs,
        )  # for kv cache

    def try_pad_prefix_video(
        self, infer_idx: int, x_chunk: torch.Tensor, t: torch.Tensor, prefix_video_start: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Non-Const Method"""
        prefix_length = self.transport_inputs[infer_idx].prefix_video.size(2)

        if prefix_length <= prefix_video_start:
            return x_chunk, t

        padding_length = min(prefix_length - prefix_video_start, x_chunk.size(2))
        prefix_video_end = prefix_video_start + padding_length
        ret = x_chunk.clone()
        ret[:, :, :padding_length] = self.transport_inputs[infer_idx].prefix_video[:, :, prefix_video_start:prefix_video_end]

        num_clean_t = (prefix_length - prefix_video_start) // self.chunk_width
        if num_clean_t > 0:
            t[:, :num_clean_t] = 1.0
        return ret, t

    def generate_default_kvrange(self, infer_idx: int, slice_point: int, denoising_range_num: int) -> torch.Tensor:
        """Const Method"""
        batch_size, chunk_token_nums = self.get_batch_size_and_chunk_token_nums(infer_idx)
        range_num = slice_point + denoising_range_num

        k_chunk_end = torch.linspace(slice_point + 1, range_num, steps=denoising_range_num).reshape((denoising_range_num, 1))
        k_chunk_start = torch.Tensor([0] * denoising_range_num).reshape((denoising_range_num, 1))
        k_chunk_range = torch.concat([k_chunk_start, k_chunk_end], dim=1)
        k_batch_range = (
            torch.concat([k_chunk_range + i * range_num for i in range(batch_size)], dim=0).to(torch.int32).to(self.device)
        )
        return k_batch_range * chunk_token_nums

    def generate_noise2clean_kvrange(
        self,
        infer_idx: int,
        slice_point: int,
        denoising_range_num: int,
        noise2clean_kvrange: List[int],
        clean_chunk_kvrange: int,
        denoise_step_of_each_chunk: List[int],
    ) -> torch.Tensor:
        """Const Method"""
        assert len(denoise_step_of_each_chunk) == denoising_range_num
        assert len(noise2clean_kvrange) > 0

        if clean_chunk_kvrange == -1:
            clean_chunk_kvrange = noise2clean_kvrange[-1]
        num_steps = self.transport_inputs[infer_idx].num_steps
        assert num_steps % len(noise2clean_kvrange) == 0
        denoise_step_per_stage = num_steps // len(noise2clean_kvrange)
        denoise_kv_range = []
        for cur_chunk_denoise_step in denoise_step_of_each_chunk:
            if cur_chunk_denoise_step == num_steps:
                denoise_kv_range.append(clean_chunk_kvrange)
            else:
                denoise_kv_range.append(noise2clean_kvrange[cur_chunk_denoise_step // denoise_step_per_stage])

        range_num = slice_point + denoising_range_num
        batch_size, chunk_token_nums = self.get_batch_size_and_chunk_token_nums(infer_idx)
        k_ranges = []
        for i in range(batch_size):
            k_batch_start = i * range_num
            for j in range(denoising_range_num):
                k_chunk_end = slice_point + j + 1
                k_chunk_start = max(0, k_chunk_end - denoise_kv_range[j])
                k_ranges.append(
                    torch.Tensor(
                        [(k_batch_start + k_chunk_start) * chunk_token_nums, (k_batch_start + k_chunk_end) * chunk_token_nums]
                    )
                    .reshape(1, 2)
                    .to(self.device)
                )
        k_range = torch.concat(k_ranges, dim=0).to(torch.int32).to(self.device)
        return k_range

    def generate_kvrange_for_denoising_video(
        self, infer_idx: int, slice_point: int, denoising_range_num: int, denoise_step_of_each_chunk: List[int]
    ) -> torch.Tensor:
        """Const Method"""
        noise2clean_kvrange = self.runtime_config.noise2clean_kvrange
        clean_chunk_kvrange = self.runtime_config.clean_chunk_kvrange
        if len(noise2clean_kvrange) == 0:
            k_range = self.generate_default_kvrange(infer_idx, slice_point, denoising_range_num)
        else:
            k_range = self.generate_noise2clean_kvrange(
                infer_idx,
                slice_point,
                denoising_range_num,
                noise2clean_kvrange,
                clean_chunk_kvrange,
                denoise_step_of_each_chunk,
            )
        return k_range

    def integrate(
        self,
        x_chunk: torch.Tensor,
        velocity: torch.Tensor,
        t_total: torch.Tensor,
        denoise_step_per_stage: int,
        t_start: int,
        t_end: int,
        i: int,
    ) -> torch.Tensor:
        """Non-Const Method"""
        t_before = self.get_timestep(t_total, denoise_step_per_stage, t_start, t_end, i)
        t_after = self.get_timestep(t_total, denoise_step_per_stage, t_start, t_end, i + 1)
        delta_t = t_after - t_before
        N, C, T, H, W = x_chunk.shape
        x_chunk = x_chunk.reshape(N, C, -1, self.chunk_width, H, W)
        velocity = velocity.reshape(N, C, -1, self.chunk_width, H, W)
        assert x_chunk.size(2) == delta_t.size(0)
        x_chunk = x_chunk + velocity * delta_t.reshape(1, 1, -1, 1, 1, 1)
        x_chunk = x_chunk.reshape(N, C, T, H, W)
        return x_chunk

    def generate_denoise_status_and_sequences(
        self, infer_idx: int, cur_denoise_step: int
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int, int]]:
        """Const Method"""
        chunk_offset = 0
        if self.transport_inputs[infer_idx].prefix_video is not None:
            chunk_offset = self.transport_inputs[infer_idx].prefix_video.size(2) // self.chunk_width

        transport_input = self.transport_inputs[infer_idx]
        denoise_step_per_stage = transport_input.num_steps // self.window_size
        denoise_stage, denoise_idx = (cur_denoise_step // denoise_step_per_stage, cur_denoise_step % denoise_step_per_stage)
        chunk_start_s, chunk_end_s, t_start_s, t_end_s = generate_sequences(
            transport_input.chunk_num, self.window_size, chunk_offset
        )
        chunk_start, chunk_end, t_start, t_end = (
            chunk_start_s[denoise_stage],
            chunk_end_s[denoise_stage],
            t_start_s[denoise_stage],
            t_end_s[denoise_stage],
        )
        return (denoise_step_per_stage, denoise_stage, denoise_idx), (chunk_offset, chunk_start, chunk_end, t_start, t_end)

    def total_forward_step(self, infer_idx: int) -> int:
        denoise_step_per_stage = self.transport_inputs[infer_idx].num_steps // self.window_size

        chunk_offset = 0
        if self.transport_inputs[infer_idx].prefix_video is not None:
            chunk_offset = self.transport_inputs[infer_idx].prefix_video.size(2) // self.chunk_width

        total_forward_step = denoise_step_per_stage * (
            self.transport_inputs[infer_idx].chunk_num + self.window_size - 1 - chunk_offset
        )
        return total_forward_step

    def forward_velocity(self, infer_idx: int, cur_denoise_step: int) -> torch.Tensor:
        # 1. Get current work status
        x = self.xs[infer_idx]
        transport_input = self.transport_inputs[infer_idx]

        # 2. Extract prefix video KV cache
        (denoise_step_per_stage, denoise_stage, denoise_idx), (
            chunk_offset,
            chunk_start,
            chunk_end,
            t_start,
            t_end,
        ) = self.generate_denoise_status_and_sequences(infer_idx, cur_denoise_step)
        model_kwargs = dict(chunk_width=self.chunk_width, fwd_extra_1st_chunk=False, num_steps=transport_input.num_steps)

        if chunk_offset > 0 and cur_denoise_step == 0:
            self.extract_prefix_video_feature(
                infer_idx, transport_input.prefix_video, transport_input.y, chunk_offset, model_kwargs
            )

        # 3. Prepare inputs
        x_chunk = x[:, :, chunk_start * self.chunk_width : chunk_end * self.chunk_width].clone()
        y_chunk = transport_input.y[:, chunk_start:chunk_end]
        mask_chunk = transport_input.emb_masks[:, chunk_start:chunk_end]
        model_kwargs.update(
            {"slice_point": chunk_start, "range_num": chunk_end, "denoising_range_num": chunk_end - chunk_start}
        )

        # 4. Forward clean chunk and get clean kv
        fwd_extra_1st_chunk = chunk_start > chunk_offset and denoise_idx == 0
        if fwd_extra_1st_chunk:
            clean_x = x[:, :, (chunk_start - 1) * self.chunk_width : chunk_start * self.chunk_width].clone()
            x_chunk = torch.cat([clean_x, x_chunk], dim=2)

            # clean feature without y embedding
            y_chunk = torch.cat([transport_input.y[1:2, 0:1].expand(y_chunk.size(0), -1, -1, -1), y_chunk], dim=1)
            mask_chunk = torch.cat([transport_input.emb_masks[1:2, 1:2].expand(mask_chunk.size(0), -1, -1), mask_chunk], dim=1)

            model_kwargs["slice_point"] = chunk_start - 1
            model_kwargs["denoising_range_num"] = chunk_end - chunk_start + 1
            model_kwargs["fwd_extra_1st_chunk"] = True

        # 5. Prepare inputs
        y_chunk_flatten = y_chunk.flatten(start_dim=0, end_dim=1).unsqueeze(1)
        mask_chunk_flatten = mask_chunk.flatten(start_dim=0, end_dim=1).unsqueeze(1)
        denoise_step_of_each_chunk = self.get_denoise_step_of_each_chunk(
            infer_idx, denoise_step_per_stage, t_start, t_end, denoise_idx, has_clean_t=fwd_extra_1st_chunk
        )
        t = self.get_timestep(
            self.ts[infer_idx], denoise_step_per_stage, t_start, t_end, denoise_idx, has_clean_t=fwd_extra_1st_chunk
        )
        t = t.unsqueeze(0).repeat(x_chunk.size(0), 1)  # [batch_size, chunk_size]
        kv_range = self.generate_kvrange_for_denoising_video(
            infer_idx=infer_idx,
            slice_point=model_kwargs["slice_point"],
            denoising_range_num=model_kwargs["denoising_range_num"],
            denoise_step_of_each_chunk=denoise_step_of_each_chunk,
        )

        # 6. Padding prefix video
        if transport_input.prefix_video is not None:
            x_chunk, t = self.try_pad_prefix_video(
                infer_idx, x_chunk, t, prefix_video_start=model_kwargs["slice_point"] * self.chunk_width
            )

        # 7. Model forward
        forward_fn = find_dit_model(self.model).forward_dispatcher
        nearly_clean_chunk_t = t[0, int(model_kwargs["fwd_extra_1st_chunk"])].item()
        model_kwargs["distill_nearly_clean_chunk"] = (
            nearly_clean_chunk_t > self.engine_config.distill_nearly_clean_chunk_threshold
        )
        model_kwargs["distill_interval"] = self.time_interval[infer_idx][denoise_idx]
        velocity = forward_fn(
            x=x_chunk,
            timestep=t,
            y=y_chunk_flatten,
            mask=mask_chunk_flatten,
            kv_range=kv_range,
            inference_params=self.inference_params[infer_idx],
            **model_kwargs,
        )
        self.x_chunks[infer_idx] = x_chunk
        self.velocities[infer_idx] = velocity
        return velocity

    def integrate_velocity(self, infer_idx: int, cur_denoise_step: int):
        transport_input = self.transport_inputs[infer_idx]
        x_chunk = self.x_chunks[infer_idx]
        velocity = self.velocities[infer_idx]
        chunk_denoise_count = self.chunk_denoise_count[infer_idx]

        (denoise_step_per_stage, denoise_stage, denoise_idx), (
            chunk_offset,
            chunk_start,
            chunk_end,
            t_start,
            t_end,
        ) = self.generate_denoise_status_and_sequences(infer_idx, cur_denoise_step)
        fwd_extra_1st_chunk = chunk_start > chunk_offset and denoise_idx == 0

        # 8. Remove clean chunk
        if fwd_extra_1st_chunk:
            x_chunk = x_chunk[:, :, self.chunk_width :]
            velocity = velocity[:, :, self.chunk_width :]

        # 9. Walk and integrate
        x_chunk = self.integrate(x_chunk, velocity, self.ts[infer_idx], denoise_step_per_stage, t_start, t_end, denoise_idx)

        # 10. chunk denoise count
        for chunk_index in range(chunk_start, chunk_end):
            chunk_denoise_count[chunk_index] += 1
        self.xs[infer_idx][:, :, chunk_start * self.chunk_width : chunk_end * self.chunk_width] = x_chunk
        self.chunk_denoise_count[infer_idx] = chunk_denoise_count

        # 11. Return clean chunk
        if chunk_denoise_count[chunk_start] == transport_input.num_steps:
            if transport_input.prefix_video is not None:
                prefix_video_length = transport_input.prefix_video.size(2)
                if (chunk_start + 1) * self.chunk_width <= prefix_video_length:
                    return None, None

                real_start = max(chunk_start * self.chunk_width, prefix_video_length)

                # Keep the first 4-frames only for I2V Job
                if chunk_start == 0 and prefix_video_length == 1:
                    real_start = 0

                clean_chunk, _ = self.xs[infer_idx][:, :, real_start : (chunk_start + 1) * self.chunk_width].chunk(2, dim=0)
                return clean_chunk, chunk_start - chunk_offset
            else:
                clean_chunk, _ = self.xs[infer_idx][
                    :, :, chunk_start * self.chunk_width : (chunk_start + 1) * self.chunk_width
                ].chunk(2, dim=0)
                return clean_chunk, chunk_start - chunk_offset
        return None, None

    def walk(self):
        event_path_timer().synced_record("begin_walk")
        infer_batch_size = len(self.transport_inputs)
        for infer_idx in range(infer_batch_size):
            velocity = self.forward_velocity(infer_idx, 0)

            if mpu.get_pp_world_size() > 1 and mpu.is_pipeline_first_stage():
                pp_scheduler().queue_irecv_prev(velocity.shape, velocity.dtype)
            if mpu.get_pp_world_size() > 1 and mpu.is_pipeline_last_stage():
                pp_scheduler().isend_next(velocity)

        while not self.work_queue.empty():
            work_status: WorkStatus = self.work_queue.get()

            if mpu.get_pp_world_size() > 1 and mpu.is_pipeline_first_stage():
                self.velocities[work_status.infer_idx] = pp_scheduler().queue_irecv_prev_data()

            clean_chunk, chunk_idx = self.integrate_velocity(work_status.infer_idx, work_status.cur_denoise_step)
            if clean_chunk is not None:
                if torch.distributed.get_rank() == 0:
                    self.time_record[work_status.infer_idx].update(1)
                yield work_status.infer_idx, chunk_idx, clean_chunk

            if work_status.cur_denoise_step + 1 == self.total_forward_step(work_status.infer_idx):
                if torch.distributed.get_rank() == 0:
                    self.time_record[work_status.infer_idx].close()
                continue
            self.work_queue.put(WorkStatus(infer_idx=work_status.infer_idx, cur_denoise_step=work_status.cur_denoise_step + 1))
            velocity = self.forward_velocity(work_status.infer_idx, work_status.cur_denoise_step + 1)

            if mpu.get_pp_world_size() > 1 and mpu.is_pipeline_first_stage():
                pp_scheduler().queue_irecv_prev(velocity.shape, velocity.dtype)
            if mpu.get_pp_world_size() > 1 and mpu.is_pipeline_last_stage():
                pp_scheduler().isend_next(velocity)


def generate_per_chunk(
    model: torch.nn.Module, prompt: str, prefix_video: torch.Tensor, caption_embs: torch.Tensor, emb_masks: torch.Tensor
) -> Generator[Tuple[int, int, int, int, int, torch.Tensor], None, None]:
    device = f"cuda:{torch.cuda.current_device()}"
    transport_inputs: InferenceInput = extract_feature_for_inference(model, prompt, prefix_video, caption_embs, emb_masks)
    sample_transport = SampleTransport(model=model, transport_inputs=[transport_inputs], device=device)
    for _, _, chunk in sample_transport.walk():
        yield chunk
    dist.barrier()
    gc.collect()
    torch.cuda.empty_cache()
