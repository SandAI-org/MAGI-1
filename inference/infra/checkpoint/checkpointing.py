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

import io
import json
import os
import re
import subprocess
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import torch
import torch.distributed
from safetensors.torch import load as load_from_bytes
from safetensors.torch import load_file
from tqdm.auto import tqdm

import inference.infra.distributed.parallel_state as mpu
from inference.common import EngineConfig, ModelConfig, RuntimeConfig, print_per_rank, print_rank_0


def _load_shard(shard_path, param_names, num_threads=None):
    zstd_path = shard_path + ".zst"
    if os.path.exists(zstd_path):
        start_time = datetime.now()
        print_per_rank(f"Decompressing {zstd_path} with {num_threads} threads")
        cmd = ["zstd", "-d"]
        if num_threads:
            cmd.extend(["-T", str(num_threads)])

        process = subprocess.Popen(cmd + ["-c", zstd_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1)

        decompressed_data = process.stdout.read()
        process.stdout.close()

        retcode = process.wait()
        if retcode != 0:
            raise RuntimeError(f"Decompression failed: {process.stderr.read().decode()}")
        print_per_rank(
            f"Decompressed {zstd_path} with {num_threads} threads, duration: {(datetime.now() - start_time).total_seconds()}s"
        )

        buffer = io.BytesIO(decompressed_data)
        start_time = datetime.now()
        print_per_rank(f"Loading {shard_path} from zstd file, start time: {start_time}")
        weights = load_from_bytes(buffer.getvalue())
        print_per_rank(f"Loaded {shard_path} from zstd file, duration: {(datetime.now() - start_time).total_seconds()}s")
        buffer.close()
    else:
        weights = load_file(shard_path)

    return {name: weights[name] for name in param_names}


def load_sharded_safetensors_parallel_with_progress(checkpoint_dir):
    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        model_file_path = os.path.join(checkpoint_dir, "model.safetensors")
        state_dict = load_file(model_file_path)
        return state_dict

    with open(index_path, "r") as f:
        index = json.load(f)

    state_dict = {}
    shard_map = {}

    # Group parameters by shard file
    for param_name, shard_file in index["weight_map"].items():
        shard_path = os.path.join(checkpoint_dir, shard_file)
        if shard_path not in shard_map:
            shard_map[shard_path] = []
        shard_map[shard_path].append(param_name)

    # Load shards in parallel with a progress bar
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(_load_shard, shard_path, param_names): shard_path for shard_path, param_names in shard_map.items()
        }
        pbar = tqdm(futures, desc="Loading shards", total=len(futures))
        for future in pbar:
            result = future.result()
            state_dict.update(result)

    return state_dict


def unwrap_model(model):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while hasattr(model_module, "module"):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def _split_state_dict_for_pp(weight_dict: OrderedDict, model_config: ModelConfig):
    num_layers = model_config.num_layers
    partition = mpu.get_pp_world_size()

    ## use partition and num_layers to get current rank layer order
    layers_for_each_stage = np.array_split(range(num_layers), partition)
    current_stage = mpu.get_pp_rank()
    allow_layer_num = layers_for_each_stage[current_stage]
    layer_offset = allow_layer_num[0]
    new_weight_dict = {}
    for k, v in weight_dict.items():
        if "videodit_blocks.layers" in k:
            layer_num = int(re.search(r"videodit_blocks\.layers\.(\d+)", k).group(1))
            if layer_num not in allow_layer_num:
                continue
            ## replace the old key name by new layer number
            new_layer_num = layer_num - layer_offset
            new_k = k.replace(f"videodit_blocks.layers.{layer_num}", f"videodit_blocks.layers.{new_layer_num}")
            new_weight_dict[new_k] = v
        else:
            new_weight_dict[k] = v
    return new_weight_dict


def load_state_dict(runtime_config: RuntimeConfig, engine_config: EngineConfig):
    load_dir = runtime_config.load

    default_subdir = "inference_weight"
    if engine_config.fp8_quant:
        default_subdir = f"{default_subdir}.fp8"
    if engine_config.distill:
        default_subdir = f"{default_subdir}.distill"
    inference_weight_dir = os.path.join(load_dir, default_subdir)

    print_rank_0(f"load {default_subdir} weight from {inference_weight_dir}")
    assert (
        os.path.exists(inference_weight_dir) and len(os.listdir(inference_weight_dir)) > 0
    ), f"Ckpt directory {inference_weight_dir} does not exist or empty. If you are using fp8_quant, please run calibration first."
    state_dict = load_sharded_safetensors_parallel_with_progress(inference_weight_dir)
    return state_dict


def load_checkpoint(model):
    state_dict = load_state_dict(model.runtime_config, model.engine_config)

    model = unwrap_model(model)
    # if we use pipeline parallelism, we need to load the state dict for each stage
    # as it always record layer from 0 -> num_layers//pipeline_parallel_size
    # so we need to choose correct layer weight when load_state_dict
    if mpu.get_pp_world_size() > 1:
        state_dict = _split_state_dict_for_pp(state_dict, model.model_config)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False, assign=True)
    model.cuda(torch.cuda.current_device())

    if mpu.get_pp_world_size() > 1:
        rank_msg = f"CP_rank={mpu.get_cp_rank()} PP_rank={mpu.get_pp_rank()}"
        print_per_rank(
            f"""[{rank_msg}] Load Weight Missing Keys: {missing_keys} Load Weight Unexpected Keys: {unexpected_keys} You should see message [missing fianl layer norm weight] except the final pipeline stage"""
        )
    else:
        print_rank_0(f"Load Weight Missing Keys: {missing_keys}")
        print_rank_0(f"Load Weight Unexpected Keys: {unexpected_keys}")

    return model
