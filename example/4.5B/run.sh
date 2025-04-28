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

export MASTER_ADDR=localhost
export MASTER_PORT=6009
export GPUS_PER_NODE=1
export NNODES=1
export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=0

export PAD_HQ=1
export PAD_DURATION=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OFFLOAD_T5_CACHE=true
export OFFLOAD_VAE_CACHE=true
export TORCH_CUDA_ARCH_LIST="8.9;9.0"

MAGI_ROOT=$(git rev-parse --show-toplevel)
LOG_DIR=log_$(date "+%Y-%m-%d_%H:%M:%S").log

export PYTHONPATH="$MAGI_ROOT:$PYTHONPATH"
python3 inference/pipeline/entry.py \
    --config_file example/4.5B/4.5B_config.json \
    --mode t2v \
    --prompt "Good Boy" \
    --output_path example/assets/output_t2v.mp4 \
    2>&1 | tee $LOG_DIR
