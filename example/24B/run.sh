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

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_ALGO=^NVLS

export PAD_HQ=1
export PAD_DURATION=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OFFLOAD_T5_CACHE=true
export OFFLOAD_VAE_CACHE=true
export TORCH_CUDA_ARCH_LIST="8.9;9.0"

GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
DISTRIBUTED_ARGS="
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:6009 \
    --nnodes=1 \
    --nproc_per_node=$GPUS_PER_NODE
"

MAGI_ROOT=$(git rev-parse --show-toplevel)
LOG_DIR=log_$(date "+%Y-%m-%d_%H:%M:%S").log

export PYTHONPATH="$MAGI_ROOT:$PYTHONPATH"
torchrun $DISTRIBUTED_ARGS inference/pipeline/entry.py \
    --config_file example/24B/24B_base_config.json \
    --mode i2v \
    --prompt "Good Boy" \
    --image_path example/assets/image.jpeg \
    --output_path example/assets/output_i2v.mp4 \
    2>&1 | tee $LOG_DIR
