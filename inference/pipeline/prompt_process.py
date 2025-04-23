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
import os
from typing import List

import numpy as np
import torch

from inference.common import MagiConfig, env_is_true, magi_logger
from inference.infra.distributed import is_last_tp_cp_rank
from inference.infra.distributed import parallel_state as mpu
from inference.model.t5 import T5Embedder

SPECIAL_TOKEN_PATH = os.getenv("SPECIAL_TOKEN_PATH", "example/assets/special_tokens.npz")
SPECIAL_TOKEN = np.load(SPECIAL_TOKEN_PATH)
CAPTION_TOKEN = torch.tensor(SPECIAL_TOKEN["caption_token"].astype(np.float16))
LOGO_TOKEN = torch.tensor(SPECIAL_TOKEN["logo_token"].astype(np.float16))
TRANS_TOKEN = torch.tensor(SPECIAL_TOKEN["other_tokens"][:1].astype(np.float16))
HQ_TOKEN = torch.tensor(SPECIAL_TOKEN["other_tokens"][1:2].astype(np.float16))
STATIC_FIRST_FRAMES_TOKEN = torch.tensor(SPECIAL_TOKEN["other_tokens"][2:3].astype(np.float16))  # static first frames
DYNAMIC_FIRST_FRAMES_TOKEN = torch.tensor(SPECIAL_TOKEN["other_tokens"][3:4].astype(np.float16))  # dynamic first frames
BORDERNESS_TOKEN = torch.tensor(SPECIAL_TOKEN["other_tokens"][4:5].astype(np.float16))
DURATION_TOKEN_LIST = [torch.tensor(SPECIAL_TOKEN["other_tokens"][i : i + 1].astype(np.float16)) for i in range(0 + 7, 8 + 7)]
THREE_D_MODEL_TOKEN = torch.tensor(SPECIAL_TOKEN["other_tokens"][15:16].astype(np.float16))
TWO_D_ANIME_TOKEN = torch.tensor(SPECIAL_TOKEN["other_tokens"][16:17].astype(np.float16))

SPECIAL_TOKEN_DICT = {
    "CAPTION_TOKEN": CAPTION_TOKEN,
    "LOGO_TOKEN": LOGO_TOKEN,
    "TRANS_TOKEN": TRANS_TOKEN,
    "HQ_TOKEN": HQ_TOKEN,
    "STATIC_FIRST_FRAMES_TOKEN": STATIC_FIRST_FRAMES_TOKEN,
    "DYNAMIC_FIRST_FRAMES_TOKEN": DYNAMIC_FIRST_FRAMES_TOKEN,
    "BORDERNESS_TOKEN": BORDERNESS_TOKEN,
    "THREE_D_MODEL_TOKEN": THREE_D_MODEL_TOKEN,
    "TWO_D_ANIME_TOKEN": TWO_D_ANIME_TOKEN,
}

for i, token in enumerate(DURATION_TOKEN_LIST):
    # DURATION_TOKEN_N represents N chunk(s) remain in the future
    SPECIAL_TOKEN_DICT[f"DURATION_TOKEN_{i+1}"] = token


def pad_duration_token_keys(special_token_keys: List[str]) -> List[str]:
    if "DURATION_TOKEN" in set(special_token_keys):
        return special_token_keys

    if env_is_true("PAD_DURATION"):
        return special_token_keys + ["DURATION_TOKEN"]
    return special_token_keys


def get_special_token_keys() -> List[str]:
    special_token_keys = []
    if env_is_true("PAD_STATIC"):
        special_token_keys.append("STATIC_FIRST_FRAMES_TOKEN")
    if env_is_true("PAD_DYNAMIC"):
        special_token_keys.append("DYNAMIC_FIRST_FRAMES_TOKEN")
    if env_is_true("PAD_BORDERNESS"):
        special_token_keys.append("BORDERNESS_TOKEN")
    if env_is_true("PAD_HQ"):
        special_token_keys.append("HQ_TOKEN")
    if env_is_true("PAD_THREE_D_MODEL"):
        special_token_keys.append("THREE_D_MODEL_TOKEN")
    if env_is_true("PAD_TWO_D_ANIME"):
        special_token_keys.append("TWO_D_ANIME_TOKEN")

    special_token_keys = pad_duration_token_keys(special_token_keys)
    return special_token_keys


def get_negative_special_token_keys() -> List[str]:
    if env_is_true("NEG_PROMPT"):
        return ["CAPTION_TOKEN", "LOGO_TOKEN", "TRANS_TOKEN", "BORDERNESS_TOKEN"]
    return None


def _pad_special_token(special_token: torch.Tensor, txt_feat: torch.Tensor, attn_mask: torch.Tensor = None):
    _device = txt_feat.device
    _dtype = txt_feat.dtype
    N, C, _, D = txt_feat.size()
    txt_feat = torch.cat(
        [special_token.unsqueeze(0).unsqueeze(0).to(_device).to(_dtype).expand(N, C, -1, D), txt_feat], dim=2
    )[:, :, :800, :]
    if attn_mask is not None:
        attn_mask = torch.cat([torch.ones(N, C, 1, dtype=_dtype, device=_device), attn_mask], dim=-1)[:, :, :800]
    return txt_feat, attn_mask


def pad_special_token(special_token_keys: List[str], caption_embs: torch.Tensor, emb_masks: torch.Tensor):
    device = f"cuda:{torch.cuda.current_device()}"
    if not special_token_keys:
        return caption_embs, emb_masks
    for special_token_key in special_token_keys:
        if special_token_key == "DURATION_TOKEN":
            new_caption_embs, new_emb_masks = [], []
            num_chunks = caption_embs.size(1)
            for i in range(num_chunks):
                chunk_caption_embs, chunk_emb_masks = _pad_special_token(
                    DURATION_TOKEN_LIST[min(num_chunks - i - 1, 7)].to(device),
                    caption_embs[:, i : i + 1],
                    emb_masks[:, i : i + 1],
                )
                new_caption_embs.append(chunk_caption_embs)
                new_emb_masks.append(chunk_emb_masks)
            caption_embs = torch.cat(new_caption_embs, dim=1)
            emb_masks = torch.cat(new_emb_masks, dim=1)
        else:
            special_token = SPECIAL_TOKEN_DICT.get(special_token_key)
            if special_token is not None:
                caption_embs, emb_masks = _pad_special_token(special_token.to(device), caption_embs, emb_masks)
    return caption_embs, emb_masks


_t5_cache = None


def _t5(model_cache_dir, model_device, model_max_length) -> T5Embedder:
    global _t5_cache
    if _t5_cache is None:
        _t5_model = T5Embedder(
            device=model_device,
            local_cache=True,
            cache_dir=model_cache_dir,
            torch_dtype=torch.float,
            model_max_length=model_max_length,
        )
        if os.environ.get("OFFLOAD_T5_CACHE") == "true":
            return _t5_model
        _t5_cache = _t5_model
    return _t5_cache


def prepare_prompt_embeddings(prompts: List[str], model_cache_dir, model_device, model_max_length):
    magi_logger.info("Precompute validation prompt embeddings")
    magi_logger.debug(
        f"rank {torch.distributed.get_rank()} memory allocated before precompute validation prompt embeddings: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )
    magi_logger.debug(
        f"rank {torch.distributed.get_rank()} memory reserved before precompute validation prompt embeddings: {torch.cuda.memory_reserved() / 1024**3:.2f} GB"
    )

    txt_embs = []
    for prompt in prompts:
        with torch.no_grad():
            caption_embs, emb_masks = _t5(model_cache_dir, model_device, model_max_length).get_text_embeddings([prompt])
            caption_embs = caption_embs.float()[:, None]
            txt_embs.append([caption_embs, emb_masks])
            magi_logger.debug(f"caption_embs.shape = {caption_embs.shape}")
            magi_logger.debug(f"emb_masks.shape = {emb_masks.shape}")

    # put everything to CPU for future broadcast
    txt_embs = [[x[0].cpu(), x[1].cpu()] for x in txt_embs]

    magi_logger.debug(
        f"rank {torch.distributed.get_rank()} memory allocated after precompute validation prompt embeddings: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )
    magi_logger.debug(
        f"rank {torch.distributed.get_rank()} memory reserved after precompute validation prompt embeddings: {torch.cuda.memory_reserved() / 1024**3:.2f} GB"
    )
    gc.collect()
    torch.cuda.empty_cache()
    return txt_embs


def get_txt_embeddings(prompt: str, config: MagiConfig):
    prompts = [prompt]
    if not torch.distributed.is_initialized():
        txt_embs = prepare_prompt_embeddings(
            prompts,
            config.runtime_config.t5_pretrained,
            config.runtime_config.t5_device,
            config.model_config.caption_max_length,
        )
    else:
        if is_last_tp_cp_rank():
            txt_embs = prepare_prompt_embeddings(
                prompts,
                config.runtime_config.t5_pretrained,
                config.runtime_config.t5_device,
                config.model_config.caption_max_length,
            )
        else:
            txt_embs = [None]
        src = mpu.get_tensor_model_parallel_last_rank(with_context_parallel=True)
        group = mpu.get_tp_group(with_context_parallel=True)
        torch.distributed.broadcast_object_list(txt_embs, src=src, group=group)

    # Only process one prompt
    assert len(txt_embs) == 1
    caption_embs, emb_masks = txt_embs[0]
    device = f"cuda:{torch.cuda.current_device()}"
    caption_embs, emb_masks = caption_embs.to(device), emb_masks.to(device)
    return caption_embs, emb_masks
