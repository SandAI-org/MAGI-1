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

import os
import random

import numpy as np
import torch


def env_is_true(env_name: str) -> bool:
    return str(os.environ.get(env_name, "0")).lower() in {"1", "true", "yes", "y", "on", "enabled"}


def divide(numerator, denominator):
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)
    return numerator // denominator


def set_random_seed(seed):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
    """
    assert seed is not None, "Please provide a seed in config.json"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed
