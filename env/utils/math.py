# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
from isaacgym.torch_utils import quat_apply, normalize
from typing import Tuple
import math
import torch
from torch import Tensor


@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


@torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2 * torch.rand(*shape, device=device) - 1
    r = torch.where(r < 0., -torch.sqrt(-r), torch.sqrt(r))
    r = (r + 1.) / 2.
    return (upper - lower) * r + lower


def scale_transform(action: torch.Tensor, action_low: torch.Tensor, action_high: torch.Tensor):
    action = torch.clip(action, -1., 1.)
    action = (action + 1.) / 2. * (action_high - action_low) + action_low
    return action

def smallest_signed_angle_between(x, y):
    """from X to Y"""
    a = (x - y) % math.tau
    b = (y - x) % math.tau
    return -a if a < b else b

def smallest_signed_angle_between_torch(alpha, beta):
    # assert alpha.ndim == 2 and alpha.shape[1] == 1 and beta.ndim == 2 and beta.shape[1] == 1, "输入的alpha和beta应为n*1维度的张量"
    # 计算角度差值
    a = beta - alpha
    # 确保math.pi也转换为与输入张量同类型和设备的张量形式
    # pi_tensor = torch.tensor(math.pi).to(alpha.device).type_as(alpha)
    # 使用torch.where来根据条件进行元素级别的操作
    a = torch.where(a > math.pi, a - 2 * math.pi, a)
    a = torch.where(a < -math.pi, a + 2 * math.pi, a)
    return a