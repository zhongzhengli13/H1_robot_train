# # -*- coding: utf-8 -*-
# import argparse
# import isaacgym
# import numpy as np
# import os
# from os.path import exists, join
# import torch.nn as nn
# from env.utils.helpers import class_to_dict

# from model import load_actor
# from env.utils import get_args
# import importlib
# from utils.yaml import ParamsProcess
# import onnxruntime as ort
# import torch

# args = get_args()
# exp_dir = join('experiments', args.name)
# model_dir = join(exp_dir, 'model')
# deploy_dir = join(exp_dir, 'deploy')
# os.makedirs(deploy_dir, exist_ok=True)

# paramsProcess = ParamsProcess()
# params = paramsProcess.read_param(join(model_dir, 'cfg.yaml'))
# cfg = getattr(importlib.import_module('.'.join(['config', params['env']['cfg']])), 'H1Config')
# cfg = paramsProcess.dict2class(cfg, params)


# def convert(name: str, model: nn.Module, input: np.ndarray):
#     print(f'\n******************************** {name} ********************************************\n')
#     deploy_path = join(deploy_dir, f'{name}.onnx')
#     torch.onnx.export(model, torch.from_numpy(input), deploy_path, verbose=False, opset_version=12, input_names=['input'], output_names=['output'])
#     print('Pytorch')
#     print(model(torch.from_numpy(input)).detach().cpu().numpy())
#     ort_session = ort.InferenceSession(deploy_path)
#     print('Onnx')
#     print(ort_session.run(None, {'input': input})[0])
#     gap = model(torch.from_numpy(input)).detach().cpu().numpy() - ort_session.run(None, {'input': input})[0]
#     print('Gap')
#     print(gap)


# actor_policy = load_actor(class_to_dict(cfg.policy), deploy=True).eval()
# policy_path = join(model_dir, 'all/policy_1000.pt')
# # policy_path = join(model_dir, 'policy.pt')
# assert exists(policy_path), policy_path
# saved_model = torch.load(policy_path, map_location='cpu')
# actor_policy.load_state_dict(saved_model['actor'], strict=False)

# for i in range(2):
#     input = torch.rand([1, cfg.policy.num_observations]).cpu().numpy()
#     convert('actor', actor_policy, input)
# input = torch.ones([1, cfg.policy.num_observations]).cpu().numpy()
# convert('actor', actor_policy, input)
# -*- coding: utf-8 -*-
import argparse
import isaacgym
import numpy as np
import os
from os.path import exists, join
import torch.nn as nn
from env.utils.helpers import class_to_dict

from model import load_actor
from env.utils import get_args
import importlib
from utils.yaml import ParamsProcess
import onnxruntime as ort
import torch

args = get_args()
exp_dir = join('experiments', args.name)
model_dir = join(exp_dir, 'model')
deploy_dir = join(exp_dir, 'deploy')
os.makedirs(deploy_dir, exist_ok=True)

paramsProcess = ParamsProcess()
params = paramsProcess.read_param(join(model_dir, 'cfg.yaml'))
cfg = getattr(importlib.import_module(
    '.'.join(['config', params['env']['cfg']])), 'H1Config')
cfg = paramsProcess.dict2class(cfg, params)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 【新增】第一步：定义对齐 SDK 所需的维度常量
#
# 为什么需要这些常量？
#   SDK 的 walk_actor.onnx 要求输入必须是 60 维：
#     前 44 维 = 机器人当前状态（观测值）
#     后 16 维 = encoder 输出的"历史记忆向量 z"
#   我们的模型只有 44 维，所以需要在外面套一层"包装壳"
#
# 来源对应关系：
#   SDK_OBS_DIM = 44  ←→  mrl_sdk/bin/config.yaml: walk_num_observations: 44
#   SDK_Z_DIM   = 16  ←→  mrl_sdk/bin/config.yaml: walk_z_dim: 16
#   SDK_INPUT_DIM=60  ←→  44 + 16 = walk_actor 实际输入维度
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SDK_OBS_DIM = 44   # 观测维度，和训练的 pure_observation() 输出一致
SDK_Z_DIM = 16   # encoder 输出的隐变量维度，我们不用但要留位置
SDK_INPUT_DIM = SDK_OBS_DIM + SDK_Z_DIM  # = 60，SDK walk_actor 的输入维度


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 【新增】第二步：定义包装类 WalkActorWrapper
#
# 这个"包装壳"做了什么？
#   1. 接收 60 维输入（和 SDK 完全一样的接口）
#   2. 只把前 44 维送给真正的策略网络
#   3. 后 16 维（z）被丢弃，因为我们没有 encoder
#
# 形象比喻：
#   就像一个适配器插头，插头形状是 60 孔的（SDK 侧），
#   但内部只用了前 44 个孔连接到实际设备（我们的策略网络）。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class WalkActorWrapper(nn.Module):
    def __init__(self, policy: nn.Module, obs_dim: int = SDK_OBS_DIM):
        super().__init__()
        self.policy = policy   # 真正的策略网络（44维→12维）
        self.obs_dim = obs_dim  # 取前多少维

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 的形状是 [1, 60]
        # x[:, :44] 取前44维，即观测部分
        # x[:, 44:] 是 z，直接丢弃
        return self.policy(x[:, :self.obs_dim])


def convert(name: str, model: nn.Module, input: np.ndarray):
    print(
        f'\n******************************** {name} ********************************************\n')
    deploy_path = join(deploy_dir, f'{name}.onnx')
    torch.onnx.export(
        model,
        torch.from_numpy(input),
        deploy_path,
        verbose=False,
        opset_version=12,
        input_names=['input'],    # 节点名必须是 'input'，SDK 里 C++ 代码按这个名字读取
        output_names=['output'],  # 节点名必须是 'output'，SDK 里 C++ 代码按这个名字读取
    )
    print('Pytorch 输出:')
    print(model(torch.from_numpy(input)).detach().cpu().numpy())
    ort_session = ort.InferenceSession(deploy_path)
    print('ONNX 输出:')
    print(ort_session.run(None, {'input': input})[0])
    gap = model(torch.from_numpy(input)).detach().cpu().numpy() - \
        ort_session.run(None, {'input': input})[0]
    print('两者差值（应该接近全零）:')
    print(gap)


actor_policy = load_actor(class_to_dict(cfg.policy), deploy=True).eval()
policy_path = join(model_dir, 'all/policy_1000.pt')
# policy_path = join(model_dir, 'policy.pt')
assert exists(policy_path), policy_path
saved_model = torch.load(policy_path, map_location='cpu')
actor_policy.load_state_dict(saved_model['actor'], strict=False)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 【修改】第三步：用包装壳套住原始策略网络
#
# 原来：直接用 actor_policy（44维输入）
# 现在：用 WalkActorWrapper 包一层（60维输入，内部用44维）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
walk_actor = WalkActorWrapper(actor_policy, obs_dim=SDK_OBS_DIM).eval()

# 打印确认信息，方便检查
print(f"\n{'='*60}")
print(f"[确认] 导出参数：")
print(f"  模型输入维度 : {SDK_INPUT_DIM}  (= {SDK_OBS_DIM}观测 + {SDK_Z_DIM}z，z部分被忽略)")
print(f"  模型输出维度 : 12  (2个相位频率 + 10个关节增量)")
print(f"  导出文件名   : walk_actor.onnx")
print(f"{'='*60}\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 【修改】第四步：导出模型
#
# 原来：
#   convert('actor', actor_policy, input)   ← 文件名 actor.onnx，44维输入
# 现在：
#   convert('walk_actor', walk_actor, input) ← 文件名 walk_actor.onnx，60维输入
#
# 为什么文件名必须是 walk_actor？
#   SDK 在 custom.hpp 中写死了加载路径：
#   rlController = new RLController(..., "walk_actor.onnx", ...)
#   文件名不对就加载失败。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
for i in range(2):
    dummy_input = torch.rand([1, SDK_INPUT_DIM]).cpu().numpy()   # 60维随机输入
    convert('walk_actor', walk_actor, dummy_input)

dummy_input = torch.ones([1, SDK_INPUT_DIM]).cpu().numpy()       # 60维全1输入
convert('walk_actor', walk_actor, dummy_input)
