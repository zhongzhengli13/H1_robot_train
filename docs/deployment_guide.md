# 部署指南：训练模型 → SDK 完整修改说明

> 适合人群：对代码不熟悉的初学者
> 目标：把训练好的模型正确部署到 SDK，让机器人能走路

---

## 理解背景：为什么需要修改？

```
训练代码（Python）                SDK（C++，跑在机器人上）
─────────────────                ─────────────────────────
产出: actor.onnx (44维输入)   →  期望: walk_actor.onnx (60维输入)
                                  ↑
                                  差了16维，直接用会崩溃
```

SDK 内部有两个网络：
- **walk_encoder**：把过去12帧历史压缩成16维的"记忆向量 z"
- **walk_actor**：接收"当前44维观测 + 16维记忆z" = 60维，输出动作

我们只有 actor，没有 encoder。解决方案：让 actor 接受60维输入，但只用前44维，把后16维z忽略掉。

---

## 需要修改的文件清单

| 文件 | 位置 | 修改内容 | 重要程度 |
|------|------|---------|---------|
| `export_pt2onnx.py` | 训练代码 | 输入从44维→60维，文件名改为walk_actor | ⭐⭐⭐ 必须 |
| `bin/config.yaml` | SDK | vx最大速度从0.4改到0.8 | ⭐⭐ 建议 |

---

## 修改一：`export_pt2onnx.py`（必须改）

**文件位置**：`/data/Downloads/hhhhh1/legged_rl-final/export_pt2onnx.py`

### 为什么要改这个文件？

这个文件的作用是：把训练好的 PyTorch 模型（`.pt`格式）转换成 ONNX 格式（`.onnx`格式），让 SDK 的 C++ 代码能读取。

问题在于：
- 现在导出的文件叫 `actor.onnx`，接受 **44维** 输入
- SDK 要的文件叫 `walk_actor.onnx`，接受 **60维** 输入

### 原始代码（完整）

```python
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
cfg = getattr(importlib.import_module('.'.join(['config', params['env']['cfg']])), 'H1Config')
cfg = paramsProcess.dict2class(cfg, params)


def convert(name: str, model: nn.Module, input: np.ndarray):
    print(f'\n******************************** {name} ********************************************\n')
    deploy_path = join(deploy_dir, f'{name}.onnx')
    torch.onnx.export(model, torch.from_numpy(input), deploy_path, verbose=False, opset_version=12, input_names=['input'], output_names=['output'])
    print('Pytorch')
    print(model(torch.from_numpy(input)).detach().cpu().numpy())
    ort_session = ort.InferenceSession(deploy_path)
    print('Onnx')
    print(ort_session.run(None, {'input': input})[0])
    gap = model(torch.from_numpy(input)).detach().cpu().numpy() - ort_session.run(None, {'input': input})[0]
    print('Gap')
    print(gap)


actor_policy = load_actor(class_to_dict(cfg.policy), deploy=True).eval()
policy_path = join(model_dir, 'all/policy_1000.pt')
# policy_path = join(model_dir, 'policy.pt')
assert exists(policy_path), policy_path
saved_model = torch.load(policy_path, map_location='cpu')
actor_policy.load_state_dict(saved_model['actor'], strict=False)

for i in range(2):
    input = torch.rand([1, cfg.policy.num_observations]).cpu().numpy()
    convert('actor', actor_policy, input)
input = torch.ones([1, cfg.policy.num_observations]).cpu().numpy()
convert('actor', actor_policy, input)
```

### 修改后代码（完整）

```python
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
cfg = getattr(importlib.import_module('.'.join(['config', params['env']['cfg']])), 'H1Config')
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
SDK_OBS_DIM   = 44   # 观测维度，和训练的 pure_observation() 输出一致
SDK_Z_DIM     = 16   # encoder 输出的隐变量维度，我们不用但要留位置
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
        self.policy  = policy   # 真正的策略网络（44维→12维）
        self.obs_dim = obs_dim  # 取前多少维

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 的形状是 [1, 60]
        # x[:, :44] 取前44维，即观测部分
        # x[:, 44:] 是 z，直接丢弃
        return self.policy(x[:, :self.obs_dim])


def convert(name: str, model: nn.Module, input: np.ndarray):
    print(f'\n******************************** {name} ********************************************\n')
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
    gap = model(torch.from_numpy(input)).detach().cpu().numpy() - ort_session.run(None, {'input': input})[0]
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
```

### 改动对比总结

| 位置 | 改之前 | 改之后 | 原因 |
|------|--------|--------|------|
| 文件顶部 | 无 | 新增3个常量 + `WalkActorWrapper` 类 | 实现60维接口适配 |
| 包装模型 | 直接用 `actor_policy` | 用 `WalkActorWrapper(actor_policy)` | 扩展到60维输入 |
| 输入维度 | `cfg.policy.num_observations`（44） | `SDK_INPUT_DIM`（60） | 匹配SDK接口 |
| 导出文件名 | `'actor'` → `actor.onnx` | `'walk_actor'` → `walk_actor.onnx` | 匹配SDK加载路径 |

---

## 修改二：SDK `config.yaml`（建议改）

**文件位置**：`/home/lzz/下载/hhhhh1/mrl_sdk/bin/config.yaml`

### 为什么要改这个文件？

模型训练时用的速度范围是 **vx ∈ [0.4, 1.0] m/s**（向前走，不转弯）。

但 SDK 的配置里速度最大只有 **0.4 m/s**：
```yaml
vx_cmd_range: [-0.4, 0.4]
```

这意味着遥控器推满也只能给 0.4 m/s，正好是模型训练范围的**最低值**。模型能工作，但速度永远处于训练范围的边缘，不够稳定。

### 原始代码

打开 `/home/lzz/下载/hhhhh1/mrl_sdk/bin/config.yaml`，找到这几行：

```yaml
# 原始内容（大约在文件中间位置，搜索 vx_cmd_range 找到）
vx_cmd_range: [-0.4, 0.4]
vy_cmd_range: [-0.6, 0.6]
yr_cmd_range: [-0.4, 0.4]
```

### 修改后代码

```yaml
# 修改后：vx 正向最大改为 0.8，覆盖训练范围 [0.4, 1.0] 的中间段
# 负向保持 0（模型没有训练后退），yr 改为 0（模型没有训练转弯）
vx_cmd_range: [0.0, 0.8]
vy_cmd_range: [0.0, 0.0]
yr_cmd_range: [0.0, 0.0]
```

### 修改说明

| 参数 | 原始值 | 修改后 | 原因 |
|------|--------|--------|------|
| `vx_cmd_range` | `[-0.4, 0.4]` | `[0.0, 0.8]` | 模型训练范围 [0.4,1.0]，给 0.0~0.8 覆盖中段 |
| `vy_cmd_range` | `[-0.6, 0.6]` | `[0.0, 0.0]` | 模型未训练侧走，强制关闭 |
| `yr_cmd_range` | `[-0.4, 0.4]` | `[0.0, 0.0]` | 模型未训练转弯，强制关闭 |

> **注意**：如果你后续训练了能转弯/侧走的模型，再把这些值改回来。

---

## 操作步骤（按顺序执行）

### 第一步：修改导出脚本

按照上面"修改一"的内容，编辑 `export_pt2onnx.py`。

可以直接用文本编辑器打开，也可以用命令：
```bash
gedit /data/Downloads/hhhhh1/legged_rl-final/export_pt2onnx.py
```

### 第二步：运行导出脚本

```bash
cd /data/Downloads/hhhhh1/legged_rl-final

# 把 <你的实验名> 替换成实际的实验文件夹名
# 例如：python export_pt2onnx.py --name exp_walk_v1
python export_pt2onnx.py --name <你的实验名>
```

**正常输出应该类似这样**（两者输出接近，差值接近0）：
```
[确认] 导出参数：
  模型输入维度 : 60  (= 44观测 + 16z，z部分被忽略)
  模型输出维度 : 12  (2个相位频率 + 10个关节增量)
  导出文件名   : walk_actor.onnx

Pytorch 输出:
[[ 0.123  -0.456  0.789 ...]]
ONNX 输出:
[[ 0.123  -0.456  0.789 ...]]
两者差值（应该接近全零）:
[[ 0.000  0.000  0.000 ...]]
```

如果差值不是接近0，说明导出有问题，检查 Python/ONNX 版本是否匹配。

### 第三步：确认导出文件

```bash
# 查看导出文件是否存在
ls -lh /data/Downloads/hhhhh1/legged_rl-final/experiments/<你的实验名>/deploy/

# 应该看到：
# walk_actor.onnx   （新文件，大小通常几百KB）
```

### 第四步：备份并替换 SDK 中的模型

```bash
# 先备份原来的模型（重要！出问题可以还原）
cp /home/lzz/下载/hhhhh1/mrl_sdk/bin/walk_actor.onnx \
   /home/lzz/下载/hhhhh1/mrl_sdk/bin/walk_actor.onnx.bak

# 复制新模型进去
cp /data/Downloads/hhhhh1/legged_rl-final/experiments/<你的实验名>/deploy/walk_actor.onnx \
   /home/lzz/下载/hhhhh1/mrl_sdk/bin/walk_actor.onnx

# 确认文件已替换（看时间戳是否是刚才）
ls -lh /home/lzz/下载/hhhhh1/mrl_sdk/bin/walk_actor.onnx
```

### 第五步：修改 SDK 速度配置

```bash
gedit /home/lzz/下载/hhhhh1/mrl_sdk/bin/config.yaml
```

找到 `vx_cmd_range` 这行，按"修改二"中的内容修改并保存。

### 第六步：重新编译 SDK（因为修改了 config.yaml 不需要，但如有其他改动则需要）

config.yaml 是运行时读取的，**不需要重新编译**。直接运行 SDK 即可。

---

## 验证清单

部署前，逐项检查：

- [ ] `export_pt2onnx.py` 中 `SDK_INPUT_DIM = 60`
- [ ] 导出文件名为 `walk_actor.onnx`（不是 `actor.onnx`）
- [ ] 导出时 PyTorch 和 ONNX 输出的差值接近 0
- [ ] SDK `bin/walk_actor.onnx` 已替换（文件时间戳是新的）
- [ ] SDK `bin/config.yaml` 中 `vx_cmd_range` 已修改
- [ ] SDK `bin/walk_actor.onnx.bak` 备份文件存在（便于还原）

---

## 如果出现问题

### 问题1：导出脚本找不到模型文件
```
AssertionError: experiments/<名字>/model/all/policy_1000.pt
```
**原因**：模型文件名或路径不对。
**解决**：
```bash
# 查看实际有哪些模型文件
ls /data/Downloads/hhhhh1/legged_rl-final/experiments/<你的实验名>/model/all/
```
然后把 `export_pt2onnx.py` 中的 `policy_1000.pt` 改成实际存在的文件名。

### 问题2：SDK 加载 ONNX 报维度错误
```
The dimension of the input size walk_observation is error!!!
```
**原因**：替换的 ONNX 文件维度不对。
**解决**：确认导出时用的是 `SDK_INPUT_DIM = 60` 的输入，重新导出。

### 问题3：机器人走路姿态很差或摔倒
**可能原因**：速度指令超出训练范围。
**解决**：
- 降低 `vx_cmd_range` 上限到 0.6
- 确认 `yr_cmd_range = [0.0, 0.0]`（不转弯）

### 问题4：想还原到原始 SDK 模型
```bash
cp /home/lzz/下载/hhhhh1/mrl_sdk/bin/walk_actor.onnx.bak \
   /home/lzz/下载/hhhhh1/mrl_sdk/bin/walk_actor.onnx
```
