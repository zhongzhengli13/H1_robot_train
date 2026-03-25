# 训练模型与 SDK 对齐分析及导出修改方案

**文件路径**
- 训练代码：`/data/Downloads/hhhhh1/legged_rl-final/`
- SDK：`/home/lzz/下载/hhhhh1/mrl_sdk/`
- 关键训练文件：`env/tasks/locomotion_task.py`、`config/loc.py`、`export_pt2onnx.py`
- 关键 SDK 文件：`source/user/rl_controller.cpp`、`bin/config.yaml`

---

## 一、完整对齐检查报告

### 1.1 关节索引顺序（最重要）

#### SDK 关节读取逻辑（`rl_controller.cpp`，`convert_dds_state2rl_state()`）

```cpp
// jointIndex2Sim 定义（第20行）
jointIndex2Sim << 6, 7, 8, 1, 2, 3, 10, 0, 5, 4, 9, 15, 16, 17, 18, 11, 12, 13, 14;

// 硬件读取（lines 457-476）
void RLController::convert_dds_state2rl_state() {
    int ii = 0;
    for (int i = 0; i < NUM_JOINTS; ++i) {
        if (i < 9)      // DDS index 9 是空置电机，跳过
            ii = i;
        else
            ii = i + 1;
        joint_pos[jointIndex2Sim[i]] = dds_motor_state->GetData()->q[ii];
        joint_vel[jointIndex2Sim[i]] = dds_motor_state->GetData()->dq[ii];
    }
}
```

**逐行追踪，推导出 SDK `joint_pos[]` 的实际内容：**

| SDK `joint_pos[]` 索引 | 硬件电机 (DDS) | 关节名称 |
|------------------------|---------------|---------|
| 0 | DDS[7] | kLeftHipYaw |
| 1 | DDS[3] | kLeftHipRoll |
| 2 | DDS[4] | kLeftHipPitch |
| 3 | DDS[5] | kLeftKnee |
| 4 | DDS[10] | kLeftAnkle |
| 5 | DDS[8] | kRightHipYaw |
| 6 | DDS[0] | kRightHipRoll |
| 7 | DDS[1] | kRightHipPitch |
| 8 | DDS[2] | kRightKnee |
| 9 | DDS[11] | kRightAnkle |
| 10 | DDS[6] | kWaistYaw |
| 11~14 | DDS[16~19] | LeftArm |
| 15~18 | DDS[12~15] | RightArm |

#### 训练代码关节顺序（从 URDF 加载，`env/legged_robot.py` 第924行）

| 训练 `joint_pos[:, i]` 索引 | URDF 关节名 |
|-----------------------------|------------|
| 0 | left_hip_yaw_joint |
| 1 | left_hip_roll_joint |
| 2 | left_hip_pitch_joint |
| 3 | left_knee_joint |
| 4 | left_ankle_joint |
| 5 | right_hip_yaw_joint |
| 6 | right_hip_roll_joint |
| 7 | right_hip_pitch_joint |
| 8 | right_knee_joint |
| 9 | right_ankle_joint |
| 10 | torso_joint (waist_yaw) |
| 11~14 | left_shoulder/elbow |
| 15~18 | right_shoulder/elbow |

#### ✅ 结论：关节顺序完全一致，无需修改

训练代码使用 URDF 顺序（左腿优先），SDK 通过 `jointIndex2Sim` 重映射后得到的 `joint_pos[]` 顺序完全相同。前10个关节均为：`[L_yaw, L_roll, L_pitch, L_knee, L_ankle, R_yaw, R_roll, R_pitch, R_knee, R_ankle]`。

---

### 1.2 参考关节位置（ref_joint_pos）

| 索引 | 关节 | 训练值 (`config/loc.py`) | SDK值 (`config.yaml`) | 一致？ |
|------|------|------------------------|----------------------|--------|
| 0 | L_HipYaw | 0.0 | 0.0 | ✅ |
| 1 | L_HipRoll | 0.0 | 0.0 | ✅ |
| 2 | L_HipPitch | -0.2 | -0.2 | ✅ |
| 3 | L_Knee | 0.4 | 0.4 | ✅ |
| 4 | L_Ankle | -0.2 | -0.2 | ✅ |
| 5 | R_HipYaw | 0.0 | 0.0 | ✅ |
| 6 | R_HipRoll | 0.0 | 0.0 | ✅ |
| 7 | R_HipPitch | -0.2 | -0.2 | ✅ |
| 8 | R_Knee | 0.4 | 0.4 | ✅ |
| 9 | R_Ankle | -0.2 | -0.2 | ✅ |

训练代码定义位置：`config/loc.py` 第85~87行
```python
ref_joint_pos = (
    [0.0, 0.0, -0.2, 0.4, -0.2] * 2 + [0] + ([0.0] * 4) * 2
)
```

SDK 定义位置：`bin/config.yaml`
```yaml
ref_joint_act: [0, 0, -0.2, 0.4, -0.2, 0, 0, -0.2, 0.4, -0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

#### ✅ 结论：参考位置完全一致

---

### 1.3 关节位置限位（act_pos）

| 索引 | 关节 | 训练 `high_ranges` | SDK `act_pos_high` | 训练 `low_ranges` | SDK `act_pos_low` | 一致？ |
|------|------|-------------------|-------------------|------------------|------------------|--------|
| 0 | L/R_HipYaw | 0.43 | 0.43 | -0.43 | -0.43 | ✅ |
| 1 | L/R_HipRoll | 0.43 | 0.43 | -0.43 | -0.43 | ✅ |
| 2 | L/R_HipPitch | 2.53 | 2.53 | -3.14 | -3.14 | ✅ |
| 3 | L/R_Knee | 2.05 | 2.05 | -0.26 | -0.26 | ✅ |
| 4 | L/R_Ankle | 0.52 | 0.52 | -0.87 | -0.87 | ✅ |

训练代码定义位置：`config/loc.py` 第82~83行
```python
high_ranges = [0.43, 0.43, 2.53, 2.05, 0.52] * 2
low_ranges  = [-0.43, -0.43, -3.14, -0.26, -0.87] * 2
```

#### ✅ 结论：关节限位完全一致

---

### 1.4 PD 增益

| 关节类型 | 训练 kp | SDK kp | 训练 kd | SDK kd | 一致？ |
|----------|---------|--------|---------|--------|--------|
| hip_yaw | 290 | 290 | 8.0 | 8 | ✅ |
| hip_roll | 290 | 290 | 8.0 | 8 | ✅ |
| hip_pitch | 250 | 250 | 8.0 | 8 | ✅ |
| knee | 290 | 290 | 9.0 | 9 | ✅ |
| ankle | 270 | 270 | 7.0 | 7 | ✅ |

训练代码定义位置：`config/loc.py` 第120~149行
```python
stiffness = {"hip_yaw": 290, "hip_roll": 290, "hip_pitch": 250, "knee": 290, "ankle": 270, ...}
damping   = {"hip_yaw": 8.0, "hip_roll": 8.0, "hip_pitch": 8.0, "knee": 9.0,  "ankle": 7.0, ...}
```

#### ✅ 结论：PD 增益完全一致

---

### 1.5 动作增量限位（act_inc）

| 索引 | 含义 | 训练 `inc_high` | SDK `act_inc_high` | 训练 `inc_low` | SDK `act_inc_low` | 一致？ |
|------|------|----------------|-------------------|---------------|------------------|--------|
| 0 | pm_f 左腿 | 3.5 | 3.5 | 0.5 | 0.5 | ✅ |
| 1 | pm_f 右腿 | 3.5 | 3.5 | 0.5 | 0.5 | ✅ |
| 2~11 | 10个关节增量 | 12.0 | 12.0 | -12.0 | -12.0 | ✅ |

训练代码定义位置：`config/loc.py` 第105~106行
```python
inc_high_ranges = [3.5, 3.5] + [12.0] * 10
inc_low_ranges  = [0.5, 0.5] + [-12.0] * 10
```

#### ✅ 结论：动作增量限位完全一致

---

### 1.6 动作缩放公式

#### SDK（`rl_controller.cpp` 第541~547行）
```cpp
Matrix<float, Dynamic, -1> RLController::inc_transform(Matrix<float, 12, -1> data) {
    auto net = (data.array() + 1.) / 2.;
    for (int i(0); i < (onnxInference.actor_walk_output_dim); i++) {
        output_joint_inc_act(i) = net(i) * (configParams.act_inc_high[i] - configParams.act_inc_low[i])
                                          + configParams.act_inc_low[i];
    }
    return output_joint_inc_act;
}
```
公式：`output = (net_out + 1) / 2 * (high - low) + low`

#### 训练代码（`env/utils/math.py` 第63~66行）
```python
def scale_transform(action, action_low, action_high):
    action = torch.clip(action, -1., 1.)
    action = (action + 1.) / 2. * (action_high - action_low) + action_low
    return action
```
公式：`output = (clip(net_out, -1, 1) + 1) / 2 * (high - low) + low`

#### ✅ 结论：动作缩放公式完全一致（SDK 的模型输出本身被 clip 到 [-1,1]）

---

### 1.7 控制时间步

| | 训练 | SDK |
|-|------|-----|
| 仿真 dt | `sim.dt = 0.001s` | — |
| decimation | `10` | — |
| 策略 dt | `0.001 × 10 = 0.01s` | `_time_step = 0.01f` (100Hz) |

训练代码：`config/loc.py` 第272行 + 第109行
```python
# sim
dt = 0.001
# pd_gains
decimation = 10
# 策略 dt = 0.001 * 10 = 0.01s
```

#### ✅ 结论：控制时间步完全一致（0.01s / 100Hz）

---

### 1.8 观测向量对比（最核心）

#### 训练代码（`locomotion_task.py` 第527~553行）
```python
def pure_observation(self):
    pm_phase = torch.cat(
        (torch.sin(self.foot_phase), torch.cos(self.foot_phase)), 1
    )  # shape: [N, 4]  (2腿 × sin/cos)
    self.obs_buf = torch.cat([
        self.commands[:, [0, 2]],                            # [0-1]  vx, yaw_cmd      2维
        (self.commands[:, [2]] - self.base_ang_vel[:, [2]]) * 0.5,  # [2]  yaw误差     1维
        self.base_euler[:, :2] * 3.0,                        # [3-4]  roll, pitch      2维
        self.base_ang_vel * 0.5,                             # [5-7]  角速度 wx,wy,wz   3维
        (self.joint_pos[:, :10] - self.ref_joint_action[:, :10]),    # [8-17] 关节位置差 10维
        self.joint_vel[:, :10] * 0.1,                        # [18-27] 关节速度        10维
        twh_joint_pos_error[:, :10],                         # [28-37] 位置误差        10维
        pm_phase * self.static_flag,                         # [38-41] 相位sin/cos     4维
        (self.pm_f * 0.3 - 1.0) * self.static_flag,         # [42-43] 相位频率        2维
    ], dim=1)
    # 合计：2+1+2+3+10+10+10+4+2 = 44维
```

#### SDK（`rl_controller.cpp` 第403~438行）
```cpp
obs << target_command,                                             // [0-1]  vx, yaw_cmd      2维
       (target_command.segment<1>(1) - base_rpy_rate(2)) * 0.5,  // [2]    yaw误差            1维
       base_rpy.segment<2>(0) * 3,                               // [3-4]  roll, pitch        2维
       base_rpy_rate * 0.5,                                      // [5-7]  角速度 wx,wy,wz    3维
       (joint_pos.segment<10>(0) - _ref_joint_act.segment<10>(0)), // [8-17] 关节位置差      10维
       joint_vel.segment<10>(0) * 0.1,                           // [18-27] 关节速度         10维
       joint_pos_error.segment<10>(0),                           // [28-37] 位置误差         10维
       pm_phase_sin_cos * high_command_coef,                     // [38-41] 相位sin/cos      4维
       (pm_f * 0.3 - one_act) * high_command_coef;              // [42-43] 相位频率          2维
// 合计：2+1+2+3+10+10+10+4+2 = 44维
```

#### ✅ 结论：观测向量结构、维度、缩放系数完全一致（均为 44 维）

---

### 1.9 模型输入输出维度对比

| | 训练导出模型 | SDK `walk_actor.onnx` |
|-|-------------|----------------------|
| **输入维度** | **44** | **60（= 44 obs + 16 z）** |
| **输出维度** | **12** | **12** |
| 输入节点名 | `"input"` | `"input"` |
| 输出节点名 | `"output"` | `"output"` |

#### ❌ 唯一不一致：模型输入维度 44 ≠ 60

SDK 的 `walk_actor` 除了接收当前 44 维观测外，还额外接收来自 `walk_encoder` 输出的 16 维隐变量 `z`。

训练代码是单一 Policy 网络，只接受 44 维观测。

SDK `walk_rl_control()` 中的输入拼接逻辑（第254~304行）：
```cpp
input << walk_observation_history.col(observation_history_len-1),  // 44维：最新观测
         walk_encoded_z;                                            // 16维：encoder输出的z
// input 总计 60 维
net_out = onnxInference.actor_walk_inference(walk_actor_session, input);
```

---

## 二、对齐结论汇总

| 检查项 | 训练代码 | SDK | 是否一致 |
|--------|---------|-----|---------|
| 关节顺序（前10个） | URDF顺序（左腿优先） | RL顺序（相同） | ✅ 一致 |
| 参考关节位置 | `[0,0,-0.2,0.4,-0.2]×2` | 相同 | ✅ 一致 |
| 关节位置限位 | `[0.43,0.43,2.53,2.05,0.52]` | 相同 | ✅ 一致 |
| PD 增益 kp | `[290,290,250,290,270]` | 相同 | ✅ 一致 |
| PD 增益 kd | `[8,8,8,9,7]` | 相同 | ✅ 一致 |
| 动作增量限位 | `[3.5,3.5]+[12]×10` | 相同 | ✅ 一致 |
| 动作缩放公式 | `(x+1)/2*(h-l)+l` | 相同 | ✅ 一致 |
| 控制时间步 | 0.01s | 0.01s | ✅ 一致 |
| 观测维度 | 44维 | 44维 | ✅ 一致 |
| 观测结构 | 完整对应 | 完整对应 | ✅ 一致 |
| **模型输入维度** | **44** | **60** | **❌ 不一致** |
| 模型输出维度 | 12 | 12 | ✅ 一致 |

**结论：仅需修改 `export_pt2onnx.py`，将导出模型的输入维度从 44 扩展为 60。**

---

## 三、修改方案（方案一：导出包装模型）

### 修改原理

SDK 的 `walk_actor` 接受 `[obs(44), z(16)]` = 60 维输入。
我们的训练模型只用 44 维观测，不需要 z。

解决办法：在 Policy 网络外包一层 `WalkActorWrapper`：
- 接受 60 维输入（与 SDK 接口完全匹配）
- 内部只取前 44 维送入真正的 Policy 网络
- 后 16 维（z）被静默忽略

这样导出的 ONNX 文件可以**直接替换** SDK 中的 `walk_actor.onnx`，无需修改任何 C++ 代码。

### 修改文件

**文件**：`export_pt2onnx.py`

---

### 原始代码

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

---

### 修改后代码

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

# ============================================================
# [新增] SDK 对齐参数
# SDK walk_actor 接受 60 维输入：前 44 维是观测，后 16 维是 encoder 输出的隐变量 z
# 我们的训练模型只使用 44 维观测，z 部分由此 Wrapper 静默忽略
# 参考：mrl_sdk/bin/config.yaml
#   walk_num_observations: 44
#   walk_z_dim:            16
#   walk_num_actions:      12
SDK_OBS_DIM = 44   # 与训练观测维度一致，来自 locomotion_task.pure_observation()
SDK_Z_DIM   = 16   # SDK walk_encoder 输出维度，本模型不使用
SDK_INPUT_DIM = SDK_OBS_DIM + SDK_Z_DIM  # = 60，SDK walk_actor 实际输入维度
# ============================================================


# ============================================================
# [新增] WalkActorWrapper：使导出模型与 SDK walk_actor 接口完全对齐
# ============================================================
class WalkActorWrapper(nn.Module):
    """
    将 44 维 Policy 包装为接受 60 维输入的模型，以匹配 SDK walk_actor.onnx 接口。

    SDK 调用方式（rl_controller.cpp walk_rl_control()）：
        input << walk_observation(44维), walk_encoded_z(16维);
        net_out = actor_walk_inference(walk_actor_session, input);  // input 共 60 维

    Wrapper 逻辑：
        - 接受 [batch, 60] 输入
        - 只取前 44 维送入真正的 Policy 网络
        - 后 16 维（z）被忽略
    """
    def __init__(self, policy: nn.Module, obs_dim: int = SDK_OBS_DIM):
        super().__init__()
        self.policy = policy
        self.obs_dim = obs_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 60]，只取前 44 维（观测部分），丢弃后 16 维（z 部分）
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
        input_names=['input'],    # 与 SDK onnx_inference.h 中的节点名一致
        output_names=['output'],  # 与 SDK onnx_inference.h 中的节点名一致
    )
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

# ============================================================
# [修改] 用 WalkActorWrapper 包装原始 Policy，使其接受 60 维输入
# ============================================================
walk_actor = WalkActorWrapper(actor_policy, obs_dim=SDK_OBS_DIM).eval()

# ============================================================
# [修改] 导出为 walk_actor.onnx（文件名与 SDK 期望一致）
#         输入维度：60（SDK_INPUT_DIM）
#         输出维度：12（walk_num_actions）
# 输入验证：用随机输入和全1输入各测试一次，打印 PyTorch vs ONNX 输出差值
# ============================================================
print(f"\n[INFO] 导出 walk_actor.onnx")
print(f"  输入维度: {SDK_INPUT_DIM}  (={SDK_OBS_DIM} obs + {SDK_Z_DIM} z，z 部分被忽略)")
print(f"  输出维度: 12  (2个相位频率 + 10个关节增量)")

for i in range(2):
    input = torch.rand([1, SDK_INPUT_DIM]).cpu().numpy()     # [修改] 60 维
    convert('walk_actor', walk_actor, input)                  # [修改] 文件名改为 walk_actor

input = torch.ones([1, SDK_INPUT_DIM]).cpu().numpy()         # [修改] 60 维
convert('walk_actor', walk_actor, input)                     # [修改] 文件名改为 walk_actor
```

---

### 修改点说明

| 修改位置 | 原始内容 | 修改后内容 | 修改原因 |
|----------|---------|-----------|---------|
| 新增常量 `SDK_OBS_DIM` | 无 | `44` | 明确标注训练观测维度，便于维护 |
| 新增常量 `SDK_Z_DIM` | 无 | `16` | SDK walk_encoder 输出的隐变量维度 |
| 新增常量 `SDK_INPUT_DIM` | 无 | `60` | SDK walk_actor 实际输入维度 |
| 新增类 `WalkActorWrapper` | 无 | 见上 | 使模型接受 60 维输入，与 SDK 接口对齐 |
| `actor_policy` → `walk_actor` | 直接用原始 policy | 用 WalkActorWrapper 包装 | 扩展输入维度 |
| 输入维度 | `cfg.policy.num_observations`（44） | `SDK_INPUT_DIM`（60） | 匹配 SDK 期望输入 |
| 导出文件名 | `actor.onnx` | `walk_actor.onnx` | 与 SDK 加载路径 `bin/walk_actor.onnx` 一致 |

---

## 四、部署步骤

```bash
# 1. 运行修改后的导出脚本（替换 <exp_name> 为你的实验名）
cd /data/Downloads/hhhhh1/legged_rl-final
python export_pt2onnx.py --name <exp_name>

# 2. 确认导出文件
ls experiments/<exp_name>/deploy/walk_actor.onnx

# 3. 备份 SDK 中原有模型
cp /home/lzz/下载/hhhhh1/mrl_sdk/bin/walk_actor.onnx \
   /home/lzz/下载/hhhhh1/mrl_sdk/bin/walk_actor.onnx.bak

# 4. 替换 SDK 中的模型
cp experiments/<exp_name>/deploy/walk_actor.onnx \
   /home/lzz/下载/hhhhh1/mrl_sdk/bin/walk_actor.onnx
```

---

## 五、注意事项

1. **walk_encoder 继续运行**：SDK 中的 encoder 网络不受影响，它会正常产生 16 维 z 并传给 walk_actor，但 walk_actor（即我们的模型）会忽略这 16 维。这不影响功能，只是 z 信息未被使用。

2. **不需要修改 SDK C++ 代码**：输入输出节点名（`"input"` / `"output"`）、维度（60 / 12）均与 SDK 期望完全一致。

3. **不需要重新训练**：现有训练好的模型权重可直接使用，只需重新执行导出脚本。

4. **关节顺序验证**：通过逐行追踪 `convert_dds_state2rl_state()` 确认，训练 URDF 顺序与 SDK RL 顺序完全一致，无需添加任何关节重排逻辑。
