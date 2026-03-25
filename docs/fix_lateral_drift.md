# 走路偏移问题分析与修复方案

> 数据来源：`experiments/try_x2/debug/debug_0.xlsx`
> 问题现象：机器人走直线任务，越走越歪

---

## 一、用数据说明到底发生了什么

从调试数据中直接读出以下事实：

### 轨迹数据（base_pos 表）
| 时刻 | X位置 | Y位置 |
|------|-------|-------|
| 起点（第0步） | -0.000026 m | 0.000116 m |
| 终点（第1999步） | **9.63 m** | **-2.18 m** |

机器人往前走了 **9.63 米**，但同时往侧面漂移了 **-2.18 米**。
侧移比例 = 2.18 ÷ 9.63 ≈ **22.6%**，即每前进1米，就会偏移23厘米！

### 偏航角速度数据（ang_vel 表）
| 方向 | 均值 | 说明 |
|------|------|------|
| x（翻滚速率） | 0.0012 rad/s | 正常 |
| y（俯仰速率） | 0.0058 rad/s | 正常 |
| **z（偏航速率）** | **0.029 rad/s** | ⚠️ 有持续偏转！ |

目标偏航速率是 **0**，但实际均值是 **0.029 rad/s**。
这个数字看起来很小，但累积起来很可怕：
```
20秒 × 0.029 rad/s = 0.58 rad ≈ 33度
```
机器人在走路过程中，头部方向慢慢转了33度，于是"前进"变成了"斜走"。

### 关节不对称数据（joint_pos 表）
| 关节 | 左腿均值 | 右腿均值 | 差值 |
|------|---------|---------|------|
| 髋关节偏转(yaw) | +0.001 rad | **-0.078 rad** | **0.079 rad** |
| 髋关节侧摆(roll) | +0.014 rad | **+0.064 rad** | 0.050 rad |
| 髋关节前倾(pitch) | -0.165 rad | **-0.277 rad** | 0.112 rad |
| 膝关节 | +0.640 rad | **+0.780 rad** | 0.140 rad |

右腿比左腿弯曲得多得多，这会导致推进力不平衡，产生旋转趋势。

---

## 二、根本原因分析（三个）

### 原因一：机器人"看不到"自己在侧移（最关键）

打开 `env/tasks/locomotion_task.py`，找到 `pure_observation()` 函数（第527行）：

```python
self.obs_buf = torch.cat([
    self.commands[:, [0, 2]],    # 只有 vx指令 和 yaw指令（yaw指令永远是0）
    (self.commands[:, [2]] - self.base_ang_vel[:, [2]]) * 0.5,
    self.base_euler[:, :2] * 3.0,
    self.base_ang_vel * 0.5,
    (self.joint_pos[:, :10] - self.ref_joint_action[:, :10]),
    self.joint_vel[:, :10] * 0.1,
    twh_joint_pos_error[:, :10],
    pm_phase * self.static_flag,
    (self.pm_f * 0.3 - 1.0) * self.static_flag,
], dim=1)
```

**观测中完全没有侧向速度 vy！**

这就像你蒙着眼睛走路——你只知道自己想往哪走，但感觉不到自己在往侧面飘。
机器人在训练时会被奖励惩罚纠正，但它没有"感知"，只能被动地等待惩罚信号。

从调试数据中可以确认，obs[1]（yaw指令）**全程都是 0.0**，这一维是完全浪费的。

### 原因二：偏航惩罚力度不足

当偏航速率很小（0.029 rad/s）时，指数型奖励公式对小偏差几乎没有惩罚：

```python
# 当 yaw误差 = 0.029 rad/s 时：
exp(-k * 0.029²) ≈ exp(-k * 0.00084) ≈ 0.999
# 几乎是满分！所以机器人学不到"要纠正小偏转"
```

### 原因三：左右腿动作不对称，没有对称性约束

右髋偏转均值 -0.078 rad，左髋仅 +0.001 rad，两边相差 **79倍**。
训练时缺少强制左右对称的奖励，导致机器人学出了偏的步态。

---

## 三、修改方案

修改只涉及一个文件：`env/tasks/locomotion_task.py`

共三处改动：
1. **加入侧向速度 vy 到观测**（替换掉永远是0的那一维）
2. **加强偏航惩罚**（加线性惩罚项）
3. **加入左右对称奖励**（新增一项）

---

## 修改一：将 vy 加入观测（替换掉无用的yaw指令维）

**文件**：`env/tasks/locomotion_task.py`

**位置**：`pure_observation()` 函数，第527~553行

### 为什么这样改？

观测的第[0]维是 vx 指令（0.4~1.0 m/s），第[1]维是 yaw指令（永远是0，完全没用）。
把第[1]维换成实际的侧向速度 vy，观测维度不变（仍然44维），但机器人现在能"感觉"到自己在往侧面飘了。

### 原始代码

```python
def pure_observation(self):
    pm_phase = torch.cat(
        (torch.sin(self.foot_phase), torch.cos(self.foot_phase)), 1
    )
    twh_joint_pos_error = self.joint_pos_error.clone()
    if self.cfg.domain_rand.randomize_joint_static_error:
        twh_joint_pos_error = self._get_observation_joint_static_error(
            self.joint_pos_error
        )
    self.obs_buf = torch.cat(
        [
            self.commands[:, [0, 2]],                           # [0-1] vx指令, yaw指令（yaw永远=0，浪费！）
            (self.commands[:, [2]] - self.base_ang_vel[:, [2]]) * 0.5,  # [2]
            self.base_euler[:, :2] * 3.0,                       # [3,4]
            self.base_ang_vel * 0.5,                            # [5-7]
            (self.joint_pos[:, :10] -
             self.ref_joint_action[:, :10]),                    # [8-17]
            self.joint_vel[:, :10] * 0.1,                      # [18-27]
            twh_joint_pos_error[:, :10],                       # [28-37]
            pm_phase * self.static_flag,                       # [38-41]
            (self.pm_f * 0.3 - 1.0) * self.static_flag,       # [42-43]
        ],
        dim=1,
    )
    return self.obs_buf
```

### 修改后代码

```python
def pure_observation(self):
    pm_phase = torch.cat(
        (torch.sin(self.foot_phase), torch.cos(self.foot_phase)), 1
    )
    twh_joint_pos_error = self.joint_pos_error.clone()
    if self.cfg.domain_rand.randomize_joint_static_error:
        twh_joint_pos_error = self._get_observation_joint_static_error(
            self.joint_pos_error
        )
    self.obs_buf = torch.cat(
        [
            self.commands[:, [0]],                              # [0] vx指令（前向速度目标）
            self.base_lin_vel[:, [1]],                          # [1] 实际侧向速度vy（原来是无用的yaw指令=0，现在换成vy）
            (self.commands[:, [2]] - self.base_ang_vel[:, [2]]) * 0.5,  # [2]
            self.base_euler[:, :2] * 3.0,                       # [3,4]
            self.base_ang_vel * 0.5,                            # [5-7]
            (self.joint_pos[:, :10] -
             self.ref_joint_action[:, :10]),                    # [8-17]
            self.joint_vel[:, :10] * 0.1,                      # [18-27]
            twh_joint_pos_error[:, :10],                       # [28-37]
            pm_phase * self.static_flag,                       # [38-41]
            (self.pm_f * 0.3 - 1.0) * self.static_flag,       # [42-43]
        ],
        dim=1,
    )
    return self.obs_buf
```

### 改动说明

| 位置 | 原来 | 改后 | 原因 |
|------|------|------|------|
| `[1]` 维 | `self.commands[:, [2]]`（yaw指令，恒为0） | `self.base_lin_vel[:, [1]]`（实际侧向速度vy） | yaw指令全程为0，完全无用；换成vy让机器人能感知侧移 |

**观测维度不变，仍然是44维，不需要改export脚本或SDK。**

---

## 修改二：加强偏航惩罚（增加线性项）

**文件**：`env/tasks/locomotion_task.py`

**位置**：`reward()` 函数，第749~758行附近

### 为什么这样改？

目前的偏航奖励只有指数项，对小偏差几乎没有感知。加入线性惩罚项后，
哪怕只有 0.029 rad/s 的偏转，也会被明显惩罚。

### 原始代码

```python
yaw_rate_rew = (
    torch.exp(
        -torch.clip(10 / lin_vel_x_norm, min=1.5, max=6.0)
        * (self.commands[:, [2]] - self.env.base_ang_vel[:, [2]]) ** 2
    )
    * balance_rew
)
```

### 修改后代码

```python
yaw_rate_rew = (
    torch.exp(
        -torch.clip(10 / lin_vel_x_norm, min=1.5, max=6.0)
        * (self.commands[:, [2]] - self.env.base_ang_vel[:, [2]]) ** 2
    )
    * balance_rew
)
# 【新增】线性惩罚：对任何实际偏转速率都施加惩罚，防止小偏转积累
# commands[:,2]=0（不转弯任务），所以直接惩罚 |yaw_rate|
yaw_rate_rew -= 2.0 * torch.abs(self.env.base_ang_vel[:, [2]]) * self.static_flag
```

### 改动说明

新增一行：`yaw_rate_rew -= 2.0 * torch.abs(self.env.base_ang_vel[:, [2]]) * self.static_flag`

- `base_ang_vel[:, [2]]`：实际偏航角速度
- 系数 `2.0`：线性惩罚强度，可根据效果调整（偏大则机器人步态僵硬，偏小则无效）
- 乘以 `static_flag`：只在有前向指令时才生效（站立时不惩罚）

---

## 修改三：加入左右对称奖励

**文件**：`env/tasks/locomotion_task.py`

**位置**：`reward()` 函数，`rew_dict` 定义部分（第1226行附近）

### 为什么这样改？

数据显示右髋偏转均值 -0.078 rad，左髋 +0.001 rad，差值高达0.079 rad。
这种不对称步态天生会导致偏转。加入对称惩罚，强制左右腿保持镜像关系。

### 原始代码

在 `reward()` 函数中，找到计算 `rew_dict` 之前的部分（约第1220行附近），和 `rew_dict` 定义（第1226行）：

```python
        # ... 上面是各种奖励计算 ...

        rew_dict = dict(
            balance=balance_rew * 0.5,
            fwd_vel=forward_vel_rew * 5.5,
            yaw_rat=yaw_rate_rew * 2,
            lateral_vel=lateral_vel_rew * 4,
            # ... 其他奖励 ...
        )
```

### 修改后代码

在 `rew_dict` 定义**之前**，新增对称奖励计算；然后在 `rew_dict` 中增加这一项：

```python
        # 【新增】左右腿对称奖励
        # 惩罚左右对应关节角度不一致的情况
        # joint_pos[:, 0:5] = 左腿 [yaw, roll, pitch, knee, ankle]
        # joint_pos[:, 5:10] = 右腿 [yaw, roll, pitch, knee, ankle]
        # 理想步态：左腿和右腿应该是镜像对称的，差值越小越好
        # 注意：髋部roll(index 1,6)和yaw(index 0,5)对称时符号相反，所以要加负号
        sym_rew = -torch.norm(
            self.env.joint_pos[:, [0, 2, 3, 4]] + self.env.joint_pos[:, [5, 7, 8, 9]],
            # hip_yaw, hip_pitch, knee, ankle：左+右应该接近0（镜像）
            dim=1, keepdim=True
        )
        sym_rew += -torch.norm(
            self.env.joint_pos[:, [1]] - self.env.joint_pos[:, [6]],
            # hip_roll：左-右应该接近0（相同方向）
            dim=1, keepdim=True
        )
        sym_rew *= self.static_flag  # 只在行走时生效

        rew_dict = dict(
            balance=balance_rew * 0.5,
            fwd_vel=forward_vel_rew * 5.5,
            yaw_rat=yaw_rate_rew * 2,          # 原来是 * 2
            lateral_vel=lateral_vel_rew * 6,    # 【修改】从 4 提高到 6，加强侧向惩罚权重
            vertical_vel=vertical_vel_rew * 0.5,
            ang_vel=ang_vel_rew * 0.8,
            twist=twist_rew * 2.5,
            foot_clr=foot_clear_rew * balance_rew * 5,
            foot_supt=foot_support_rew * balance_rew * 0.7,
            foot_heit=foot_height_rew * balance_rew * 0.8,
            leg_width_rew=leg_width_rew * balance_rew * 2,
            act_const=action_constraint_rew * balance_rew * 0.4,
            sa_const=sa_constraint_rew * balance_rew * 0.2,
            foot_phase=foot_phase_rew * balance_rew * 4,
            jnt_pos_err=joint_pos_error_rew * balance_rew * 0.3,
            act_smo=action_smooth_rew * balance_rew * 0.15,
            net_smo=net_out_smooth_rew * balance_rew * 0.0005,
            net_out_val=net_out_val_rew * balance_rew * 0.00001,
            foot_slip=foot_slip_rew * balance_rew * 1.2,
            foot_vz=foot_vz_rew * 0.3 * balance_rew,
            foot_acc=foot_acc_rew * balance_rew * 0.05,
            foot_sft=foot_soft_rew * 2 * balance_rew,
            jnt_vel=joint_velocity_rew * balance_rew * 0.05,
            feet_py=foot_py_rew * balance_rew * 0.5,
            feet_frc=feet_contact_frc_rew * 0.003,
            joint_tor=joint_tor_rew * 0.01,
            pmf=pmf_rew * balance_rew * 0.03,
            symmetry=sym_rew * balance_rew * 1.0,  # 【新增】左右对称奖励
        )
```

### 改动说明

| 位置 | 原来 | 改后 | 原因 |
|------|------|------|------|
| `sym_rew` 计算 | 无 | 新增10行 | 惩罚左右腿不对称 |
| `lateral_vel` 权重 | `* 4` | `* 6` | 加强侧移惩罚50% |
| `symmetry` 项 | 无 | 新增 `* 1.0` | 将对称奖励加入训练目标 |

---

## 四、还需要同步修改 SDK 观测（一行C++代码）

因为修改一把观测[1]从"yaw指令"换成了"侧向速度vy"，**SDK的C++代码也需要做同样的替换**，否则部署时观测[1]仍然是0，模型表现会变差。

**文件**：`/home/lzz/下载/hhhhh1/mrl_sdk/source/user/rl_controller.cpp`

**位置**：`get_walk_observation()` 函数，约第403~438行

### 原始代码（C++）

```cpp
obs << target_command,                          // [0,1]: vx_cmd, yr_cmd（yr_cmd永远是0）
       (target_command.segment<1>(1) - base_rpy_rate.segment<1>(2))*0.5,  // [2]
       base_rpy.segment<2>(0)*3,                // [3,4]
       base_rpy_rate * 0.5,                     // [5-7]
       ...
```

### 修改后代码（C++）

```cpp
obs << target_command.segment<1>(0),            // [0]: vx_cmd（只取vx，不要yr_cmd）
       base_vel.segment<1>(1),                  // [1]: 实际侧向速度vy（从硬件读取）
       (target_command.segment<1>(1) - base_rpy_rate.segment<1>(2))*0.5,  // [2]
       base_rpy.segment<2>(0)*3,                // [3,4]
       base_rpy_rate * 0.5,                     // [5-7]
       ...
```

**修改后需要重新编译SDK**：
```bash
cd /home/lzz/下载/hhhhh1/mrl_sdk
mkdir -p build && cd build
cmake .. && make -j4
```

---

## 五、所有修改文件汇总

| 文件 | 修改内容 | 是否必须 |
|------|---------|---------|
| `env/tasks/locomotion_task.py` | 修改一：vy进观测（第538行） | ⭐⭐⭐ 必须 |
| `env/tasks/locomotion_task.py` | 修改二：偏航线性惩罚（第755行后） | ⭐⭐ 建议 |
| `env/tasks/locomotion_task.py` | 修改三：对称奖励（第1220行后，rew_dict） | ⭐⭐ 建议 |
| `mrl_sdk/source/user/rl_controller.cpp` | SDK同步：get_walk_observation()改vy（第404行） | ⭐⭐⭐ 必须（部署时） |

---

## 六、操作步骤

### 步骤1：修改训练代码

按照上面三处改动，编辑 `env/tasks/locomotion_task.py`：

```bash
gedit /data/Downloads/hhhhh1/legged_rl-final/env/tasks/locomotion_task.py
```

### 步骤2：重新训练（从断点继续）

```bash
cd /data/Downloads/hhhhh1/legged_rl-final

# 从现有模型继续训练（--resume 指定要接着跑的实验名）
python train.py --name try_x3 --resume try_x2
```

说明：`--resume try_x2` 表示加载 try_x2 的最新模型权重继续训练，不用从零开始。

### 步骤3：验证效果

训练一段时间后（几百次迭代），运行可视化查看是否还走歪：
```bash
python play.py --name try_x3
```

观察以下指标是否改善：
- 侧向速度 vy 是否接近 0
- 偏航角速度是否接近 0
- 左右腿是否更对称

### 步骤4（部署前）：同步修改SDK

参照"四"中的C++修改，修改SDK并重新编译，再按 `deployment_guide.md` 的流程导出模型并部署。

---

## 七、预期效果

| 指标 | 修改前（当前数据） | 修改后预期 |
|------|-----------------|-----------|
| X方向前进 9.63m 后的Y偏移 | -2.18 m（22.6%） | < 0.5 m（< 5%） |
| 平均偏航角速度 | 0.029 rad/s | < 0.005 rad/s |
| 右髋yaw vs 左髋yaw 差值 | 0.079 rad | < 0.020 rad |
| 侧向速度 vy 标准差 | 0.118 m/s | < 0.05 m/s |

---

## 八、如果效果还不够好

可以调大以下参数继续尝试（在 `rew_dict` 里修改系数）：

```python
# 在 rew_dict 中：
lateral_vel = lateral_vel_rew * 8,    # 从6继续加到8
symmetry    = sym_rew * balance_rew * 2.0,  # 从1.0加到2.0
```

同时在偏航线性惩罚里：
```python
yaw_rate_rew -= 3.0 * torch.abs(...)  # 从2.0加到3.0
```

> 注意：系数不能无限加大，过大会导致机器人动作过于僵硬，反而走路不自然。建议每次调整一项，观察效果后再继续调整。
