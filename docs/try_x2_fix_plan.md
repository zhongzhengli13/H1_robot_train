# try_x2 问题诊断与修复方案

> 基于 `debug_0.xlsx`（1999步推理轨迹）分析，代码版本：try_x2

---

## 一、数据总览

| 指标 | 数值 | 问题等级 |
|------|------|----------|
| 前向速度跟踪 | 0.498 m/s（目标 0.4） | ✓ 良好 |
| 侧向漂移 | 最终 -2.25 m / 前进 9.63 m = **23.4%** | ✗ 严重 |
| Yaw 角累积偏转 | 平均 0.502 rad，最大 **0.785 rad（≈45°）** | ✗ 极严重 |
| 右脚平均接触力 | 307 N（左脚仅 209 N，右腿多承重 47%） | ✗ 严重不对称 |
| 膝关节不对称 | 左 0.640 rad，右 0.780 rad，**差值 0.140 rad（8°）** | ✗ 明显 |
| 髋俯仰不对称 | 左 -0.165 rad，右 -0.277 rad，**差值 0.112 rad（6.4°）** | ✗ 明显 |
| 左脚平均高度 | 0.0355 m | ✓ 可接受 |
| 右脚平均高度 | 0.0388 m | ✓ 可接受 |

---

## 二、问题一：直线行走差（侧向漂移 23.4%）

### 原因链分析

```
右腿承重更多（307N vs 209N）
    ↓
右腿推力更大，产生向左的转矩
    ↓
机器人向左偏转（yaw_rate 均值 0.029 rad/s）
    ↓
Yaw 角在整个 episode 内从 0 积累到 45°
    ↓
运动方向持续偏离，侧向漂移累积到 2.25m
```

### 当前 Yaw 惩罚为何不足

**指数项几乎无效：**
```
yaw_rate_rew = exp(-k * (0.0 - 0.029)²) * balance * 2.0
当 yaw_rate = 0.029 rad/s, k ≈ 3 时：
exp(-3 * 0.029²) = exp(-0.0025) ≈ 0.9975
→ 几乎满分，惩罚极微弱
```

**侧向速度惩罚被观测"盲区"削弱：**

当前观测向量（44维）的第2维是 `commands[:, 2]`（yaw 指令，始终为0）。
网络看到的是"一个永远为0的信号"，实际侧向速度 `vy` 没有进入观测，网络无法感知侧漂。

### 修复方案（按顺序执行）

**修复 A：把侧向速度 `vy` 加入观测**

在构建观测向量时，将原来填充 yaw command（始终为0）的通道替换为实际的侧向速度：

```python
# 找到观测构建位置，通常形如：
# obs[:, 1] = commands[:, 2]  # yaw command (always 0)
# 改为：
obs[:, 1] = base_lin_vel[:, 1]  # 实际侧向速度
```

这是让网络感知侧漂的最低成本修复，不改变网络结构，只改变信息输入。

**修复 B：对 Yaw 角速度增加线性惩罚**

在 `yaw_rat` 奖励计算结束后追加：

```python
# 在 yaw_rat_rew 计算之后添加：
yaw_rate_rew -= 6.0 * torch.abs(base_ang_vel[:, 2])
```

线性惩罚对小偏差（0.029 rad/s）有实质性惩罚，而指数形式在此量级几乎无效。
系数 6.0 的选择依据：`yaw_rat` 当前权重为 2.0，均值约 1.76；添加 `6.0 * 0.029 ≈ 0.17` 的额外惩罚量级适当。

**修复 C：对侧向速度增强线性惩罚系数**

当前代码中已有：
```python
lateral_vel_rew -= 3.0 * torch.abs(base_lin_vel[:, 1])
```

建议提高到：
```python
lateral_vel_rew -= 5.0 * torch.abs(base_lin_vel[:, 1])
```

---

## 三、问题二：步态不对称 / 姿态难看

### 原因分析

try_x2 **没有任何左右对称性奖励**。当前 27 个奖励组件中：
- `sa_const`：只惩罚支撑相关节角度大小，不要求左右对称
- `leg_width_rew`：只管腿间距，不管左右动作是否镜像
- `foot_phase`：只管相位差（反相位），不管关节角镜像

网络自由探索找到了一个"不对称但稳定"的局部最优——右腿承重更多，推力更大，换来了较高的 `fwd_vel` 奖励，而没有任何惩罚制约这种不对称。

**膝关节不对称量化：**
```
knee_left  均值: 0.640 rad（较直）
knee_right 均值: 0.780 rad（较弯，多弯 8°）
→ 右腿蹲伏更深，提供更大推力，同时也让右脚有效离地高度降低
```

### 修复方案

**修复 D：添加左右对称奖励**

在奖励函数末尾添加新组件 `symmetry`：

```python
# 镜像关节对：左腿关节 = 右腿相应关节的负值（因为对称）
# 假设关节顺序为：[L_hip_yaw, L_hip_roll, L_hip_pitch, L_knee, L_ankle,
#                  R_hip_yaw, R_hip_roll, R_hip_pitch, R_knee, R_ankle]
# 镜像对称要求：L_hip_yaw ≈ -R_hip_yaw，L_hip_roll ≈ -R_hip_roll，
#              L_hip_pitch ≈ R_hip_pitch，L_knee ≈ R_knee，L_ankle ≈ R_ankle

mirror_signs = torch.tensor([-1, -1, 1, 1, 1], device=device, dtype=torch.float)
left_joints  = joint_pos[:, :5]           # 左腿 5 个关节
right_joints = joint_pos[:, 5:10]         # 右腿 5 个关节
sym_error = left_joints + right_joints * mirror_signs   # 理想时应为 0
sym_rew = -torch.norm(sym_error, dim=-1)

rew_dict['symmetry'] = sym_rew * balance_rew * 6.0
```

权重建议 **6.0**（与 `fwd_vel` 的 5.5 量级对齐，确保对称性不会被速度奖励完全压制）。

> **注意 mirror_signs 的符号：** 需要根据实际关节定义确认。如果正向 hip_yaw 对左腿是向外，对右腿是向内，则应为 -1；hip_pitch 和 knee 通常同向，则为 +1。请核对 URDF 确认。

---

## 四、问题三：抬脚高度不一致

### 原因分析

右脚均值高度 0.0388 m，左脚 0.0355 m，差值约 3mm，本身差距较小。
但主观感受"一条腿抬得低"更可能来源于以下两点：

1. **步频/相位不对称**：右腿承重 307N vs 左腿 209N，支撑时间更长，摆动相更短，视觉上显得"卡顿"
2. **`foot_heit` 奖励只惩罚过高，不惩罚过低**：

```python
# 当前设计：
foot_heit = 50.0 * clip(foot_h, 0, 0.1) * swing_mask   # 鼓励0~10cm
foot_heit -= 1.0 * (foot_h - 0.15).clip(min=0)          # 惩罚超过15cm

# 问题：如果摆动脚只抬高 2cm，奖励 = 50*0.02 = 1.0
#       如果抬高 10cm，奖励 = 50*0.10 = 5.0（最大）
# → 有正向激励，但左右腿共享同一奖励公式，不对称无法被感知
```

### 修复方案

**修复 E：增加最小抬脚高度惩罚**

```python
# 在摆动相内，脚高低于 0.04m 时给予额外惩罚
min_clearance = 0.04  # 目标最低离地高度 4cm
clearance_deficit = (min_clearance - foot_height).clip(min=0) * foot_swing_mask
min_clear_rew = -torch.sum(clearance_deficit, dim=-1) * 10.0

rew_dict['min_clearance'] = min_clear_rew * balance_rew * 1.5
```

**修复 F（可选）：惩罚左右接触时间差**

```python
# 计算当前 episode 窗口内的接触频率差
contact_diff = torch.abs(contact_frac_left - contact_frac_right)
contact_sym_rew = -contact_diff * 2.0

rew_dict['contact_symmetry'] = contact_sym_rew * balance_rew * 1.0
```

---

## 五、修复优先级与实施顺序

建议分两批实施，每批训练 1000～1500 iter 后用 debug 验证效果：

### 第一批（直接解决直线行走问题，改动最小）

| 编号 | 修改内容 | 预期效果 | 风险 |
|------|---------|---------|------|
| **A** | 观测中 `vy` 替换零值 yaw 指令 | 网络感知侧漂，为 B/C 提供前提 | 低，不影响网络结构 |
| **B** | 追加 `yaw_rate_rew -= 6.0 * \|ang_vel[2]\|` | 直接惩罚 yaw 偏转 | 低 |
| **C** | 侧向速度线性系数 3.0 → 5.0 | 增强侧漂惩罚 | 低 |

**验证指标：** `base_pos[1]` 最终值、Yaw 角均值

---

### 第二批（解决不对称问题，改动较大）

| 编号 | 修改内容 | 预期效果 | 风险 |
|------|---------|---------|------|
| **D** | 添加 `symmetry` 奖励，权重 6.0 | 强制左右腿对称 | 中，可能减慢收敛 |
| **E** | 添加 `min_clearance` 奖励，权重 1.5 | 保证最低抬脚高度 | 低 |
| **F** | 添加接触时间对称惩罚 | 步频均匀化 | 中 |

**验证指标：** `knee_left` vs `knee_right` 均值差、接触力左右比值

---

## 六、每次 Debug 需重点核查的指标

每次训练完生成 debug.xlsx 后，重点检查以下 5 个数字：

1. `base_pos[1]` 最终值 / `base_pos[0]` 最终值 → **漂移率**（目标 < 5%）
2. `euler_z` 末端值 → **Yaw 偏转**（目标 < 5°）
3. `knee_left` 均值 - `knee_right` 均值 → **膝关节对称性**（目标 < 0.05 rad）
4. 左右接触力均值比 → **承重对称性**（目标 < 1.1）
5. 左右脚高度均值 → **抬脚对称性**（目标差值 < 0.005 m）

---

## 七、当前 try_x2 基准参考值（修改前）

| 指标 | try_x2 基准值 |
|------|--------------|
| 漂移率 | 23.4% |
| Yaw 偏转（末端） | ~45° |
| 膝关节不对称 | 0.140 rad |
| 接触力比（右/左） | 307/209 = 1.47 |
| 抬脚高度差 | 3.3 mm |

---

*文档生成于 2026-03-25，基于 try_x2 推理数据分析*
