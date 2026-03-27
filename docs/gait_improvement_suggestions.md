# 机器人步态姿态改进建议

## 📊 问题诊断总结

根据 `debug_0.xlsx` 的数据分析，发现以下主要问题：

| 问题类型 | 具体数据 | 严重程度 |
|---------|---------|---------|
| 左右脚接地时间不平衡 | 左脚66.8% vs 右脚35.1% | ⚠️ 严重 |
| 髋关节Yaw角度差异 | 左23.42° vs 右8.16° (差15°) | ⚠️ 严重 |
| 膝关节力矩差异 | 左-39.29 vs 右-18.05 (差21) | ⚠️ 严重 |
| 足部Yaw角度差异 | 左26.68° vs 右12.02° (差15°) | ⚠️ 中等 |
| 足部Roll不对称 | 左-5.56°(内翻) vs 右1.89°(外翻) | ⚠️ 中等 |

---

## 🎯 修改建议

### 1. 新增：左右对称性奖励 (最高优先级)

**问题原因**：当前奖励函数没有惩罚左右脚发力不均，导致策略学会"偏心"使用左脚。

**修改位置**：`locomotion_task.py` 的 `reward()` 函数

#### 源代码 (无此奖励)

```python
# 当前代码中没有左右对称性奖励
```

#### 修改后代码

在 `reward()` 函数中添加（建议放在 `leg_width_rew` 附近，约第1395行后）：

```python
# ========== 新增：左右对称性奖励 ==========
# 1. 关节力矩对称性 (惩罚左右膝关节、髋关节力矩差异)
left_leg_taus = self.joint_tau[:, [2, 3]]   # 左髋pitch, 左膝
right_leg_taus = self.joint_tau[:, [7, 8]]  # 右髋pitch, 右膝
tau_symmetry_rew = -0.5 * torch.norm(
    left_leg_taus - right_leg_taus, dim=1, keepdim=True
) * self.static_flag

# 2. 足部接触力对称性 (惩罚左右脚接触力差异)
contact_force_diff = torch.abs(self.foot_frc[:, [0]] - self.foot_frc[:, [1]])
foot_force_symmetry_rew = -0.002 * contact_force_diff * self.static_flag

# 3. 接触时间对称性 (基于支撑相位)
left_support = self.foot_support_mask[:, [0]].float()
right_support = self.foot_support_mask[:, [1]].float()
contact_time_diff = torch.abs(left_support - right_support)
contact_symmetry_rew = -0.3 * contact_time_diff * self.static_flag

# 合并为总对称性奖励
symmetry_rew = tau_symmetry_rew + foot_force_symmetry_rew + contact_symmetry_rew
# ========== 对称性奖励结束 ==========
```

然后在 `rew_dict` 中添加：

```python
rew_dict = dict(
    # ... 其他奖励 ...
    symmetry=symmetry_rew * balance_rew * 1.5,  # 新增
)
```

---

### 2. 新增：髋关节Yaw角度限制奖励 (高优先级)

**问题原因**：当前 `action_constraint_rew` 对髋关节Yaw的惩罚不够，导致脚尖外撇严重。

#### 源代码 (约第1261-1271行)

```python
action_constraint_rew = (
    -0.3
    * torch.clip(1.0 / lin_vel_x_norm, 0, 1.5)
    * torch.norm((self.env.joint_pos), dim=1, keepdim=True)
)
action_constraint_rew += (
    -0.5
    * torch.clip(1.0 / lin_vel_x_norm, 0, 1.5)
    * torch.norm((self.env.joint_pos[:, [0, 5]]), dim=1, keepdim=True)
)
```

#### 修改后代码

```python
action_constraint_rew = (
    -0.3
    * torch.clip(1.0 / lin_vel_x_norm, 0, 1.5)
    * torch.norm((self.env.joint_pos), dim=1, keepdim=True)
)
action_constraint_rew += (
    -0.5
    * torch.clip(1.0 / lin_vel_x_norm, 0, 1.5)
    * torch.norm((self.env.joint_pos[:, [0, 5]]), dim=1, keepdim=True)
)

# ========== 新增：髋关节Yaw角度限制 ==========
# hip_yaw_joint 索引: 左0, 右5
# 限制目标: 保持在 ±10° (约0.174 rad) 范围内
HIP_YAW_TARGET = 0.1  # 目标角度 (rad)，约5.7°
HIP_YAW_TOLERANCE = 0.15  # 容差 (rad)，约8.6°

left_hip_yaw = self.env.joint_pos[:, [0]]
right_hip_yaw = self.env.joint_pos[:, [5]]

# 惩罚偏离目标角度
hip_yaw_penalty = -2.0 * torch.clip(
    1.0 / lin_vel_x_norm, 0, 2.0
) * (
    torch.abs(left_hip_yaw - HIP_YAW_TARGET) +
    torch.abs(right_hip_yaw - HIP_YAW_TARGET)
)

# 惩罚左右不对称
hip_yaw_asymmetry = -1.5 * torch.clip(
    1.0 / lin_vel_x_norm, 0, 2.0
) * torch.abs(left_hip_yaw - right_hip_yaw)

action_constraint_rew += hip_yaw_penalty + hip_yaw_asymmetry
# ========== 髋关节Yaw限制结束 ==========
```

---

### 3. 增强：足部Yaw角度对齐奖励 (高优先级)

**问题原因**：当前 `foot_py_rew` 只惩罚脚与身体的偏航差，没有惩罚左右脚之间的差异。

#### 源代码 (约第1373-1393行)

```python
foot_py_rew = -0.5 * (
    torch.norm(
        smallest_signed_angle_between_torch(
            self.env.foot_euler[:, [2]], self.env.base_euler[:, [2]]
        ),
        dim=1,
        keepdim=True,
    )
)
foot_py_rew += -0.5 * (
    torch.norm(
        smallest_signed_angle_between_torch(
            self.env.foot_euler[:, [5]], self.env.base_euler[:, [2]]
        ),
        dim=1,
        keepdim=True,
    )
)
```

#### 修改后代码

```python
foot_py_rew = -0.5 * (
    torch.norm(
        smallest_signed_angle_between_torch(
            self.env.foot_euler[:, [2]], self.env.base_euler[:, [2]]
        ),
        dim=1,
        keepdim=True,
    )
)
foot_py_rew += -0.5 * (
    torch.norm(
        smallest_signed_angle_between_torch(
            self.env.foot_euler[:, [5]], self.env.base_euler[:, [2]]
        ),
        dim=1,
        keepdim=True,
    )
)

# ========== 新增：左右脚Yaw对齐 ==========
# 惩罚左右脚朝向不一致
left_foot_yaw = self.env.foot_euler[:, [2]]
right_foot_yaw = self.env.foot_euler[:, [5]]
foot_yaw_diff = smallest_signed_angle_between_torch(left_foot_yaw, right_foot_yaw)
foot_yaw_symmetry_rew = -0.8 * torch.abs(foot_yaw_diff) * self.static_flag

# 惩罚双脚外撇 (相对于前进方向)
# 理想情况: 脚尖朝前 (yaw ≈ 0)
FOOT_YAW_TARGET = 0.0  # 目标朝向正前方
foot_forward_rew = -0.3 * (
    torch.abs(left_foot_yaw - FOOT_YAW_TARGET) +
    torch.abs(right_foot_yaw - FOOT_YAW_TARGET)
) * self.static_flag

foot_py_rew += foot_yaw_symmetry_rew + foot_forward_rew
# ========== 足部Yaw对齐结束 ==========
```

---

### 4. 新增：足部Roll角度控制 (中等优先级)

**问题原因**：左脚内翻(-5.56°)，右脚外翻(1.89°)，导致脚掌着地不平稳。

#### 源代码 (无此奖励)

```python
# 当前代码中没有足部Roll角度控制
```

#### 修改后代码

在 `reward()` 函数中添加：

```python
# ========== 新增：足部Roll角度控制 ==========
# foot_euler: [L_roll, L_pitch, L_yaw, R_roll, R_pitch, R_yaw]
left_foot_roll = self.env.foot_euler[:, [0]]
right_foot_roll = self.env.foot_euler[:, [3]]

# 惩罚脚内翻/外翻 (理想值接近0)
FOOT_ROLL_TARGET = 0.0
foot_roll_penalty = -0.4 * torch.clip(1.0 / lin_vel_x_norm, 0, 1.5) * (
    torch.abs(left_foot_roll - FOOT_ROLL_TARGET) +
    torch.abs(right_foot_roll - FOOT_ROLL_TARGET)
) * self.static_flag

# 惩罚左右不对称
foot_roll_asymmetry = -0.3 * torch.clip(1.0 / lin_vel_x_norm, 0, 1.5) * (
    torch.abs(left_foot_roll - right_foot_roll)
) * self.static_flag

foot_roll_rew = foot_roll_penalty + foot_roll_asymmetry
# ========== 足部Roll控制结束 ==========
```

然后在 `rew_dict` 中添加：

```python
rew_dict = dict(
    # ... 其他奖励 ...
    foot_roll=foot_roll_rew * balance_rew * 0.5,  # 新增
)
```

---

### 5. 修改：支撑相力分配奖励 (中等优先级)

**问题原因**：当前 `feet_contact_frc_rew` 的支撑相惩罚逻辑可能导致策略偏向使用某一只脚。

#### 源代码 (约第1099-1102行)

```python
feet_contact_frc_rew += -torch.sum(
    (20.0 - self.env.foot_frc).clip(min=0.0) * self.foot_support_mask,
    dim=1, keepdim=True
)
```

#### 修改后代码

```python
# ========== 修改：支撑相力分配 ==========
# 原代码只惩罚力不足，新增：鼓励双腿均衡承重
TARGET_SUPPORT_FORCE = 250.0  # 目标单脚支撑力 (N)

# 惩罚支撑力不足
feet_contact_frc_rew += -torch.sum(
    (20.0 - self.env.foot_frc).clip(min=0.0) * self.foot_support_mask,
    dim=1, keepdim=True
)

# 新增：鼓励双腿支撑时力量均衡
both_support = (self.foot_support_mask[:, [0]] & self.foot_support_mask[:, [1]]).float()
force_balance_rew = -0.005 * torch.abs(
    self.env.foot_frc[:, [0]] - self.env.foot_frc[:, [1]]
) * both_support * self.static_flag

feet_contact_frc_rew += force_balance_rew
# ========== 支撑相力分配结束 ==========
```

---

### 6. 修改：动作空间限制 (可选，动作层面硬约束)

**修改位置**：`locomotion_task.py` 的 `action()` 函数

#### 源代码 (约第576-596行)

```python
if self.cfg.action.use_increment:
    act = (
        self.current_joint_act[:, :10]
        + net_out[:, self.num_legs:] * self.env.dt
    )
    act = torch.clip(
        act,
        self.joint_action_limit_low[:, :10],
        self.joint_action_limit_high[:, :10],
    )
else:
    act = torch.clip(
        net_out[:, self.num_legs:],
        self.joint_action_limit_low,
        self.joint_action_limit_high,
    )
```

#### 修改后代码

```python
# 髋关节Yaw的硬限制
HIP_YAW_MIN = -0.25  # 约-14°
HIP_YAW_MAX = 0.35   # 约20°

if self.cfg.action.use_increment:
    act = (
        self.current_joint_act[:, :10]
        + net_out[:, self.num_legs:] * self.env.dt
    )
    act = torch.clip(
        act,
        self.joint_action_limit_low[:, :10],
        self.joint_action_limit_high[:, :10],
    )
else:
    act = torch.clip(
        net_out[:, self.num_legs:],
        self.joint_action_limit_low,
        self.joint_action_limit_high,
    )

# ========== 新增：髋关节Yaw硬限制 ==========
# 限制左右髋关节Yaw范围，防止过度外撇
act[:, [0]] = torch.clip(act[:, [0]], HIP_YAW_MIN, HIP_YAW_MAX)  # 左髋Yaw
act[:, [5]] = torch.clip(act[:, [5]], HIP_YAW_MIN, HIP_YAW_MAX)  # 右髋Yaw
# ========== 硬限制结束 ==========
```

---

## 📋 完整修改汇总

### 需要在 `rew_dict` 中新增的奖励项

```python
rew_dict = dict(
    balance=balance_rew * 0.5,
    fwd_vel=forward_vel_rew * 5.5,
    yaw_rat=yaw_rate_rew * 3,
    lateral_vel=lateral_vel_rew * 4,
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
    # ========== 新增奖励 ==========
    symmetry=symmetry_rew * balance_rew * 1.5,      # 左右对称性
    foot_roll=foot_roll_rew * balance_rew * 0.5,    # 足部Roll控制
)
```

---

## ⚖️ 权重调优建议

根据当前各奖励的均值，建议按以下顺序调整：

| 优先级 | 修改 | 说明 |
|--------|------|------|
| 1 | `symmetry` 权重: 1.0 → 2.0 | 逐步增加，观察左右平衡改善 |
| 2 | `act_const` 权重: 0.4 → 0.6 | 加强髋关节Yaw惩罚 |
| 3 | `feet_py` 权重: 0.5 → 0.8 | 加强脚尖朝向约束 |
| 4 | `foot_phase` 权重: 4.0 → 3.0 | 适当降低，避免过度约束步态 |

---

## 🔧 训练策略建议

1. **课程学习**：先用较低的速度命令(0.3-0.5 m/s)训练对称性，再逐步提高速度
2. **奖励热身**：前1000次迭代将新增奖励权重从0逐渐增加到目标值
3. **早停机制**：如果某只脚接地时间超过65%，触发重置

---

## 📈 预期效果

修改后预期改善：
- 左右脚接地时间比例：66.8%/35.1% → ~50%/50%
- 髋关节Yaw差异：15° → <5°
- 膝关节力矩差异：21 → <10
- 整体步态更自然，减少"坡脚"现象
