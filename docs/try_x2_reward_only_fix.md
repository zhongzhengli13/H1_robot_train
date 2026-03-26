# try_x2 纯奖励修复方案（不改 obs，SDK 对齐）

> 约束：观测向量 44 维结构不变，所有修改仅在奖励函数内进行

---

## 一、为什么之前的方案效果一般

当前 `yaw_rat` 奖励惩罚的是**偏转速率**，但实际问题是**累积偏转角度**：

```
当前方案（惩罚速率）：
  yaw_rate_rew = exp(-k * yaw_rate²) * balance * 2.0
  当 yaw_rate = 0.029 rad/s 时 → 奖励 ≈ 0.9975（几乎满分）
  → 网络认为"慢慢偏转"是合法的

真实问题：
  每步 0.029 rad/s × 0.1s = 0.0029 rad
  1999 步累积 → Yaw 偏转 0.785 rad（45°）→ 侧移 2.25m
```

只惩罚速率≈只惩罚"拐弯太快"，不惩罚"一直慢慢转"。需要一个**惩罚累积角度**的奖励。

---

## 二、核心方案：Heading 误差奖励

### 原理

代码中 `self.commands` 已有第 4 维 `commands[:, 3]`（heading 指令），目前生成了但从未连接到任何奖励。

这个量存的是"期望朝向角"，可以直接用来约束机器人不偏离目标方向：

```
heading_error = smallest_angle(base_euler[:, 2], commands[:, 3])
heading_rew   = -|heading_error|  × weight
```

- `base_euler[:, 2]`：机器人当前 Yaw 角（已有）
- `commands[:, 3]`：期望朝向角（已有，只是没用）
- **不需要改观测，不需要改网络结构，不需要改 SDK**

### 实现代码

在 `reward()` 函数内，找到 `yaw_rate_rew` 计算附近，新增：

```python
# === Heading 误差奖励（新增）===
# commands[:, 3] 为期望朝向角（直线行走时通常为 0）
# base_euler[:, 2] 为当前朝向角
heading_error = torch.atan2(
    torch.sin(self.base_euler[:, 2] - self.commands[:, 3]),
    torch.cos(self.base_euler[:, 2] - self.commands[:, 3])
)  # 结果在 [-π, π] 之间，处理角度回绕

heading_rew = -torch.abs(heading_error)
```

在 `rew_dict` 中添加：

```python
heading=heading_rew * balance_rew * 4.0,
```

### 权重选择依据

| 参考量 | 当前 try_x2 均值 | 乘以权重后量级 |
|--------|----------------|--------------|
| `yaw_rat`（现有）| 1.764 | 1.764 × 2.0 = 3.53 |
| `heading`（新增）| -0.502 rad（当前均值）| -0.502 × 4.0 = **-2.01**（显著惩罚）|

权重 4.0 让 heading 误差均值贡献约 -2.0，与主要速度奖励量级相当，既有效又不会压制前进行为。

### 与现有 `yaw_rat` 的互补关系

```
yaw_rat（现有）：惩罚"正在拐弯的速度"  → 微分项
heading（新增）：惩罚"偏离了多少角度"  → 积分项

两者结合 = PD 控制逻辑：
  → 偏转过快会被 yaw_rat 惩罚（不要快转）
  → 已经偏了的角度会被 heading 惩罚（必须回正）
```

---

## 三、补充方案：左右对称奖励

不依赖 obs，纯粹约束关节角度：

```python
# 镜像对称：理想情况下 L_joint + R_joint * sign = 0
# 符号取决于关节定义，通常：
#   hip_yaw:   L + R（两腿外展对称） → sign = +1
#   hip_roll:  L - R（左外展=右内收）→ sign = -1
#   hip_pitch: L + R（前后对称）     → sign = +1
#   knee:      L + R（弯曲对称）     → sign = +1
#   ankle:     L + R（踝关节对称）   → sign = +1
mirror_signs = torch.tensor([1, -1, 1, 1, 1], device=self.device)

left_joints  = self.joint_pos[:, :5]
right_joints = self.joint_pos[:, 5:10]
sym_error = left_joints - right_joints * mirror_signs  # 理想时为 0

sym_rew = -torch.norm(sym_error, dim=-1)
```

在 `rew_dict` 中添加：

```python
symmetry=sym_rew * balance_rew * 5.0,
```

> **注意**：`mirror_signs` 的正负号需根据实际 URDF 关节定义核对，可先打印一个理想站立姿态下的 `sym_error`，确认其接近零后再启用。

---

## 四、辅助调整：增强现有 Yaw 惩罚（可选）

不改 obs，只改奖励系数：

**方案 4a：提高 `yaw_rat` 权重**

```python
# 现有：
yaw_rat=yaw_rate_rew * 2,
# 改为：
yaw_rat=yaw_rate_rew * 4,
```

**方案 4b：对 Yaw 角速度追加线性惩罚**

```python
# 在 yaw_rate_rew 计算结束后追加（不依赖 static_flag）：
yaw_rate_rew -= 6.0 * torch.abs(self.base_ang_vel[:, 2])
```

> 方案 4a/4b 效果有限（因为指数项对小偏差本身就不灵敏），建议作为 heading 奖励的辅助，而非主要手段。

---

## 五、方案汇总与优先级

| 优先级 | 方案 | 修改位置 | 预期效果 | obs 改动 |
|--------|------|---------|---------|---------|
| **★★★** | Heading 误差奖励（权重 4.0） | `reward()` + `rew_dict` | 直接约束累积 Yaw 漂移，解决直线行走 | **无** |
| **★★☆** | 左右对称奖励（权重 5.0） | `reward()` + `rew_dict` | 改善姿态、抬脚不对称 | **无** |
| **★☆☆** | Yaw 线性惩罚（系数 6.0） | `yaw_rate_rew` 计算内 | 辅助增强 | **无** |

---

## 六、建议实施顺序

```
第一步：单独加 Heading 奖励（权重 4.0）
        训练 1000 iter → debug 检查漂移率和 Yaw 末端角度

第二步：若步态仍不对称，加对称奖励（权重 5.0）
        训练 1000 iter → debug 检查膝关节左右差值

第三步：若仍有残余 Yaw 漂移，叠加线性惩罚
        （通常第一步就能解决大部分漂移问题）
```

**验证时 debug 重点关注：**
- `euler_z` 末端值（目标 < 0.1 rad）
- `base_pos[1]` / `base_pos[0]` 漂移率（目标 < 5%）
- `knee_left` vs `knee_right` 均值差（目标 < 0.05 rad）

---

*文档生成于 2026-03-25*
