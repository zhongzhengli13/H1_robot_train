# H1 人形机器人强化学习行走训练

基于 PPO 算法，在 NVIDIA Isaac Gym 仿真环境中训练 Unitree H1 人形机器人行走。

## 环境要求

- Ubuntu 18.04 / 20.04+
- NVIDIA GPU（8GB+ 显存），驱动 470+，CUDA 11.4+
- Python 3.8，PyTorch 2.0.0+
- Isaac Gym Preview 4

## 安装

```bash
# 1. 创建 conda 环境
conda create -n isaac python==3.8 && conda activate isaac

# 2. 安装 PyTorch
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.0

# 3. 安装 Isaac Gym
tar -zxvf IsaacGym_Preview_4_Package.tar.gz
cd isaacgym/python && pip install -e .

# 4. 安装依赖
pip install -r requirements.txt
pip install matplotlib pandas tensorboard opencv-python numpy==1.23.5 openpyxl
```

## 训练

```bash
# 基本训练
python train.py --name <实验名>

# 指定迭代次数
python train.py --name try_x8 --max_iterations 4000

# 从断点恢复训练
python train.py --name try_x8 --resume try_x8

# 查看训练日志
tensorboard --logdir experiments/<实验名>/log --bind_all
```

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--name` | test3 | 实验名，保存在 `experiments/<name>/` |
| `--config` | loc | 配置文件（`config/loc.py`） |
| `--max_iterations` | 8000 | 最大训练轮数 |
| `--resume` | 无 | 从指定实验的 checkpoint 恢复 |
| `--render` | False | 训练时开启可视化窗口 |
| `--rl_device` | cuda:0 | 训练设备 |

## 演示

```bash
# 可视化演示
python play.py --name <实验名> --render

# 录制视频（保存到 experiments/<name>/debug/<name>.mp4）
python play.py --name <实验名> --video

# 指定演示时长（秒）
python play.py --name <实验名> --render --time 30

# 加载指定 checkpoint
python play.py --name <实验名> --render --iter 2000

# 保存关节数据到 Excel
python play.py --name <实验名> --render --debug

# CPU 模式（GPU 显存不足时）
python play.py --name <实验名> --video --sim_device cpu --rl_device cpu
```

### 演示参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--name` | test3 | 实验名 |
| `--render` | False | 开启可视化窗口 |
| `--video` | False | 录制视频 |
| `--time` | 20 | 演示时长（秒） |
| `--iter` | 无 | 加载指定迭代的 checkpoint（默认加载最新） |
| `--epochs` | 1 | 评估轮数 |
| `--debug` | False | 保存数据到 Excel |
| `--fix_cam` | True | 相机跟随机器人 |

## 导出模型

```bash
# 导出为 ONNX 格式
python export_pt2onnx.py --name <实验名>
```

## 项目结构

```
legged_rl-final/
├── config/loc.py              # 超参数配置（PD增益、奖励权重、关节范围等）
├── env/
│   ├── legged_robot.py        # Isaac Gym 仿真环境封装
│   ├── gym_env_wrapper.py     # 环境包装器
│   └── tasks/
│       └── locomotion_task.py # 核心：观测、动作、奖励、终止条件
├── model/simple_policy.py     # Actor-Critic 网络定义
├── rl/alg/ppo.py              # PPO 算法实现
├── train.py                   # 训练入口
├── play.py                    # 演示/推理入口
└── assets/h1/urdf/h1.urdf    # H1 机器人模型
```

## 配置说明

主要配置在 `config/loc.py` 中：

- **pd_gains**: PD 控制器刚度和阻尼
- **action**: 关节增量范围、参考姿态
- **command**: 速度指令范围
- **algorithm**: PPO 超参数（学习率、clip、折扣因子等）
- **domain_rand**: 域随机化（摩擦力、质量、延迟等）

## 英文版

详见 [README_en.md](README_en.md)
