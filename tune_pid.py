# This file is used to tune PID without RL learning, run it to see how well it will behave
# Parameters 'p' & 'd' can be changed in line 118 & 119
# Data will be saved in foldr 'excel/tune_PID'
import time
import math
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
import os
from math import pi
import matplotlib.pyplot as plt
import pandas as pd
from env.utils.trajectory_generator import VerticalTrajectoryGenerator
import warnings
from os.path import join
import torch

warnings.filterwarnings("ignore")


# Z-up axis in this file, (x,y,z)


def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)


# initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = 0.005  # dt*action_repeat=0.01 #单位秒（s） #仿真时间步---》可以得到仿真频率 #物理仿真积分步长 = 5 ms，物理积分频率=1/f=200hz
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)  # -9.81
sim_params.substeps = 1
sim_params.use_gpu_pipeline = False
print("WARNING: Forcing CPU pipeline.")

# physx parameters
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4  # 位置迭代
sim_params.physx.num_velocity_iterations = 0  # 速度迭代
sim_params.physx.num_threads = 10
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0
sim_params.physx.bounce_threshold_velocity = 0.5
sim_params.physx.max_depenetration_velocity = 1.0
sim_params.physx.max_gpu_contact_pairs = 2**23
sim_params.physx.default_buffer_size_multiplier = 5
sim_params.physx.use_gpu = True

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
plane_params.restitution = 0
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load asset
asset_path = "assets/h1/urdf/h1.urdf"

asset_root = os.path.dirname(asset_path)
asset_file = os.path.basename(asset_path)

asset_options = gymapi.AssetOptions()
asset_options.angular_damping = 0.0  # 为物体提供一个全局的旋转空气阻力。设置为 0.0 表示不人为添加这种额外的阻力，完全依赖关节自带的 damping 参数。
asset_options.collapse_fixed_joints = True  # 含义：合并固定关节（大大减少了仿真中的物体数量（Links），从而显著提高计算效率）
asset_options.default_dof_drive_mode = 3  # 0: DOF_MODE_NONE (自由运动)；1: DOF_MODE_POS (位置控制)；2: DOF_MODE_VEL (速度控制)；3: DOF_MODE_EFFORT (力矩控制)
asset_options.density = 0.001
asset_options.max_angular_velocity = 100.0  # 这是一个安全阈值。如果由于物理计算错误导致关节飞速旋转，系统会将其限制在 100 rad/s，防止仿真因数值爆炸（Explosion）而直接卡死或崩溃。
asset_options.replace_cylinder_with_capsule = True
asset_options.thickness = 0.01
asset_options.flip_visual_attachments = False  # 有些 URDF 导出的网格模型（Mesh）坐标系是反的，设置为 True 可以纠正显示错误，但不影响物理表现。
render = True
asset_options.fix_base_link = (
    False  # todo "on rack  if True" #fix_base_link = False：浮动基座-》被挂起来
)
asset_options.use_mesh_materials = False  # color!!! False have color
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

action_repeat = 4
dt = (
    action_repeat * sim_params.dt
)  # 控制周期 = 0.005 × 4 = 0.02 s ； 控制频率 = 50 Hz （控制指令（动作/力矩）的更新频率）

action, motor_position, motor_torque, motor_velocity = [], [], [], []

T = 2  # total simulation time
mode = "sin"  # real,sin

pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 1.05)  # todo base init position
pose.r = gymapi.Quat(0, 0, 0, 1)  # actor init orientation

ts = np.linspace(0, T, int(T / (sim_params.dt * action_repeat)))

# set up the env grid
num_envs = 1
num_per_row = 1
spacing = 3.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
cam_pos = gymapi.Vec3(2.0, 2, 2.0)
cam_target = gymapi.Vec3(0, 0, 1.8)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# get array of DOF properties
init_dof_props = gym.get_asset_dof_properties(asset)

init_dof_props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)

lower_limits = init_dof_props["lower"]
upper_limits = init_dof_props["upper"]
torque_lower_limits = -init_dof_props["effort"]
torque_upper_limits = init_dof_props["effort"]

# create an array of DOF states that will be used to update the actors
num_dofs = gym.get_asset_dof_count(asset)
names_dofs = gym.get_asset_dof_names(asset)

kp = torch.zeros(num_dofs, dtype=torch.float, requires_grad=False)
kd = torch.zeros(num_dofs, dtype=torch.float, requires_grad=False)
kp_real = torch.zeros(num_dofs, dtype=torch.float, requires_grad=False)
kd_real = torch.zeros(num_dofs, dtype=torch.float, requires_grad=False)

plt_id = [0]  # list(range(0, num_dofs))  # [0,1,2,3,4]

# τ = Kp · (q_des − q) − Kd · q̇ #Kp-〉stiffness；Kd-〉damping
stiffness = {  # 关节刚度；含义：关节角度每偏差 1 rad，控制器会多施加多少力矩（越大，关节越“硬”，越不愿偏离目标）
    "hip_yaw": 300,
    "hip_roll": 300,
    "hip_pitch": 250,
    "knee": 300,
    "ankle": 100,  # "ankle": 100, damping: 2.0 这里ankle小的原因是因为脚踝是接触调节器，太硬 → 接触冲击大，容易数值震荡，容易炸仿真 / 伤实机
    "torso": 300,
    "shoulder": 200,
    "elbow": 200,
}  # [N*m/rad]
damping = {  # 关节阻尼；含义：关节角速度每增加 1 rad/s，会被施加多少“刹车力矩”（阻尼越大，越不容易振荡，但反应越慢）
    "hip_yaw": 6.0,
    "hip_roll": 15.0,
    "hip_pitch": 10.0,
    "knee": 8,
    "ankle": 2.0,
    "torso": 2,
    "shoulder": 2,
    "elbow": 2,
}  # [N*m*s/rad]
for i in range(num_dofs):
    for gain_name in stiffness:
        if gain_name in names_dofs[i]:
            kp[i] = stiffness[gain_name]
            kd[i] = damping[gain_name]
            kp_real[i] = stiffness[gain_name]
            kd_real[i] = damping[gain_name]

debug_dir = join("experiments/tune_pid", "real")
os.makedirs(debug_dir, exist_ok=True)
if mode == "real":
    xl = (
        pd.read_csv(join(debug_dir, "general.txt"), sep="\t+", header=None)
        .values[1:, :]
        .astype(float)
    )
    joint_act_real = xl[:, :19]
    joint_pos_real = xl[:, 19:38]
    joint_vel_real = xl[:, 38:57]
    joint_tau_real = (
        kp_real * (joint_act_real - joint_pos_real) - kd_real * joint_vel_real
    )
    ts_real = np.linspace(0, T, len(joint_act_real))

torque_limits = torch.zeros(num_dofs, dtype=torch.float, requires_grad=False)
for i in range(num_dofs):
    torque_limits[i] = init_dof_props["effort"][i].item()

env = gym.create_env(sim, env_lower, env_upper, num_per_row)
actor_handle = gym.create_actor(env, asset, pose, "actor", 0, 0)
# reload parameters
gym.set_actor_dof_properties(env, actor_handle, init_dof_props)
dof_props = gym.get_actor_dof_properties(env, actor_handle)
gym.enable_actor_dof_force_sensors(env, actor_handle)


def compute_torques(
    actions, joint_pos, joint_vel
):  # actions 目标关节角；joint_pos 当前关节角；joint_vel 当前关节角速度
    error = actions - joint_pos  # 计算还差多少角度（error = “目标 − 现实”）
    torques = (
        kp * error - kd * joint_vel
    )  # kp * error ：其中的kp就是上面的stiffness（刚度），相乘的含义是：离目标越远，拉得越狠（比作弹簧）；- kd * joint_vel：其中kd就是上面的damping（阻尼），这这部分的含义是：转得越快，刹得越狠
    return torch.clip(
        torques, torch.tensor(torque_lower_limits), torch.tensor(torque_upper_limits)
    )  # 安全保护。剪裁。防止：数值爆炸，仿真炸飞，实机烧电机


if mode == "real":
    init_dof_states = np.array(joint_pos_real[0, :], dtype=gymapi.DofState.dtype)
    init_dof_vel = np.array(joint_vel_real[0, :], dtype=gymapi.DofState.dtype)
else:
    init_dof_states = np.array([0] * num_dofs, dtype=gymapi.DofState.dtype)
    init_dof_vel = np.array([0] * num_dofs, dtype=gymapi.DofState.dtype)

gym.set_actor_dof_states(env, actor_handle, init_dof_states, gymapi.STATE_POS)
# set init dof velocity
gym.set_actor_dof_states(env, actor_handle, init_dof_vel, gymapi.STATE_VEL)
dof_state_tensor = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(dof_state_tensor)

joint_pos = dof_states.view(num_dofs, 2)[..., 0]
joint_vel = dof_states.view(num_dofs, 2)[..., 1]

start = time.time()
count = 0
# act = torch.tensor([0.] * num_dofs, dtype=torch.float)
act = torch.tensor(
    [0.0, 0.0, -0.2, 0.4, -0.2] * 2 + [0] + ([0.0] * 4) * 2, dtype=torch.float
)

for t in ts:  # 控制循环（50 Hz）
    if mode == "sin":
        for i in range(num_dofs):
            act[i] = math.sin(0.5 * (2.0 * pi) * t) * 0.2
    elif mode == "real":
        for i in range(num_dofs):
            act[i] = joint_act_real[count, i]
    count += 1
    for i in range(
        action_repeat
    ):  # action_repeat这个值越大，模拟现实可能会越准确，但是吃性能
        torques = compute_torques(
            torch.tensor(act, dtype=torch.float), joint_pos.clone(), joint_vel.clone()
        )
        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torques))
        gym.simulate(sim)  # 物理积分（200 Hz）
        gym.fetch_results(sim, True)
        gym.refresh_dof_force_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
    action.append(act.tolist().copy())
    mp = joint_pos.tolist().copy()
    motor_position.append(mp)
    mv = joint_vel.tolist().copy()
    motor_velocity.append(mv)
    mt = gym.get_actor_dof_forces(env, actor_handle).tolist().copy()
    motor_torque.append(torques.tolist().copy())

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

end = time.time()
print(end - start)
print("Done")

# convert them from 1 dimension to 2 dimension
joint_action = np.stack(action.copy())
joint_position = np.stack(motor_position.copy())
joint_velocity = np.stack(motor_velocity.copy())
joint_torque = np.stack(motor_torque.copy())

# draw a picture
num = (
    min(len(ts / sim_params.dt), len(ts_real / sim_params.dt))
    if mode == "real"
    else len(ts / dt)
)
colors = ["r", "g", "b"]
for j in plt_id:  # range(num_dofs):
    for i, motor_id in enumerate([j]):
        plt.plot(ts[:num], joint_action[:num, motor_id], linestyle="-.", c="k")
        plt.plot(ts[:num], joint_position[:num, motor_id], linestyle="-", c="b")
        if mode == "real":
            plt.plot(ts[:num], joint_act_real[:num, motor_id], linestyle=":", c="k")
            plt.plot(ts[:num], joint_pos_real[:num, motor_id], c="r")
            # plt.plot(ts, joint_position[:, motor_id + 5], linestyle='-', c='r')
            # plt.plot(ts_real, joint_pos_real[:, motor_id + 5], c='r')
        plt.title(names_dofs[j] + ":" + "joint position")
        plt.grid()
        plt.show()

        # plt.plot(ts[:num], joint_action[:num, motor_id] - joint_position[:num, motor_id], c='b')
        # if mode == 'real':
        #     plt.plot(ts[:num], joint_act_real[:num, motor_id] - joint_pos_real[:num, motor_id], linestyle=':', c='r')
        # plt.title(names_dofs[j] + ':' + 'joint position error')
        # plt.grid()
        # plt.show()

        plt.plot(ts[:num], joint_velocity[:num, motor_id], linestyle="-", c="b")
        if mode == "real":
            plt.plot(ts[:num], joint_vel_real[:num, motor_id], c="r")
        plt.title(names_dofs[j] + ":" + "joint velocity")
        plt.grid()
        plt.show()

        plt.plot(ts[:num], joint_torque[:num, motor_id], c="b")
        if mode == "real":
            plt.plot(ts[:num], joint_tau_real[:num, motor_id], c="r")
        plt.title(names_dofs[j] + ":" + "joint torque")
        plt.grid()
        plt.show()
path = join(debug_dir, f"sim_pid_{mode}.xlsx")
with pd.ExcelWriter(path) as f:
    pd.DataFrame(np.hstack([joint_action]), columns=names_dofs).to_excel(
        f, "joint_act", index=False
    )
    pd.DataFrame(np.hstack([joint_position]), columns=names_dofs).to_excel(
        f, "joint_pos", index=False
    )
    pd.DataFrame(np.hstack([joint_velocity]), columns=names_dofs).to_excel(
        f, "joint_vel", index=False
    )
    pd.DataFrame(np.hstack([joint_torque]), columns=names_dofs).to_excel(
        f, "joint_tau", index=False
    )
print(f"Debug data has been saved to {path}.")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
