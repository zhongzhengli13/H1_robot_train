# This file is used to check out the order & orientation of motors, just run it and see what happen
# Data will be saved in foldr 'excel/tune_URDF'
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

warnings.filterwarnings('ignore')

# Z-up axis in this file, (x,y,z)
# positive direction: hip:left; thigh:backward; calf:backward


LEG_LENGTH = np.array([0.083, 0.25, 0.25])
FOOT_POSITION_REFRERNCE = np.asarray([0.083, 0, -0.35355])  # [0.084, -0.018, -0.3723]


def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)


# compute dof_position from foot_position



# initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = 0.001  # dt*action_repeat=0.01 #越小精度越高，进而要求性能越好
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)  # -9.81
sim_params.substeps = 1
sim_params.use_gpu_pipeline = False
print("WARNING: Forcing CPU pipeline.")

# physx parameters
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 0
sim_params.physx.num_threads = 10
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0
sim_params.physx.bounce_threshold_velocity = 0.5
sim_params.physx.max_depenetration_velocity = 1.0
sim_params.physx.max_gpu_contact_pairs = 2 ** 23
sim_params.physx.default_buffer_size_multiplier = 5
sim_params.physx.use_gpu = True

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
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
asset_options.angular_damping = 0.
asset_options.collapse_fixed_joints = True
asset_options.default_dof_drive_mode = 1
asset_options.density = 0.001
asset_options.max_angular_velocity = 1000.
asset_options.replace_cylinder_with_capsule = True
asset_options.thickness = 0.01
asset_options.flip_visual_attachments = False
asset_options.fix_base_link = True  # "on rack"
asset_options.use_mesh_materials = False  # color!!! False have color
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

action_repeat = 10
frequency = 2  # 1/frequency waves per second
action, motor_position, motor_torque, motor_velocity = [], [], [], []

T = 1  # total simulation time
ts = np.linspace(0, T, int(T / (sim_params.dt * action_repeat)))

# set up the env grid
num_envs = 1
num_per_row = 1
spacing = 2.5
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# position the camera
cam_pos = gymapi.Vec3(2, 0, 3.)
cam_target = gymapi.Vec3(0, 0, 2.)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# get array of DOF properties
init_dof_props = gym.get_asset_dof_properties(asset)

init_dof_props['driveMode'].fill(gymapi.DOF_MODE_POS)

lower_limits = init_dof_props['lower']
upper_limits = init_dof_props['upper']

# create an array of DOF states that will be used to update the actors
num_dofs = gym.get_asset_dof_count(asset)
dof_names = gym.get_asset_dof_names(asset)


# create env
env = gym.create_env(sim, env_lower, env_upper, num_per_row)

# add actor
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0., 2.)  # actor init position
pose.r = gymapi.Quat(0, 0, 0, 0.707107)  # actor init orientation

actor_handle = gym.create_actor(env, asset, pose, "actor", 0, 0)

# reload parameters
gym.set_actor_dof_properties(env, actor_handle, init_dof_props)
dof_props = gym.get_actor_dof_properties(env, actor_handle)

gym.enable_actor_dof_force_sensors(env, actor_handle)

# get the limit-related slices of the DOF properties array
stiffnesses = dof_props['stiffness']  # kp parameter
dampings = dof_props['damping']  # kd parameter
drivemode = dof_props['driveMode']

# set init dof states
ref_act = [0] * num_dofs  # [0., 0., -0.2, 0.4, -0.2] * 2 + [0] + [0.] * 4 * 2

init_dof_states = np.array(ref_act, dtype=gymapi.DofState.dtype)
gym.set_actor_dof_states(env, actor_handle, init_dof_states, gymapi.STATE_POS)
# set init dof velocity
init_dof_vel = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
gym.set_actor_dof_states(env, actor_handle, init_dof_vel, gymapi.STATE_VEL)

start = time.time()
act = [0] * num_dofs  # [0., 0., -0.2, 0.4, -0.2] * 2 + [0] + [0.] * 4 * 2
i = 0
count = 0

while not gym.query_viewer_has_closed(viewer):
    # for i in range(num_dofs):
    #     act[i] += 0.001  # positive direction
    # act[i] -= 0.01   # reverse direction
    targets = act.copy()
    gym.set_actor_dof_position_targets(env, actor_handle, targets)
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.refresh_dof_force_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

    dof_states = gym.get_actor_dof_states(env, actor_handle, gymapi.STATE_ALL)
    dof_pos = torch.as_tensor(dof_states['pos'])
    dof_vel = torch.as_tensor(dof_states['vel'])
    action.append(act)
    mp = dof_pos.tolist().copy()
    motor_position.append(mp)
    mv = dof_vel.tolist().copy()
    motor_velocity.append(mv)
    mt = gym.get_actor_dof_forces(env, actor_handle)  # .copy().reshape(4, 3)[[2, 0, 3, 1]].flatten().tolist().copy()
    motor_torque.append(mt)
    count += 1

    # if act[i] >= upper_limits[i]:  # positive direction
    #     # if act[i] <= lower_limits[i]:    # reverse direction
    #     if i == num_dofs:
    #         break
    #     act = np.zeros(num_dofs).astype('f')
    #     i += 1
    #     if i == num_dofs:
    #         break

end = time.time()
print(end - start)
print("Done")

# convert them from 1 dimension to 2 dimension
action = np.stack(action.copy())
motor_position = np.stack(motor_position.copy())
motor_velocity = np.stack(motor_velocity.copy())
motor_torque = np.stack(motor_torque.copy())

# save data to excel
debug_dir = join('experiments', 'tune_urdf')
os.makedirs(debug_dir, exist_ok=True)
path = join(debug_dir, f'tune_urdf.xlsx')
with pd.ExcelWriter(path) as f:
    pd.DataFrame(np.hstack([action]), columns=dof_names).to_excel(f, 'act', index=False)
    pd.DataFrame(np.hstack([motor_position]), columns=dof_names).to_excel(f, 'pos', index=False)
    pd.DataFrame(np.hstack([motor_velocity]), columns=dof_names).to_excel(f, 'vel', index=False)
    pd.DataFrame(np.hstack([motor_torque]), columns=dof_names).to_excel(f, 'tau', index=False)
print(f"Debug data has been saved to {path}.")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
