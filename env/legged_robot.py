import collections
import math
import queue

# from isaacgym.torch_utils import quat_rotate_inverse, torch_rand_float, get_axis_params, to_torch
# import numpy as np
import os
import sys

from math import cos, sin, pi
from typing import Dict
from config.loc import H1Config
from env.utils.math import quat_apply_yaw
import random
from isaacgym.torch_utils import *
from scipy.spatial.transform import Rotation as R
from isaacgym import gymtorch, gymapi, gymutil
from env.utils.terrain import Terrain
from collections import deque
from env.utils.delay_torch_deque import DelayDeque
import torch


class LeggedRobotEnv:
    def __init__(
        self,
        cfg: H1Config,
        sim_params,
        physics_engine,
        sim_device,
        render,
        fix_cam,
        residual_cfg=None,
        tcn_name=None,
        debug=False,
        epochs=1,
    ):
        """Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initializes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            render (bool): Run without rendering if True
        """
        self.gym = gymapi.acquire_gym()
        self.viewer = None
        self.cfg = cfg
        self.residual_cfg = residual_cfg if residual_cfg is not None else None
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        self.render = render
        self.debug = debug
        self.fix_cam = fix_cam
        self.tcn_name = tcn_name
        self.epochs = epochs
        self.num_legs = self.cfg.init_state.num_legs
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.device = (
            self.sim_device
            if sim_device_type == "cuda" and sim_params.use_gpu_pipeline
            else "cpu"
        )
        self.graphics_device_id = (
            -1 if not self.render else self.sim_device_id
        )  # graphics device for rendering, -1 for no rendering

        self.height_samples = None
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.num_envs = self.cfg.env.num_envs
        self.num_observations = self.cfg.policy.num_observations
        self.num_actions = self.cfg.policy.num_actions
        self.torques = None
        self.react_tau = None
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.time_out_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.reset_time_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.dt = cfg.pd_gains.decimation * sim_params.dt
        self.max_episode_length_s = cfg.env.episode_length_s
        # self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        self.max_episode_length = int(self.max_episode_length_s / self.dt)
        self.push_interval = np.ceil(cfg.domain_rand.push_interval_s / self.dt)

        self.foot_pos_hd = torch.zeros(
            self.num_envs, self.num_legs * 3, device=self.device, dtype=torch.float32
        )
        self.hand_pos_hd = torch.zeros(
            self.num_envs, self.num_legs * 3, device=self.device, dtype=torch.float32
        )

        self.base_pos_hd = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=torch.float32
        )

        self.start_push = False
        self.push_count = 1
        self.push_body_id = torch.randint(low=0, high=10, size=(1, 5))
        if cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            cfg.terrain.curriculum = False

        self.render_count = 0
        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # create sim, env, buffer and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)
        self._init_buffers()

        if self.render:
            # if running with a viewer, set up keyboard shortcuts and camera
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            # self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
            fixed_robot_id = self.cfg.viewer.fixed_robot_id
            fixed_robot_pos = self.base_pos.cpu().numpy().copy()[fixed_robot_id]
            fixed_robot_pos0 = self.base_pos.cpu().numpy().copy()[fixed_robot_id]
            fixed_robot_pos[0] = (
                fixed_robot_pos0[0] - 0
            )  # * sin(self.base_euler[0, [2]])
            fixed_robot_pos[1] = (
                fixed_robot_pos0[1] - 2.3
            )  # * cos(self.base_euler[0, [2]])
            fixed_robot_pos[2] = 1.0
            fixed_robot_pos0[2] = 1.0
            # self.set_camera(fixed_robot_pos + np.array(self.cfg.viewer.fixed_offset), fixed_robot_pos)
            self.set_camera(fixed_robot_pos, fixed_robot_pos0)
            # subscribe to keyboard shortcuts
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync"
            )
            # self.rendering()

    #
    def reset(self, env_ids, reset_phy_ids):
        """Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), and Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._reset_body_props(env_ids, reset_phy_ids)
        self.episode_length_buf[env_ids] = 0

        self.base_lin_vel_his.reset(env_ids, self.base_lin_vel[env_ids, :])
        self.base_ang_vel_his.reset(env_ids, self.base_ang_vel[env_ids, :])
        self.base_eul_his.reset(env_ids, self.base_euler[env_ids, :])
        self.joint_pos_his.reset(env_ids, self.joint_pos[env_ids, :].clone())
        self.joint_vel_his.reset(env_ids, self.joint_vel[env_ids, :].clone())
        # self.joint_vel_his.append(0.001 * ((self.joint_pos_his.delay(1) - self.joint_pos_his.delay(2))).clone())

        self.foot_frc_his.reset(env_ids, self.foot_frc[env_ids, :].clone())
        self.foot_pos_hd_his.reset(env_ids, self.foot_pos_hd[env_ids, :].clone())
        self.foot_vel_hd_his.reset(env_ids, self.foot_vel[env_ids, :].clone())

    def reset_counters(self, env_ids):
        self.episode_length_buf[env_ids] = 0
        self.render_count = 0
        self.common_step_counter = 0

    def step_torques(self, joint_actions):
        self.torques = self._compute_torques(joint_actions).to(self.device)
        if self.cfg.domain_rand.randomize_torque:
            self.tau_gains = torch_rand_float(
                self.cfg.domain_rand.torque_range[0],
                self.cfg.domain_rand.torque_range[1],
                (self.num_envs, self.num_dofs),
                device=self.device,
            )
            self.torques *= self.tau_gains
            self.torques = torch.clip(
                self.torques, -self.torque_limits, self.torque_limits
            )
        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(self.torques)
        )
        self.gym.simulate(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.joint_pos_his.append(self.joint_pos.clone())
        self.joint_vel_his.append(self.joint_vel.clone())
        # self.joint_vel_his.append(0.001 * ((self.joint_pos_his.delay(1) - self.joint_pos_his.delay(2))).clone())

        ### refresh body state
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.base_pos = self.root_states[:, :3]  # positions
        self.base_quat = self.root_states[:, 3:7]  # quaternions
        self.base_lvel[:] = self.root_states[:, 7:10]  # linear velocities
        self.base_avel[:] = self.root_states[:, 10:13]  # angular velocities

        self.base_euler = self._get_euler_from_quat(self.base_quat)

        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.base_lvel)
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.base_avel)

        self.projected_gravity[:] = quat_rotate_inverse(
            self.base_quat, self.gravity_vec
        )
        self.foot_frc = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)

        self.foot_euler = torch.cat(
            [
                torch.squeeze(
                    self._get_euler_from_quat(self.rigid_body_param[:, foot, 3:7])
                )
                for foot in self.feet_indices
            ],
            dim=-1,
        ).view(self.num_envs, -1)

        self.foot_pos = torch.cat(
            [
                torch.squeeze(self.rigid_body_param[:, foot, :3])
                for foot in self.feet_indices
            ],
            dim=-1,
        ).view(self.num_envs, -1)
        self.foot_pos[:, [2, 5]] -= 0.065
        # self.foot_pos_front = self.foot_pos[:, [2, 5]] - 0.19 * torch.sin(self.foot_euler[:, [1, 4]])
        # self.foot_pos_rear = self.foot_pos[:, [2, 5]] + 0.1 * torch.sin(self.foot_euler[:, [1, 4]])
        # self.foot_pos[:, [2, 5]] = torch.min(self.foot_pos_front, self.foot_pos_rear)

        base_yaw = self.base_euler[:, 2:3].clone()
        quat_tmp = quat_from_euler_xyz(
            torch.zeros_like(base_yaw), torch.zeros_like(base_yaw), base_yaw
        ).squeeze(1)

        self.base_pos_hd = quat_rotate_inverse(quat_tmp, self.base_pos)  # heading frame
        Rm = R.from_quat(self.base_quat.cpu().numpy())
        self.matrix = torch.as_tensor(
            torch.from_numpy(Rm.as_matrix()), device=self.device
        )

        self.foot_pos_hd = torch.cat(
            [
                torch.squeeze(
                    quat_rotate_inverse(
                        quat_tmp, self.foot_pos[:, 3 * foot : 3 * foot + 3]
                    )
                )
                for foot in range(self.num_legs)
            ],
            dim=-1,
        ).view(self.num_envs, -1)

        self.foot_vel = torch.cat(
            [
                torch.squeeze(
                    quat_rotate_inverse(quat_tmp, self.rigid_body_param[:, foot, 7:10])
                )
                for foot in self.feet_indices
            ],
            dim=-1,
        ).view(
            self.num_envs, -1
        )  # heading frame

        self.shoulder_roll_pos = torch.cat(
            [
                torch.squeeze(self.rigid_body_param[:, shoulder, :3])
                for shoulder in self.shoulder_roll_indices
            ],
            dim=-1,
        ).view(self.num_envs, -1)

        self.hand_pos = torch.cat(
            [
                torch.squeeze(self.rigid_body_param[:, hand, :3])
                for hand in self.hand_indices
            ],
            dim=-1,
        ).view(self.num_envs, -1)

        self.hand_pos_hd = torch.cat(
            [
                torch.squeeze(
                    quat_rotate_inverse(
                        quat_tmp, self.hand_pos[:, 3 * foot : 3 * foot + 3]
                    )
                )
                for foot in range(self.num_legs)
            ],
            dim=-1,
        ).view(self.num_envs, -1)

        self.base_lin_vel_his.append(self.base_lin_vel.clone())
        self.base_ang_vel_his.append(self.base_ang_vel.clone())
        self.base_eul_his.append(self.base_euler.clone())
        self.foot_frc_his.append(self.foot_frc.clone())
        self.foot_pos_hd_his.append(self.foot_pos_hd.clone())
        self.foot_vel_hd_his.append(self.foot_vel.clone())
        ### refresh body state

        if self.device == "cpu":
            self.gym.fetch_results(self.sim, True)

    def step_states(self, target_command):

        if self.cfg.terrain.mesh_type == "plane":
            self.cfg.terrain.measure_heights = False
            self.foot_scanned_height = self.foot_pos.clone()
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        self.render_count += 1
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_time_buf = self.episode_length_buf < 2
        self.time_out_buf = (
            self.episode_length_buf >= self.max_episode_length
        )  # no terminal reward for time-outs

        if self.cfg.domain_rand.push_robots and (
            self.common_step_counter % self.push_interval == 0
        ):
            self.push_body_id = self.push_body_id = torch.randint(
                low=0, high=self.num_bodies, size=(1, 5)
            )
            self.force_x = self.cfg.domain_rand.max_push_force * (
                2.0 * torch.rand_like(self.rb_positions[:, self.push_body_id, [0]])
                - 1.0
            )
            self.force_y = self.cfg.domain_rand.max_push_force * (
                2.0 * torch.rand_like(self.rb_positions[:, self.push_body_id, [0]])
                - 1.0
            )
            self.force_z = self.cfg.domain_rand.max_push_force * (
                2.0 * torch.rand_like(self.rb_positions[:, self.push_body_id, [0]])
                - 1.0
            )
            # self.force_x = self.cfg.domain_rand.max_push_force * torch.ones([1,1],device='cuda:0')
            # self.force_y = self.cfg.domain_rand.max_push_force * torch.zeros([1,1],device='cuda:0')
            # self.force_z = self.cfg.domain_rand.max_push_force * torch.zeros([1,1],device='cuda:0')
            self.force_positions = self.rb_positions.clone()
            self.start_push = True
            self.push_count = 0
        # self.cfg.domain_rand.push_robots = True
        # self.cfg.domain_rand.push_duration_step = 150
        # self.force_x = -150
        if self.start_push and (
            self.push_count <= self.cfg.domain_rand.push_duration_step
        ):
            self._push_force_robots(
                self.force_x,
                self.force_y,
                self.force_z,
                self.force_positions,
                self.push_body_id,
                (torch.norm(target_command[:, :3], dim=1, keepdim=True) >= 0.11),
            )
            self.push_count = self.push_count + 1
            if self.push_count == self.cfg.domain_rand.push_duration_step:
                self.start_push = False
        # if self.common_step_counter % self.push_interval == 0:
        #     self.tau_gains = torch_rand_float(self.cfg.domain_rand.torque_range[0],
        #                                       self.cfg.domain_rand.torque_range[1], (self.num_envs, self.num_dofs),
        #                                       device=self.device)

        if self.cfg.domain_rand.push_robots and (
            self.common_step_counter % self.push_interval == 0
        ):
            self.push_body_id = torch.randint(low=0, high=self.num_bodies, size=(1, 5))
            # ratio = 1. + min(iter / 3000., 1.) if iter is not None else 1.
            self.push_force = self.cfg.domain_rand.max_push_force * (
                2.0 * torch.rand_like(self.rb_positions) - 1.0
            )
            self.start_push = True
            self.push_count = 0
        if target_command is not None and self.cfg.domain_rand.push_robots:
            zero_command_env_ids = (
                (torch.norm(target_command[:, :3], dim=1, keepdim=True) < 0.11)
                .nonzero(as_tuple=False)[:, [0]]
                .flatten()
            )
            if len(zero_command_env_ids) > 0:
                if self.cfg.domain_rand.push_robots and (
                    (2.0 * self.common_step_counter) % self.push_interval == 0
                ):
                    self._push_root_state_robots(
                        env_ids=zero_command_env_ids, iter=iter
                    )

        # if self.start_push and (self.push_count <= self.cfg.domain_rand.push_duration_step):
        #     self._push_force_robots(self.push_force, self.rb_positions.clone(), self.push_body_id)
        #     self.push_count += 1
        #     if self.push_count == self.cfg.domain_rand.push_duration_step:
        #         self.start_push = False
        if self.cfg.domain_rand.push_robots and (
            self.common_step_counter % self.push_interval == 0
        ):
            self._push_root_state_robots(iter=iter)

        if self.render:
            if self.render_count % 2 == 0:
                if self.fix_cam:
                    fixed_robot_id = self.cfg.viewer.fixed_robot_id
                    fixed_robot_pos = self.base_pos.cpu().numpy().copy()[fixed_robot_id]
                    fixed_robot_pos0 = (
                        self.base_pos.cpu().numpy().copy()[fixed_robot_id]
                    )
                    fixed_robot_pos[0] = (
                        fixed_robot_pos0[0] - 1.5
                    )  # * sin(self.base_euler[0, [2]])
                    fixed_robot_pos[1] = (
                        fixed_robot_pos0[1] - 1.5
                    )  # * cos(self.base_euler[0, [2]])
                    fixed_robot_pos[2] = 2.0
                    # self.set_camera(fixed_robot_pos + np.array(self.cfg.viewer.fixed_offset), fixed_robot_pos)
                    self.set_camera(fixed_robot_pos, fixed_robot_pos0)
                self.rendering()
                self.render_count = 0

    def rendering(self, sync_frame_time=True):
        if self.viewer:
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.motor_action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.motor_action == "toggle_viewer_sync" and evt.value > 0:
                    self.render = not self.render
            if self.device != "cpu":
                self.gym.fetch_results(self.sim, True)
            if self.render:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)

    def set_camera(self, position, lookat):
        """Set camera position and direction."""
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ------------- Callback --------------
    def _process_rigid_shape_props(self, props, env_id):
        """Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                self.friction_coeffs = torch_rand_float(
                    friction_range[0],
                    friction_range[1],
                    (self.num_envs, 1),
                    device="cpu",
                )
                # self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
                props[s].restitution = self.friction_coeffs[env_id] - 0.4
                # props[s].rolling_friction = self.friction_coeffs[env_id]
        return props

    def _process_rigid_body_props(self, props):
        if self.cfg.domain_rand.randomize_mass:
            mrng = self.cfg.domain_rand.added_mass_range
            irng = self.cfg.domain_rand.added_inertia_range
            for i in range(len(props)):
                # dm = np.random.uniform(mrng[0], mrng[1])
                if random.random() > 0.5:
                    index_id = 15
                else:
                    index_id = 19
                if i == 11 and self.cfg.domain_rand.added_body_mass:  # 11
                    irng_body = self.cfg.domain_rand.added_body_inertia_range
                    props[i].inertia.x.x *= np.random.uniform(
                        irng_body[0], irng_body[1]
                    )
                    props[i].inertia.x.y *= np.random.uniform(
                        irng_body[0], irng_body[1]
                    )
                    props[i].inertia.x.z *= np.random.uniform(
                        irng_body[0], irng_body[1]
                    )
                    props[i].inertia.y.x *= np.random.uniform(
                        irng_body[0], irng_body[1]
                    )
                    props[i].inertia.y.y *= np.random.uniform(
                        irng_body[0], irng_body[1]
                    )
                    props[i].inertia.y.z *= np.random.uniform(
                        irng_body[0], irng_body[1]
                    )
                    props[i].inertia.z.x *= np.random.uniform(
                        irng_body[0], irng_body[1]
                    )
                    props[i].inertia.z.y *= np.random.uniform(
                        irng_body[0], irng_body[1]
                    )
                    props[i].inertia.z.z *= np.random.uniform(
                        irng_body[0], irng_body[1]
                    )
                    twh_inertia = np.array(
                        [
                            [
                                props[i].inertia.x.x,
                                props[i].inertia.x.y,
                                props[i].inertia.x.z,
                            ],
                            [
                                props[i].inertia.y.x,
                                props[i].inertia.y.y,
                                props[i].inertia.y.z,
                            ],
                            [
                                props[i].inertia.z.x,
                                props[i].inertia.z.y,
                                props[i].inertia.z.z,
                            ],
                        ]
                    )
                    inv_twh_inertia = np.linalg.inv(twh_inertia)
                    props[i].invInertia.x.x = float(inv_twh_inertia[0][0])
                    props[i].invInertia.x.y = float(inv_twh_inertia[0][1])
                    props[i].invInertia.x.z = float(inv_twh_inertia[0][2])
                    props[i].invInertia.y.x = float(inv_twh_inertia[1][0])
                    props[i].invInertia.y.y = float(inv_twh_inertia[1][1])
                    props[i].invInertia.y.z = float(inv_twh_inertia[1][2])
                    props[i].invInertia.z.x = float(inv_twh_inertia[2][0])
                    props[i].invInertia.z.y = float(inv_twh_inertia[2][1])
                    props[i].invInertia.z.z = float(inv_twh_inertia[2][2])
                    props[i].mass += np.random.uniform(
                        self.cfg.domain_rand.added_body_mass_range[0],
                        self.cfg.domain_rand.added_body_mass_range[1],
                    )
                    props[i].invMass = 1 / props[i].mass
                else:
                    props[i].inertia.x.x *= np.random.uniform(irng[0], irng[1])
                    props[i].inertia.x.y *= np.random.uniform(irng[0], irng[1])
                    props[i].inertia.x.z *= np.random.uniform(irng[0], irng[1])
                    props[i].inertia.y.x *= np.random.uniform(irng[0], irng[1])
                    props[i].inertia.y.y *= np.random.uniform(irng[0], irng[1])
                    props[i].inertia.y.z *= np.random.uniform(irng[0], irng[1])
                    props[i].inertia.z.x *= np.random.uniform(irng[0], irng[1])
                    props[i].inertia.z.y *= np.random.uniform(irng[0], irng[1])
                    props[i].inertia.z.z *= np.random.uniform(irng[0], irng[1])
                    twh_inertia = np.array(
                        [
                            [
                                props[i].inertia.x.x,
                                props[i].inertia.x.y,
                                props[i].inertia.x.z,
                            ],
                            [
                                props[i].inertia.y.x,
                                props[i].inertia.y.y,
                                props[i].inertia.y.z,
                            ],
                            [
                                props[i].inertia.z.x,
                                props[i].inertia.z.y,
                                props[i].inertia.z.z,
                            ],
                        ]
                    )
                    inv_twh_inertia = np.linalg.inv(twh_inertia)
                    props[i].invInertia.x.x = float(inv_twh_inertia[0][0])
                    props[i].invInertia.x.y = float(inv_twh_inertia[0][1])
                    props[i].invInertia.x.z = float(inv_twh_inertia[0][2])
                    props[i].invInertia.y.x = float(inv_twh_inertia[1][0])
                    props[i].invInertia.y.y = float(inv_twh_inertia[1][1])
                    props[i].invInertia.y.z = float(inv_twh_inertia[1][2])
                    props[i].invInertia.z.x = float(inv_twh_inertia[2][0])
                    props[i].invInertia.z.y = float(inv_twh_inertia[2][1])
                    props[i].invInertia.z.z = float(inv_twh_inertia[2][2])
                    props[i].mass *= np.random.uniform(mrng[0], mrng[1])
                    props[i].invMass = 1 / props[i].mass
        if self.cfg.domain_rand.randomize_mass_com:
            rng_high = self.cfg.domain_rand.added_mass_com_high
            rng_low = self.cfg.domain_rand.added_mass_com_low
            # k = np.random.uniform(rng[0], rng[1])
            k_body = np.random.uniform(
                self.cfg.domain_rand.added_body_mass_com_low,
                self.cfg.domain_rand.added_body_mass_com_high,
            )
            for i in range(len(props)):
                k = np.random.uniform(rng_low, rng_high)
                if i == 11 and self.cfg.domain_rand.randomize_body_com:
                    props[i].com.x += k_body[0]
                    props[i].com.y += k_body[1]
                    props[i].com.z += k_body[2]
                else:
                    props[i].com.x += k[0]
                    props[i].com.y += k[1]
                    props[i].com.z += k[2]
                pass
        return props

    def _process_dof_props(self, props, env_id):
        """Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(
                self.num_dofs,
                2,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            self.dof_vel_limits = torch.zeros(
                self.num_dofs,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            self.torque_limits = torch.zeros(
                self.num_dofs,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
        return props

    def _compute_torques(self, joint_action):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        error = joint_action - self.joint_pos
        # self.integral += error * self.cfg.sim.dt
        # if self.cfg.domain_rand.randomize_gains: #增益随机化，目的：让策略别记死“某一组 PD 参数”，增强 sim2real
        #     self.p_gains_rand = torch_rand_float(self.cfg.domain_rand.gains_range[0],
        #                                          self.cfg.domain_rand.gains_range[1], (self.num_envs, self.num_dofs),
        #                                          device=self.device)
        #     self.d_gains_rand = torch_rand_float(self.cfg.domain_rand.gains_range[0],
        #                                          self.cfg.domain_rand.gains_range[1], (self.num_envs, self.num_dofs),
        #                                          device=self.device)
        torques = (
            self.p_gains * self.p_gains_rand * error
            - self.d_gains * self.d_gains_rand * self.joint_vel
        )
        torques += self.tau_residual
        return torch.clip(torques, -self.torque_limits, self.torque_limits).view(
            self.torques.shape
        )

    # ------------- Reset --------------
    def _reset_dofs(self, env_ids):
        """Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.joint_pos[env_ids] = self.reset_joint_pos[env_ids, :] + torch_rand_float(
            -0.05, 0.05, (len(env_ids), self.num_dofs), device=self.device
        )
        self.joint_vel[env_ids] = (
            4 * torch.rand_like(self.reset_joint_vel[env_ids, :])
        ) - 2
        self.last_dof_vel[:] = self.joint_vel[:].clone()

        env_id_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_states),
            gymtorch.unwrap_tensor(env_id_int32),
            len(env_id_int32),
        )

    def _reset_body_props(self, env_ids, reset_phy_ids):
        pass
        if self.cfg.runner.epoch > 1000:
            if random.random() > 0.995:
                for i in env_ids:
                    rigid_body_props = self.gym.get_actor_rigid_body_properties(
                        self.envs[i], self.actor_handles[i]
                    )
                    rigid_body_props = self._process_rigid_body_props(rigid_body_props)
                    self.gym.set_actor_rigid_body_properties(
                        self.envs[i],
                        self.actor_handles[i],
                        rigid_body_props,
                        recomputeInertia=False,
                    )

    def _reset_root_states(self, env_ids):
        """Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        init_rad = 0.2
        rot_r = (
            random.uniform(-init_rad, init_rad)
            if self.cfg.init_state.random_rot
            else 0.0
        )
        rot_p = (
            random.uniform(-init_rad, init_rad)
            if self.cfg.init_state.random_rot
            else 0.0
        )
        rot_y = random.uniform(0, 2 * pi) if self.cfg.init_state.random_rot else 0.0

        # self.cfg.init_state.pos[2] = max(abs(1.06 * cos(rot_r)), abs(1.06 * cos(rot_p))) + 0.05

        rot_quat = gymapi.Quat.from_euler_zyx(rot_r, rot_p, rot_y)
        rot = [rot_quat.x, rot_quat.y, rot_quat.z, rot_quat.w]
        lin_vel_x = random.uniform(-0.4, 0.4) if self.cfg.init_state.random_rot else 0.0
        lin_vel_y = random.uniform(-0.4, 0.4) if self.cfg.init_state.random_rot else 0.0
        lin_vel_z = random.uniform(-0.1, 0.1) if self.cfg.init_state.random_rot else 0.0
        lin_vel = [lin_vel_x, lin_vel_y, lin_vel_z]
        ang_vel_x = random.uniform(-0.9, 0.9) if self.cfg.init_state.random_rot else 0.0
        ang_vel_y = random.uniform(-0.9, 0.9) if self.cfg.init_state.random_rot else 0.0
        ang_vel_z = random.uniform(-0.9, 0.9) if self.cfg.init_state.random_rot else 0.0
        ang_vel = [ang_vel_x, ang_vel_y, ang_vel_z]
        # base_init_state_list = self.cfg.init_state.pos + rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        base_init_state_list = self.cfg.init_state.pos + rot + lin_vel + ang_vel
        reset_state = to_torch(
            base_init_state_list, device=self.device, requires_grad=False
        )
        if self.custom_origins:
            self.root_states[env_ids] = reset_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(
                -1.0, 1.0, (len(env_ids), 2), device=self.device
            )  # xy position within 1m of the center
        else:
            self.root_states[env_ids] = reset_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        # self.root_states[env_ids, 7:13] = torch_rand_float(-0.1, 0.1, (len(env_ids), 6),
        #                                                    device=self.device)  # [7:10]: lin vel, [10:13]: ang vel
        self.last_root_vel[:] = self.root_states[:, 7:13]

        env_id_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_id_int32),
            len(env_id_int32),
        )

    def _reset_dof_inertia(self):
        for i in range(self.num_envs):
            rigid_body_props = self.gym.get_actor_rigid_body_properties(
                self.envs[i], self.actor_handles[i]
            )
            rigid_body_props = self._process_rigid_body_props(rigid_body_props)
            self.gym.set_actor_rigid_body_properties(
                self.envs[i],
                self.actor_handles[i],
                rigid_body_props,
                recomputeInertia=True,
            )

    def _push_root_state_robots(self, env_ids=None, iter=None):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        ratio = 1.0  # + min(iter / 3000., 0.5) if iter is not None else 1.
        max_vel = 1
        max_rate = 1
        if env_ids is not None:
            self.root_states[env_ids, 7:9] = torch_rand_float(
                -max_vel * ratio, max_vel * ratio, (len(env_ids), 2), device=self.device
            )  # lin vel x/y
            self.root_states[env_ids, 9:10] = torch_rand_float(
                -max_vel * ratio, max_vel * ratio, (len(env_ids), 1), device=self.device
            )  # lin vel x/y
            self.root_states[env_ids, 10:13] = torch_rand_float(
                -max_rate * ratio,
                max_rate * ratio,
                (len(env_ids), 3),
                device=self.device,
            )  # lin vel x/y

            # self.root_states[env_ids, 3:7] = self.rot[env_ids, :]
        else:
            self.root_states[:, 7:9] = torch_rand_float(
                -max_vel, max_vel, (self.num_envs, 2), device=self.device
            )  # lin vel x/y
            self.root_states[:, [9]] = torch_rand_float(
                -max_vel, max_vel, (self.num_envs, 1), device=self.device
            )  # lin vel x/y
            self.root_states[:, 10:13] = torch_rand_float(
                -max_rate, max_rate, (self.num_envs, 3), device=self.device
            )  # lin vel x/y
            # self.root_states[:, 3:7] = self.rot[:, :]
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states)
        )

    def _push_force_robots(
        self, force_x, force_y, force_z, force_positions, push_body_id, high_command
    ):
        forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float
        )
        forces[:, push_body_id, [0]] = force_x * torch.logical_not(
            high_command
        ).unsqueeze(2)
        forces[:, push_body_id, [1]] = force_y * torch.logical_not(
            high_command
        ).unsqueeze(2)
        forces[:, push_body_id, [2]] = force_z * torch.logical_not(
            high_command
        ).unsqueeze(2)
        self.gym.apply_rigid_body_force_at_pos_tensors(
            sim=self.sim,
            forceTensor=gymtorch.unwrap_tensor(forces),
            posTensor=gymtorch.unwrap_tensor(force_positions),
            space=gymapi.ENV_SPACE,
        )
        # self.gym.apply_rigid_body_force_at_pos_tensors(sim=self.sim, forceTensor=gymtorch.unwrap_tensor(forces), posTensor=gymtorch.unwrap_tensor(force_positions),
        #                                                space=gymapi.CoordinateSpace.ENV_SPACE)

    def create_sim(self):
        """Creates simulation, terrain and environments
        Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        self.sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ["heightfield", "trimesh"]:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == "plane":
            self._create_ground_plane()
        elif mesh_type == "heightfield":
            self._create_heightfield()
        elif mesh_type == "trimesh":
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]"
            )

        asset_path = self.cfg.asset.file
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = (
            self.cfg.asset.replace_cylinder_with_capsule
        )
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )

        self.num_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)

        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        feet_names = [
            s for name in self.cfg.asset.foot_name for s in self.body_names if name in s
        ]
        termination_contact_names = [
            s
            for name in self.cfg.asset.terminate_after_contacts_on
            for s in self.body_names
            if name in s
        ]

        base_init_state_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(
            base_init_state_list, device=self.device, requires_grad=False
        )
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower, env_upper = gymapi.Vec3(0.0, 0.0, 0.0), gymapi.Vec3(0.0, 0.0, 0.0)
        self.envs, self.actor_handles = [], []
        self.dof_propss, self.rigid_body_propss = [], []

        for i in range(self.num_envs):
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=self.device).squeeze(
                1
            )
            pos += to_torch(
                self.cfg.init_state.pos, device=self.device, requires_grad=False
            )
            start_pose.p = gymapi.Vec3(*pos)
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs))
            )
            actor_handle = self.gym.create_actor(
                env_handle,
                robot_asset,
                start_pose,
                "robot",
                i,
                self.cfg.asset.self_collisions,
            )  # todo!!! i
            rigid_shape_props = self._process_rigid_shape_props(
                rigid_shape_props_asset, i
            )
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            dof_props = self._process_dof_props(dof_props_asset, i)
            if self.cfg.domain_rand.randomize_damping:
                dof_props["friction"] *= np.random.uniform(
                    self.cfg.domain_rand.added_friction_range[0],
                    self.cfg.domain_rand.added_friction_range[1],
                    (self.num_dofs,),
                )
                dof_props["damping"] *= np.random.uniform(
                    self.cfg.domain_rand.added_damping_range[0],
                    self.cfg.domain_rand.added_damping_range[1],
                    (self.num_dofs,),
                )
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            rigid_body_props = self.gym.get_actor_rigid_body_properties(
                env_handle, actor_handle
            )
            rigid_body_props = self._process_rigid_body_props(rigid_body_props)
            self.gym.set_actor_rigid_body_properties(
                env_handle, actor_handle, rigid_body_props, recomputeInertia=False
            )

            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            self.dof_propss.append(dof_props)
            self.rigid_body_propss.append(rigid_body_props)

        self.feet_indices = torch.zeros(
            len(feet_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], feet_names[i]
            )

        self.shoulder_roll_indices = torch.zeros(
            len(feet_names), dtype=torch.long, device=self.device, requires_grad=False
        )

        self.hand_indices = torch.zeros(
            len(feet_names), dtype=torch.long, device=self.device, requires_grad=False
        )

        shoulder_roll_names = [
            s
            for name in self.cfg.asset.shoulder_roll_name
            for s in self.body_names
            if name in s
        ]
        for i in range(len(feet_names)):
            self.shoulder_roll_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], shoulder_roll_names[i]
            )

        hand_names = [s for name in ["elbow"] for s in self.body_names if name in s]
        for i in range(len(feet_names)):
            self.hand_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], hand_names[i]
            )

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )

    def _get_env_origins(self):
        """Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
        Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(
                self.num_envs, 3, device=self.device, requires_grad=False
            )
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum:
                max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(
                0, max_init_level + 1, (self.num_envs,), device=self.device
            )
            self.terrain_types = torch.div(
                torch.arange(self.num_envs, device=self.device),
                (self.num_envs / self.cfg.terrain.num_cols),
                rounding_mode="floor",
            ).to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = (
                torch.from_numpy(self.terrain.env_origins)
                .to(self.device)
                .to(torch.float)
            )
            self.env_origins[:] = self.terrain_origins[
                self.terrain_levels, self.terrain_types
            ]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(
                self.num_envs, 3, device=self.device, requires_grad=False
            )
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
            self.env_origins[:, 2] = 0.0

    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # positions orientations linvels angvels
        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        self.react_tau = gymtorch.wrap_tensor(dof_force_tensor).view(
            self.num_envs, self.num_dofs
        )

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3
        )
        self.rigid_body_param = gymtorch.wrap_tensor(rigid_body_state).view(
            self.num_envs, -1, 13
        )
        self.rb_positions = gymtorch.wrap_tensor(rigid_body_state)[:, 0:3].view(
            self.num_envs, self.num_bodies, 3
        )

        self.joint_pos = self.dof_states.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.joint_vel = self.dof_states.view(self.num_envs, self.num_dofs, 2)[..., 1]

        self.common_step_counter = 0
        self.gravity_vec = to_torch(
            get_axis_params(-1.0, self.up_axis_idx), device=self.device
        ).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.torques = torch.zeros(
            self.num_envs,
            self.num_dofs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.p_gains = torch.zeros(
            self.num_envs,
            self.num_dofs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.d_gains = torch.zeros(
            self.num_envs,
            self.num_dofs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        # self.i_gains = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)

        self.tau_residual = torch.zeros(
            self.num_envs,
            self.num_dofs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.tau_gains = torch.ones(
            self.num_envs,
            self.num_dofs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.p_gains_rand = torch.ones(
            self.num_envs,
            self.num_dofs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.d_gains_rand = torch.ones(
            self.num_envs,
            self.num_dofs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        if self.cfg.domain_rand.randomize_gains:
            self.p_gains_rand = torch_rand_float(
                self.cfg.domain_rand.gains_range[0],
                self.cfg.domain_rand.gains_range[1],
                (self.num_envs, self.num_dofs),
                device=self.device,
            )
            self.d_gains_rand = torch_rand_float(
                self.cfg.domain_rand.gains_range[0],
                self.cfg.domain_rand.gains_range[1],
                (self.num_envs, self.num_dofs),
                device=self.device,
            )

        self.base_pos = self.root_states[:, :3]  # positions
        self.base_quat = self.root_states[:, 3:7]  # quaternions
        self.base_lvel = self.root_states[:, 7:10]  # linear velocities
        self.base_avel = self.root_states[:, 10:13]  # angular velocities

        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.base_lvel)
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.base_avel)
        self.base_euler = self._get_euler_from_quat(self.base_quat)

        self.foot_pos = torch.cat(
            [
                torch.squeeze(self.rigid_body_param[:, foot, :3])
                for foot in self.feet_indices
            ],
            dim=-1,
        ).view(self.num_envs, -1)
        self.foot_vel = torch.cat(
            [
                torch.squeeze(self.rigid_body_param[:, foot, 7:10])
                for foot in self.feet_indices
            ],
            dim=-1,
        ).view(self.num_envs, -1)

        self.shoulder_roll_pos = torch.cat(
            [
                torch.squeeze(self.rigid_body_param[:, shoulder, :3])
                for shoulder in self.shoulder_roll_indices
            ],
            dim=-1,
        ).view(self.num_envs, -1)

        self.hand_pos = torch.cat(
            [
                torch.squeeze(self.rigid_body_param[:, hand, :3])
                for hand in self.hand_indices
            ],
            dim=-1,
        ).view(self.num_envs, -1)

        self.foot_euler = torch.cat(
            [
                torch.squeeze(
                    self._get_euler_from_quat(self.rigid_body_param[:, foot, 3:7])
                )
                for foot in self.feet_indices
            ],
            dim=-1,
        ).view(self.num_envs, -1)

        self.foot_frc = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)

        self.base_lin_vel_his = DelayDeque(maxlen=500)
        self.base_eul_his = DelayDeque(maxlen=500)

        self.base_ang_vel_his = DelayDeque(maxlen=500)
        self.foot_frc_his = DelayDeque(maxlen=500)

        self.joint_pos_his = DelayDeque(maxlen=20)
        self.joint_vel_his = DelayDeque(maxlen=20)

        self.foot_pos_hd_his = DelayDeque(maxlen=20)
        self.foot_vel_hd_his = DelayDeque(maxlen=20)

        for _ in range(self.base_eul_his.maxlen):
            self.base_lin_vel_his.append(self.base_lin_vel.clone())
            self.base_eul_his.append(self.base_euler.clone())
        for _ in range(self.base_ang_vel_his.maxlen):
            self.base_ang_vel_his.append(self.base_ang_vel.clone())
            self.joint_pos_his.append(self.joint_pos.clone())
            self.joint_vel_his.append(self.joint_vel.clone())

            self.foot_frc_his.append(self.foot_frc.clone())
            self.foot_pos_hd_his.append(self.foot_pos_hd.clone())
            self.foot_vel_hd_his.append(self.foot_vel.clone())

        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.last_dof_vel = torch.zeros_like(self.joint_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_contacts = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            device=self.device,
            requires_grad=False,
        )
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0
        self.foot_height_points = self._init_foot_height_points()
        self.default_dof_pos = torch.zeros(
            self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.integral = torch.zeros(
            self.num_envs,
            self.num_dofs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.cfg.init_state.reset_joint_pos = self.cfg.init_state.reset_joint_pos
        for i in range(self.num_dofs):
            dof_name = self.dof_names[i]
            self.default_dof_pos[i] = self.cfg.init_state.reset_joint_pos[i]
            found = False

            for gain_name in self.cfg.pd_gains.stiffness:
                if gain_name in dof_name:
                    found = True
                    if self.residual_cfg is not None:
                        self.p_gains[:, i] = self.residual_cfg.pd_gains.stiffness[
                            gain_name
                        ]
                        self.d_gains[:, i] = self.residual_cfg.pd_gains.damping[
                            gain_name
                        ]
                        # self.i_gains[:, i] = self.residual_cfg.pd_gains.integration[gain_name]
                    else:
                        self.p_gains[:, i] = self.cfg.pd_gains.stiffness[gain_name]
                        self.d_gains[:, i] = self.cfg.pd_gains.damping[gain_name]
                        # self.i_gains[:, i] = self.cfg.pd_gains.integration[gain_name]

            # assert found, f'PD gain of {gain_name} joint was not defined'

        self.reset_joint_pos = self.default_dof_pos.repeat(self.num_envs, 1).clone()
        self.reset_joint_vel = torch.zeros(
            self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False
        ).repeat(self.num_envs, 1)
        self.reset_base_quat = self.base_quat

    def _create_ground_plane(self):
        """Adds a ground plane to the simulation, sets friction and restitution based on the cfg."""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """Adds a heightfield terrain to the simulation, sets parameters based on the cfg."""
        hf_params = gymapi.HeightFieldProperties()
        hf_params.column_scale = self.terrain.horizontal_scale
        hf_params.row_scale = self.terrain.horizontal_scale
        hf_params.vertical_scale = self.terrain.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.border_size
        hf_params.transform.p.y = -self.terrain.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = (
            torch.as_tensor(torch.from_numpy(self.terrain.heightsamples))
            .view(self.terrain.tot_rows, self.terrain.tot_cols)
            .to(self.device)
        )

    def _create_trimesh(self):
        """Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        #"""
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(
            self.sim,
            self.terrain.vertices.flatten(order="C"),
            self.terrain.triangles.flatten(order="C"),
            tm_params,
        )
        self.height_samples = torch.as_tensor(
            self.terrain.heightsamples, device=self.device
        )
        # self.height_samples = torch.as_tensor(torch.from_numpy(self.terrain.heightsamples)).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _init_height_points(self):
        """Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.as_tensor(self.cfg.terrain.measured_points_y, device=self.device)
        x = torch.as_tensor(self.cfg.terrain.measured_points_x, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(
            self.num_envs,
            self.num_height_points,
            3,
            device=self.device,
            requires_grad=False,
        )
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def get_foot_height_to_ground(self):
        x, y, z = (
            self.foot_pos[:, [0, 3]],
            self.foot_pos[:, [1, 4]],
            self.foot_pos[:, [2, 5]],
        )
        px = (
            (x.squeeze() + self.terrain.cfg.border_size)
            / self.terrain.cfg.horizontal_scale
        ).long()
        py = (
            (y.squeeze() + self.terrain.cfg.border_size)
            / self.terrain.cfg.horizontal_scale
        ).long()
        px = torch.clip(px, 0, self.height_samples.shape[0] - 1)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 1)
        foot_sample = self.height_samples[px, py] * self.terrain.cfg.vertical_scale
        foot_heights = z - foot_sample
        return foot_heights

    def _get_heights(self, env_ids=None):
        """Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(
                self.num_envs,
                self.num_height_points,
                device=self.device,
                requires_grad=False,
            )
        elif self.cfg.terrain.mesh_type == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(
                self.base_quat[env_ids].repeat(1, self.num_height_points),
                self.height_points[env_ids],
            ) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(
                self.base_quat.repeat(1, self.num_height_points), self.height_points
            ) + (self.root_states[:, :3]).unsqueeze(1)
        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _init_foot_height_points(self):
        x = torch.as_tensor(self.cfg.terrain.measured_points_x, device=self.device)  #
        y = torch.as_tensor(self.cfg.terrain.measured_points_y, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y)
        self.num_foot_height_points = 1  # grid_x.numel()
        foot_points = torch.zeros(
            self.num_envs,
            self.num_foot_height_points,
            3,
            device=self.device,
            requires_grad=False,
        )
        foot_points[:, :, 0] = grid_x.flatten()[0]
        foot_points[:, :, 1] = grid_y.flatten()[0]
        return foot_points

    # def _get_base_euler(self):
    #     return to_torch(torch.cat([self._from_2pi_to_pi(v) for v in get_euler_xyz(self.base_quat)]), device=self.device).view(self.num_envs, -1)

    def _get_euler_from_quat(self, quat):
        base_rpy = get_euler_xyz(quat)
        r = to_torch(self._from_2pi_to_pi(base_rpy[0]), device=self.device)
        p = to_torch(self._from_2pi_to_pi(base_rpy[1]), device=self.device)
        y = to_torch(self._from_2pi_to_pi(base_rpy[2]), device=self.device)
        return torch.t(torch.vstack((r, p, y)))

    def _from_2pi_to_pi(self, rpy):
        return rpy.cpu() - 2 * pi * np.floor((rpy.cpu() + pi) / (2 * pi))
