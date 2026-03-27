from math import pi, sin, cos, exp, tau
import numpy as np
from env.legged_robot import LeggedRobotEnv
from env.utils.helpers import class_to_dict
from env.utils.math import wrap_to_pi, smallest_signed_angle_between
from env.utils.phase_modulator import PhaseModulator
from env.tasks.base_task import BaseTask, register
from isaacgym.torch_utils import *
from scipy.spatial.transform import Rotation as R
import random
from env.utils.math import scale_transform, smallest_signed_angle_between_torch
from collections import deque
import statistics
import torch

"""
这是一个用于“腿式机器人行走”的强化学习任务类
它负责：
生成速度指令（走多快、怎么转）
构造观测（给神经网络看的状态）
把网络输出转成“关节目标位置”
计算奖励（走得好不好）
判断什么时候失败（摔倒、越界）
"""


@register
class LocomotionTask(BaseTask):
    def __init__(self, env: LeggedRobotEnv):
        super(LocomotionTask, self).__init__(env)
        self.env = env
        self.cmd_id = 0
        self.rew_names = None
        self.num_envs = env.num_envs
        self.num_legs = 2  # 可以判断出是双足机器人
        self.commands = torch.zeros(
            self.num_envs,
            self.cfg.command.num_commands,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # x vel, y vel, yaw vel, heading ---》 command中的参数
        self.smooth_commands = torch.zeros(
            self.num_envs,
            self.cfg.command.num_commands,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # x vel, y vel, yaw vel, heading

        self.command_cfgs = class_to_dict(self.cfg.command)
        self.resampling_interval = int(
            self.cfg.command.resampling_time / self.env.dt
        )  # 重采样时间
        self._resample_commands(torch.arange(env.num_envs, device=self.device))

        if self.cfg.domain_rand.delay_observation:
            self.delay_joint_steps = random.randint(
                self.cfg.domain_rand.delay_joint_ranges[0],
                self.cfg.domain_rand.delay_joint_ranges[1],
            )
            self.delay_rate_steps = random.randint(
                self.cfg.domain_rand.delay_rate_ranges[0],
                self.cfg.domain_rand.delay_rate_ranges[1],
            )
            self.delay_angle_steps = random.randint(
                self.cfg.domain_rand.delay_angle_ranges[0],
                self.cfg.domain_rand.delay_angle_ranges[1],
            )
        else:
            self.delay_joint_steps = 1
            self.delay_rate_steps = 1
            self.delay_angle_steps = 1
        self.convert_phi = 1.0 * pi
        self.phase_modulator = PhaseModulator(
            time_step=env.dt,
            num_envs=self.num_envs,
            num_legs=self.num_legs,
            device=self.device,
            # 维护每条腿的 相位 φ；判断：哪条腿在支撑（support），哪条腿在摆动（swing）。φ∈[0,π)  → 支撑相；φ∈[π,2π) → 摆动相
        )
        self.phase_modulator.reset(
            convert_phi=self.convert_phi,
            env_ids=torch.arange(self.num_envs),
            render=self.env.render
            or self.env.debug
            or self.env.epochs > 1
            or self.env.tcn_name is not None,
        )
        self.foot_phase = self.phase_modulator.phase
        if self.cfg.action.use_increment:
            self.action_low = to_torch(
                self.cfg.action.inc_low_ranges, device=self.device
            )
            self.action_high = to_torch(
                self.cfg.action.inc_high_ranges, device=self.device
            )
        else:
            self.action_low = to_torch(
                self.cfg.action.low_ranges, device=self.device)
            self.action_high = to_torch(
                self.cfg.action.high_ranges, device=self.device)
        self.current_joint_act = to_torch(
            self.env.default_dof_pos, device=self.device
        ).repeat(self.num_envs, 1)
        self.ref_joint_action = to_torch(
            self.cfg.action.ref_joint_pos, device=self.device
        ).repeat(self.num_envs, 1)
        self.motor_position_reference = torch.as_tensor(
            [0.0] * self.env.num_dofs
        ).repeat(self.num_envs, 1)
        self.joint_action_limit_low_over = torch.as_tensor(
            self.env.dof_pos_limits[:, 0]
        ).repeat(self.num_envs, 1)
        self.joint_action_limit_high_over = torch.as_tensor(
            self.env.dof_pos_limits[:, 1]
        ).repeat(self.num_envs, 1)
        self.joint_pos_stastic_error = (
            2 * torch.rand((self.cfg.env.num_envs, 19)) - 1
        ).to(self.device)

        # self.joint_action_limit_low = torch.as_tensor(self.cfg.action.low_ranges[self.num_legs:], device=self.device).repeat(self.num_envs, 1)
        # self.joint_action_limit_high = torch.as_tensor(self.cfg.action.high_ranges[self.num_legs:], device=self.device).repeat(self.num_envs, 1)
        self.joint_action_limit_low = torch.as_tensor(
            self.env.dof_pos_limits[:, 0], device=self.device
        ).repeat(self.num_envs, 1)
        self.joint_action_limit_high = torch.as_tensor(
            self.env.dof_pos_limits[:, 1], device=self.device
        ).repeat(self.num_envs, 1)

        self.action_history = deque(maxlen=3)
        self.net_out_history = deque(maxlen=3)
        self.debug_net_out_history = deque(maxlen=3)
        for _ in range(self.action_history.maxlen):
            self.action_history.append(self.current_joint_act)
        for _ in range(self.net_out_history.maxlen):
            self.net_out_history.append(
                torch.zeros_like(self.action_low[:12]).repeat(self.num_envs, 1)
            )
            self.debug_net_out_history.append(
                torch.zeros_like(self.action_low[:12]).repeat(self.num_envs, 1)
            )
        self.obs_history = deque(maxlen=1)
        self.ground_impact_force = None
        Rm = R.from_quat(self.env.base_quat.cpu().numpy())
        self.matrix = torch.as_tensor(
            torch.from_numpy(Rm.as_matrix()), device=self.device
        )
        foot_support_mask_1 = torch.where(self.foot_phase >= 0, True, False)
        foot_support_mask_2 = torch.where(
            self.foot_phase < self.convert_phi, True, False
        )
        self.foot_support_mask = torch.logical_and(
            foot_support_mask_1, foot_support_mask_2
        )
        self.foot_swing_mask = torch.logical_not(self.foot_support_mask)
        self.pm_f = self.phase_modulator.frequency.clone()

        self.last_joint_taus = torch.zeros(
            self.num_envs,
            self.env.num_dofs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_joint_vels = torch.zeros(
            self.num_envs,
            self.env.num_dofs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.last_foot_frc = torch.zeros(
            self.num_envs,
            self.num_legs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.foot_frc_acc = torch.zeros(
            self.num_envs,
            self.num_legs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.last_foot_vel = torch.zeros(
            self.num_envs,
            self.num_legs * 3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # self.joint_vel = torch.clip(self.env.joint_vel, -self.env.dof_vel_limits, self.env.dof_vel_limits)
        # self.joint_pos = torch.clip(self.env.joint_pos, self.env.dof_pos_limits[:, 0], self.env.dof_pos_limits[:, 1])
        self.joint_vel = self.env.joint_vel_his.delay(self.delay_joint_steps)
        self.joint_pos = self.env.joint_pos_his.delay(self.delay_joint_steps)

        self.joint_pos_error = self.current_joint_act - self.joint_pos
        self.joint_tau = (
            self.env.p_gains * self.joint_pos_error - self.env.d_gains * self.joint_vel
        )
        self.foot_pos_hd = self.env.foot_pos_hd
        self.foot_height = (
            self.env.get_foot_height_to_ground()
            if self.cfg.terrain.mesh_type in ["trimesh", "heightfield"]
            else self.env.foot_pos_hd[:, [2, 5]]
        )

        self.foot_vel = self.env.foot_vel_hd_his.delay(self.delay_joint_steps)

        self.foot_frc = self.env.foot_frc_his.delay(self.delay_rate_steps)
        self.base_ang_vel = self.env.base_ang_vel_his.delay(
            self.delay_rate_steps)

        self.base_euler = self.env.base_eul_his.delay(self.delay_angle_steps)
        self.base_lin_vel = self.env.base_lin_vel_his.delay(
            self.delay_angle_steps)

        self.joint_pos_history, self.foot_fre_history, self.pmf_history = (
            [0.0] * 200,
            [0.0] * 200,
            [0.0] * 100,
        )
        self.joint_pos_err_history = [0.0] * 10
        self.static_flag = torch.where(
            torch.norm(self.commands[:, :3], dim=1,
                       keepdim=True) < 0.11, False, True
        ).float()
        for _ in range(len(self.joint_pos_err_history)):
            self.joint_pos_err_history.pop(0)
            self.joint_pos_err_history.append(self.joint_pos_error.clone())
        for _ in range(len(self.joint_pos_history)):
            self.joint_pos_history.pop(0)
            self.joint_pos_history.append(self.joint_pos.clone())
        for _ in range(len(self.pmf_history)):
            self.pmf_history.pop(0)
            self.pmf_history.append(self.pm_f.clone())
        for _ in range(len(self.foot_fre_history)):
            self.foot_fre_history.pop(0)
            self.foot_fre_history.append(self.foot_frc.clone())
        for _ in range(self.obs_history.maxlen):
            self.obs_history.append(self.pure_observation())
        self.noise_values = torch.zeros(
            self.num_envs,
            len(self.pure_observation()[0]),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # x vel, y vel, yaw vel, heading
        self.command_boundary = 0.2

    def reset(self, env_ids):
        self.joint_vel = torch.clip(
            self.env.joint_vel, -self.env.dof_vel_limits, self.env.dof_vel_limits
        )
        self.joint_pos = torch.clip(
            self.env.joint_pos,
            self.env.dof_pos_limits[:, 0],
            self.env.dof_pos_limits[:, 1],
        )
        self.current_joint_act[env_ids] = self.env.default_dof_pos
        self.joint_pos_error = self.current_joint_act - self.joint_pos
        self.phase_modulator.reset(
            convert_phi=self.convert_phi,
            env_ids=env_ids,
            render=self.env.render
            or self.env.epochs > 1
            or self.env.tcn_name is not None,
        )
        self.static_flag = torch.where(
            torch.norm(self.commands[:, :3], dim=1,
                       keepdim=True) < 0.11, False, True
        ).float()
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        self._resample_commands(env_ids)
        self.extra_info["task"] = {}
        if self.cfg.terrain.curriculum:
            self.extra_info["task"]["terrain_level"] = torch.mean(
                self.env.terrain_levels.float()
            )
        if self.cfg.env.send_timeouts:
            self.extra_info["timeouts"] = self.env.time_out_buf
        for _ in range(self.action_history.maxlen):
            self.action_history.append(self.current_joint_act)
        for _ in range(self.net_out_history.maxlen):
            self.net_out_history.append(
                torch.zeros_like(self.action_low[:12]).repeat(self.num_envs, 1)
            )
            self.debug_net_out_history.append(
                torch.zeros_like(self.action_low[:12]).repeat(self.num_envs, 1)
            )
        foot_support_mask_1 = torch.where(self.foot_phase >= 0, True, False)
        foot_support_mask_2 = torch.where(
            self.foot_phase < self.convert_phi, True, False
        )
        self.foot_support_mask = torch.logical_and(
            foot_support_mask_1, foot_support_mask_2
        )
        self.foot_swing_mask = torch.logical_not(self.foot_support_mask)
        self.pm_f = self.phase_modulator.frequency.clone()
        # self.sym_joint_pos = self.joint_pos[:, -5:] if np.random.uniform() < 0.5 else self.joint_pos[:, :5]
        # self.joint_pos_nstep_his = self.joint_pos[:, -5:] if np.random.uniform() < 0.5 else self.joint_pos[:, :5]
        # self.sym_foot_frc = self.foot_frc[:, [0]] if np.random.uniform() < 0.5 else self.foot_frc[:, [1]]

    def _update_terrain_curriculum(
        self, env_ids
    ):  # env_ids:在这一帧需要被重置（reset）的一批环境编号
        # 用于决定 下一个 episode，机器人站在哪种地形上
        """Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        distance = torch.norm(
            self.env.root_states[env_ids, :2] - self.env.env_origins[env_ids, :2], dim=1
        )  # distance = 机器人在这个 episode 中“实际走了多远” （单位：m）
        move_up = (
            distance > self.env.terrain.env_length / 2
        )  # 2 #如果机器人至少走过“半块地形” （没摔，还往前走了，说明能驾驭当前难度）
        """
        torch.norm(self.commands[env_ids, :2], dim=1) 这是：|| [vx, vy] || = 期望线速度大小（m/s）
        * self.env.max_episode_length_s -> 期望距离 = 期望速度 × episode 时长
        * 0.3 这是一个 宽容系数：你只要完成了 30% 的期望距离，就不算太差
        ~move_up 就是按位取反（NOT） 作用：同一帧里，已经决定升级的环境，不再允许被降级。
        """
        move_down = (
            distance
            < torch.norm(self.commands[env_ids, :2], dim=1)
            * self.env.max_episode_length_s
            * 0.3
        ) * ~move_up  # 0.5
        self.env.terrain_levels[env_ids] += (
            1 * move_up - 1 * move_down
        )  # 根据上面的计算自动判断升级 or 降级
        self.env.terrain_levels[env_ids] = torch.where(
            self.env.terrain_levels[env_ids] >= self.env.max_terrain_level,
            # 格式：randint_like(input, low=0, high, \*, dtype=None, layout=torch.strided, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor
            torch.randint_like(
                self.env.terrain_levels[env_ids], self.env.max_terrain_level
                # 下届默认为0；self.env.terrain_levels[env_ids]对应的是input（只拿它的“外壳”——shape、dtype、device——作为生成新张量的模板，完全不看里面的数值。；high对应的是 self.env.max_terrain_level
            ),
            torch.clip(
                self.env.terrain_levels[env_ids], 0
            ),  # 地形等级最小是 0（最平地）
        )  # (the minimum level is zero)
        self.env.env_origins[env_ids] = self.env.terrain_origins[
            self.env.terrain_levels[env_ids], self.env.terrain_types[env_ids]
        ]  # terrain_origins[level, type] → 某种地形、某个难度，对应的世界坐标起点

    def designed_command(self):
        # 修改 直接给一个恒定的速度
        self.commands[:, [0]] = 0.4
        self.commands[:, [1]] = 0.0
        self.commands[:, [2]] = 0.0  # 不转弯
        self.commands[:, [3]] = 0.0
        # if self.env.common_step_counter < 200:
        #     self.commands[:, [0]] = -0.0
        #     self.commands[:, [2]] = 0
        # else:
        #     self.commands[:, [0]] = 0.2
        #     self.commands[:, [2]] = (
        #         -0.0
        #     )  # torch.clip(5. * smallest_signed_angle_between(self.env.base_euler[:, [2]],
        #     # -pi / 2. * sin(5. * (self.env.common_step_counter - 60) * self.env.dt)), min=-4., max=4.)

        self.static_flag[:] = torch.where(
            torch.norm(self.commands[:, :3], dim=1,
                       keepdim=True) < 0.11, False, True
        ).float()
        self.commands[:, :3] *= self.static_flag[:]

        self.commands[:, 0:1] *= torch.where(
            torch.norm(self.commands[:, 0:1], dim=1,
                       keepdim=True) < 0.11, False, True
        ).float()
        self.commands[:, 2:3] *= torch.where(
            torch.norm(self.commands[:, 2:3], dim=1,
                       keepdim=True) < 0.11, False, True
        ).float()

        # self.command_boundary = 0.00000001
        self.low_command = torch.logical_and(
            (torch.abs(self.commands[:, [0]]) < self.command_boundary),
            (torch.abs(self.commands[:, [2]]) < self.command_boundary),
        )
        self.high_command = torch.logical_not(self.low_command)
        # elif self.env.common_step_counter < 310:
        #     self.commands[:, [0]] = 1.6
        #     self.commands[:, [2]] = -3.8  # torch.clip(4. * smallest_signed_angle_between(self.env.base_euler[:, [2]],
        #     # -pi / 2. * sin(5. * (self.env.common_step_counter - 90) * self.env.dt)), min=-4., max=4.)
        # elif self.env.common_step_counter < 600:
        #     self.commands[:, [0]] = 4.
        #     self.commands[:, [2]] = torch.clip(4.5 * smallest_signed_angle_between(self.env.base_euler[:, [2]], 0.),
        #                                        min=-4., max=4.)

    def step(self):

        self.joint_vel = self.env.joint_vel_his.delay(self.delay_joint_steps)
        self.joint_pos = self.env.joint_pos_his.delay(self.delay_joint_steps)
        self.joint_pos_error = self.current_joint_act - self.joint_pos
        self.joint_tau = (
            self.env.p_gains * self.joint_pos_error - self.env.d_gains * self.joint_vel
        )
        self.foot_pos_hd = self.env.foot_pos_hd
        self.foot_height = (
            self.env.get_foot_height_to_ground()
            if self.cfg.terrain.mesh_type in ["trimesh", "heightfield"]
            else self.env.foot_pos_hd[:, [2, 5]]
        )

        self.foot_vel = self.env.foot_vel_hd_his.delay(self.delay_joint_steps)

        self.foot_frc = self.env.foot_frc_his.delay(self.delay_rate_steps)
        self.base_ang_vel = self.env.base_ang_vel_his.delay(
            self.delay_rate_steps)

        self.base_euler = self.env.base_eul_his.delay(self.delay_angle_steps)
        self.base_lin_vel = self.env.base_lin_vel_his.delay(
            self.delay_angle_steps)

        self.shoulder_height = (
            self.env.shoulder_roll_pos[:, [2]] +
            self.env.shoulder_roll_pos[:, [5]]
        ) * 0.5

        Rm = R.from_quat(self.env.base_quat.cpu().numpy())
        self.matrix = torch.as_tensor(
            torch.from_numpy(Rm.as_matrix()), device=self.device
        )
        self.foot_phase = self.phase_modulator.phase
        foot_support_mask_1 = torch.where(self.foot_phase >= 0.0, True, False)
        foot_support_mask_2 = torch.where(
            self.foot_phase < self.convert_phi, True, False
        )
        self.foot_support_mask = torch.logical_and(
            foot_support_mask_1, foot_support_mask_2
        )
        self.foot_swing_mask = torch.logical_not(self.foot_support_mask)
        self.pm_f = self.phase_modulator.frequency.clone().detach()
        if self.env.render or self.env.epochs > 1:
            self.designed_command()
        else:
            env_ids = (
                ((self.env.episode_length_buf) % self.resampling_interval == 0)
                .nonzero(as_tuple=False)
                .flatten()
            )
            if len(env_ids) > 0:
                self._resample_commands(env_ids)
        if (
            self.cfg.domain_rand.delay_observation
            and self.env.common_step_counter % 20 == 0
        ):  # 20 #观测延迟系统
            self.delay_joint_steps = random.randint(
                self.cfg.domain_rand.delay_joint_ranges[0],
                self.cfg.domain_rand.delay_joint_ranges[1],
            )
            self.delay_rate_steps = random.randint(
                self.cfg.domain_rand.delay_rate_ranges[0],
                self.cfg.domain_rand.delay_rate_ranges[1],
            )
            self.delay_angle_steps = random.randint(
                self.cfg.domain_rand.delay_angle_ranges[0],
                self.cfg.domain_rand.delay_angle_ranges[1],
            )

        self.pmf_history.pop(0)
        self.joint_pos_history.pop(0)
        self.foot_fre_history.pop(0)
        self.joint_pos_err_history.pop(0)
        self.pmf_history.append(self.pm_f.detach().clone())
        self.joint_pos_history.append(self.joint_pos.detach().clone())
        self.foot_fre_history.append(self.foot_frc.detach().clone())
        self.joint_pos_err_history.append(
            self.joint_pos_error.detach().clone())

    def observation(self):
        self.obs_buf_pure = self.pure_observation()
        if self.cfg.noise_values.randomize_noise:
            self.noise_values = (
                2.0 * torch.rand_like(self.obs_buf) - 1.0
            ) * self._get_observation_noise_scales(len_vec=len(self.obs_buf[0]))
            self.obs_buf = self.obs_buf_pure + self.noise_values  # add noise to obs_buf
            self.obs_history.append(self.obs_buf)
        else:
            self.obs_history.append(self.obs_buf_pure)
        # return torch.ones_like(self.obs_history[-1])  #
        estimation_value = torch.cat([self.env.base_lin_vel], dim=1)
        return (
            torch.cat([obs for obs in self.obs_history], dim=-1),
            estimation_value,
        )  # #self.obs_history[-1]  #

    def critic_observation(self):
        pm_phase = torch.cat(
            (torch.sin(self.foot_phase), torch.cos(self.foot_phase)), 1
        )
        obs_buf = torch.cat(
            [
                self.commands[:, [0, 2]],
                self.commands[:, [0]] - self.env.base_lin_vel[:, [0]],
                self.commands[:, [2]] - self.env.base_ang_vel[:, [2]],
                self.env.base_lin_vel,
                self.env.base_euler[:, :2] * 3.0,  # 4-6
                self.env.base_ang_vel / 2.0,  # 6-9
                self.env.joint_pos - self.ref_joint_action,  # 9-21
                self.env.joint_vel / 10.0,  # 21-33
                self.current_joint_act - self.ref_joint_action,  # 33-45
                self.joint_pos_error,  # 45-57
                pm_phase * self.static_flag,  # 57-65
                (self.pm_f * 0.3 - 1.0) * self.static_flag,  # 65-69
                # self.env.foot_pos_hd[:, [1, 4]] - self.env.base_pos_hd[:, [1]],
                self.env.foot_frc.clip(max=1000.0) / 500.0,
                self.env.base_pos_hd[:, [1, 2]],
                # torch.norm(self.env.contact_forces[:, self.env.termination_contact_indices, :], dim=-1).clip(max=5.) / 5.,
                self.foot_height * 10.0,
            ],
            dim=1,
        )
        return obs_buf

    def pure_observation(self):
        pm_phase = torch.cat(
            (torch.sin(self.foot_phase), torch.cos(self.foot_phase)), 1
        )
        # pm_phase = torch.cat(
        #     (torch.sin(self.foot_phase), torch.cos(self.foot_phase)), 1)
        twh_joint_pos_error = self.joint_pos_error.clone()
        if self.cfg.domain_rand.randomize_joint_static_error:
            twh_joint_pos_error = self._get_observation_joint_static_error(
                self.joint_pos_error
            )
        self.obs_buf = torch.cat(
            [
                self.commands[:, [0, 2]],  # 0-1
                (self.commands[:, [2]] - self.base_ang_vel[:, [2]]) * 0.5,  # 2
                self.base_euler[:, :2] * 3.0,  # 3,4 #Roll/Pitch
                self.base_ang_vel * 0.5,  # 5-7 #角速度
                (self.joint_pos[:, :10] -
                 self.ref_joint_action[:, :10]),  # 8-17
                self.joint_vel[:, :10] * 0.1,  # 18-27
                twh_joint_pos_error[:, :10],  # 28-37
                pm_phase * self.static_flag,  # 57-65
                (self.pm_f * 0.3 - 1.0) * self.static_flag,  # 65-69
            ],
            dim=1,
        )
        return self.obs_buf

    def _get_observation_noise_scales(self, len_vec):
        noise_vec = torch.zeros(len_vec, device=self.device, dtype=torch.float)
        noise_values = self.cfg.noise_values
        # noise_vec[0:2] = noise_values.lin_vel  # commands
        noise_vec[2:3] = noise_values.ang_vel  # yaw rate error
        noise_vec[3:5] = noise_values.gravity  # rp
        noise_vec[5:8] = noise_values.ang_vel  # rpy rate
        noise_vec[8:18] = noise_values.dof_pos  # joint position
        noise_vec[18:28] = noise_values.dof_vel  # joint velocity
        noise_vec[28:38] = noise_values.dof_pos  # joint error
        return noise_vec  # * ratio

        pos_over = self.env.base_pos[:, [2]] < 0.15
        pos_over |= self.env.base_pos[:, [2]] >= 0.65

    def action(self, net_out, step_num):
        self.debug_net_out_history.append(net_out)
        net_out = scale_transform(
            net_out, self.action_low[:12], self.action_high[:12])
        self.net_out_history.append(net_out)
        self.phase_modulator.compute(net_out[:, : self.num_legs])
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
        zero_joint_act = torch.zeros(act.shape)[:, :9].to(self.device)
        act = torch.cat([act, zero_joint_act], 1)
        self.current_joint_act = act
        self.action_history.append(act.clone())
        return act

    def _resample_commands(self, env_ids):
        """完全重写：强制仅生成 X 轴直线指令，屏蔽侧向和旋转"""
        # 1. 清除旧命令
        self.commands[env_ids, :] = 0.

        # 2. 强制仅给定 X 方向速度 (0.4 ~ 1.0 m/s)
        # 必须要给一个明确的正向速度，不能太小，否则机器人会因为想“偷懒”而站着不动
        self.commands[env_ids, 0] = torch_rand_float(
            0.4, 1.0, (len(env_ids), 1), device=self.device
        ).squeeze(1)

        # 3. 强制 Y (侧向) 和 Yaw (旋转) 为 0
        self.commands[env_ids, 1] = 0.
        self.commands[env_ids, 2] = 0.

        # 4. 锚点环境 (保持前400个环境静止，防止课程崩塌)
        static_env_count = 400
        static_indices = env_ids[env_ids < static_env_count]
        if len(static_indices) > 0:
            self.commands[static_indices, :] = 0.

        # 5. 更新静止标志位
        self.static_flag[env_ids] = torch.where(
            torch.norm(self.commands[env_ids, :3], dim=1, keepdim=True) < 0.11,
            False, True
        ).float()

        # 同步命令与标志位
        self.commands[env_ids, :3] *= self.static_flag[env_ids]

    def terminate(self):
        time_out = torch.unsqueeze(self.env.time_out_buf, 1)
        twist_over = torch.abs(self.env.base_euler[:, 0:1]) > 1.6
        twist_over |= torch.abs(self.env.base_euler[:, 1:2]) > 1.6

        pos_over = self.env.base_pos[:, [2]] < 0.4
        shoulder_over = self.shoulder_height < 0.4

        lateral_over = torch.abs(self.env.base_lin_vel[:, [1]]) > 1.0

        action_over = (
            torch.sum(
                torch.abs(self.current_joint_act -
                          self.joint_action_limit_low_over)
                < 0.02,
                dim=1,
                keepdim=True,
            )
            >= 2
        )
        action_over |= (
            torch.sum(
                torch.abs(self.current_joint_act -
                          self.joint_action_limit_high_over)
                < 0.02,
                dim=1,
                keepdim=True,
            )
            >= 2
        )
        # jpos_over = torch.sum(torch.abs(self.joint_pos - self.joint_action_limit_low_over) < 0.05, dim=1,
        #                       keepdim=True) >= 1
        # jpos_over |= torch.sum(torch.abs(self.joint_pos - self.joint_action_limit_high_over) < 0.05, dim=1,
        #                        keepdim=True) >= 1
        con_over = torch.where(
            torch.sum(
                torch.where(
                    torch.norm(
                        self.env.contact_forces[
                            :, self.env.termination_contact_indices, :
                        ],
                        dim=-1,
                    )
                    > 10.0,
                    True,
                    False,
                ),
                dim=1,
                keepdim=True,
            )
            >= 1,
            True,
            False,
        )
        if self.env.render or self.env.epochs > 1 or self.env.tcn_name is not None:
            done = (
                action_over | pos_over | twist_over | time_out
            )  # todo for training tcn and evaluation
        else:
            done = con_over | action_over | pos_over | twist_over | time_out
        return done, time_out

    def reward(self, target_pos=None, target_vel=None, real_pos=None, real_vel=None):
        """
        状态 s_t
            ↓
        策略网络 π(s_t) → action
            ↓
        仿真器 → 得到新状态 s_{t+1}
            ↓
        reward(s_t, a_t, s_{t+1})  ← 就是你这个函数

        """
        constant_rew = to_torch([1.0]).repeat(self.num_envs, 1)
        lin_vel_x_norm = (
            torch.clip(
                torch.abs(self.commands[:, [0]]), min=0.3, max=2.0) + 0.2
        )  # 归一化尺度 #torch.abs(self.commands[:, [0]])->|cmd_vx|
        # lin_vel_y_norm = torch.clip(torch.abs(self.commands[:, [1]]), min=0.3, max=2.) + 0.2
        yaw_rate_norm = (
            torch.clip(
                torch.abs(self.commands[:, [2]]), min=0.3, max=1.5) + 0.2
        )
        """
        lateral_vel_rew:
        k = torch.clip(5.0 / lin_vel_x_norm, 3.0, 15.0) 
        - 当 vx_cmd 很大 → lin_vel_x_norm 大 → k 小 → 指数衰减慢 → 对同样的 vy 惩罚更轻
        - 当 vx_cmd 很小 → lin_vel_x_norm 小 → k 大 → 指数衰减速 → 对同样的 vy 惩罚更重
        物理含义：高速奔跑时：身体自然会有一定侧摆，策略不必过度“僵直”地去抑制 vy，否则容易僵硬、摔倒。低速或原地踏步时：任何不必要的侧移都是“能量浪费”或“平衡差”，惩罚要更严厉。
        """
        lateral_vel_rew = torch.exp(
            -torch.clip(20 / lin_vel_x_norm, min=3.0, max=15.0)
            * torch.norm(self.env.base_lin_vel[:, [1]], dim=1, keepdim=True) ** 2
        )  # lateral_vel_rew = exp( −k * vy² ) #原始是5.0/...
        # 【新增】线性惩罚项！ #修改
        # 只要有侧向速度，就直接扣分。这样即使漂移很快，梯度依然存在，逼迫网络修正。
        # lateral_vel_rew -= 2.0 * torch.abs(self.env.base_lin_vel[:, [1]]) #原始
        lateral_vel_rew -= 3.0 * \
            torch.abs(self.env.base_lin_vel[:, [1]])  # 修改 new

        base_heit_rew = torch.exp(
            -60 * (self.env.base_pos[:, [2]] - 1.0) ** 2
        )  # self.env.base_pos[:, [2]]实际高度；1m 为期望高度
        balance_rew = 0.5 * (
            base_heit_rew
            * torch.exp(
                -torch.clip(4.0 / lin_vel_x_norm, min=2, max=8.0)
                * torch.norm(self.env.base_euler[:, :2], dim=-1, keepdim=True)
            )
            + 1.0
        )  # balance_rew = 0.5 * (高度 × 姿态惩罚 + 1). 站稳！

        forward_vel_rew = (
            torch.exp(
                -torch.clip(3.0 / lin_vel_x_norm, min=2.0, max=10.0)
                * (self.commands[:, [0]] - self.env.base_lin_vel[:, [0]]) ** 2
            )
            * balance_rew
        )  # exp(-k * (cmd_vx - real_vx)^2) * balance_rew #原始为4.0/...

        # 原始
        # yaw_rate_rew = (
        #     torch.exp(
        #         -torch.clip(10 / lin_vel_x_norm, min=1.5, max=6.0)
        #         * (self.commands[:, [2]] - self.env.base_ang_vel[:, [2]]) ** 2
        #     )
        #     * balance_rew
        # )  # exp(-k * (cmd_yaw - real_yaw)^2) * balance_rew #原始2.5/...
        # ggg
        yaw_rate_rew = (
            torch.exp(
                -torch.clip(10 / lin_vel_x_norm, min=1.5, max=6.0)
                * (self.commands[:, [2]] - self.env.base_ang_vel[:, [2]]) ** 2
            )
            * balance_rew
        )
        # 追加线性惩罚（去掉 static_flag 约束，始终生效）
        yaw_rate_rew -= 8.0 * torch.abs(self.env.base_ang_vel[:, [2]])

        lateral_vel_rew += (
            -0.1
            / lin_vel_x_norm
            * torch.norm(self.env.base_lin_vel[:, [1]], dim=1, keepdim=True)
            * self.static_flag
        )  # 侧移速度奖励 #乘 static_flag：只有命令速度较大时才启用；原地站直时这条线性惩罚直接归零，避免“僵直”站立也扣分。

        ang_vel_rew = torch.exp(
            -torch.clip(1.5 / lin_vel_x_norm, min=0.7, max=6.0)
            * torch.norm(self.env.base_ang_vel[:, :2], dim=1, keepdim=True) ** 2
        )
        # base_acc_rew = -0.4 / lin_vel_x_norm * torch.norm(
        #     (self.env.base_acc - to_torch([0, 0, 9.81], device=self.device)) * 0.1,
        #     dim=1, keepdim=True)
        #
        # base_acc_rew *= self.static_flag
        vertical_vel_rew = torch.exp(
            -torch.clip(5.0 / lin_vel_x_norm, min=3.0, max=15.0)
            * torch.norm(self.env.base_lin_vel[:, [2]], dim=1, keepdim=True) ** 2
        )
        vertical_vel_rew -= (
            0.8
            / lin_vel_x_norm
            * torch.norm(self.env.base_lin_vel[:, 1:], dim=1, keepdim=True)
            * self.static_flag
        )

        """
        swing_foot_index		        脚离地 → True（离地 >1 N）
        support_foot_index		    脚承重 → True（受力 >20 N）
        self.foot_swing_mask		相位说该摆 → True（φ∈[π,2π)）
        self.foot_support_mask		相位说该撑 → True（φ∈[0,π)）

        """

        support_foot_index = torch.where(
            self.env.foot_frc >= 20.0, True, False)
        swing_foot_index = torch.where(self.env.foot_frc < 1.0, True, False)

        foot_clear_rew = (
            torch.sum(
                torch.logical_and(swing_foot_index, self.foot_swing_mask),
                dtype=torch.float,
                dim=1,
                keepdim=True,
            )
            / self.num_legs
        )  # 意义：防止“该摆不摆”或“拖着地跑”。比例越高说明步态越干净。

        foot_support_rew = (
            torch.sum(
                torch.logical_and(support_foot_index, self.foot_support_mask),
                dtype=torch.float,
                dim=1,
                keepdim=True,
            )
            / self.num_legs
        )  # 意义：防止“该撑不撑”或“虚踩”——支撑脚必须真发力。
        foot_support_rew *= self.static_flag
        foot_clear_rew *= self.static_flag

        foot_support_rew += (
            torch.sum(support_foot_index, dtype=torch.float,
                      dim=1, keepdim=True)
            / self.num_legs
        )

        foot_heit_score = 50.0 * torch.clip(self.foot_height, min=0.0, max=0.1)
        foot_height_rew = (
            torch.sum(self.foot_swing_mask * foot_heit_score, dim=1, keepdim=True).clip(
                max=5.0
            )
            * self.static_flag
        )
        # ------------------- 修改这里 -------------------
        # 原代码: -20.0 * ... (惩罚太重，导致它不敢抬腿，从而跛脚)
        # 建议修改: -5.0 * ... (降低惩罚，允许偶尔抬高一点)
        # 同时: 0.1 改为 0.13 (稍微放宽高度阈值)
        foot_height_rew += -1 * torch.sum(
            (self.foot_height - 0.15).clip(min=0.0), dim=1, keepdim=True
        )  # foot_height ≈ 0.1m #惩罚“过度抬脚”

        foot_height_rew += (
            -0.5
            * torch.sum(self.foot_support_mask * foot_heit_score, dim=1, keepdim=True)
            * self.static_flag
        )  # 惩罚“支撑相却抬脚”
        foot_height_rew += -0.5 * torch.sum(
            support_foot_index * foot_heit_score, dim=1, keepdim=True
        )  # 惩罚“承重脚却抬脚”
        foot_height_rew += (
            -0.5
            * torch.sum(foot_heit_score, dim=1, keepdim=True)
            * torch.logical_not(self.static_flag)
        )  # 惩罚“静止时任何脚离地”

        twist_rew = - \
            torch.norm(self.env.base_euler[:, :2], dim=-1, keepdim=True)

        self.foot_frc_acc = (self.env.foot_frc - self.last_foot_frc).clone()
        foot_soft_rew = (
            -0.1
            * torch.clip(1.0 / lin_vel_x_norm, min=0.0, max=1.5)
            * torch.norm(self.foot_frc_acc, dim=1, keepdim=True)
            / 100.0
        )

        self.last_foot_frc = self.env.foot_frc.clone().detach()

        # 1. 摆动相惩罚：摆动脚不应受力（保持原样）
        feet_contact_frc_rew = (
            -torch.norm(self.env.foot_frc * self.foot_swing_mask,
                        dim=1, keepdim=True)
            * self.static_flag
        )

        # 2. 支撑相惩罚（修复BUG）：
        # 原代码使用了 support_foot_index (sensor) 导致逻辑互斥归零。
        # 修复：必须使用 self.foot_support_mask (phase)，意思是“相位要求你支撑时，如果力不够(小于20N)，就扣分”。
        # 作用：迫使支撑腿用力踩地，从而释放摆动腿进行抬步，解决拖腿问题。
        feet_contact_frc_rew += -torch.sum(
            (20.0 - self.env.foot_frc).clip(min=0.0) * self.foot_support_mask,
            dim=1, keepdim=True
        )

        # 3. 静止模式惩罚（修复BUG）：
        # 原代码逻辑反了（变成了禁止力小）。
        # 修复：改为 (force - 250).clip(min=0) * -1。
        # 作用：只有当力超过 250N 时才扣分，允许轻柔站立。
        # feet_contact_frc_rew += -torch.sum(
        #     (self.env.foot_frc - 250.0).clip(min=0.0) * torch.logical_not(self.static_flag),
        #     dim=1, keepdim=True
        # )
        # 修改为：
        # feet_contact_frc_rew += -torch.sum(
        #     (20.0 - self.env.foot_frc).clip(min=0.0) * self.foot_support_mask, # <--- 修正为 mask
        #     dim=1, keepdim=True
        # )
        feet_contact_frc_rew += -torch.sum(
            (self.env.foot_frc - 350.0).clip(min=0.0),
            dim=1, keepdim=True
        ) * torch.logical_not(self.static_flag)  # 修改 new

        clip_foot_h = torch.abs(self.foot_height) + 0.03

        """
        foot_vel[..., 0] → x 方向速度（前后）
        foot_vel[..., 1] → y 方向速度（左右）
        foot_vel[..., 2] → z 方向速度（上下）

        foot_swing_mask = 1 → 这是摆动脚
        foot_swing_mask = 0 → 这是支撑脚

        static_flag = 1 → 正在走
        static_flag = 0 → 静止 / 站立

        """
        foot_slip_rew = (
            lin_vel_x_norm
            * torch.sum(
                (self.env.foot_vel.view(
                    self.num_envs, self.num_legs, -1)[:, :, 0])
                * self.commands[:, [0]].sign()
                * self.foot_swing_mask,
                dim=1,
                keepdim=True,
            )
        ).clip(min=0.0, max=1.5) * self.static_flag

        foot_slip_rew += (
            -0.5
            * torch.norm(
                torch.norm(
                    self.env.foot_vel.view(
                        self.num_envs, self.num_legs, -1)[:, :, [1]],
                    dim=-1,
                ),
                dim=1,
                keepdim=True,
            )
            * self.static_flag
        )

        foot_slip_rew += (
            0.2
            * torch.norm(
                0.02
                * torch.norm(
                    self.env.foot_vel.view(
                        self.num_envs, self.num_legs, -1)[:, :, :2],
                    dim=-1,
                )
                / clip_foot_h,
                dim=1,
                keepdim=True,
            )
            * (self.static_flag - 1.0)
        )

        foot_slip_rew += (
            -0.1
            / lin_vel_x_norm
            * torch.norm(
                0.02
                * torch.norm(
                    self.env.foot_vel.view(
                        self.num_envs, self.num_legs, -1)[:, :, :2],
                    dim=-1,
                )
                / clip_foot_h,
                dim=1,
                keepdim=True,
            )
            * self.static_flag
            # [:, :, :2]-》平面速度，vx && vy #torch.norm(self.env.foot_vel[... , :2], dim=-1)->slip_speed #命令速度越小，对脚掌在地面上的 任何二维滑移 越不能容忍。
        )

        foot_vz_rew = (
            -0.1
            * torch.clip(1.0 / lin_vel_x_norm, min=0.0, max=1.0)
            * torch.norm(
                torch.norm(
                    self.env.foot_vel.view(self.num_envs, self.num_legs, -1)[
                        :, :, [2]
                    ].clip(max=0.0),
                    dim=-1,
                )
                / clip_foot_h,
                dim=1,
                keepdim=True,
            )
            * self.static_flag
        )

        foot_vz_rew += (
            0.5
            * torch.clip(1.0 / lin_vel_x_norm, min=0.0, max=1.0)
            * torch.norm(
                torch.norm(
                    self.env.foot_vel.view(self.num_envs, self.num_legs, -1)[
                        :, :, [2]
                    ].clip(max=0.0),
                    dim=-1,
                ),
                dim=1,
                keepdim=True,
            )
            * (self.static_flag - 1.0)
        )

        foot_acc_rew = (
            -0.4
            * torch.clip(1.0 / lin_vel_x_norm, min=0.0, max=2.0)
            * torch.norm(self.env.foot_vel[:, [2, 5]], dim=1, keepdim=True)
        )

        action_smooth_rew = (
            -0.3
            * torch.clip(1.0 / lin_vel_x_norm, min=0.0, max=2.0)
            * torch.norm(
                self.action_history[-3]
                - 2.0 * self.action_history[-2]
                + self.action_history[-1],
                dim=1,
                keepdim=True,
            )
        )
        net_out_smooth_rew = (
            -0.2
            * torch.clip(1.0 / lin_vel_x_norm, min=0.0, max=2.0)
            * torch.norm(
                (
                    self.net_out_history[-3]
                    - 2 * self.net_out_history[-2]
                    + self.net_out_history[-1]
                )[:, self.num_legs:],
                dim=1,
                keepdim=True,
            )
            ** 2
        )

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
        # action_constraint_rew += -2. * torch.norm((self.env.joint_pos[:, [1, 2, 7, 8, 5, 11]]), dim=1, keepdim=True) * self.static_flag

        sa_constraint_rew = (
            -0.1
            * torch.clip(1.0 / lin_vel_x_norm, min=0.0, max=1.5)
            * torch.norm(self.env.joint_pos, dim=1, keepdim=True) ** 2
            * self.static_flag
        )

        sa_constraint_rew += (
            -self.static_flag
            * torch.clip(1.0 / lin_vel_x_norm, 0, 2)
            * torch.norm(
                (self.env.joint_pos[:, :5] * support_foot_index[:, [0]]),
                dim=1,
                keepdim=True,
            )
            ** 2
        )
        sa_constraint_rew += (
            -self.static_flag
            * torch.clip(1.0 / lin_vel_x_norm, 0, 2)
            * torch.norm(
                (self.env.joint_pos[:, 5:10] * support_foot_index[:, [1]]),
                dim=1,
                keepdim=True,
            )
            ** 2
        )

        joint_pos_error_rew = (
            -0.4
            * torch.clip(1.0 / lin_vel_x_norm, min=0.0, max=2.0)
            * torch.norm(
                (self.current_joint_act - self.env.joint_pos)[:, :10],
                dim=1,
                keepdim=True,
            )
            ** 2
        )
        # joint_pos_error_rew *= self.static_flag

        joint_velocity_rew = (
            -0.4
            * torch.clip(1.0 / lin_vel_x_norm, min=0.0, max=1.5)
            * torch.norm(self.env.joint_vel[:, :], dim=1, keepdim=True) ** 2
        )
        # joint_velocity_rew += -torch.clip(1. / lin_vel_x_norm, 0, 2) * torch.norm(self.env.joint_vel[:, [1, 2, 5, 7, 8, 11]], dim=1, keepdim=True) ** 2
        # joint_velocity_rew *= self.static_flag

        self.last_joint_vels = self.env.joint_vel.clone().detach()

        joint_tor_rew = (
            -0.4
            * torch.clip(1.0 / lin_vel_x_norm, min=0.0, max=2.0)
            * torch.sum(
                (torch.abs(self.env.react_tau[:, :]) - self.env.torque_limits[:]).clip(
                    min=0.0
                ),
                dim=1,
                keepdim=True,
            )
        )

        joint_tor_rew *= self.static_flag

        self.last_foot_vel = self.env.foot_vel.clone().detach()
        pmf_rew = (
            -0.02
            * torch.clip(1.0 / lin_vel_x_norm, min=0.0, max=1.5)
            * torch.norm(
                (
                    self.net_out_history[-3]
                    - 2 * self.net_out_history[-2]
                    + self.net_out_history[-1]
                )[:, : self.num_legs],
                dim=1,
                keepdim=True,
            )
        )
        pmf_rew += (
            -1.5
            * torch.clip(1 / lin_vel_x_norm, 0, 1.5)
            * torch.norm(
                self.net_out_history[-1][:,
                                         : self.num_legs] * self.foot_support_mask,
                dim=1,
                keepdim=True,
            )
            ** 2
        )
        pmf_rew *= self.static_flag

        net_out_val_rew = (
            -0.4
            * torch.clip(1.0 / lin_vel_x_norm, min=0.0, max=1.5)
            * torch.norm(
                self.net_out_history[-1][:, self.num_legs:], dim=1, keepdim=True
            )
            ** 2
        )
        # net_out_val_rew *= self.static_flag
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

        # foot_py_rew += 0.5 * (torch.norm(self.env.foot_euler[:, [1, 4]] * support_foot_index, dim=1, keepdim=True)) * (self.static_flag - 1.)
        # foot_py_rew += 0.5 * (torch.norm(self.env.foot_euler[:, [0, 3]] * support_foot_index, dim=1, keepdim=True)) * (self.static_flag - 1.)

        leg_width_rew = -torch.norm(
            torch.abs(self.env.foot_pos_hd[:, [
                      1, 4]] - self.env.base_pos_hd[:, [1]])
            - 0.25,
            dim=1,
            keepdim=True,
        )
        # ggg
       # ========== 修复版：左右对称性奖励 ==========

        # 1. 关节力矩对称性 (仅在【静止/站立】时生效)
        # 站立时，两腿发力应该一样。走路时允许不一样。
        left_leg_taus = self.joint_tau[:, [2, 3]]   # 左髋pitch, 左膝
        right_leg_taus = self.joint_tau[:, [7, 8]]  # 右髋pitch, 右膝
        tau_symmetry_rew = -0.5 * torch.norm(
            left_leg_taus - right_leg_taus, dim=1, keepdim=True
        ) * torch.logical_not(self.static_flag)  # 修改：用 not 翻转，让它只在站立(0)时生效

        # 2. 足部接触力对称性 (仅在【静止/站立】时生效)
        # 站立时，两只脚应该平分体重。走路时必定是一只脚受力大。
        contact_force_diff = torch.abs(
            self.foot_frc[:, [0]] - self.foot_frc[:, [1]])
        foot_force_symmetry_rew = -0.002 * contact_force_diff * \
            torch.logical_not(self.static_flag)  # 修改

        # 3. 接触时间对称性 (这段代码逻辑不适用于走路，建议直接废弃或仅用于站立)
        # 走路时相位差是 180 度，一个是 1 一个肯定是 0，算出来 diff 永远是 1，惩罚毫无意义。
        left_support = self.foot_support_mask[:, [0]].float()
        right_support = self.foot_support_mask[:, [1]].float()
        contact_time_diff = torch.abs(left_support - right_support)
        contact_symmetry_rew = -0.3 * contact_time_diff * \
            torch.logical_not(self.static_flag)  # 修改

        # --- 新增 4：走路时的“空间步幅对称性”（用来治瘸腿的特效药）---
        # 走路时 (static_flag=1)，虽然左右脚不可能同时迈出，但它们相对身体中心的距离和应该接近 0。
        # left_foot_rel_x = self.env.foot_pos[:, 0] - self.env.base_pos[:, 0]
        # right_foot_rel_x = self.env.foot_pos[:, 3] - self.env.base_pos[:, 0]
        # step_symmetry_rew = -5.0 * \
        #     torch.abs(left_foot_rel_x + right_foot_rel_x) * self.static_flag
        # --- 新增 4：走路时的”空间步幅对称性”（治瘸腿特效药，升级版）---
        # 使用 _hd (航向坐标系)，这样即使未来机器人转弯，X轴也永远代表它当前的正前方！
        # 注意：使用 [0:1] 切片保持维度为 (num_envs, 1)，与 static_flag 形状匹配
        left_foot_rel_x = self.env.foot_pos_hd[:,
                                               0:1] - self.env.base_pos_hd[:, 0:1]
        right_foot_rel_x = self.env.foot_pos_hd[:,
                                                3:4] - self.env.base_pos_hd[:, 0:1]
        step_symmetry_rew = -5.0 * \
            torch.abs(left_foot_rel_x + right_foot_rel_x) * self.static_flag
        # 合并为总对称性奖励
        symmetry_rew = tau_symmetry_rew + foot_force_symmetry_rew + \
            contact_symmetry_rew + step_symmetry_rew
        # end ------------------------------------------------------

        lsin = torch.sin(self.foot_phase.clone())
        lcos = torch.cos(self.foot_phase.clone())
        foot_phase_rew = (
            -torch.norm(lsin[:, [0]] + lsin[:, [1]], dim=1, keepdim=True) ** 2
        )
        foot_phase_rew += (
            -torch.norm(lcos[:, [0]] + lcos[:, [1]], dim=1, keepdim=True) ** 2
        )
        foot_phase_rew *= self.static_flag

        # is_push = torch.norm(self.env.push_force[:, self.env.push_body_id, :].view(self.num_envs, -1), dim=1, keepdim=True) > 100.

        rew_dict = dict(
            symmetry=symmetry_rew * balance_rew * 1.5,  # ggg
            balance=balance_rew * 0.5,
            fwd_vel=forward_vel_rew * 5.5,
            # yaw_rat=yaw_rate_rew * 2, #原始
            yaw_rat=yaw_rate_rew * 3,  # ggg
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
        )  # act_smo 是惩罚“动作突变”。虽然为了平滑，但如果权重太大，机器人会觉得“我不动就不会有突变”，导致它不愿意快速响应指令。
        if self.debug:
            self.rew_names = [name for name in rew_dict.keys()]
            self.debug = None
        rewards = torch.cat(
            [
                torch.clip(value.to(self.device), min=-
                           4.0, max=5.0) * self.env.dt
                for value in rew_dict.values()
            ],
            dim=1,
        )
        return rewards
