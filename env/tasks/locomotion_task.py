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
        self.obs_history = deque(maxlen=3)
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

        # ========== 新增：步幅追踪变量（用于检测瘸腿）==========
        # 记录每只脚上一次落地时的前向位置（相对于当时身体）
        self.last_stride_x = torch.zeros(
            self.num_envs, self.num_legs,
            dtype=torch.float,
            device=self.device,
        )
        # 记录上一步的支撑相状态，用于检测"摆动→支撑"的切换
        self.last_support_mask = torch.ones(
            self.num_envs, self.num_legs,
            dtype=torch.bool,
            device=self.device,
        )
        # 记录每只脚的累计步幅（用于奖励计算）
        self.stride_length = torch.zeros(
            self.num_envs, self.num_legs,
            dtype=torch.float,
            device=self.device,
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

        # 重置步幅追踪变量
        self.last_support_mask[env_ids] = self.foot_support_mask[env_ids].clone()
        self.stride_length[env_ids] = 0.0

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
        # 评估时给一个恒定速度
        self.commands[:, [0]] = 0.5
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

        # ========== 新增：步幅追踪（检测摆动→支撑切换）==========
        # 检测"摆动→支撑"的切换：上一帧是摆动(0)，这一帧是支撑(1)
        swing_to_support = torch.logical_and(
            torch.logical_not(self.last_support_mask),
            self.foot_support_mask
        )
        # 获取当前脚相对于身体的前向位置
        left_foot_rel_x = self.env.foot_pos_hd[:, 0:1] - self.env.base_pos_hd[:, 0:1]
        right_foot_rel_x = self.env.foot_pos_hd[:, 3:4] - self.env.base_pos_hd[:, 0:1]
        foot_rel_x = torch.cat([left_foot_rel_x, right_foot_rel_x], dim=1)

        # 当脚从摆动切换到支撑时，记录这个位置作为步幅
        self.stride_length = torch.where(
            swing_to_support,
            foot_rel_x,
            self.stride_length
        )

        # 更新上一帧的支撑状态
        self.last_support_mask = self.foot_support_mask.clone()
        # ========== 步幅追踪结束 ==========

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

        # 2. 强制仅给定 X 方向速度 (0.3 ~ 0.8 m/s)
        self.commands[env_ids, 0] = torch_rand_float(
            0.3, 0.8, (len(env_ids), 1), device=self.device
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
        # Simplified reward: 14 core terms
        lin_vel_x_norm = (
            torch.clip(torch.abs(self.commands[:, [0]]), min=0.3, max=2.0) + 0.2
        )

        # ========== 1. 平衡奖励（站稳是前提）==========
        base_heit_rew = torch.exp(-60 * (self.env.base_pos[:, [2]] - 1.0) ** 2)
        balance_rew = 0.5 * (
            base_heit_rew
            * torch.exp(
                -torch.clip(4.0 / lin_vel_x_norm, min=2, max=8.0)
                * torch.norm(self.env.base_euler[:, :2], dim=-1, keepdim=True)
            )
            + 1.0
        )

        # ========== 2. 前进速度跟踪（核心任务）==========
        forward_vel_rew = torch.exp(
            -torch.clip(3.0 / lin_vel_x_norm, min=2.0, max=10.0)
            * (self.commands[:, [0]] - self.env.base_lin_vel[:, [0]]) ** 2
        )

        # ========== 3. 侧向速度惩罚 ==========
        lateral_vel_rew = -2.0 * torch.abs(self.env.base_lin_vel[:, [1]])

        # ========== 4. 偏航角速度跟踪 ==========
        yaw_rate_rew = torch.exp(
            -torch.clip(10 / lin_vel_x_norm, min=1.5, max=6.0)
            * (self.commands[:, [2]] - self.env.base_ang_vel[:, [2]]) ** 2
        )

        # ========== 5. 垂直速度惩罚（不跳）==========
        vertical_vel_rew = torch.exp(
            -5.0 * torch.norm(self.env.base_lin_vel[:, [2]], dim=1, keepdim=True) ** 2
        )

        # ========== 6. 姿态惩罚（不歪）==========
        twist_rew = -torch.norm(self.env.base_euler[:, :2], dim=-1, keepdim=True)

        # ========== 7. 步态相位奖励 ==========
        support_foot_index = torch.where(self.env.foot_frc >= 20.0, True, False)
        swing_foot_index = torch.where(self.env.foot_frc < 1.0, True, False)

        # 抬脚：摆动相时脚应该离地
        foot_clear_rew = torch.sum(
            torch.logical_and(swing_foot_index, self.foot_swing_mask),
            dtype=torch.float, dim=1, keepdim=True,
        ) / self.num_legs

        # 踩地：支撑相时脚应该受力
        foot_support_rew = torch.sum(
            torch.logical_and(support_foot_index, self.foot_support_mask),
            dtype=torch.float, dim=1, keepdim=True,
        ) / self.num_legs

        # 步态反相：左右脚相位差应为 π
        lsin = torch.sin(self.foot_phase.clone())
        lcos = torch.cos(self.foot_phase.clone())
        foot_phase_rew = (
            -torch.norm(lsin[:, [0]] + lsin[:, [1]], dim=1, keepdim=True) ** 2
            - torch.norm(lcos[:, [0]] + lcos[:, [1]], dim=1, keepdim=True) ** 2
        )

        # ========== 8. 动作平滑（二阶差分惩罚）==========
        action_smooth_rew = -torch.norm(
            self.action_history[-3] - 2.0 * self.action_history[-2] + self.action_history[-1],
            dim=1, keepdim=True,
        )

        # ========== 9. 关节跟踪误差 ==========
        joint_pos_error_rew = -torch.norm(
            (self.current_joint_act - self.env.joint_pos)[:, :10],
            dim=1, keepdim=True,
        ) ** 2

        # ========== 10. 对称性（治瘸腿）==========
        left_foot_rel_x = self.env.foot_pos_hd[:, 0:1] - self.env.base_pos_hd[:, 0:1]
        right_foot_rel_x = self.env.foot_pos_hd[:, 3:4] - self.env.base_pos_hd[:, 0:1]
        position_symmetry_rew = -torch.abs(left_foot_rel_x + right_foot_rel_x)
        hip_pitch_diff = torch.abs(self.joint_pos[:, [2]] - self.joint_pos[:, [7]])
        joint_symmetry_rew = -hip_pitch_diff

        # ========== 11. 脚间距 ==========
        leg_width_rew = -torch.norm(
            torch.abs(self.env.foot_pos_hd[:, [1, 4]] - self.env.base_pos_hd[:, [1]]) - 0.25,
            dim=1, keepdim=True,
        )

        # ========== 12. 支撑脚力惩罚 ==========
        feet_contact_frc_rew = -torch.sum(
            (20.0 - self.env.foot_frc).clip(min=0.0) * self.foot_support_mask,
            dim=1, keepdim=True,
        )

        # ========== 汇总 ==========
        rew_dict = dict(
            fwd_vel=forward_vel_rew * 5.0,
            balance=balance_rew * 1.0,
            lateral_vel=lateral_vel_rew * 2.0,
            yaw_rat=yaw_rate_rew * 2.0,
            vertical_vel=vertical_vel_rew * 0.5,
            twist=twist_rew * 1.5,
            foot_clr=foot_clear_rew * 3.0,
            foot_supt=foot_support_rew * 1.0,
            foot_phase=foot_phase_rew * 2.0,
            act_smo=action_smooth_rew * 0.3,
            jnt_pos_err=joint_pos_error_rew * 0.2,
            symmetry=(position_symmetry_rew + joint_symmetry_rew) * 1.0,
            leg_width=leg_width_rew * 1.0,
            feet_frc=feet_contact_frc_rew * 0.005,
        )
        if self.debug:
            self.rew_names = [name for name in rew_dict.keys()]
            self.debug = None
        rewards = torch.cat(
            [
                torch.clip(value.to(self.device), min=-4.0, max=5.0) * self.env.dt
                for value in rew_dict.values()
            ],
            dim=1,
        )
        return rewards