from collections import OrderedDict
import numpy as np
import pandas as pd
import os

import env
from .legged_robot import LeggedRobotEnv
from .tasks import LocomotionTask
import collections
from env.utils.math import scale_transform
from isaacgym.torch_utils import *
from .tasks import BaseTask
import torch


class GymEnvWrapper:
    def __init__(self, env: LeggedRobotEnv, task: LocomotionTask, residual_task=None,dynamic_params=None, debug: bool = False):
        self.env = env
        self.task = task
        self.task.debug = debug
        self.device = env.device
        self.debug = debug
        self.debug_data = {name: [] for name in self.debug_name} if debug else None
        if dynamic_params is not None:
            self.dynamic_params = dynamic_params
            self.tcn_obs_buf = collections.deque(maxlen=dynamic_params['seq_length'])
        self.residual_task = residual_task if residual_task is not None else None

    def reset(self, env_ids, reset_phy_ids, reset_joint_pos=None, reset_joint_vel=None, reset_base_quat=None):
        if reset_joint_pos is not None:
            self.env.reset_joint_pos = reset_joint_pos
        if reset_joint_vel is not None:
            self.env.reset_joint_vel = reset_joint_vel
        if reset_base_quat is not None:
            self.env.reset_base_quat = reset_base_quat
        self.env.step_states(self.task.commands)
        obs, estimation_value = self.task.observation()
        cri_obs = self.task.pure_observation()
        obs_real = self.task.pure_observation()
        if self.debug:
            self.clear_debug_data()
        self.task.reset(env_ids)
        self.env.reset(env_ids, reset_phy_ids)
        return obs, cri_obs, obs_real, estimation_value



    def step(self, net_out, step_num):
        joint_act = self.task.action(net_out, step_num)
        for _ in range(self.env.cfg.pd_gains.decimation):
            self.env.step_torques(joint_act)
        self.env.step_states(self.task.commands)
        self.task.step()
        obs,estimation_value = self.task.observation()
        cri_obs = self.task.pure_observation()
        obs_real = cri_obs.clone()
        rew = self.task.reward()
        done, time_done = self.task.terminate()
        info = self.task.info()
        rew_buf = torch.clip(rew.sum(dim=1), min=0.)
        # rew_buf = rew.sum(dim=1)
        if self.debug:
            self.record_debug_data(joint_act, rew, obs)
        reset_env_ids = (done > 0).nonzero(as_tuple=False)[:, [0]].flatten()
        reset_phy_ids = (time_done > 0).nonzero(as_tuple=False)[:, [0]].flatten()
        if len(reset_env_ids) > 0:
            self.task.reset(reset_env_ids)
            self.env.reset(reset_env_ids, reset_phy_ids)
        return obs, cri_obs, rew_buf, done, info, obs_real, estimation_value, reset_phy_ids


    def step_obs_for_critic(self):
        return self.task.pure_observation()

    def close(self):
        pass

    def record_debug_data(self, act, rew, obs):
        self.debug_data['reward'].append(rew / self.env.dt)
        self.debug_data['command'].append(self.task.commands.clone())
        self.debug_data['lin_vel'].append(self.env.base_lin_vel.clone())
        self.debug_data['base_eul'].append(self.env.base_euler.clone())
        self.debug_data['ang_vel'].append(self.env.base_ang_vel.clone())
        self.debug_data['base_pos'].append(self.env.base_pos_hd.clone())  #todo
        self.debug_data['foot_pos'].append(self.env.foot_pos_hd.clone())
        self.debug_data['foot_vel'].append(self.env.foot_vel.clone())
        self.debug_data['foot_frc'].append(self.env.foot_frc.clone())
        self.debug_data['foot_rpy'].append(self.env.foot_euler.clone())
        self.debug_data['foot_phs'].append(self.task.phase_modulator.phase.clone())
        self.debug_data['joint_act'].append(act)
        self.debug_data['joint_pos'].append(self.env.joint_pos.clone())
        self.debug_data['joint_vel'].append(self.env.joint_vel.clone())
        self.debug_data['joint_tau'].append(self.env.torques.clone())
        self.debug_data['net_out'].append(self.task.debug_net_out_history[-1].clone())
        self.debug_data['obs_state'].append(obs)


    @property
    def debug_name(self):
        d = OrderedDict()
        axises = ['x', 'y', 'z']
        foot_names = ['L', 'R']
        d['reward'] = self.task.rew_names
        d['command'] = ['fwd_vel', 'lat_vel', 'yaw_rate', 'heading']
        d['lin_vel'] = [n for n in axises]
        d['base_eul'] = [n for n in axises]
        d['ang_vel'] = [n for n in axises]
        d['joint_act'] = self.env.dof_names
        d['joint_pos'] = [n for n in self.env.dof_names]
        d['net_out'] = [f'{f}_f' for f in foot_names] + [n for n in self.env.dof_names[:10]]
        # d['net_out'] = [n for n in self.env.dof_names]
        d['foot_frc'] = [n for n in foot_names]
        d['foot_pos'] = [f'{o}_{n}' for o in foot_names for n in axises]
        d['obs_state'] = ['obs' + '_' + str(i) for i in range(self.env.num_observations)]
        d['foot_rpy'] = [f'{o}_{n}' for o in foot_names for n in axises]
        d['base_pos'] = [n for n in axises]
        d['foot_vel'] = [f'{o}_{n}' for o in foot_names for n in axises]
        d['foot_phs'] = [n for n in foot_names]
        d['joint_vel'] = [n for n in self.env.dof_names]
        d['joint_tau'] = [n for n in self.env.dof_names]
        return d

    def clear_debug_data(self):
        for k, v in self.debug_data.items():
            self.debug_data[k].clear()

    def save_debug_data(self, debug_dir: str):
        debug_data = {key: torch.stack(self.debug_data[key], dim=1).cpu().numpy() for key in self.debug_name.keys()}
        for i in range(self.env.num_envs):
            data_path = os.path.join(debug_dir, f'debug_{i}.xlsx')
            with pd.ExcelWriter(data_path) as f:
                for key in self.debug_name.keys():
                    pd.DataFrame(np.asarray(debug_data[key][i]), columns=self.debug_name[key]).to_excel(f, key,
                                                                                                        index=False)
            print(f'#The debug data has been written into `{data_path}`.')
        return debug_data['joint_act'], debug_data['joint_pos'], debug_data['joint_vel']

