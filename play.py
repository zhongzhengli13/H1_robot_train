import time
import cv2
import importlib
import isaacgym
import numpy as np
import os
from os.path import join, exists
import shutil
import torch

from env import LeggedRobotEnv, GymEnvWrapper
from env.tasks import load_task_cls
from env.utils import get_args
from env.utils.helpers import class_to_dict, set_seed, parse_sim_params
from model import load_actor, load_critic
from utils.yaml import ParamsProcess
from env.utils.math import scale_transform
import pandas as pd
from collections import deque
from isaacgym.torch_utils import *
import matplotlib.pyplot as plt

import collections
from rl.alg import PPO

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def play(args):
    device = args.rl_device
    exp_dir = join('experiments', args.name)

    model_dir = join(exp_dir, 'model')
    debug_dir = join(exp_dir, 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    if args.cmp_real:
        debug_real_dir = debug_dir  # join(debug_dir, 'real')
        os.makedirs(debug_real_dir, exist_ok=True)
        xl = pd.read_csv(join(debug_real_dir, 'general.txt'),
                         sep='\t+', header=None, engine='python').values[1:, :].astype(float)
        joint_act_real = xl[:, :19]
        joint_pos_real = xl[:, 19:38]
        joint_vel_real = xl[:, 38:57]
        # joint_tau_real = kp_real * (joint_act_real - joint_pos_real) - kd_real * joint_vel_real
        ts_real = np.linspace(0, len(joint_act_real[:, [0]]), len(
            joint_act_real[:, [0]]))  # .reshape(1, -1)
    if args.video:
        args.render = True
        pic_folder = os.path.join(debug_dir, 'picture')
        folder = os.path.exists(pic_folder)
        if not folder:
            os.makedirs(pic_folder)
    paramsProcess = ParamsProcess()
    params = paramsProcess.read_param(join(model_dir, 'cfg.yaml'))
    cfg = getattr(importlib.import_module(
        '.'.join(['config', params['env']['cfg']])), 'H1Config')
    cfg = paramsProcess.dict2class(cfg, params)
    cfg.env.num_envs = min(cfg.env.num_envs, 1)
    cfg.terrain.mesh_type = 'plane'

    cfg.terrain.num_rows = 5
    cfg.terrain.num_cols = 5

    # set_seed(cfg.runner.seed)
    # set_seed(seed=None)
    set_seed(seed=3985)  # 3985
    cfg.env.episode_length_s = args.time
    cfg.domain_rand.push_robots = False
    # cfg.terrain.mesh_type = 'trimesh'
    cfg.noise_values.randomize_noise = False
    # 修改
    cfg.init_state.random_rot = False  # 关闭随机旋转，避免初始倾斜导致掉落
    cfg.init_state.pos[2] = 1.065  # 降低初始高度
    #结束
    sim_params = parse_sim_params(args, class_to_dict(cfg.sim))
    env = LeggedRobotEnv(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=args.physics_engine,
                         sim_device=args.sim_device,
                         render=args.render,
                         debug=args.debug,
                         fix_cam=args.fix_cam,
                         tcn_name=args.tcn,
                         epochs=args.epochs)
    task = load_task_cls(cfg.env.task)(env)

    cfg_dict = collections.OrderedDict()
    cfg_dict.update(paramsProcess.class2dict(cfg))
    cfg_dict['policy'].update({'num_observations': task.num_observations,
                               'num_actions': task.num_actions})

    gym_env = GymEnvWrapper(env, task, debug=args.debug)
    task.num_observations = len(gym_env.task.pure_observation()[
                                0]) * gym_env.task.obs_history.maxlen
    task.num_actions = len(gym_env.task.action_low)

    actor = load_actor(class_to_dict(cfg.policy), device).train()
    # args.iter = 2000
    if args.iter is not None:
        saved_model_state_dict = torch.load(
            join(model_dir, f'all/policy_{args.iter}.pt'))
    else:
        saved_model_state_dict = torch.load(join(model_dir, 'policy.pt'))
    actor.load_state_dict(saved_model_state_dict['actor'])
    critic = load_critic(cfg_dict['policy'], device).train()
    alg = PPO(actor, critic, device=device, **class_to_dict(cfg.algorithm))
    alg.init_storage(env.num_envs, cfg.runner.num_steps_per_env, [len(gym_env.task.pure_observation()[0])],
                     [task.num_observations], [task.num_actions])

    print(f'--------------------------------------------------------------------------------------')
    print(f'Start to evaluate policy `{exp_dir}`.')

    # loaded_rl_data = pd.read_csv(join(join(exp_dir, 'real'), 'rl.txt'), sep='\t+', header=None, engine='python').values[1:, :].astype(float)
    # rl_data_list = to_torch(np.array(loaded_rl_data), dtype=torch.float, device=device, requires_grad=False)
    live_time_count = []
    from rl.storage import Transition
    transition = Transition()
    for epoch in range(args.epochs):
        print(
            f'#The `{epoch + 1}st/(total {args.epochs} times)` rollout......................................')

        obs, cri_obs, obs_real, estimation_value = gym_env.reset(torch.arange(env.num_envs, device=device).detach(),
                                                                 torch.arange(env.num_envs, device=device).detach())
        obs = obs.type(torch.float32)
        cri_obs = cri_obs.type(torch.float32)
        for i in range(int(args.time / (cfg.sim.dt * cfg.pd_gains.decimation))):
            with torch.inference_mode():
                # act = actor(rl_data_list[[i], 16:])['act'].detach().clone()
                if i % cfg.runner.num_steps_per_env == 0:
                    alg.storage.clear()
                act = actor(obs)['act'].detach().clone()

                transition.observations = obs
                transition.observations_real = obs_real
                transition.estimation_value = estimation_value
            if args.cmp_real:
                obs, cri_obs, rew, done, info = gym_env.step(act,
                                                             to_torch(joint_act_real[i],
                                                                      device=device).repeat((cfg.env.num_envs, 1)))
            else:
                obs, cri_obs, rew, done, info, obs_real, estimation_value, reset_phy_ids = gym_env.step(
                    act, i)
                transition.clear()

            if args.video:
                if i >= 0:
                    filename = os.path.join(pic_folder, f"{i}.png")
                    env.gym.write_viewer_image_to_file(env.viewer, filename)
            if any(done):
                break
        print(
            f'Evaluation finished after {i} timesteps ({i * gym_env.env.dt:.1f} seconds).')
        if args.epochs > 1:
            live_time_count.append(i)
            if epoch == (args.epochs - 1):
                data_path = os.path.join(debug_dir, f'live_time.xlsx')
                with pd.ExcelWriter(data_path) as f:
                    pd.DataFrame({'live_time': live_time_count}).to_excel(
                        f, sheet_name='live time', index=False)
                    print(
                        f'#The live count data has been written into `{data_path}`.')
        if args.debug:
            joint_act, joint_pos, joint_vel = gym_env.save_debug_data(
                debug_dir)
            if args.cmp_real:
                num = min(len(joint_act[0, :]) - 1, len(ts_real))
                for j in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                    for i, motor_id in enumerate([j]):
                        plt.plot(ts_real[:num], joint_act[0,
                                 :num, motor_id], linestyle='-.', c='k')
                        plt.plot(ts_real[:num], joint_pos[0,
                                 :num, motor_id], linestyle='-', c='b')
                        plt.plot(
                            ts_real[:num], joint_act_real[:num, motor_id], linestyle=':', c='k')
                        plt.plot(
                            ts_real[:num], joint_pos_real[:num, motor_id], linestyle='-', c='r')
                        plt.title(gym_env.env.dof_names[j] + ':' + 'joint_pos')
                        plt.legend(
                            ['sim_act', 'sim_pos', 'real_act', 'real_pos'])
                        plt.grid()
                        plt.show()
                for j in [0, 1, 2, 3, 4]:
                    for i, motor_id in enumerate([j]):
                        plt.plot(ts_real[:num], joint_vel[0,
                                 :num, motor_id], linestyle='-', c='b')
                        plt.plot(
                            ts_real[:num], joint_vel_real[:num, motor_id], linestyle='-', c='r')
                        plt.title(gym_env.env.dof_names[j] + ':' + 'joint_vel')
                        plt.legend(['sim_vel', 'real_vel'])
                        plt.grid()
                        plt.show()
        if args.video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # low quality mp4
            # fourcc = cv2.VideoWriter_fourcc('X', '2', '6', '4')
            cap_fps = int(1 / (cfg.sim.dt * cfg.pd_gains.decimation))
            video_path = join(debug_dir, f'{args.name}.mp4')
            video = cv2.VideoWriter(video_path, fourcc, cap_fps, (1600, 900))
            file_lst = os.listdir(pic_folder)
            file_lst.sort(key=lambda x: int(x[:-4]))
            for filename in file_lst:
                img = cv2.imread(join(pic_folder, filename))
                video.write(img)
            video.release()
            shutil.rmtree(pic_folder)
            print(f'#The video has been saved into `{video_path}`.')
    print(f'--------------------------------------------------------------------------------------')


if __name__ == '__main__':
    args = get_args()
    play(args)
