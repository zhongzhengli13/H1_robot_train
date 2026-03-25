import importlib
import os
from os.path import join

from env.utils import get_args
from env.utils.helpers import update_cfg_from_args, class_to_dict, set_seed, parse_sim_params
from env import LeggedRobotEnv, GymEnvWrapper
from env.tasks import load_task_cls
from model import load_actor, load_critic
from rl.alg import PPO
from env.utils.math import scale_transform
import time
from collections import deque
import collections
import statistics
from utils.common import clear_dir
from utils.yaml import ParamsProcess
from isaacgym.torch_utils import *
from torch.utils.tensorboard import SummaryWriter
import torch

# os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def train():
    torch.cuda.empty_cache()
    args = get_args()
    device = args.rl_device
    # cfg = getattr(importlib.import_module(args.config), 'H1Config')
    cfg = getattr(importlib.import_module('.'.join(['config', args.config])), 'H1Config')
    cfg = update_cfg_from_args(cfg, args)
    exp_dir = join('experiments', args.name)
    model_dir = join(exp_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    all_model_dir = join(exp_dir, 'model', 'all')
    os.makedirs(all_model_dir, exist_ok=True)
    log_dir = join(exp_dir, 'log')
    clear_dir(log_dir)
    writer = SummaryWriter(log_dir, flush_secs=10)
    num_steps_per_env = cfg.runner.num_steps_per_env
    num_learning_iterations = cfg.runner.max_iterations
    # set_seed(cfg.runner.seed)
    # set_seed(seed=None)
    set_seed(seed=8091)  # 3985 # 8091

    sim_params = parse_sim_params(args, class_to_dict(cfg.sim))
    env = LeggedRobotEnv(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=args.physics_engine,
                         sim_device=args.sim_device,
                         render=args.render,
                         fix_cam=args.fix_cam)
    task = load_task_cls(cfg.env.task)(env)
    gym_env = GymEnvWrapper(env, task)
    # dim_z = cfg.algorithm.dim_z
    task.num_observations = len(gym_env.task.pure_observation()[0]) * gym_env.task.obs_history.maxlen
    task.num_actions = len(gym_env.task.action_low)#-9


    cfg_dict = collections.OrderedDict()
    paramProcess = ParamsProcess()
    cfg_dict.update(paramProcess.class2dict(cfg))
    cfg_dict['policy'].update({'num_observations': task.num_observations,
                               'num_actions': task.num_actions,
                               'num_critic_obs': len(gym_env.task.pure_observation()[0])})
    cfg_dict['action'].update(
        {'action_limit_low': cfg.action.low_ranges[2:], 'action_limit_up': cfg.action.high_ranges[2:]})
    paramProcess.write_param(join(model_dir, "cfg.yaml"), cfg_dict)
    actor = load_actor(cfg_dict['policy'], device).train()
    critic = load_critic(cfg_dict['policy'], device).train()
    alg = PPO(actor, critic, device=device, **class_to_dict(cfg.algorithm))
    alg.init_storage(env.num_envs, num_steps_per_env, [len(gym_env.task.pure_observation()[0])],
                     [task.num_observations], [task.num_actions])
    # args.resume = 'test10'
    if args.resume is not None:
        resume_model_dir = join(join('experiments', args.resume), 'model')
        saved_model_state_dict = torch.load(join(resume_model_dir, 'policy.pt'))
        alg.actor.load_state_dict(saved_model_state_dict['actor'])
        alg.critic.load_state_dict(saved_model_state_dict['critic'])
        alg.optimizer1.load_state_dict(saved_model_state_dict['optimizer1'])
        current_learning_iteration = saved_model_state_dict['iteration']
        print(f"成功加载模型！旧迭代次数为: {current_learning_iteration}") # 添加这一行
    else:
        current_learning_iteration = 1


    total_time, total_timesteps = 0., 0
    total_iteration = current_learning_iteration + num_learning_iterations
    rew_buffer, len_buffer = deque(maxlen=100), deque(maxlen=100)
    cur_reward_sum = torch.zeros(env.num_envs, dtype=torch.float, device=device)
    cur_episode_length = torch.zeros(env.num_envs, dtype=torch.float, device=device)

    obs, cri_obs, obs_real, estimation_value = gym_env.reset(torch.arange(env.num_envs, device=device),
                                                             torch.arange(env.num_envs, device=device))
    for it in range(current_learning_iteration, total_iteration):
        env.cfg.runner.epoch = it
        start = time.time()
        for i in range(num_steps_per_env):
            act = alg.act(obs, cri_obs, obs_real, estimation_value)
            obs, cri_obs, rew, done, info, obs_real, estimation_value, reset_phy_ids = gym_env.step(act, i)
            alg.process_env_step(rew, done, info)
            cur_reward_sum += rew
            cur_episode_length += 1
            reset_env_ids = (done > 0).nonzero(as_tuple=False)[:, [0]].flatten()
            if len(reset_env_ids) > 0:
                rew_buffer.extend(cur_reward_sum[reset_env_ids].cpu().numpy().tolist())
                len_buffer.extend(cur_episode_length[reset_env_ids].cpu().numpy().tolist())
                cur_reward_sum[reset_env_ids] = 0
                cur_episode_length[reset_env_ids] = 0
        alg.compute_returns(cri_obs)
        stop = time.time()
        collection_time = stop - start
        start = stop
        mean_value_loss, mean_surrogate_loss, mean_kl, los1,los2 = alg.update()
        saved_model_state_dict = {
            'actor': alg.actor.state_dict(),
            'critic': alg.critic.state_dict(),
            'optimizer1': alg.optimizer1.state_dict(),
            # 'optimizer2': alg.optimizer2.state_dict(),
            'iteration': current_learning_iteration,
        }
        try:
            torch.save(saved_model_state_dict, join(model_dir, 'policy.pt'))
        except OSError as e:
            print('Failed to save policy.')
            print(e)
        if it % cfg.runner.save_interval == 0:
            try:
                torch.save(saved_model_state_dict, join(all_model_dir, f'policy_{it}.pt'))
            except OSError as e:
                print('Failed to save policy.')
                print(e)
        stop = time.time()
        learn_time = stop - start
        iteration_time = collection_time + learn_time
        total_time += iteration_time
        total_timesteps += num_steps_per_env * env.num_envs
        fps = int(num_steps_per_env * env.num_envs / iteration_time)
        mean_std = alg.actor.std.mean()
        mean_reward = statistics.mean(rew_buffer) if len(rew_buffer) > 0 else 0.
        mean_episode_length = statistics.mean(len_buffer) if len(len_buffer) > 0 else 0.
        writer.add_scalar('1:Train/mean_reward', mean_reward, it)
        writer.add_scalar('1:Train/mean_episode_length', mean_episode_length, it)
        writer.add_scalar('1:Train/mean_episode_time', mean_episode_length * gym_env.env.dt, it)
        writer.add_scalar('2:Loss/value', mean_value_loss, it)
        writer.add_scalar('2:Loss/surrogate', mean_surrogate_loss, it)
        writer.add_scalar('2:Loss/learning_rate', alg.learning_rate, it)
        writer.add_scalar('2:Loss/mean_kl', mean_kl, it)
        writer.add_scalar('2:Loss/mean_noise_std', mean_std.item(), it)
        writer.add_scalar('3:Perf/total_fps', fps, it)
        writer.add_scalar('3:Perf/collection_time', collection_time, it)
        writer.add_scalar('3:Perf/learning_time', learn_time, it)

        print(f"{args.name} #{it}: ",
              f"{'time'} {total_time / 60:.1f}m({iteration_time:.1f}s)",
              f"col {collection_time:.2f}s",
              f"lrn {learn_time:.2f}s",
              f"st_nm {fps:.0f}",
              f"mn_kl {mean_kl:.3f}",
              f"{'v_lss:'} {mean_value_loss:.3f}",
              f"{'a_lss:'} {mean_surrogate_loss:.3f}",
              f"len_t {mean_episode_length * gym_env.env.dt:.2f}s",
              f"len_l {int(mean_episode_length)}",
              f"rew {mean_reward:.2f}",
              # f"los1 {los1:.2f}",
              # f"los2 {los2:.2f}",
              sep='  ')


if __name__ == '__main__':
    train()
