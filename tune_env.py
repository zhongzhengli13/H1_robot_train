import importlib
import os
from os.path import join

from env.utils import get_args
from env.utils.helpers import update_cfg_from_args, class_to_dict, parse_sim_params
from env import LeggedRobotEnv, GymEnvWrapper
from env.tasks import load_task_cls
import collections
from utils.yaml import ParamsProcess
import torch


def run():
    args = get_args()
    args.render = True
    cfg = getattr(importlib.import_module('.'.join(['config', args.config])), 'H1Config')
    cfg = update_cfg_from_args(cfg, args)
    exp_dir = join('experiments', args.name)
    model_dir = join(exp_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    cfg.env.num_envs = min(cfg.env.num_envs, 1)
    cfg.asset.fix_base_link = True
    cfg_dict = collections.OrderedDict()
    paramProcess = ParamsProcess()
    cfg_dict.update(paramProcess.class2dict(cfg))
    paramProcess.write_param(join(model_dir, "cfg.yaml"), cfg_dict)
    sim_params = parse_sim_params(args, class_to_dict(cfg.sim))
    env = LeggedRobotEnv(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=args.physics_engine,
                         sim_device=args.sim_device,
                         render=args.render,
                         fix_cam=args.fix_cam)
    task = load_task_cls(cfg.env.task)(env)
    gym_env = GymEnvWrapper(env, task)
    task.num_observations = len(gym_env.task.pure_observation()[0]) #* 3#gym_env.task.obs_history.maxlen
    task.num_actions = len(gym_env.task.action_low)
    obs, cri_obs, obs_real, estimation_value = gym_env.reset(torch.arange(env.num_envs, device=args.rl_device).detach(),
                                                             torch.arange(env.num_envs, device=args.rl_device).detach())
    while True:
        act = torch.as_tensor([0.] * task.num_actions).repeat(task.num_envs, 1).to(args.rl_device)
        obs, cri_obs, rew, done, info, obs_real, estimation_value, reset_phy_ids = gym_env.step(act, 0)
    gym_env.close()


if __name__ == '__main__':
    run()
