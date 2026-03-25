from env.legged_robot import LeggedRobotEnv
import torch

TASKS = {}


def register(cls):
    cls.name = cls.__name__
    assert cls.name not in TASKS, cls.name
    TASKS[cls.name] = cls
    return cls


def check(cls):
    assert any([cls == v for v in TASKS.values()]), \
        f"Please register the task:'{cls.name}'"


def load_task_cls(name):
    if name in TASKS:
        return TASKS[name]
    else:
        raise KeyError(f'Not exist task named {name}.')


class BaseTask:
    name: str = 'test'

    def __init__(self, env: LeggedRobotEnv):
        check(self.__class__)
        self.env = env
        self.cfg = env.cfg
        self.debug = None
        self.device = env.device
        self.extra_info = {}
        self.num_observations = self.cfg.policy.num_observations
        self.num_actions = self.cfg.policy.num_actions
        self.command_boundary = 0.2
        self.num_envs = env.num_envs

        self.commands = torch.zeros(self.num_envs, self.cfg.command.num_commands, dtype=torch.float, device=self.device,
                                    requires_grad=False)  # x vel, y vel, yaw vel, heading
        self.static_flag = torch.where(torch.norm(self.commands[:, :3], dim=1, keepdim=True) < 0.11, False,
                                       True).float()

    def step(self):
        raise NotImplementedError

    def reset(self, env_ids):
        raise NotImplementedError

    def observation(self):
        raise NotImplementedError

    def action(self, net_out):
        raise NotImplementedError

    def reward(self, target_pos=None, target_vel=None):
        raise NotImplementedError

    def terminate(self):
        raise NotImplementedError

    def info(self):
        return self.extra_info
