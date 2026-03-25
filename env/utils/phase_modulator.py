from math import sin, pi, tau
import numpy as np
from isaacgym.torch_utils import to_torch
import torch


class PhaseModulator:
    def __init__(self, time_step, num_envs, num_legs, device):
        self.num_legs = num_legs
        self._phase = torch.zeros(num_envs, num_legs, dtype=torch.float, device=device, requires_grad=False)
        self._frequency = torch.ones(num_envs, num_legs, dtype=torch.float, device=device, requires_grad=False) * 0.5
        self._time_step = time_step
        self.device = device
        self.reset(env_ids=torch.arange(num_envs))

    def reset(self, convert_phi=pi, env_ids=None, render=False):
        if render:
            init_phase = to_torch([[0. for _ in range(self.num_legs)]], device=self.device)
        else:
            init_phase = to_torch([[np.random.uniform(0, 2 * pi) for _ in range(self.num_legs)]], device=self.device)
        self._phase[env_ids] = init_phase % tau
        self._frequency[env_ids] = torch.ones(len(env_ids), self.num_legs, dtype=torch.float, device=self.device, requires_grad=False) * 0.5

    def compute(self, frequency):
        self._frequency = frequency
        self._phase = (self._phase + tau * frequency * self._time_step) % tau
        return self._phase

    @property
    def frequency(self):
        return self._frequency

    @property
    def phase(self):
        return self._phase
