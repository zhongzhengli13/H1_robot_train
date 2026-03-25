# -*- coding: utf-8 -*-
import numpy as np
from typing import Dict
import yaml
from .attrdict import AttrDict
from config.loc import H1Config


class ParamsProcess:
    def _parse_param(self, param: dict):
        _param = {}
        for k, v in param.items():
            if isinstance(v, AttrDict):
                v = self._parse_param(v.to_dict())
            elif isinstance(v, dict):
                v = self._parse_param(v)
            elif isinstance(v, np.ndarray):
                v = v.tolist()
            elif isinstance(v, tuple):
                v = list(v)
            _param[k] = v
        return _param

    def write_param(self, file, param: Dict):
        with open(file, 'w', encoding='utf-8') as f:
            yaml.dump(self._parse_param(param), f, default_flow_style=False, allow_unicode=True)

    def read_param(self, file) -> Dict:
        with open(file, 'r', encoding='utf-8') as f:
            param = yaml.load(f, Loader=yaml.FullLoader)
        return param

    def update_param(self, file, param: Dict) -> Dict:
        _param = self.read_param(file)
        _param.update(param)
        self.write_param(file, _param)
        return _param

    def class2dict(self, obj):
        dict, tmp_dict = {}, {}
        for name, value in vars(obj).items():
            if not name.startswith('__') and name not in ['sim', 'asset', 'viwer']:
                for sub_name, sub_value in vars(value).items():
                    if not sub_name.startswith('__'):
                        tmp_dict[sub_name] = sub_value
                dict[name] = tmp_dict.copy()
            tmp_dict.clear()
        return dict

    def dict2class(self, cfg: H1Config, params: Dict):
        cfg.runner.set_dict(cfg.runner, dict=params['runner'])
        cfg.policy.set_dict(cfg.policy, dict=params['policy'])
        cfg.env.set_dict(cfg.env, dict=params['env'])
        cfg.terrain.set_dict(cfg.terrain, dict=params['terrain'])
        cfg.init_state.set_dict(cfg.init_state, dict=params['init_state'])
        cfg.pd_gains.set_dict(cfg.pd_gains, dict=params['pd_gains'])
        cfg.action.set_dict(cfg.action, dict=params['action'])
        cfg.domain_rand.set_dict(cfg.domain_rand, dict=params['domain_rand'])
        cfg.command.set_dict(cfg.command, dict=params['command'])
        return cfg
