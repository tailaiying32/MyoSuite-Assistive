import os
import collections
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils import gym; register=gym.register
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf


class KinovaArm(BaseV0):
    DEFAULT_OBS_KEYS = ['qpos', 'qvel', 'time']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "sparse": 1.0,
        "dense": 1.0,
    }

    def __init__(self, model_path=None, obsd_model_path=None, seed=None, cfg={}, **kwargs):
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        if model_path is None:
            model_path = os.path.join("myosuite", "envs", "myo", "assets", "kinova", "robot_arm", "kinova.xml")

        self.goal = self.sample_goal()
        self.cfg = cfg

        
        # two step construction is required for pickling to work correctly idk what that means but it's required
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)
        
        self._setup(**kwargs)
        
        
    def _setup(self,
               obs_keys: list = DEFAULT_OBS_KEYS,
               weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
               sites = None,
               frame_skip=1,
               muscle_condition="",
               fatigue_reset_vec=None,
               fatigue_reset_random=False,
               **kwargs):
        super()._setup(
            obs_keys=obs_keys, 
            weighted_reward_keys=weighted_reward_keys,
            sites=sites,
            frame_skip=frame_skip,
            muscle_condition=muscle_condition,
            fatigue_reset_vec=fatigue_reset_vec,
            fatigue_reset_random=fatigue_reset_random,
            **kwargs
        )


    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        self.goal = self.sample_goal()
        return super().reset()

    def sample_goal(self):
        # sample a reachable 3D target in action space
        return np.random.uniform(low=[0.9, 0, 1.8], high=[0.9, 0, 1.8])

    def get_achieved_goal(self):
        # return palm site
        return self.sim.data.site_xpos[self.sim.model.site_name2id('S_grasp')].copy() 

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict["time"] = np.array([sim.data.time])
        obs_dict["qpos"] = sim.data.qpos[:].copy()
        obs_dict["qvel"] = sim.data.qvel[:].copy() * self.dt
        if sim.model.na > 0:
            obs_dict["act"] = sim.data.act[:].copy()

        # goal-conditioned additions
        obs_dict["desired_goal"] = self.goal.copy()
        obs_dict["achieved_goal"] = self.get_achieved_goal()
        return obs_dict

    def get_reward_dict(self, obs_dict):   
        reach_dist = np.linalg.norm(obs_dict["achieved_goal"] - obs_dict["desired_goal"]) ** self.cfg["reward"]["reward_scale"]

        rwd_dict = collections.OrderedDict((
            ('reach_dist', -reach_dist),
            ('dense', -reach_dist),  
            ('sparse', 1.0 if reach_dist < 0.05 else 0.0),        
            ('solved', float(reach_dist < 0.05)),      
            ('done', 0.0), 
        ))
        return rwd_dict

    # render environment    
    def render(self):
        self.mj_render()