import os
from myosuite.envs.env_base import MujocoEnv
from myosuite.utils import gym; register=gym.register
import numpy as np


class Kinova(MujocoEnv):
    def __init__(self, **kwargs):
        model_path = os.path.join("myosuite", "envs", "myo", "assets", "kinova", "robot_arm", "kinova.xml")
        super().__init__(model_path=model_path, **kwargs)
        self._setup(
            obs_keys={"time": 1}, 
            weighted_reward_keys={},
        )

    def get_obs_dict(self, sim):
        return {
            "time": np.array([sim.data.time]), 
        }

    def get_reward_dict(self, obs_dict):
        return {
            "dense": 0.0,
            "sparse": 0.0,
            "solved": 0.0,
            "done": 0.0,
        }
    
    def render(self):
        self.mj_render()

