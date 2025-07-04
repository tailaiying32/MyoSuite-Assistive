# myosuite/envs/my_custom/my_custom_env.py

import os
from myosuite.envs.env_base import MujocoEnv
from myosuite.utils import gym; register=gym.register


class Kinova(MujocoEnv):
    def __init__(self, **kwargs):
        model_path = os.path.join("myosuite", "envs", "myo", "assets", "kinova", "robot_arm", "kinova.xml")
        super().__init__(model_path=model_path, **kwargs)

