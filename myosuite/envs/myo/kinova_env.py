import os
import collections
from myosuite.envs.env_base import MujocoEnv
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils import gym; register=gym.register
import numpy as np


class Kinova(BaseV0):
    DEFAULT_OBS_KEYS = ['qpos', 'qvel', 'time', 'gripper_pos', 'cube_pos', 'reach_err']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "sparse": 1.0,
        "dense": 1.0,
    }

    def __init__(self, model_path=None, obsd_model_path=None, seed=None, **kwargs):
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        if model_path is None:
            model_path = os.path.join("myosuite", "envs", "myo", "assets", "kinova", "robot_arm", "kinova.xml")
        
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

    # get observation vector
    def get_obs_vec(self):
        self.obs_dict['time'] = np.array([self.sim.data.time])
        self.obs_dict['qpos'] = self.sim.data.qpos[:].copy()
        self.obs_dict['qvel'] = self.sim.data.qvel[:].copy() * self.dt
        if self.sim.model.na > 0:
            self.obs_dict['act'] = self.sim.data.act[:].copy()

        self.obs_dict['gripper_pos'] = self.sim.data.site_xpos[self.sim.model.site_name2id('pinch_site')].copy()
        self.obs_dict['cube_pos'] = self.sim.data.body_xpos[self.sim.model.body_name2id('cube')].copy()
        self.obs_dict['reach_err'] = self.obs_dict['gripper_pos'] - self.obs_dict['cube_pos']


        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    # get observation dictionary
    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict["time"] = np.array([sim.data.time])
        obs_dict["qpos"] = sim.data.qpos[:].copy()
        obs_dict["qvel"] = sim.data.qvel[:].copy() * self.dt
        if sim.model.na > 0:
            obs_dict["act"] = sim.data.act[:].copy()
        
        obs_dict['gripper_pos'] = sim.data.site_xpos[sim.model.site_name2id('pinch_site')].copy()
        obs_dict['cube_pos'] = sim.data.body_xpos[sim.model.body_name2id('cube')].copy()
        obs_dict['reach_err'] = obs_dict['gripper_pos'] - obs_dict['cube_pos']

        # print(sim.data.body_xpos[sim.model.body_name2id('cube')].copy())

        return obs_dict

    # get reward dictionary, define reward
    def get_reward_dict(self, obs_dict):
        # get gripper and cube position
        gripper_pos = self.sim.data.site_xpos[self.sim.model.site_name2id('pinch_site')]
        cube_pos = self.sim.data.body_xpos[self.sim.model.body_name2id('cube')]

        # calculate distance from gripper to cube
        reach_dist = np.linalg.norm(gripper_pos - cube_pos)

        rwd_dict = collections.OrderedDict((
            # negative reach dist means closer to cube
            ('reach_dist', -reach_dist),
            ('dense', -reach_dist),  
            ('sparse', 1.0 if reach_dist < 0.05 else 0.0),        
            ('solved', reach_dist < 0.05),      
            ('done', reach_dist < 0.05 or self.sim.data.time > 10.0), 
        ))
        
        return rwd_dict

    # render environment    
    def render(self):
        self.mj_render()
