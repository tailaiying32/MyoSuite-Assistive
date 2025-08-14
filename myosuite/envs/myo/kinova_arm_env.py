import os
import collections
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils import gym; register=gym.register
import numpy as np
import wandb


class KinovaArm(BaseV0):
    # Include all keys that will appear in obs_dict every step so the
    # flattened observation size is constant across reset/steps & workers.
    # Order here defines concatenation order in BaseV0/ObsVecDict.
    DEFAULT_OBS_KEYS = ['qpos', 'qvel', 'desired_goal', 'achieved_goal', 'act', 'time']
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


    def reset(self, seed=None, options=None):
        np.random.seed(seed)

        self.goal = self.sample_goal()
        return super().reset()

    def sample_goal(self):
        # sample a reachable 3D target in action space
        self.sim.model.body_pos[self.sim.model.body_name2id("cube")] = np.random.uniform(low=[0.7, -0.4, 0.7], high=[2.0, 0.75, 2.0])
        return self.sim.model.body_pos[self.sim.model.body_name2id("cube")]


    def get_achieved_goal(self):
        # return palm site
        return self.sim.data.body_xpos[self.sim.model.body_name2id('firstmc')].copy() 

    def get_obs_dict(self, sim):
        obs_dict = {}
        # core kinematics
        obs_dict["qpos"] = sim.data.qpos[:].copy()
        obs_dict["qvel"] = sim.data.qvel[:].copy() * self.dt
 
        # ensure goal exists
        if not hasattr(self, "goal"):
            self.goal = self.sample_goal()
        obs_dict["desired_goal"] = self.goal.copy()
        obs_dict["achieved_goal"] = self.get_achieved_goal()

        # actions (always include for consistent shape; zeros if no actuators)
        if sim.model.na > 0:
            obs_dict["act"] = sim.data.act[:].copy()
        else:
            obs_dict["act"] = np.zeros(0, dtype=np.float32)

        # timestamp last (order controlled by DEFAULT_OBS_KEYS anyway)
        obs_dict["time"] = np.array([sim.data.time])
        return obs_dict

    def get_reward_dict(self, obs_dict):   
        reach_dist = np.linalg.norm(obs_dict["achieved_goal"] - obs_dict["desired_goal"])
        rwd_dict = collections.OrderedDict((
            ('reach_dist', -reach_dist),
            ('dense', -reach_dist),  
            ('sparse', 1.0 if reach_dist < 0.05 else 0.0),        
            ('solved', float(reach_dist < 0.05)),      
            ('done', float(reach_dist < 0.05)), 
        ))
        
        return rwd_dict

    # render environment    
    def render(self):
        self.mj_render()
