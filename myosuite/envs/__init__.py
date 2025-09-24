from myosuite.utils import gym; register=gym.register
import os
import numpy as np

curr_dir = os.path.dirname(os.path.abspath(__file__))

register(id='Kinova-v0',
    entry_point='myosuite.envs.myo.kinova_env:Kinova',
)

register(id='Kinova-v1',
        entry_point='myosuite.envs.myo.kinova_arm_env:KinovaArm',
        max_episode_steps=1000,
        kwargs={
            'model_path': curr_dir+'/myo/assets/kinova/robot_arm/kinova.xml',
            # 'normalize_act': True,
            # 'frame_skip': 5,
            # 'pos_th': 0.1,              # cover entire base of the receptacle
            # 'rot_th': np.inf,           # ignore rotation errors
            # 'target_xyz_range': {'high':[0.2, -.1, 0.9], 'low':[0.0, -.35, 0.9]},
            # 'target_rxryrz_range': {'high':[0.0, 0.0, 0.0], 'low':[0.0, 0.0, 0.0]}
        }
    )