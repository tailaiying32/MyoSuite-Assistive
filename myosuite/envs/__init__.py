from myosuite.utils import gym; register=gym.register

register(id='Kinova-v0',
    entry_point='myosuite.envs.myo.kinova_env:Kinova',
)