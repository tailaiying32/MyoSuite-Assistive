from myosuite.utils import gym; register=gym.register

register(id='Kinova',
    entry_point='myosuite.envs.myo.kinova_env:Kinova',
)