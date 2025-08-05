from myosuite.utils import gym; register=gym.register

register(id='Kinova-v0',
    entry_point='myosuite.envs.myo.kinova_env:Kinova',
)

register(id='Kinova-v1',
    entry_point='myosuite.envs.myo.kinova_arm_env:KinovaArm',
)