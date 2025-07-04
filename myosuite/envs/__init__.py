from myosuite.utils import gym; register=gym.register

register(id='Kinova',
    entry_point='myosuite.envs.myo.kinova_env:Kinova', # where to find the new Environment Class
    # max_episode_steps=200, # duration of the episode
    # kwargs={
    #     'model_path': 'assets/kinova/kinova.xml', # where the xml file of the environment is located
    #     'target_reach_range': {'IFtip': ((0.1, 0.05, 0.20), (0.2, 0.05, 0.20)),}, # this is used in the setup to define the goal e.g. rando position of the team between 0.1 and 0.2 in the x coordinates
    #     'normalize_act': True, # if to use normalized actions using a sigmoid function.
    #     'frame_skip': 5, # collect a sample every 5 iteration step
    # }
)