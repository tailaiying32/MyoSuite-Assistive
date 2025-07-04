from myosuite.utils import gym

env = gym.make("Kinova")
env.reset()

for _ in range(1000):
    env.step(env.action_space.sample())
    env.render()
