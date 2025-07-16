from myosuite.utils import gym

env = gym.make("Kinova-v0")
print(env.action_space)
env.reset()

for _ in range(100000):
    env.step(env.action_space.sample())
    env.render()
