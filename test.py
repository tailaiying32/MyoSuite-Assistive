from myosuite.utils import gym

env = gym.make("Kinova-v0")
print("action space:" , env.action_space)
env.reset()

for _ in range(10000):
    env.step(env.action_space.sample())
    # env.step([0.1] * 70)
    env.render()
