from stable_baselines3 import SAC
from myosuite.utils import gym

env = gym.make("Kinova-v0")
model = SAC.load("robot_arm/kinova_sac")

obs, info = env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()