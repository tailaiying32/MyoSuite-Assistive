from stable_baselines3 import SAC
from myosuite.utils import gym
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

TOTAL_TIMESTEPS = 100000

class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.pbar = tqdm(total=total_timesteps, desc="Training Progress")
        self.last_num_timesteps = 0

    def _on_step(self) -> bool:
        steps = self.num_timesteps - self.last_num_timesteps
        self.pbar.update(steps)
        self.last_num_timesteps = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
        self.pbar.close()


env = gym.make("Kinova-v0")

model = SAC("MlpPolicy", env, verbose=1, device="cuda")

model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=TqdmCallback(TOTAL_TIMESTEPS))

model.save("robot_arm/kinova_sac")