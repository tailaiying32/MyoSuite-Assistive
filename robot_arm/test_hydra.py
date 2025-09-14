import hydra
from omegaconf import DictConfig, OmegaConf
from myosuite.envs.myo.kinova_arm_env import KinovaArm
from pprint import pprint
import json

@hydra.main(config_path="../robot_arm", config_name="train.yaml")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    env_args = config.get("env_args", {})
    model_path = "/home/tty6/MyoSuite-Assistive/myosuite/envs/myo/assets/kinova/robot_arm/kinova.xml"
    env = KinovaArm(cfg=config, model_path=model_path)
    print(json.dumps(env.cfg, indent=2))
    # Optionally, check a specific value
    assert env.cfg["reward"]["reward_scale"] == config["reward"]["reward_scale"]

if __name__ == "__main__":
    main()