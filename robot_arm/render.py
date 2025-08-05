from stable_baselines3 import SAC
from myosuite.utils import gym
import numpy as np
import time

def test():
    model = SAC.load("robot_arm/kinova_sac")
    env = gym.make("Kinova-v1")
    
    for episode in range(10):
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result
        current_goal = env.goal.copy()
        
        print(f"\nEpisode {episode + 1}")
        print(f"Target position: [{current_goal[0]:.2f}, {current_goal[1]:.2f}, {current_goal[2]:.2f}]")
        
        episode_reward = 0
        done = False
        step_count = 0
        
        while not done and step_count < 200:
            # get action from trained policy
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
            episode_reward += reward
            step_count += 1
            
            # render the environment
            env.render()

            # print progress every 50 steps
            if step_count % 50 == 0:
                current_pos = env.get_achieved_goal()
                distance = np.linalg.norm(current_pos - current_goal)
                print(f"  Step {step_count}: Distance to goal = {distance:.3f}")
        
        # final results
        final_pos = env.get_achieved_goal()
        final_distance = np.linalg.norm(final_pos - current_goal)
        success = final_distance < 0.05
        
        if success:
            print("Goal reached!")
        else:
            print("Goal not reached")
        
        time.sleep(2)
    
    env.close()

if __name__ == "__main__":
    test()