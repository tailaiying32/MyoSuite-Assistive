from myosuite.utils import gym
import numpy as np

env = gym.make("Kinova-v1")
env.reset()
positions = []

try:
    site_id = env.sim.model.site_name2id("S_grasp")
except:
    print("S_grasp site not found. Available sites:")
    for i in range(env.sim.model.nsite):
        print(f"  {i}: {env.sim.model.site_id2name(i)}")
    site_id = 0  

joint_ranges = env.sim.model.jnt_range 

for _ in range(1):
    env.mj_render()

    for _ in range(3000):
        q = [np.random.uniform(low, high) for low, high in joint_ranges]
        env.sim.data.qpos[:] = q
        env.sim.forward()
        
        # Correct way to access site position
        pos = env.sim.data.site_xpos[site_id].copy()
        positions.append(pos)

print(f"Collected {len(positions)} positions")
print("Sample positions:", positions[:5])
min_x = min(positions, key=lambda pos: pos[0])[0]
min_y = min(positions, key=lambda pos: pos[1])[1]
min_z = min(positions, key=lambda pos: pos[2])[2]
max_x = max(positions, key=lambda pos: pos[0])[0]
max_y = max(positions, key=lambda pos: pos[1])[1]
max_z = max(positions, key=lambda pos: pos[2])[2]

print(f"min x: {min_x}")
print(f"min y: {min_y}")
print(f"min z: {min_z}")
print(f"max x: {max_x}")
print(f"max y: {max_y}")
print(f"max z: {max_z}")
env.close()