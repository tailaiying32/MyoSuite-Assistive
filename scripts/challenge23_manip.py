from myosuite.utils import gym
import myosuite, deprl

env = gym.make("myoChallengeRelocateP1-v0")
policy = deprl.load_baseline(env)

for ep in range(5):
    print(f"Episode: {ep} of 5")
    state = env.reset()[0]
    while True:
        action = policy(state)
        env.mj_render()
        next_state, reward, terminated, truncated, info = env.step(action)
        state = next_state
        if terminated or truncated:
            break