import gym
import air_gym

env = gym.make('air_gym:airsim-drone-v0',ip_address='10.24.197.27', control_type='continuous',step_length=1, image_shape=(608,608,3), goal=[20,20,-20])

#env.render()
episodes = 10
steps = 20
episode_rewards = []
for ep in range(episodes):
    ep_reward = 0
    print("Episode {}".format(ep))
    for s in range(steps):
        act = env.action_space.sample()
        obs, reward, done, state = env.step(act)
        if done:
            break
        ep_reward += reward
    episode_rewards.append(ep_reward)
    env.reset()
#   env.render()
