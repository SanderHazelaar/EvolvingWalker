import numpy as np
import gym


def run_bipedal_walker(agent, simulation_seed=0, n_episodes=1, max_steps=300,
                        graphics=False):
    env = gym.make("BipedalWalker-v3")
    env.seed(simulation_seed)

    reward = 0
    cumulative_reward = 0
    done = False
    step = 0

    for i in range(n_episodes):
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)  # perform action by agent
            ob, reward, done, _ = env.step(action)  # do the action and retrieve next observation and reward
            cumulative_reward += reward
            step += 1
            if (step >= max_steps):
                done = True
            if (graphics):
                env.render()
            if done:
                break

    env.close()

    return cumulative_reward;