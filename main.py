import pygame
from multi_agent_rps_env import MultiAgentRockPaperScissorsEnv
import numpy as np
import matplotlib.pyplot as plt

def compute_average_reward(rewards_list):
    return sum(rewards_list) / len(rewards_list)

if __name__ == "__main__":
    pygame.init()
    env = MultiAgentRockPaperScissorsEnv(num_actions=5)

    num_episodes = 1000
    episodes_to_print = 50
    epsilon = 0.1
    learning_rate = 0.1
    discount_factor = 0.8

    q_values = [[np.zeros(env.num_actions) for _ in range(env.num_actions)] for _ in range(env.players)]

    episode_rewards = []
    average_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            actions = []
            for i in range(env.players):
                if np.random.uniform(0, 1) < epsilon:
                    action = np.random.choice(env.num_actions)
                else:
                    action = np.argmax(q_values[i][state])
                actions.append(action)

            next_state, rewards, done, _ = env.step(actions)

            for i in range(env.players):
                q_values[i][state][actions[i]] += learning_rate * (
                        rewards[i] + discount_factor * np.max(q_values[i][next_state]) - q_values[i][state][actions[i]])

            state = next_state
            total_reward += sum(rewards)

        episode_rewards.append(total_reward)

        if (episode + 1) % episodes_to_print == 0:
            average_reward = compute_average_reward(episode_rewards[-episodes_to_print:])
            average_rewards.append(average_reward)
            print(f"Episodes {episode + 1 - episodes_to_print + 1}-{episode + 1}: Average Reward {average_reward}")

    env.close()

    # Plot the average rewards over episodes
    plt.plot(range(episodes_to_print, num_episodes + 1, episodes_to_print), average_rewards, marker='o')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Model Growth Over Episodes')
    plt.show()