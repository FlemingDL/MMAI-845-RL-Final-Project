"""
SARSA code from: PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 4: Temporal Difference and Q-Learning
Author: Yuxi (Hayden) Liu
"""

import torch
import gym
import gym_fleming
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
from gym import wrappers
import os
import pandas as pd
import csv
import numpy as np


def gen_epsilon_greedy_policy(n_action, epsilon):
    def policy_function(state, Q):
        probs = torch.ones(n_action) * epsilon / n_action
        best_action = torch.argmax(Q[state]).item()
        probs[best_action] += 1.0 - epsilon
        action = torch.multinomial(probs, 1).item()
        return action
    return policy_function


def sarsa(env, gamma, n_episode, alpha):
    """
    Obtain the optimal policy with on-policy SARSA algorithm
    @param env: OpenAI Gym environment
    @param gamma: discount factor
    @param n_episode: number of episodes
    @return: the optimal Q-function, and the optimal policy
    """
    n_action = env.action_space.n
    Q = defaultdict(lambda: torch.zeros(n_action))
    for episode in tqdm(range(n_episode)):
        state = env.reset()
        is_done = False
        action = epsilon_greedy_policy(state, Q)
        while not is_done:
            next_state, reward, is_done, info = env.step(action)
            next_action = epsilon_greedy_policy(next_state, Q)
            td_delta = reward + gamma * Q[next_state][next_action] - Q[state][action]
            Q[state][action] += alpha * td_delta
            length_episode[episode] += 1
            total_reward_episode[episode] += reward
            if is_done:
                break
            state = next_state
            action = next_action
    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()
    return Q, policy

def test_policy(env, n_episode, policy):
    """
    Test the agent
    @param env: OpenAI Gym environment
    @param n_episode: number of episodes
    @param policy: the policy to follow
    """
    for episode in tqdm(range(n_episode)):
        # print('###################### Episode {} ######################'.format(episode))
        state = env.reset()
        is_done = False
        while not is_done:
            action = policy[state]
            next_state, reward, is_done, info = env.step(action)
            length_test_episode[episode] += 1
            total_reward_test_episode[episode] += reward
            # print('Reward: {}'.format(reward))
            if (length_test_episode[episode] >= max_episode_steps):
                env.stats_recorder.save_complete()
                env.stats_recorder.done = True
                break
            if is_done:
                break
            state = next_state
    return


environments = [gym.make('Taxi-v3'), gym.make('taxi_fleming-10x10-v0'),
                gym.make('taxi_fleming-15x15-v0'), gym.make('taxi_fleming-20x20-v0')]
save_to_dirs = ['./sarsa_outputs_5x5/', './sarsa_outputs_10x10/',
                './sarsa_outputs_15x15/', './sarsa_outputs_20x20/']

for i, environment in enumerate(environments):

    env = environment
    save_to_dir = save_to_dirs[i]
    print('working on: {}'.format(save_to_dir))

    video_dir = os.path.join(save_to_dir, 'video')
    env = gym.wrappers.Monitor(env, video_dir, force=True)
    # env = gym.wrappers.Monitor(env, video_dir, video_callable=lambda episode_id: True, force=True)
    n_episode = 2000
    length_episode = [0] * n_episode
    total_reward_episode = [0] * n_episode

    gamma = 1
    alpha = 0.4
    epsilon = 0.1
    epsilon_greedy_policy = gen_epsilon_greedy_policy(env.action_space.n, epsilon)

    optimal_Q, optimal_policy = sarsa(env, gamma, n_episode, alpha)
    env.close()

    # Plot the length and total reward for each episode over time
    plt.plot(length_episode)
    plt.title('Episode length over time')
    plt.xlabel('Episode')
    plt.ylabel('Length')
    # plt.show()
    plt.savefig(os.path.join(save_to_dir, "episode_length.png"))
    plt.clf()

    plt.plot(total_reward_episode)
    plt.title('Episode reward over time')
    plt.xlabel('Episode')
    plt.ylabel('Total reward')
    # plt.show()
    plt.savefig(os.path.join(save_to_dir, "episode_rewards.png"))
    plt.clf()

    # Export data
    pd.DataFrame(length_episode).to_csv(os.path.join(save_to_dir, "length_episode.csv"))
    pd.DataFrame(total_reward_episode).to_csv(os.path.join(save_to_dir, "total_reward_episode.csv"))
    w = csv.writer(open(os.path.join(save_to_dir, "optimal_policy.csv"), "w"))
    w.writerow(['taxi_row', 'taxi_col', 'pass_loc', 'dest_idx', 'action'])
    for key, val in optimal_policy.items():
        to_write = list(env.decode(key))
        to_write.append(val)
        to_write = np.array(to_write)
        w.writerow(np.array(list(to_write)))

    # Test the policy
    env = environment
    video_dir = os.path.join(save_to_dir, 'trained_agent_videos')
    env = gym.wrappers.Monitor(env, video_dir, video_callable=lambda episode_id: True, force=True)
    n_episode = 100
    length_test_episode = [0] * n_episode
    total_reward_test_episode = [0] * n_episode

    max_episode_steps = 8000  # Set the max episode length in case the agent get stuck

    test_policy(env, n_episode, optimal_policy)

    pd.DataFrame(length_test_episode).to_csv(os.path.join(save_to_dir, "trained_length_episode.csv"))
    pd.DataFrame(total_reward_test_episode).to_csv(os.path.join(save_to_dir, "trained_total_reward_episode.csv"))