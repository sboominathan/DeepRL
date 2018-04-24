#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import pickle
import os
import timeit
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def run_episodes(agent):
    config = agent.config
    window_size = 100
    plot_interval = 25

    ep = 0
    max_episodes = 100 

    rewards = []
    steps = []
    avg_train_rewards = []
    avg_test_rewards = []

    episode_times = []
    avg_episode_times = []

    train_rewards_filename = 'avg_train_rewards_breakout_ddpg_1.png'
    ep_times_filename = 'ep_times_breakout_ddpg_1.png'

    agent_type = agent.__class__.__name__

    while ep < max_episodes:
        ep += 1
        episode_start = timeit.default_timer()
        reward, step = agent.episode()

        episode_duration = timeit.default_timer() - episode_start
        episode_times.append(episode_duration)
        avg_ep_time = np.mean(episode_times[-10:])
        avg_episode_times.append(avg_ep_time)

        rewards.append(reward)
        steps.append(step)
        avg_reward = np.mean(rewards[-window_size:])
        avg_train_rewards.append(avg_reward)

        config.logger.info('episode %d, reward %f, avg reward %f, total steps %d, episode step %d' % (
            ep, reward, avg_reward, agent.total_steps, step))

        if config.save_interval and ep % config.save_interval == 0:
            with open('data/%s-%s-online-stats-%s.bin' % (
                    agent_type, config.tag, agent.task.name), 'wb') as f:
                pickle.dump([steps, rewards], f)

        if config.episode_limit and ep > config.episode_limit:
            break

        if config.max_steps and agent.total_steps > config.max_steps:
            break

        # Plot average training rewards
        if ep % plot_interval == 0:
            plot_rewards(range(ep), avg_train_rewards, train_rewards_filename, ylabel='Avg. Train Rewards/Episode')
            plot_rewards(range(ep), avg_episode_times, ep_times_filename, ylabel='Time/Episode')

        if config.test_interval and ep % config.test_interval == 0:
            config.logger.info('Testing...')
            agent.save('data/%s-%s-model-%s.bin' % (agent_type, config.tag, agent.task.name))
            test_rewards = []
            for _ in range(config.test_repetitions):
                test_rewards.append(agent.episode(deterministic=True)[0])
            avg_reward = np.mean(test_rewards)
            avg_test_rewards.append(avg_reward)
            config.logger.info('Avg reward %f(%f)' % (
                avg_reward, np.std(test_rewards) / np.sqrt(config.test_repetitions)))
            with open('data/%s-%s-all-stats-%s.bin' % (agent_type, config.tag, agent.task.name), 'wb') as f:
                pickle.dump({'rewards': rewards,
                             'steps': steps,
                             'test_rewards': avg_test_rewards}, f)
            if avg_reward > config.success_threshold:
                break

    
    agent.close()
    return steps, rewards, avg_test_rewards

def run_test_episodes(agent, iter_num):
    config = agent.config

    max_episodes = 100
    test_rewards = []
    avg_rewards = []

    for ep in range(max_episodes):
        reward = agent.episode(deterministic=True, record_actions=True)[0]
        test_rewards.append(reward)

        avg_reward = np.mean(test_rewards)
        avg_rewards.append(avg_reward)

    config.logger.info('Avg reward %f(%f)' % (
                avg_reward, np.std(test_rewards) / np.sqrt(max_episodes)))

    # policy_history_filename = 'policy_history_breakout_1_trial_%s.npy' % iter_num
    action_history_filename = 'action_history_breakout_ddpg_1_trial_%s.npy' % iter_num
    # state_policy_dict_filename = 'state_policy_dict_breakout_1_trial_%s.npy' % iter_num

    # agent.save_policy_history("policy_action_data/" + policy_history_filename)
    agent.save_action_history("policy_action_data/" + action_history_filename)
    # agent.save_state_policy_dict("policy_action_data/" + state_policy_dict_filename)

    action_distribution = get_action_proportions(np.array(agent.action_history)) 
    print(action_distribution)
        
    agent.clear_action_history()
    # agent.clear_policy_history()    
    return action_distribution 

def get_action_proportions(action_data):
   actions, counts = np.unique(action_data, return_counts=True)
   action_proportions = {}
   for action, count in zip(actions, counts):
       action_proportions[action] = count/float(action_data.shape[0])
   return action_proportions

def run_iterations(agent):
    config = agent.config
    agent_name = agent.__class__.__name__
    iteration = 0
    max_iters = 1000

    steps = []
    rewards = []
    avg_rewards = []

    train_rewards_filename = 'avg_train_rewards_breakout_a2c_1.png'

    while iteration <= max_iters:
        agent.iteration()
        steps.append(agent.total_steps)
        rewards.append(np.mean(agent.last_episode_rewards))
        if iteration % config.iteration_log_interval == 0:
            config.logger.info('total steps %d, mean/max/min reward %f/%f/%f' % (
                agent.total_steps, np.mean(agent.last_episode_rewards),
                np.max(agent.last_episode_rewards),
                np.min(agent.last_episode_rewards)
            ))
        if iteration % (config.iteration_log_interval * 100) == 0:
            with open('data/%s-%s-online-stats-%s.bin' % (agent_name, config.tag, agent.task.name), 'wb') as f:
                pickle.dump({'rewards': rewards,
                             'steps': steps}, f)
            agent.save('data/%s-%s-model-%s.bin' % (agent_name, config.tag, agent.task.name))

        if iteration % plot_interval == 0:
            plot_rewards(range(iteration), rewards, train_rewards_filename, ylabel='Avg. Train Rewards/Episode')

        iteration += 1


def run_test_iterations(agent, iter_num, max_iters=100):
    config = agent.config
    steps = []
    rewards = []
    avg_rewards = []
    
    for i in range(max_iters):
    	total_reward,  _ = agent.evaluate(record_actions=True)
    	rewards.append(total_reward)
    	avg_reward = np.mean(rewards)
    	avg_rewards.append(avg_reward)

    config.logger.info('Avg reward %f(%f)' % (
                avg_reward, np.std(rewards) / np.sqrt(max_iters)))

    # policy_history_filename = 'policy_history_breakout_1_a2c_trial_%s.npy' % iter_num
    action_history_filename = 'action_history_breakout_1_a2c_trial_%s.npy' % iter_num

    # agent.save_policy_history("policy_action_data/" + policy_history_filename)
    agent.save_action_history("policy_action_data/" + action_history_filename)

    action_distribution = get_action_proportions(np.array(agent.action_history))
    print(action_distribution)

    agent.clear_action_history()
    return action_distribution


def plot_rewards(x, y, filename, ylabel, xlabel='Episode #', color='red'):
    matplotlib.rcParams['figure.figsize'] = (13.0, 8.0)
    matplotlib.rcParams['lines.linewidth'] = 3
    matplotlib.rcParams['axes.grid'] = True
    matplotlib.rcParams['grid.linestyle'] = '--'
    matplotlib.rcParams['grid.color'] = '#aaaaaa'
    matplotlib.rcParams['xtick.major.size'] = 0
    matplotlib.rcParams['ytick.major.size'] = 0
    matplotlib.rcParams['xtick.labelsize'] = 24
    matplotlib.rcParams['ytick.labelsize'] = 24
    matplotlib.rcParams['axes.labelsize'] = 24
    matplotlib.rcParams['axes.titlesize'] = 24
    matplotlib.rcParams['legend.fontsize'] = 32
    matplotlib.rcParams['legend.frameon'] = False
    matplotlib.rcParams['figure.subplot.top'] = 0.85
    matplotlib.rcParams['axes.facecolor'] = 'white'
    matplotlib.rcParams['axes.linewidth'] = 0.8

    plt.plot(x, y, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0, 80)
    plt.savefig("plots/"+filename)
    plt.gcf().clear()

def calculate_mean_action_dist(action_dists):
    mean_proportions = {0: 0, 1: 0, 2: 0, 3: 0} 
    for distribution in action_dists:
        print(distribution)
        for action in distribution:
        mean_proportions[action] += distribution[action]
    mean_proportions = {action: value/len(action_dists) for action, value in mean_proportions.items()}
    return mean_proportions

def sync_grad(target_network, src_network):
    for param, src_param in zip(target_network.parameters(), src_network.parameters()):
        param._grad = src_param.grad.clone()

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

class Batcher:
    def __init__(self, batch_size, data):
        self.batch_size = batch_size
        self.data = data
        self.num_entries = len(data[0])
        self.reset()

    def reset(self):
        self.batch_start = 0
        self.batch_end = self.batch_start + self.batch_size

    def end(self):
        return self.batch_start >= self.num_entries

    def next_batch(self):
        batch = []
        for d in self.data:
            batch.append(d[self.batch_start: self.batch_end])
        self.batch_start = self.batch_end
        self.batch_end = min(self.batch_start + self.batch_size, self.num_entries)
        return batch

    def shuffle(self):
        indices = np.arange(self.num_entries)
        np.random.shuffle(indices)
        self.data = [d[indices] for d in self.data]
