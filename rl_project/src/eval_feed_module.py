"""
Created on May 16, 2017

Evaluative Feedback

Reinforcement Learning, Chapter 2
(Sutton, Barto, 1998)

"""
__author__ = 'amm'
__date__  = "May 16, 2017"
__version__ = 0.0

import numpy as np
import pylab as plt

np.set_printoptions(linewidth = 100, edgeitems = 'all', suppress = True, 
                 precision = 4)

def n_armed_bandit(num_bandits, num_arms, num_plays):
    """
    N-armed bandit problem.  
    
    Parameters
    ----------
    In    : num_bandits, num_arms, num_plays
    Out   : 
    
    Examples
    --------
    avg_reward = n_armed_bandit(num_bandits, num_arms, num_plays)
    """
    np.random.seed(1)
    qstar = np.random.normal(0, 1, (num_bandits, num_arms))
    print "qstar = \n", qstar
    eps_list = [0.0, 0.1, 0.01]
    avg_reward = []
    for eps in eps_list[:1]:
        print eps
        q_estimated = np.zeros((num_bandits, num_arms))
        q_counter = np.ones((num_bandits, num_arms))
        q_aggregated = q_estimated

        rewards_array = np.zeros((num_bandits, num_plays))
        optimal_action_flag = np.zeros((num_bandits, num_plays))
        
        for i in range(num_bandits):
            for j in range(num_plays):
                if np.random.uniform(0, 1) > eps:
                    # select greedy strategy (arm with highest q_estimated)
                    arm_idx = q_estimated[i, :].argmax()
                    print "argmax arm_idx = %d with j = %d"%(arm_idx, j)
                else:
                    # select random arm
                    arm_idx = np.random.choice(num_arms, 1)
                    print "random arm_idx = %d with j = %d"%(arm_idx, j)
                
                if arm_idx == qstar[i, :].argmax():
                    optimal_action_flag[i, j] = 1
                
                sigma = 1.0
                reward = qstar[i, arm_idx] + sigma * np.random.normal(0, 1)
                rewards_array[i, j] = reward
                
                q_counter[i, arm_idx] += 1
                q_aggregated[i, arm_idx] += reward
                
                q_estimated[i, arm_idx] = q_aggregated[i, arm_idx] / \
                                          q_counter[i, arm_idx]
                
        avg_reward_per_play = rewards_array.mean(axis=0)
        print "avg_reward_per_play = ", avg_reward_per_play
        avg_reward.append(avg_reward_per_play.T)
                
                             
    return avg_reward
    

if __name__ == '__main__':
    """
    execfile('C:\\Users\\amalysch\\git\\rl_repository\\rl_project\\src\\eval_feed_module.py')
    """
    num_bandits = 20
    num_arms = 5
    num_plays = 7
    print "num_bandits = ", num_bandits
    print "num_arms = ", num_arms
    print "num_plays = ", num_plays
    avg_reward = n_armed_bandit(num_bandits, num_arms, num_plays)
    print "avg_reward = \n", avg_reward
    