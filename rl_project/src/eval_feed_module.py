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
from copy import deepcopy

np.set_printoptions(linewidth = 100, edgeitems = 'all', suppress = True, 
                 precision = 4)

def n_armed_bandit(num_bandits, num_arms, num_plays, eps_list, verbose=False):
    """
    N-armed bandit problem.  
    
    Interesting is the comparison of the given reward (qstar, Q*) and
    the estimated reward, q_estimated.  q_estimated is produced for each
    epsilon.  
    
    Parameters
    ----------
    In    : num_bandits, num_arms, num_plays, eps_list
    Out   : avg_reward
    
    Examples
    --------
    avg_reward = n_armed_bandit(num_bandits, num_arms, num_plays, eps_list)
    """
    np.random.seed(1)
    # true reward Q*
    qstar = np.random.normal(0, 1, (num_bandits, num_arms))
    avg_reward = []
    opt_action = []
    for eps in eps_list:
        q_estimated = np.zeros((num_bandits, num_arms))
        q_counter = np.zeros((num_bandits, num_arms))
        q_aggregated = deepcopy(q_estimated)

        rewards_array = np.zeros((num_bandits, num_plays))
        optimal_action_flag = np.zeros((num_bandits, num_plays))
        
        for i in range(num_bandits):
            for j in range(num_plays):
                if np.random.uniform(0, 1) > eps:
                    # select greedy strategy (arm with highest q_estimated)
                    arm_idx = q_estimated[i, :].argmax()
                else:
                    # select random arm
                    arm_idx = np.random.choice(num_arms, 1)
                
                if arm_idx == qstar[i, :].argmax():
                    optimal_action_flag[i, j] = 1
                
                sigma = 1.0
                reward = qstar[i, arm_idx] + sigma * np.random.normal(0, 1)
                rewards_array[i, j] = reward
                
                q_counter[i, arm_idx] += 1
                q_aggregated[i, arm_idx] += reward
                
                q_estimated[i, arm_idx] = \
                    deepcopy(q_aggregated[i, arm_idx] / q_counter[i, arm_idx])
        if verbose:
            a = raw_input("""
This may produce significant screen output. Are you sure? (y/n) """)
            if a.lower() == 'y':
                print "optimal_action_flag = \n", optimal_action_flag
                print "q_counter = \n", q_counter
                print "q_aggregated = \n", q_aggregated
                print "q_estimated = \n", q_estimated
                print "qstar = \n", qstar
                print "rewards_array = \n", rewards_array
        avg_reward_per_play = rewards_array.mean(axis=0)
        opt_action_per_play = optimal_action_flag.sum(axis=0) / num_bandits
        avg_reward.append(avg_reward_per_play.T)
        opt_action.append(opt_action_per_play.T)
        
    avg_reward = np.array(avg_reward)
    opt_action = np.array(opt_action)
    
    fig, ax = plt.subplots(2, 1)
    num_fig = ax.shape[0]

    for k in range(len(eps_list)):
        ax[0].plot(range(1, num_plays + 1), avg_reward[k, :], \
                   label='$\epsilon=%0.3f$'%(eps_list[k]))
    ax[0].set_title('Average reward')
    ax[0].set_xlabel('Plays')
    ax[0].set_ylabel('Average reward')
    ax[0].legend(loc="upper left")
    ax[0].grid()
    
    for k in range(len(eps_list)):
        ax[1].plot(range(1, num_plays + 1), opt_action[k, :], \
                   label='$\epsilon=%0.3f$'%(eps_list[k]))
    ax[1].set_title('Optimal action in percent')
    ax[1].set_xlabel('Plays')
    ax[1].set_ylabel('Optimal action in percent')
    ax[1].legend(loc="upper left")
    ax[1].grid()
    
    plt.show()
    
    return avg_reward, opt_action

if __name__ == '__main__':
    """
    execfile('C:\\Users\\amalysch\\git\\rl_repository\\rl_project\\src\\eval_feed_module.py')
    """
    num_bandits = 2000
    num_arms = 10
    num_plays = 1000
    eps_list = [0.0, 0.01, 0.1]
#     eps_list = [0.1]    
    print "num_bandits = ", num_bandits
    print "num_arms = ", num_arms
    print "num_plays = ", num_plays
    print "eps_list = ", eps_list
    avg_reward, opt_action = n_armed_bandit(num_bandits, num_arms, 
                                            num_plays, eps_list)
#     print "avg_reward = \n", avg_reward
    