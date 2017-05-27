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
from scipy import exp
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
        # Initial reward estimate is set to zero: Qt=q_estimate
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
#                     arm_idx = np.random.choice(num_arms, 1)
                    
                    # softmax for gibbs_temp
                    gibbs_temp = 100000.0
                    softmax = exp(q_estimated[i, :]/gibbs_temp) / \
                                    exp(q_estimated[i, :]/gibbs_temp).sum()
                    arm_idx = np.random.choice(range(num_arms), p=softmax)
               
                if arm_idx == qstar[i, :].argmax():
                    optimal_action_flag[i, j] = 1
                
                # Add noise, so learner receives distorted signal from qstar
                sigma = 1.0
                noise = sigma * np.random.normal(0, 1)
                reward = qstar[i, arm_idx] + noise
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
    ax[0].set_title('Average reward for %d bandits and %d plays (sigma (noise) = %d)'%(num_bandits, num_plays, sigma))
    ax[0].set_xlabel('Plays')
    ax[0].set_ylabel('Average reward')
    ax[0].legend(loc="upper left")
    ax[0].grid()
    
    for k in range(len(eps_list)):
        ax[1].plot(range(1, num_plays + 1), opt_action[k, :], \
                   label='$\epsilon=%0.3f$'%(eps_list[k]))
    ax[1].set_title('Optimal action in percent for %d bandits and %d plays (sigma (noise) = %d)'%(num_bandits, num_plays, sigma))
    ax[1].set_xlabel('Plays')
    ax[1].set_ylabel('Optimal action in percent')
    ax[1].legend(loc="upper left")
    ax[1].grid()
    
    plt.show()
    
    return avg_reward, opt_action

def supervised_binary_bandit(num_bandits, num_plays, p, eps_list, verbose=False):
    """
    Binary bandit problem.  Test whether the better of two actions can be
    identified. 
    
    Supervised learning approach: failure is taken as an indicator that the
    other action was the correct one.  Leads to oscillation for low 
    (but distinct) probabilities of success for both actions (e.g. 0.1, 0.2). 
    For high probabilities of success (e.g. 0.8, 0.9) the wrong action may
    be selected consistently.  
    
    Comparison
    
    Evaluation                vs.         Instruction
    Learning by selection     vs.         Learning by instruction
    Reinforcement learning    vs.         Supervised learning
    
    Parameters
    ----------
    In    : num_bandits, num_plays, p
    Out   : opt_action
    
    Examples
    --------
    opt_action = supervised_binary_bandit(num_bandits, num_plays, p, eps_list, verbose=False)
    """
    np.random.seed(1)
    num_arms = 2
    if type(p) == list: p = np.array(p)
    arm_best = p.argmax()
    # Probability of success (defines the better action)
    print "p = ", p
    print "arm_best = ", arm_best
    opt_action = []
    alpha = 0.1
    for eps in eps_list:
        # Initial reward estimate is set to zero: Qt=q_estimate
        optimal_action_flag = np.zeros((num_bandits, num_plays))
        
        for i in range(num_bandits):
            # select random arm
            arm_idx = np.random.choice(num_arms, 1)
            q_estimated = np.array([0.5, 0.5])
            for j in range(num_plays):
                rnd_num = np.random.uniform(0, 1)
                if not(isinstance(eps, basestring)) and rnd_num <= eps:
                    # select random arm
                    arm_idx = np.random.choice(num_arms, 1)
                elif isinstance(eps, basestring): 
                    arm_idx = np.random.choice(np.arange(0, num_arms), p=q_estimated)

                if arm_idx == arm_best:
                    optimal_action_flag[i, j] = 1
                
                change_prob_flag = False
                if isinstance(eps, basestring):
                    change_prob_flag = True

                # high p_win makes switch less likely
                p_win = p[arm_idx]
                if p_win < np.random.uniform(0, 1):
                    # try other arm
                    arm_idx = np.mod(arm_idx+1, 2)
                    if eps == 'L-PI':
                        change_prob_flag = False
                        
                if change_prob_flag:
                    arm_other = np.mod(arm_idx+1, 2)
                    q_estimated[arm_idx] += alpha * (1.0 - q_estimated[arm_idx])
                    q_estimated[arm_other] = 1.0 - q_estimated[arm_idx]

        if verbose:
            a = raw_input("""
This may produce significant screen output. Are you sure? (y/n) """)
            if a.lower() == 'y':
                print "optimal_action_flag = \n", optimal_action_flag

        opt_action_per_play = optimal_action_flag.mean(axis=0)
        opt_action.append(opt_action_per_play.T)

    opt_action = np.array(opt_action)
    
    fig, ax = plt.subplots(1, 1)
#     num_fig = ax.shape[0]
    for k in range(len(eps_list)):
        if isinstance(eps_list[k], float):
            ax.plot(range(1, num_plays + 1), opt_action[k, :], \
                   label='$\epsilon=%0.3f$'%(eps_list[k]))
        elif isinstance(eps_list[k], basestring):
            ax.plot(range(1, num_plays + 1), opt_action[k, :], \
                   label='%s'%(eps_list[k]))
    ax.set_title('Optimal action in percent for %d bandits and %d plays'%(num_bandits, num_plays))
    ax.set_xlabel('Plays')
    ax.set_ylabel('Optimal action in percent')
    ax.legend(loc="upper left")
    ax.grid()
     
    plt.show()
     
    return opt_action
    
if __name__ == '__main__':
    """
    execfile('C:\\Users\\amalysch\\git\\rl_repository\\rl_project\\src\\eval_feed_module.py')
    """
    selection = raw_input("""\n\nWhich experiment do you want to run? \n
    n_armed_bandit    (1)
    binary_bandit     (2) \n
    Enter number: \n""")
    
    if selection.lower() == '1': 
        print "\n(1) Running n-armed bandit:"
        num_bandits = 2000
        num_arms = 10
        num_plays = 1000
        eps_list = [0.0, 0.01, 0.1]
        print "num_bandits = ", num_bandits
        print "num_arms = ", num_arms
        print "num_plays = ", num_plays
        print "eps_list = ", eps_list
        avg_reward, opt_action = n_armed_bandit(num_bandits, num_arms, 
                                                num_plays, eps_list)
#         print "avg_reward = \n", avg_reward

    elif selection.lower() == '2':
        print "\n(2) Running binary bandit:"
        num_bandits = 2000
        num_plays = 500
        eps_list = [0.0, 0.1, '$L_{R-P}$', '$L_{R-I}$']
        # Probability of success (defines the better action)
        p = [0.9, 0.8]
#         p = [0.1, 0.2]
        print "num_bandits = ", num_bandits
        print "num_plays = ", num_plays
        print "eps_list = ", eps_list
        print "p = ", p
        opt_action = supervised_binary_bandit(num_bandits, num_plays, \
                                              p, eps_list)
#         print "opt_action = \n", opt_action
    else:
        print "Please selected from menu. Exiting. "
        
        