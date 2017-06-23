"""
Created on June 21, 2017

Black Jack using Monte Carlo Methods

Reinforcement Learning, Chapter 5, Monte Carlo Methods
(Sutton, Barto, 1998)

"""
__author__ = 'amm'
__date__  = "June 21, 2017"
__version__ = 0.0

import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import axes3d
# from mpl_toolkits.mplot3d.axes3d import Axes3D

np.set_printoptions(linewidth = 100, edgeitems = 'all', suppress = True, 
                 precision = 4)


def get_shuffled_cards():
    """
    Get a deck of 52 shuffled cards.
    
    Parameters
    ----------
    In    : ----
    Out   : deck, shuffled
    
    Examples
    --------
    deck_shuffled = get_shuffled_cards()
    """
    deck = np.arange(1, 53)
    # shuffle in place
    np.random.shuffle(deck)
    return deck

def get_standard_hand(hand):
    """
    Converts the higher card numbers (e.g. 14, 52) to standard range between
    one and 13 (e.g. 1, 13).  
    
    Parameters
    ----------
    In    : hand
    Out   : standard_hand
    
    Examples
    --------
    standard_hand = get_standard_hand(hand)
    """
    return np.mod(hand-1, 13) + 1
    
def deal(deck_shuffled):
    """
    Deal cards to player and dealer.  
    
    Parameters
    ----------
    In    : deck_shuffled
    Out   : hand_player, hand_dealer, card_dealer, deck
    
    Examples
    --------
    hand_player, hand_dealer, card_dealer, deck = deal(deck_shuffled)
    """
    hand_player = get_standard_hand(deck_shuffled[0:2])
    deck_shuffled = deck_shuffled[2:]
    hand_dealer = get_standard_hand(deck_shuffled[0:2])
    deck = deck_shuffled[2:]
    card_dealer = hand_dealer[0]
    return hand_player, hand_dealer, card_dealer, deck

def get_hand_value(hand):
    """
    Find the value of the player hand and the dealer hand.  The dealer hand
    is not revealed.  Do not need to distinguish between clubs, diamonds, 
    hearts, and spades. 
    
    Actual hand:
    A----2-3-4-5-6-7-8-9-10-J--Q--K
    
    Index (1-based indexing):
    1----2-3-4-5-6-7-8-9-10-11-12-13

    Value per card
    1/11-2-3-4-5-6-7-8-9-10-10-10-10
    
    Returns the hand_value (e.g. 14 for hand = [1, 13]) and usable_ace as 
    boolean (e.g. usable_ace = True for hand = [1, 13]).
    
    Parameters
    ----------
    In    : hand, an array of length 2 out of 52 integers 1, ..., 52
    Out   : hand_value, usable_ace
    
    Examples
    --------
    hand_value, usable_ace, standard_hand = get_hand_value(hand)
    """
    standard_hand = get_standard_hand(hand)
    value_per_card = np.minimum(standard_hand, 10)
    hand_value = value_per_card.sum()
    if (standard_hand == 1).any() and hand_value <= 11:
        usable_ace = True
        hand_value += 10
    else:
        usable_ace = False
    return hand_value, usable_ace, standard_hand

def get_player_state(hand_player, card_dealer):
    """
    S&B define 200 states represented in a vector of length three.  The 
    first element indicates the value of the hand of the player.  There are 
    ten possible states (12, ..., 21).  The second element describes the 
    value of the hand of the dealer, a total of ten states (1/11, ..., 10). 
    The third state indicates, whether the player has a usable ace (true/false). 
    
    Parameters
    ----------
    In    : hand_player, card_dealer
    Out   : hand_value, dealer_value, usable_ace
    
    Examples
    --------
    state = get_player_state(hand_player, card_dealer)
    """
    player_value, usable_ace_player, standard_hand = get_hand_value(hand_player)
    state = player_value, card_dealer, usable_ace_player
    return state

def hit(hand, deck):
    """
    Player requests card.  
    
    Parameters
    ----------
    In    : hand, deck
    Out   : hand, deck
    
    Examples
    --------
    hand, deck = hit(hand, deck)
    """
    hand = np.append(hand, get_standard_hand(deck[0]))
    deck = deck[1:]
    return hand, deck

def get_payoff(player_value, dealer_value):
    """
    Determine the winter.  
    
    Parameters
    ----------
    In    : player_value, dealer_value
    Out   : player wins (+1), dealer wins (-1), tie (0)
    
    Examples
    --------
    payoff = get_payoff(player_value, dealer_value)
    """
    if player_value > 21:
        payoff = -1
    elif player_value <= 21 and dealer_value > 21:
        payoff = +1
    elif player_value < dealer_value:
        payoff = -1
    elif player_value == dealer_value:
        payoff = 0
    else: 
        # player_value > dealer_value
        payoff = +1
    return payoff
    
def run_sequential(num_experiments, player_stick, verbose=False):
    """
    Run num_experiments experiments sequentially.  
    
    Parameters
    ----------
    In    : num_experiments
    Out   : payoff_array, payoff_array.mean()
    
    Examples
    --------
    payoff_array, payoff_array.mean() = run_sequential(num_experiments, player_stick, verbose=False)
    """
    np.random.seed(1)
    payoff_list = []
    for i in range(1, num_experiments+1):
        if verbose: print "\nExperiment: %d"%i
        deck = get_shuffled_cards()
    
        hand_player, hand_dealer, card_dealer, deck = deal(deck)
        state = []
        num_hits = 0
        dealer_value, usable_ace_void, standard_hand = get_hand_value(hand_dealer)
        state.append(get_player_state(hand_player, card_dealer))
        
        if verbose: 
            print "hand_player = ", hand_player, \
            "; hand_dealer = ", hand_dealer, \
            "; card_dealer = ", card_dealer
        
        player_value = state[num_hits][0]
        if verbose: 
            print "player_value = ", player_value, \
            "; card_dealer = ", state[num_hits][1], \
            "; usable_ace_player = ", state[num_hits][2]
        
        # Implement player policy
        while player_value < player_stick:
            # hit player
            num_hits += 1
            hand_player, deck = hit(hand_player, deck)
            state.append(get_player_state(hand_player, card_dealer))
            player_value = state[num_hits][0]
            if verbose: 
                print "player_value = ", player_value, \
                "; card_dealer = ", state[num_hits][1], \
                "; usable_ace_player = ", state[num_hits][2]
            
        # Implement dealer policy
        if verbose: print "dealer_value = ", dealer_value
        while dealer_value < 17:
            # hit dealer
            hand_dealer, deck = hit(hand_dealer, deck)
            dealer_value, usable_ace_void, standard_hand = get_hand_value(hand_dealer)
            if verbose: 
                print "dealer_value = ", dealer_value
            
        payoff = get_payoff(player_value, dealer_value)
        if verbose: print "Experiment %d payoff: %d"%(i, payoff)
        payoff_list.append(payoff)
    
    payoff_array = np.array(payoff_list)
    return payoff_array, payoff_array.mean(), player_stick

def get_state_value_function(num_experiments, player_stick):
    """
    Approximate the state-value function for various blackjack policies. The 
    value 'player_stick' indicates the hand value when the player stops 
    playing (hitting).  
    
    I do not seem to obtain an initial player value 'player_value_init' of 21 
    without having an initial usable ace in the hand.  B&S appear to show 
    values for that state in Fig. 5.2. 
    
    Parameters
    ----------
    In    : num_experiments, player_stick
    Out   : value_function dictionary (card_dealer, player_value, useable_ace) 
            with wins and losses
    
    Examples
    --------
    value_dict = get_state_value_function(num_experiments, player_stick)
    value_dict = get_state_value_function(10, 17)
    """
    np.random.seed(1)
    value_dict = {}
    for i in range(1, num_experiments+1):
        deck = get_shuffled_cards()
        hand_player, hand_dealer, card_dealer, deck = deal(deck)
        dealer_value, usable_ace_void, standard_hand = get_hand_value(hand_dealer)
        
        # Compute the initial state and update value_dict key, if necessary
        player_value_init, card_dealer, usable_ace_player_init = \
                                get_player_state(hand_player, card_dealer)
        value_dict.setdefault((card_dealer, player_value_init, \
                               usable_ace_player_init), [])
        player_value = player_value_init
        
        # Implement player policy
        while player_value < player_stick:
            # hit player
            hand_player, deck = hit(hand_player, deck)
            player_value, card_dealer, usable_ace_player = \
                                get_player_state(hand_player, card_dealer)
        
        # Implement dealer policy
        while dealer_value < 17:
            # hit dealer
            hand_dealer, deck = hit(hand_dealer, deck)
            dealer_value, usable_ace_void, standard_hand = get_hand_value(hand_dealer)
        
        payoff = get_payoff(player_value, dealer_value)
        value_dict[(card_dealer, player_value_init, usable_ace_player_init)].append(int(payoff))
    return value_dict

def split_and_average(value_dict):
    """
    Splitting value_dict and plotting.  
    
    Parameters
    ----------
    In    : value_dict
    Out   : value_dict_ace, value_dict_no_ace
    
    Examples
    --------
    value_dict_ace, value_dict_no_ace = split_and_average(value_dict)
    """
    value_dict_ace = {k: v for k, v in value_dict.items() if k[2] == True}
    value_dict_no_ace = {k: v for k, v in value_dict.items() if k[2] == False and k[1] > 11}
    
    for k in sorted(value_dict_ace.iterkeys()):
        value_dict_ace[k] = np.mean(value_dict_ace[k])
    for k in sorted(value_dict_no_ace.iterkeys()):
        value_dict_no_ace[k] = np.mean(value_dict_no_ace[k])
    
    return value_dict_ace, value_dict_no_ace
    
def prepare_plot_data(value_dict_ace_no_ace, flag):
    """
    Prepare data for 3d plotting.  Need to assign functional values that 
    are only available in tabular format to the data received 
    from 'np.meshgrid'. 
    
    Parameters
    ----------
    In    : value_dict_ace_no_ace, flag=True/False
    Out   : xx, yy, zz
    
    Examples
    --------
    xx, yy, zz = prepare_plot_data(value_dict_ace_no_ace, flag)
    """
    if value_dict_ace_no_ace == value_dict_ace: 
        assert flag==True, "Check input"
    if value_dict_ace_no_ace == value_dict_no_ace: 
        assert flag==False, "Check input"
    xx, yy = np.meshgrid(np.arange(1, 14), np.arange(12, 22))
    zz = np.zeros((xx.shape))
    for c in xx[0, :]:
        for r in yy[:, 0]:
            try:
                zz[r-12, c-1] = value_dict_ace_no_ace[(c, r, flag)]
            except:
                zz[r-12, c-1] = 0.0
    return xx, yy, zz

if __name__ == '__main__':
    """
    execfile('C:\\Users\\amalysch\\git\\rl_repository\\rl_project\\src\\blackjack_module.py')
    """
    num_experiments = 10000
    player_stick = 17
    verbose = False
#     payoff_array, payoff_array_mean, player_stick = \
#             run_sequential(num_experiments, player_stick, verbose)
    
    value_dict = get_state_value_function(num_experiments, player_stick)
    value_dict_ace, value_dict_no_ace = split_and_average(value_dict)
    xxa, yya, zza = prepare_plot_data(value_dict_ace, True)
    xxn, yyn, zzn = prepare_plot_data(value_dict_no_ace, False)
    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')
    ax.plot_wireframe(xxa, yya, zza, rstride=1, cstride=1)
    ax.set_title('Usable Ace with ' + str(num_experiments) + \
                 ' Experiments/Episodes ' + '(Player sticks at ' + str(player_stick) + ' )')
    ax.set_xticks(np.arange(1, 14, 1))
    ax.set_yticks(np.arange(12, 22, 1))
    ax.set_xlabel('Card Shown')
    ax.set_ylabel('Player Value')
    ax.set_zlabel('Average Winner \n (+1 Player / -1 Dealer)')
    ax = fig.add_subplot(212, projection='3d')
    ax.plot_wireframe(xxn, yyn, zzn, rstride=1, cstride=1)
    ax.set_title('No Usable Ace with ' + str(num_experiments) + \
                 ' Experiments/Episodes ' + '(Player sticks at ' + str(player_stick) + ' )')
    ax.set_xticks(np.arange(1, 14, 1))
    ax.set_yticks(np.arange(12, 22, 1))
    ax.set_xlabel('Card Shown')
    ax.set_ylabel('Player Value')
    ax.set_zlabel('Average Winner \n (+1 Player / -1 Dealer)')
    plt.show()
           
    
  
    
    
