import numpy as np
import random

def q_learn(initial_Q,initial_state,transition,
          num_episodes,gamma, alpha, epsilon=0.1):
              
    """
    This function implements Q-learning. It returns learned Q values.
    To create 6.4, the function also returns number of steps, and 
    the total rewards in each episode.
        
    Notes on inputs:    
    -transition: function. It takes current state s and action a as parameters 
                and returns next state s', immediate reward R, and a boolean 
                variable indicating whether s' is a terminal state. 
                (See windy_setup as an example)
    -epsilon: exploration rate as in epsilon-greedy policy
    
    """    
    
    """ 
    Your code
    """
    
    Q = np.copy(initial_Q)
    num_states, num_actions = Q.shape    
       
    steps = np.zeros(num_episodes,dtype=int) # store #steps in each episode
    rewards = np.zeros(num_episodes) # store total rewards for each episode
    
    
    for ep in range(num_episodes):

        ## epsilon greedy
        uniformScl = epsilon/num_actions
        greedySlc = (1-epsilon) + epsilon/num_actions
        
        crnState = initial_state
        
        cnt = 0        
        imdRewards = 0
        while True: 
           cnt += 1 
           
           if random.random() < epsilon: 
               crnAction = random.randint(0, num_actions-1)
           else: 
               crnAction = Q[crnState].argmax()
           
           nxtState, imdReward, terminal = transition(crnState,crnAction) 
           imdRewards += imdReward           
           
                      
           actIdx = np.argmax(Q[nxtState,:])
           mxQnxtState = Q[nxtState,actIdx]

           Q[crnState,crnAction] += alpha * (imdReward + gamma * mxQnxtState - Q[crnState,crnAction])
           
           crnState = nxtState

           if terminal:
               break

           
           
        rewards[ep] += imdRewards
        steps[ep] += cnt  
            
    return Q,  steps, rewards


# RESULT
#===============================================
#path learned from Sarsa:
#state: [0, 0]   action: up
#state: [0, 1]   action: up
#state: [0, 2]   action: up
#state: [0, 3]   action: right
#state: [1, 3]   action: right
#state: [2, 3]   action: right
#state: [3, 3]   action: right
#state: [4, 3]   action: right
#state: [5, 3]   action: right
#state: [6, 3]   action: right
#state: [7, 3]   action: right
#state: [8, 3]   action: right
#state: [9, 3]   action: right
#state: [10, 3]   action: right
#state: [11, 3]   action: down
#state: [11, 2]   action: down
#state: [11, 1]   action: down
#state: [7,3]
#number of steps:  17
#--------------------------------------
#path learned from Q-learning:
#state: [0, 0]   action: up
#state: [0, 1]   action: right
#state: [1, 1]   action: right
#state: [2, 1]   action: right
#state: [3, 1]   action: right
#state: [4, 1]   action: right
#state: [5, 1]   action: right
#state: [6, 1]   action: right
#state: [7, 1]   action: right
#state: [8, 1]   action: right
#state: [9, 1]   action: right
#state: [10, 1]   action: right
#state: [11, 1]   action: down
#state: [7,3]
#number of steps:  13