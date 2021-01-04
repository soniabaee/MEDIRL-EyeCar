import numpy as np
import random

def sarsa(initial_Q,initial_state,transition,
          num_episodes,gamma, alpha, epsilon=0.1):
    """
    This function implements Sarsa. It returns learned Q values.
    To crete Figure 6.3 and 6.4, the function also returns number of steps, and 
    the total rewards in each episode.
        
    Notes on inputs:    
    -transition: function. It takes current state s and action a as parameters 
                and returns next state s', immediate reward R, and a boolean 
                variable indicating whether s' is a terminal state. 
                (See windy_setup as an example)
    -epsilon: exploration rate as in epsilon-greedy policy
    
    """    
    
     # initialization    
    Q = np.copy(initial_Q)
    num_states, num_actions = Q.shape    
       
    steps = np.zeros(num_episodes,dtype=int) # store #steps in each episode
    rewards = np.zeros(num_episodes) # store total rewards for each episode
    
    for ep in range(num_episodes):

        ## epsilon greedy
        uniformScl = epsilon/num_actions
        greedySlc = (1-epsilon) + epsilon/num_actions
        
        crnState = initial_state
             
        ## greedy selection
        actionPrb = uniformScl*np.ones(num_actions, dtype=float)
        actionPrb[np.argmax(Q[crnState])] = greedySlc
        crnAction = np.random.choice(num_actions, p = actionPrb)

        cnt = 0        
        imdRewards = 0
        while True: 
           cnt += 1 
           
           nxtState, imdReward, terminal = transition(crnState,crnAction) 
           imdRewards += imdReward           
           
           if terminal:
               break
           
           nxtActPrb = uniformScl * np.ones(num_actions, dtype=float)
           nxtActPrb[np.argmax(Q[nxtState])] = greedySlc
           nxtAction = np.random.choice(num_actions, p = nxtActPrb)
                  
         
           Q[crnState,crnAction] += alpha * (imdReward + gamma * Q[nxtState, nxtAction] - Q[crnState,crnAction])
           
           crnState = nxtState
           crnAction = nxtAction
           
           
        rewards[ep] += imdRewards
        steps[ep] += cnt  
       
          
    return Q,  steps, rewards
        
    