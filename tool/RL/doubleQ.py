import numpy as np
import random


def doubleQ(initial_Q1,initial_Q2,initial_state,transition,
           num_episodes,gamma, alpha, epsilon=0.1):
    #This function implements double Q-learning. It returns Q1, Q2 and their sum Q
    
    """
    Your code
    """
    
    Q1 = np.copy(initial_Q1)
    Q2 = np.copy(initial_Q2)
    
    num_states, num_actions = Q1.shape  
    
    for ep in range(num_episodes):
        
        uniformScl = epsilon/num_actions
        greedySlc = (1-epsilon) + epsilon/num_actions
        
        crnState = initial_state
        
        cnt = 0
        imdRewards = 0
        while True:
            cnt += 1
            
            actionPrb = uniformScl*np.ones(num_actions, dtype=float)
            Q = Q1 + Q2
            actionPrb[np.argmax(Q[crnState])] = greedySlc
            crnAction = np.random.choice(num_actions, p = actionPrb)
            
            nxtState, imdReward, terminal = transition(crnState,crnAction) 
            imdRewards += imdReward 
            coinRand = random.uniform(0, 1)
            
            actIdx1 = np.argmax(Q1[nxtState,:])
            mxQ2nxtState = Q2[nxtState,actIdx1]
            
            actIdx2 = np.argmax(Q2[nxtState,:])
            mxQ1nxtState = Q1[nxtState,actIdx2]
            
            if coinRand < 0.5:
                Q1[crnState,crnAction] += alpha * (imdReward + gamma * mxQ2nxtState - Q1[crnState,crnAction])
            else:
                Q2[crnState,crnAction] += alpha * (imdReward + gamma * mxQ1nxtState - Q2[crnState,crnAction])
            
            crnState = nxtState
        
            if terminal:
                break
            

    

    return Q1, Q2, Q #,  steps#, rewards
           