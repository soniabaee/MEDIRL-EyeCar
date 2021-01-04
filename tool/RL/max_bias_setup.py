import numpy as np

def transition(s,a):
     
    # states: terminal (0), A (1), B(2)

    if s == 2: # state B
        reward = np.random.normal(-0.1,1)
            #reward = np.random.rand()-0.6 
        next_state = 0           
        terminal = True
            
    elif s == 1: # state A    
        if a == 1: # right
            next_state = 0 # terminal state
            reward = 0
            terminal = True
        elif a== 0 : # left
            next_state = 2 
            reward = 0
            terminal = False
        else: # 
            next_state = 0
            reward = -100000
            terminal = True
    return next_state, reward, terminal        
        
