import numpy as np

stateSpace = np.zeros([48,2], dtype=int)
stateSpace[:,0] = np.tile(range(12),4)
stateSpace[:,1] = np.tile(np.repeat(range(4),12),1)
stateSpace = stateSpace.tolist()
terminal_state = [11,0]
cliff_states = np.arange(1,11) #i.e. states [1,0],[2,0], ..., [10,0]

def transition(state,action):
    # get coordinates of the current state
    state_x, state_y = stateSpace[state]
    
    if action == 0: # left
        next_state = [max(state_x-1,0), state_y]
   
    elif action == 1: # up
        next_state = [state_x, min(state_y + 1, 3)]

    elif action == 2: # right
        next_state = [min(state_x+1, 11), state_y]

    else: # down
        next_state = [state_x, max(state_y - 1, 0)]

    terminal = next_state == terminal_state

    next_state = stateSpace.index(next_state)
    if next_state in cliff_states:
        reward = -100
        next_state = 0
    else:
        reward = -1
        
    return next_state, reward, terminal
