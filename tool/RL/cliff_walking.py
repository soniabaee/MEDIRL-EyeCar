import numpy as np
import matplotlib.pyplot as plt
from sarsa import sarsa
from q_learn import q_learn
from cliff_setup import *

# initialization 
initial_Q = np.zeros([48,4])
initial_state = 0
gamma = 1
alpha = 0.5
epsilon = 0.1
num_episodes = 500
action_str=['left','up', 'right', 'down']  

# using sarsa 
Q_sarsa, steps_sarsa,rewards_sarsa = sarsa(initial_Q, initial_state, transition, 
                         num_episodes, gamma,alpha, epsilon)

Q_q, steps_q, rewards_q = q_learn(initial_Q, initial_state, transition, 
                         num_episodes, gamma,alpha, epsilon)


def print_learned_path(Q):
    actions = np.argmax(Q,axis=1)
    state = initial_state
    num_steps = 0    
    terminal = False
    while not terminal and num_steps < 50:        
        action = actions[state]
        print('state:',stateSpace[state],'  action:',action_str[action])
        next_state,_,terminal = transition(state,action)
    
        state = next_state
        num_steps += 1
    if num_steps < 30:
        print('state:', '[7,3]')
        print('number of steps: ', num_steps)
    else:
        print('Cannot terminate. Run again')  

# print learned path
print('path learned from Sarsa:')   
print_learned_path(Q_sarsa)
print('--------------------------------------')
print('path learned from Q-learning:')
print_learned_path(Q_q)

# generate Figure 6.4
window = 100 
smoothed_rewards_sarsa = np.zeros(num_episodes-window) 
smoothed_rewards_q = np.zeros(num_episodes-window) 
for i in range(num_episodes-window):
    smoothed_rewards_q[i]=np.mean(rewards_q[i:i+window+1])
    smoothed_rewards_sarsa[i]=np.mean(rewards_sarsa[i:i+window+1])

fig = plt.figure()    
plt.plot(np.arange(window+1,num_episodes+1),smoothed_rewards_q,'r')
plt.plot(np.arange(window+1,num_episodes+1),smoothed_rewards_sarsa,'b')
plt.ylim(-100)
plt.xlabel('Episodes')
plt.ylabel('total rewards during episode')
plt.legend(['Q-learning','Sarsa'],loc=0)
#fig.savefig('cliff_walking.jpg')