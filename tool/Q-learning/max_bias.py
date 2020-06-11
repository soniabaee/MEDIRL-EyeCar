import numpy as np
import matplotlib.pyplot as plt
from q_learn import q_learn
from doubleQ import doubleQ
from max_bias_setup import transition

# initialization
num_actions_B = 30 # number of actions available for state B
initial_Q = np.zeros([3,num_actions_B])  # action 0: left
initial_Q[1,2:] = -1e6
initial_state = 1 # states: 0- terminal; 1 - A; 2-B
gamma = 1
alpha = 0.1
epsilon = 0.1
episodes = 500
runs = 100
num_left_Q = np.zeros(episodes)
num_left_2Q = np.zeros(episodes)

def prob_left(Q):
    if Q[1,0] > Q[1,1]:
        prob_left = 1-epsilon + epsilon /2
    elif Q[1,0] == Q[1,1]:
        prob_left = 0.5
    else:
        prob_left = epsilon /2
    return prob_left
    
for run in range(runs):
    Q = np.copy(initial_Q)
    Q1 = np.copy(initial_Q)
    Q2 = np.copy(initial_Q)
    for ep in range(episodes):
       Q,_,_ = q_learn(Q, initial_state, transition, 1, gamma, alpha, epsilon) 
       Q1,Q2,DQ = doubleQ(Q1,Q2,initial_state,transition,1,gamma,alpha,epsilon)
       
       num_left_Q[ep] += np.random.binomial(1, prob_left(Q))
       num_left_2Q[ep] += np.random.binomial(1,prob_left(DQ))
       
fig = plt.figure()  
plt.plot(num_left_Q/runs,'b',label='Q-learning')
plt.plot(num_left_2Q/runs,'r',label='Double Q-learning')
plt.legend(loc=0)
plt.xlabel('Episodes')
plt.ylabel('% left actions from A')
plt.xlim([-5,episodes])
plt.grid()
#fig.savefig('doubleQ_learning'+str(num_actions_B)+'.jpg')