import numpy as np
import matplotlib.pyplot as plt

import maximumEtropyDeepIRL.py as deep_maxent
import medirl

from itertools import product

import numpy as np
import numpy.random as rn
import theano as th
import theano.tensor as T

import maximumEntropy

FLOAT = th.config.floatX

def find_svf(n_states, trajectories):
    """
    Find the state vistiation frequency from trajectories.
    """

    svf = np.zeros(n_states)

    for trajectory in trajectories:
        for state, _, _ in trajectory:
            svf[state] += 1

    svf /= trajectories.shape[0]

    return th.shared(svf, "svf", borrow=True)

def optimal_value(n_states, n_actions, transition_probabilities, reward,
                  discount, threshold=1e-2):
    """
    Find the optimal value function.
    """

    v = T.zeros(n_states, dtype=FLOAT)

    def update(s, prev_diff, v, reward, tps):
        max_v = float("-inf")
        v_template = T.zeros_like(v)
        for a in range(n_actions):
            tp = tps[s, a, :]
            max_v = T.largest(max_v, T.dot(tp, reward + discount*v))
        new_diff = abs(v[s] - max_v)
        if T.lt(prev_diff, new_diff):
            diff = new_diff
        else:
            diff = prev_diff
        return (diff, T.set_subtensor(v_template[s], max_v)), {}

    def until_converged(diff, v):
        (diff, vs), _ = th.scan(
                fn=update,
                outputs_info=[{"initial": diff, "taps": [-1]},
                              None],
                sequences=[T.arange(n_states)],
                non_sequences=[v, reward, transition_probabilities])
        return ((diff[-1], vs.sum(axis=0)), {},
                th.scan_module.until(diff[-1] < threshold))

    (_, vs), _ = th.scan(fn = until_converged,
                         outputs_info=[
                            # Need to force an inf into the right Theano
                            # data type and this seems to be the only way that
                            # works.
                            {"initial": getattr(np, FLOAT)(float("inf")),
                             "taps": [-1]},
                            {"initial": v,
                             "taps": [-1]}],
                         n_steps=1000)

    return vs[-1]

def find_policy(n_states, n_actions, transition_probabilities, reward, discount,
                threshold=1e-2, v=None):
    """
    Find the optimal policy.
    """

    if v is None:
        v = optimal_value(n_states, n_actions, transition_probabilities, reward,
                          discount, threshold)

    # Get Q using equation 9.2 from Ziebart's thesis.
    Q = T.zeros((n_states, n_actions))
    def make_Q(i, j, tps, Q, reward, v):
        Q_template = T.zeros_like(Q)
        tp = transition_probabilities[i, j, :]
        return T.set_subtensor(Q_template[i, j], tp.dot(reward + discount*v)),{}

    prod = np.array(list(product(range(n_states), range(n_actions))))
    state_range = th.shared(prod[:, 0])
    action_range = th.shared(prod[:, 1])
    Qs, _ = th.scan(fn=make_Q,
                    outputs_info=None,
                    sequences=[state_range, action_range],
                    non_sequences=[transition_probabilities, Q, reward, v])
    Q = Qs.sum(axis=0)
    Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
    Q = T.exp(Q)/T.exp(Q).sum(axis=1).reshape((n_states, 1))
    return Q

def find_expected_svf(n_states, r, n_actions, discount,
                      transition_probability, trajectories):
    """
    Find the expected state visitation frequencies using algorithm 1 from
    Ziebart et al. 2008.
    """

    n_trajectories = trajectories.shape[0]
    trajectory_length = trajectories.shape[1]

    policy = find_policy(n_states, n_actions,
                         transition_probability, r, discount)

    start_state_count = T.extra_ops.bincount(trajectories[:, 0, 0],
                                             minlength=n_states)
    p_start_state = start_state_count.astype(FLOAT)/n_trajectories

    def state_visitation_step(i, j, prev_svf, policy, tps):
        """
        The sum of the outputs of a scan over this will be a row of the svf.
        """

        svf = prev_svf[i] * policy[i, j] * tps[i, j, :]
        return svf, {}

    prod = np.array(list(product(range(n_states), range(n_actions))))
    state_range = th.shared(prod[:, 0])
    action_range = th.shared(prod[:, 1])
    def state_visitation_row(prev_svf, policy, tps, state_range, action_range):
        svf_t, _ = th.scan(fn=state_visitation_step,
                           sequences=[state_range, action_range],
                           non_sequences=[prev_svf, policy, tps])
        svf_t = svf_t.sum(axis=0)
        return svf_t, {}

    svf, _ = th.scan(fn=state_visitation_row,
                     outputs_info=[{"initial": p_start_state, "taps": [-1]}],
                     n_steps=trajectories.shape[1]-1,
                     non_sequences=[policy, transition_probability, state_range,
                                 action_range])

    return svf.sum(axis=0) + p_start_state

def irl(structure, feature_matrix, n_actions, discount, transition_probability,
        trajectories, epochs, learning_rate, initialisation="normal", l1=0.1,
        l2=0.1):
    """
    Find the reward function for the given trajectories.
    """

    n_states, d_states = feature_matrix.shape
    transition_probability = th.shared(transition_probability, borrow=True)
    trajectories = th.shared(trajectories, borrow=True)

    # Initialise W matrices; b biases.
    n_layers = len(structure)-1
    weights = []
    hist_w_grads = []  # For AdaGrad.
    biases = []
    hist_b_grads = []  # For AdaGrad.
    for i in range(n_layers):
        # W
        shape = (structure[i+1], structure[i])
        if initialisation == "normal":
            matrix = th.shared(rn.normal(size=shape), name="W", borrow=True)
        else:
            matrix = th.shared(rn.uniform(size=shape), name="W", borrow=True)
        weights.append(matrix)
        hist_w_grads.append(th.shared(np.zeros(shape), name="hdW", borrow=True))

        # b
        shape = (structure[i+1], 1)
        if initialisation == "normal":
            matrix = th.shared(rn.normal(size=shape), name="b", borrow=True)
        else:
            matrix = th.shared(rn.uniform(size=shape), name="b", borrow=True)
        biases.append(matrix)
        hist_b_grads.append(th.shared(np.zeros(shape), name="hdb", borrow=True))

    # Initialise α weight, β bias.
    if initialisation == "normal":
        α = th.shared(rn.normal(size=(1, structure[-1])), name="alpha",
                      borrow=True)
    else:
        α = th.shared(rn.uniform(size=(1, structure[-1])), name="alpha",
                      borrow=True)
    hist_α_grad = T.zeros(α.shape)  # For AdaGrad.

    adagrad_epsilon = 1e-6  # AdaGrad numerical stability.

    # Symbolic input.
    s_feature_matrix = T.matrix("x")
    # Feature matrices.
    φs = [s_feature_matrix.T]
    # Forward propagation.
    for W, b in zip(weights, biases):
        φ = T.nnet.sigmoid(th.compile.ops.Rebroadcast((0, False), (1, True))(b)
                           + W.dot(φs[-1]))
        φs.append(φ)
        # φs[1] = φ1 etc.
    # Reward.
    r = α.dot(φs[-1]).reshape((n_states,))
    # Engineering hack: z-score the reward.
    r = (r - r.mean())/r.std()
    # Associated feature expectations.
    expected_svf = find_expected_svf(n_states, r,
                                     n_actions, discount,
                                     transition_probability,
                                     trajectories)
    svf = maximumEntropy.find_svf(n_states, trajectories.get_value())
    # Derivatives (backward propagation).
    updates = []
    α_grad = φs[-1].dot(svf - expected_svf).T
    hist_α_grad += α_grad**2
    adj_α_grad = α_grad/(adagrad_epsilon + T.sqrt(hist_α_grad))
    updates.append((α, α + adj_α_grad*learning_rate))

    def grad_for_state(s, theta, svf_diff, r):
        """
        Calculate the gradient with respect to theta for one state.
        """

        regularisation = abs(theta).sum()*l1 + (theta**2).sum()*l2
        return svf_diff[s] * T.grad(r[s], theta) - regularisation, {}

    for i, W in enumerate(weights):
        w_grads, _ = th.scan(fn=grad_for_state,
                             sequences=[T.arange(n_states)],
                             non_sequences=[W, svf - expected_svf, r])
        w_grad = w_grads.sum(axis=0)
        hist_w_grads[i] += w_grad**2
        adj_w_grad = w_grad/(adagrad_epsilon + T.sqrt(hist_w_grads[i]))
        updates.append((W, W + adj_w_grad*learning_rate))
    for i, b in enumerate(biases):
        b_grads, _ = th.scan(fn=grad_for_state,
                             sequences=[T.arange(n_states)],
                             non_sequences=[b, svf - expected_svf, r])
        b_grad = b_grads.sum(axis=0)
        hist_b_grads[i] += b_grad**2
        adj_b_grad = b_grad/(adagrad_epsilon + T.sqrt(hist_b_grads[i]))
        updates.append((b, b + adj_b_grad*learning_rate))

    train = th.function([s_feature_matrix], updates=updates, outputs=r)
    run = th.function([s_feature_matrix], outputs=r)

    for e in range(epochs):
        reward = train(feature_matrix)

    return reward.reshape((n_states,))



class Reward(chainer.Chain):
    def __init__(self, n_input, n_hidden):
        super(Reward, self).__init__(
            l1=L.Linear(n_input, n_hidden),
            l2=L.Linear(n_hidden, n_hidden),
            l3=L.Linear(n_hidden, 1)
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

def max_ent_deep_irl(feature_matrix, trans_probs, trajs,
                     gamma=0.9, n_epoch=30):
    n_states, d_states = feature_matrix.shape
    _, n_actions, _ = trans_probs.shape
    reward_func = Reward(d_states, 64)
    optimizer = optimizers.AdaGrad(lr=0.01)
    optimizer.setup(reward_func)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))
    optimizer.add_hook(chainer.optimizer.GradientClipping(100.0))

    feature_exp = np.zeros((d_states))
    for episode in trajs:
        for step in episode:
            feature_exp += feature_matrix[step[0], :]
    feature_exp = feature_exp / len(trajs)

    fmat = chainer.Variable(feature_matrix.astype(np.float32))
    for _ in range(n_epoch):
        reward_func.zerograds()
        r = reward_func(fmat)
        v = value_iteration(trans_probs, r.data.reshape((n_states,)), gamma)
        pi = best_policy(trans_probs, v)
        exp_svf = expected_svf(trans_probs, trajs, pi)
        grad_r = feature_exp - exp_svf
        r.grad = -grad_r.reshape((n_states, 1)).astype(np.float32)
        r.backward()
        optimizer.update()

    return reward_func(fmat).data.reshape((n_states,))



def attention(visualOutput, drivingOutput, eyeFixations):
    trajectory_length = 21
    l1 = l2 = 0, 0.1
    discount = 0.9
    learning_rate =  0.01
    structure = (52, 34, 20, 20)


    dirInd = "./eyeCar-master/Data/videoPos-demogData.csv"
    dirDep = "./eyeCar-master/Data/rameData.csv"
    dirHzrd = "./eyeCar-master/Data/AOIData.csv"
    
    medirl = medirl(dirInd, dirDep, dirHzrd)
    medirl.setup_mdp()
    
    # inital the reward, action and state for each participant during the study
    medirl.calcualteState(visualOutput, drivingOutput, eyeFixations)
    medirl.actionValue(eyeFixations)
    medirl.pattern()
    medirl.distPattern()
    medirl.irlComponent(visualOutput, drivingOutput, eyeFixations)

    medirl.generate_expert_trajectories()

    ow = medirl.calcualteState()
    ground_r = np.array([ow.reward(s) for s in range(ow.n_states)])
    policy = find_policy(ow.n_states, ow.n_actions, trans_mat(grid)[0],
                         ground_r, ow.discount, stochastic=False)
    trajectories = medirl.pattern()
    feature_matrix = ow.feature_matrix(discrete=False)
    r = irl((feature_matrix.shape[1],) + structure, feature_matrix,
        ow.n_actions, discount, ow.transition_probability, trajectories, epochs,
        learning_rate, l1=l1, l2=l2)
    medirl.feature_expectation_from_trajectories()
    medirl.initial_probabilities_from_trajectories()
    medirl.compute_expected_svf()
    medirl.maxent_irl()
   