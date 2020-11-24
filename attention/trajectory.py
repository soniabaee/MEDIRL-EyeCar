"""
Trajectories representing expert demonstrations and automated generation
thereof.
"""

import numpy as np
from itertools import chain


class Trajectory:
    """
    A trajectory consisting of states, corresponding actions, and outcomes.
    """
    def __init__(self, transitions):
        self._t = transitions

    def transitions(self):
        """
        The transitions of this trajectory.
        """
        return self._t

    def states(self):
        """
        The states visited in this trajectory.
        """
        return map(lambda x: x[0], chain(self._t, [(self._t[-1][2], 0, 0)]))

    def __repr__(self):
        return "Trajectory({})".format(repr(self._t))

    def __str__(self):
        return "{}".format(self._t)


def generate_trajectory(world, policy, start, final):
    """
    Generate a single trajectory.
    """

    state = start

    trajectory = []
    while state not in final:
        action = policy(state)

        next_s = range(world.n_states)
        next_p = world.p_transition[state, :, action]

        next_state = np.random.choice(next_s, p=next_p)

        trajectory += [(state, action, next_state)]
        state = next_state

    return Trajectory(trajectory)


def generate_trajectories(n, world, policy, start, final):
    """
    Generate multiple trajectories.
    """
    start_states = np.atleast_1d(start)

    def _generate_one():
        if len(start_states) == world.n_states:
            s = np.random.choice(range(world.n_states), p=start_states)
        else:
            s = np.random.choice(start_states)

        return generate_trajectory(world, policy, s, final)

    return (_generate_one() for _ in range(n))


def policy_adapter(policy):
    return lambda state: policy[state]


def stochastic_policy_adapter(policy):
    return lambda state: np.random.choice([*range(policy.shape[1])], p=policy[state, :])
