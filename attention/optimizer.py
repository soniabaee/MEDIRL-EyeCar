"""
Generic stochastic gradient-ascent based optimizers.
"""

import numpy as np


class Optimizer:
    """
    Optimizer base-class.
    """
    def __init__(self):
        self.parameters = None

    def reset(self, parameters):
        """
        Reset this optimizer.
        """
        self.parameters = parameters

    def step(self, grad, *args, **kwargs):
        """
        Perform a single optimization step.
        """
        raise NotImplementedError

    def normalize_grad(self, ord=None):
        """
        Create a new wrapper for this optimizer which normalizes the
        gradient before each step.
        """
        return NormalizeGrad(self, ord)


class Sga(Optimizer):
    """
    Basic stochastic gradient ascent.

    """
    def __init__(self, lr):
        super().__init__()
        self.lr = lr
        self.k = 0

    def reset(self, parameters):
        """
        Reset this optimizer.
        """
        super().reset(parameters)
        self.k = 0

    def step(self, grad, *args, **kwargs):
        """
        Perform a single optimization step.
        """
        lr = self.lr if not callable(self.lr) else self.lr(self.k)
        self.k += 1

        self.parameters += lr * grad


class ExpSga(Optimizer):
    """
    Exponentiated stochastic gradient ascent.

    """
    def __init__(self, lr, normalize=False):
        super().__init__()
        self.lr = lr
        self.normalize = normalize
        self.k = 0

    def reset(self, parameters):
        """
        Reset this optimizer.
        """
        super().reset(parameters)
        self.k = 0

    def step(self, grad, *args, **kwargs):
        """
        Perform a single optimization step.
        """
        lr = self.lr if not callable(self.lr) else self.lr(self.k)
        self.k += 1

        self.parameters *= np.exp(lr * grad)

        if self.normalize:
            self.parameters /= self.parameters.sum()


class NormalizeGrad(Optimizer):
    """
    A wrapper wrapping another Optimizer, normalizing the gradient before
    each step.
    """
    def __init__(self, opt, ord=None):
        super().__init__()
        self.opt = opt
        self.ord = ord

    def reset(self, parameters):
        """
        Reset this optimizer.
        """
        super().reset(parameters)
        self.opt.reset(parameters)

    def step(self, grad, *args, **kwargs):
        """
        Perform a single optimization step.

        """
        return self.opt.step(grad / np.linalg.norm(grad, self.ord), *args, **kwargs)


def linear_decay(lr0=0.2, decay_rate=1.0, decay_steps=21):
    """
    Linear learning-rate decay.

    """
    def _lr(k):
        return lr0 / (1.0 + decay_rate * np.floor(k / decay_steps))

    return _lr


def power_decay(lr0=0.2, decay_rate=1.0, decay_steps=1, power=2):
    """
    Power-based learning-rate decay.

    """
    def _lr(k):
        return lr0 / (decay_rate * np.floor(k / decay_steps) + 1.0)**power

    return _lr


def exponential_decay(lr0=0.2, decay_rate=0.5, decay_steps=21):
    """
    Exponential learning-rate decay.

    C
    """
    def _lr(k):
        return lr0 * np.exp(-decay_rate * np.floor(k / decay_steps))

    return _lr


class Initializer:
    """
    Base-class for an Initializer, specifying a strategy for parameter
    initialization.
    """
    def __init__(self):
        pass

    def initialize(self, shape):
        """
        Create an initial set of parameters.

        """
        raise NotImplementedError

    def __call__(self, shape):
        """
        Create an initial set of parameters.
        """
        return self.initialize(shape)


class Uniform(Initializer):
    """
    An Initializer, initializing parameters according to a specified uniform
    """
    def __init__(self, low=0.0, high=1.0):
        super().__init__()
        self.low = low
        self.high = high

    def initialize(self, shape):
        """
        Create an initial set of uniformly random distributed parameters.
        """
        return np.random.uniform(size=shape, low=self.low, high=self.high)


class Constant(Initializer):
    """
    An Initializer, initializing parameters to a constant value.
    """
    def __init__(self, value=1.0):
        super().__init__()
        self.value = value

    def initialize(self, shape):
        """
        Create set of parameters with initial fixed value.
        """
        if callable(self.value):
            return np.ones(shape) * self.value(shape)
        else:
            return np.ones(shape) * self.value
