"""
Implements a custom sampler for the CEM controller
- Normally, our CEM controller draws from a multi-variate gaussian with 0 mean and diagonal sigma
- This class allows for a custom sampling distrubtion to be used in CEM instead
"""

class Sampler:
    def __init__(self, sigma, mean, hp, repeat, adim):
        pass

    def sample(self, itr, M, current_state, current_mean, current_sigma, close_override):
        pass
