import numpy as np
from thesis import BaseModel


class MovingAverageModel(BaseModel, object):
    def __init__(self, error_threshold=0.01):
        super(MovingAverageModel, self).__init__(error_threshold)

    @staticmethod
    def moving_average(a, window=20) :
        ret = np.cumsum(a, dtype=float)
        ret[window:] = ret[window:] - ret[:-window]
        ret = ret[window - 1:] / window
        return ret

    def run(self, x):
        pass
