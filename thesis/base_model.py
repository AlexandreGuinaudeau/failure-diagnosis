import numpy as np
import matplotlib.pyplot as plt
from thesis import BaseGenerator


class BaseModel(object):
    def __init__(self, error_threshold):
        self.error_threshold = error_threshold

    def run(self, x, return_full_prediction=True):

        """
        Given a sequence of events (one-hot represented),
        Returns a predicted sequence of underlying probabilities
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    @staticmethod
    def change_points_to_sequence(change_points, values):
        assert len(change_points) == len(values)+1, "%i != %i+1" % (len(change_points), len(values))
        return np.concatenate([[values[i]] * (change_points[i + 1] - change_points[i])
                               for i in range(len(change_points)-1)])

    def plot_prediction(self, x, generator=None, show=True, save_fig_path=None):
        y = self(x)
        ind = np.arange(len(x))
        plt.plot(ind, x, 'b+', label="events")
        plt.plot(ind, y, 'g', label="predicted")

        if generator is not None:
            assert isinstance(generator, BaseGenerator)
            z = self.change_points_to_sequence([0] + list(np.cumsum(generator.durations[:-1])) + [len(x)],
                                               generator.probabilities)
            plt.plot(ind, z, 'r', label="true")

        plt.ylim(-0.1, 1.1)
        # plt.legend()
        if show:
            plt.show()
        if save_fig_path is not None:
            plt.savefig(save_fig_path)
        return y
