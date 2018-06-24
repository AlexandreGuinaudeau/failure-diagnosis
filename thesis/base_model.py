import numpy as np
import matplotlib.pyplot as plt
from thesis import BaseGenerator
from sklearn.metrics import pairwise


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

    def has_change_point(self, x):
        raise NotImplementedError

    @staticmethod
    def change_points_to_sequence(change_points, values):
        assert len(change_points) == len(values)+1, "%i != %i+1" % (len(change_points), len(values))
        return np.concatenate([[values[i]] * (change_points[i + 1] - change_points[i])
                               for i in range(len(change_points)-1)])

    def score(self, x, generator, kind="l2", best_possible=True):
        """
        kind is 'cosine', 'euclidean', 'l2', 'l1', 'manhattan' or 'cityblock'
        if best_possible, uses change points instead of original probability
        """
        assert kind in pairwise.PAIRED_DISTANCES.keys()
        y = self(x)
        change_points = [0] + list(np.cumsum(generator.durations[:-1])) + [len(x)]
        if best_possible:
            z = self.change_points_to_sequence(change_points,
                                               [np.mean(x[change_points[i]:change_points[i + 1]])
                                                for i in range(len(change_points) - 1)])
        else:
            z = self.change_points_to_sequence(change_points, generator.probabilities)

        y = np.array(y).reshape(1, -1)
        z = np.array(z).reshape(1, -1)

        return pairwise.PAIRED_DISTANCES[kind](z, y)

    def plot_prediction(self, x, generator=None, show=True, save_fig_path=None, color=None, plot_x=True, label=None,
                        **plot_kwargs):
        if color is None:
            color = 'g'
        if label is None:
            label = "predicted"
        y = self.run(x)
        ind = np.arange(len(x))
        if plot_x:
            plt.plot(ind, x, 'b+', label="events", **plot_kwargs)
        plt.plot(ind, y, color, label=label, **plot_kwargs)

        if generator is not None:
            assert isinstance(generator, BaseGenerator)
            change_points = [0] + list(np.cumsum(generator.durations[:-1])) + [len(x)]
            z = self.change_points_to_sequence(change_points,
                                               [np.mean(x[change_points[i]:change_points[i+1]])
                                                for i in range(len(change_points) - 1)])
            if plot_x:
                plt.plot(ind, z, 'r', label="true", **plot_kwargs)

        plt.ylim(-0.1, 1.1)
        if label is not None:
            plt.legend()
        if show:
            plt.show()
        if save_fig_path is not None:
            plt.savefig(save_fig_path)
        return y
