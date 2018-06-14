import numpy as np
from scipy.stats import norm
import time
from datetime import timedelta
from thesis import BaseModel, BaseGenerator


def pretty_print_progress(progress, start):
    elapsed = time.time() - start
    if progress == 0:
        return {"elapsed": pretty_print_time(elapsed),
                "progress": progress}
    return {"elapsed": pretty_print_time(elapsed),
            "progress": progress,
            "total": pretty_print_time(elapsed/progress),
            "remaining": pretty_print_time(elapsed*(1/progress-1))}


def pretty_print_time(t):
    return str(timedelta(seconds=t)).split(".")[0]


def cast_all_args(dtype, ignore_first=True):
    def decorator(f):
        def wrapper(self, *args):
            args = [dtype(i) for i in args]
            if not ignore_first:
                self = dtype(self)
            return f(self, *args)
        return wrapper
    return decorator


class BinarySegmentationModel(BaseModel):
    def __init__(self, error_threshold=0.01, window=None, gauss_threshold_function=None):
        assert 0 < error_threshold <= 1
        super(BinarySegmentationModel, self).__init__(error_threshold)
        # self.gauss_threshold = norm.ppf(1 - error_threshold / 2)
        if window is None:
            window = self._min_window()
        self.window = window
        self.memoization = {}
        self.gauss_threshold_function = gauss_threshold_function

    def gauss_threshold(self, p, t):
        if self.gauss_threshold_function is not None:
            return self.gauss_threshold_function(p, t) * norm.ppf(1 - self.error_threshold / 2)
        k = 1 + (0.5 - 12 * (p - 0.5) ** 6) * np.sqrt(t) * (0.4 * self.error_threshold + 0.01)
        # k = 1
        return k * norm.ppf(1 - self.error_threshold / 2)

    def _min_window(self):
        gauss_threshold = norm.ppf(1 - self.error_threshold / 2)
        for n in range(1, int(1e6)):
            if self.test_statistic(n, n, 0, 1) > gauss_threshold:
                return n
        raise AttributeError("Gauss threshold is too large: %f" % self.gauss_threshold)

    @cast_all_args(float)
    def test_statistic(self, n1, n2, p1, p2):
        p = (n1 * p1 + n2 * p2) / (n1 + n2)
        if p == 0 or p == 1:
            return 0
        return abs(p1 - p2) / np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))

    def segmentation(self, l, i=0, j=None, return_full_segmentation=True):
        if (i, j) in self.memoization.keys():
            return self.memoization[i, j]
        if j is None:
            j = len(l)
        max_score = 0
        score = 0
        seg = [j]
        p = np.mean(l[:j])
        for k in range(i + self.window, j - self.window, self.window):
            n1 = k - i
            n2 = j - k
            p1 = np.mean(l[i:k])
            p2 = np.mean(l[k:j])
            z = self.test_statistic(n1, n2, p1, p2)
            if z > self.gauss_threshold(p, j):  # split at k?
                if return_full_segmentation is False:
                    return True
                # score1, seg1 = self.segmentation(l, i, k)
                score2, seg2 = self.segmentation(l, k, j)
                # score2, seg2 = self._segmentation(l, k, j, t)
                score = score2 + abs(p1-p2)
                # score = score1/len(seg1) + score2/len(seg2) + z
                if score > max_score:
                    max_score = score
                    seg = [k] + seg2
                    # seg = seg1 + [k] + seg2

        if return_full_segmentation is False:
            return False

        self.memoization[i, j] = score, seg
        return score, seg

    def run(self, x, return_full_prediction=True):
        x = np.array(x)
        if return_full_prediction is False:
            return self.segmentation(x, return_full_segmentation=False)
        change_points = [0] + self.segmentation(x)[1]
        n_change_points = len(change_points)-1
        probabilities = [np.mean(x[change_points[i]:change_points[i+1]]) for i in range(n_change_points)]
        return self.change_points_to_sequence(change_points, probabilities)

    def fit(self, probability, max_observations=1000, n_tries=100):
        all_max = []
        start = time.time()
        for i in range(n_tries):
            bg = BaseGenerator(probability, max_observations)
            observations = bg.generate(max_observations)
            max_discrepancy = max([max([self.test_statistic(k, t - k, np.mean(observations[:k]), np.mean(observations[k:t]))
                                        for k in range(self.window, t - self.window, self.window)])
                                   for t in range(2*self.window+1, max_observations, self.window)])
            all_max.append(max_discrepancy)
            print i, max_discrepancy, pretty_print_progress(float(i)/n_tries, start)
        all_max = sorted(all_max)
        for confidence in [0.8, 0.9, 0.95, 0.99, 0.999]:
            print "%f%% Max :" % (confidence*100), all_max[int(n_tries*confidence)]
        return all_max


if __name__ == "__main__":
    from scipy.stats import binom_test, fisher_exact
    from statsmodels.stats.proportion import proportions_ztest

    n1_ = 100
    n2_ = 20
    n11_ = 20
    n12_ = 15
    p1_ = float(n11_) / n1_
    p2_ = float(n12_) / n2_
    alt_ = 'two-sided'  # , 'greater', 'less'

    def alt(p1, p2, statsmodel=False):
        if statsmodel:
            return 'larger' if p1 > p2 else 'smaller'
        return 'greater' if p1 > p2 else 'less'

    def test_statistic(n1, n2, n11, n12):
        n11 = float(n11)
        n12 = float(n12)
        p = (n11 + n12) / (n1 + n2)
        p1 = n11 / n1
        p2 = n12 / n2
        test1 = binom_test(n11, n1, p, alternative=alt(p1, p2))
        test2 = binom_test(n12, n2, p, alternative=alt(p2, p1))
        return test1 * test2


    def get_example(method):
        bsm = BinarySegmentationModel()
        x = []
        start = time.time()
        for n11_ in range(1, n1_):
            p1_ = float(n11_) / n1_
            if method == "binom_test":
                x.append(test_statistic(n1_, n2_, n11_, n12_))
            elif method == "manual":
                x.append(bsm.test_statistic(n1_, n2_, p1_, p2_))
            elif method == "fisher":
                x.append(fisher_exact([[n11_, n12_], [n1_ - n11_, n2_ - n12_]], alt(p1_, p2_))[1])
            elif method == "proportions_ztest":
                x.append(proportions_ztest([n11_, n12_], [n1_, n2_])[0])
            else:
                raise NotImplementedError
        print time.time() - start
        x = np.array(x)
        if method in ("manual", "proportions_ztest"):
            x = norm.pdf(np.array(x))
        return x

    import matplotlib.pyplot as plt
    plt.plot(get_example("manual"), label="manual")
    plt.plot(get_example("proportions_ztest"), label="prop_ztest")
    plt.legend()
    plt.show()
