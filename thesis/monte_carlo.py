import os
import numpy as np
import time
from datetime import timedelta
import pandas as pd
from thesis import BaseGenerator, BinarySegmentationModel
import matplotlib.pyplot as plt


class MonteCarloResult:
    def __init__(self, arrays=None, tags=None, out=None, **kwargs):
        dir_path = os.path.join(os.getcwd(), "monte_carlo")
        if out is None:
            for i in range(1000):
                out = os.path.join(dir_path, "%i.csv" % i)
                if not os.path.exists(out):
                    break
        else:
            out = os.path.join(dir_path, out)
        self.out = out
        if arrays is not None:
            self.arrays = pd.DataFrame(data=arrays, columns=tags)
        else:
            assert out is not None
            self.arrays = self.restore()
        if tags is None:
            tags = self.arrays.columns
        self.tags = tags
        self.kwargs = dict(kwargs)
        self.save()

    @staticmethod
    def exists(out):
        dir_path = os.path.join(os.getcwd(), "monte_carlo")
        out = os.path.join(dir_path, out)
        return os.path.exists(out)

    def __getattr__(self, item):
        if item in self.tags:
            return self.arrays[item]

    def save(self):
        self.arrays.to_csv(self.out, **self.kwargs)

    def restore(self):
        return pd.read_csv(self.out)

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


def timeit(f):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        print(time.time() - start)
        return result
    return wrapper


def cast_all_args(dtype, ignore_first=True):
    def decorator(f):
        def wrapper(self, *args):
            args = [dtype(i) for i in args]
            if not ignore_first:
                self = dtype(self)
            return f(self, *args)
        return wrapper
    return decorator


@cast_all_args(float)
def test_statistic(n1, n2, p1, p2):
    p = (n1 * p1 + n2 * p2) / (n1 + n2)
    if p == 0 or p == 1:
        return 0
    return (p1 - p2) / np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))


@timeit
def monte_carlo_constant(p, max_observations, n_tries=100, window=5):
    """
    Using Monte Carlo method, find the constant k such that
    z_alpha = z_theory*k*sqrt(t)
    """
    out = "p%i_obs%i_tries%i.csv" % (int(100*p), max_observations, n_tries)
    if MonteCarloResult.exists(out):
        return MonteCarloResult(out=out)
    bg = BaseGenerator(p, max_observations)
    k_extremum = {}
    for n_try in range(n_tries):
        k_extremum[n_try] = np.array([[-1, -1], [0, 0]])

    def memoized_test_statistic(t, w, observations):
        return max(np.abs([test_statistic(k, t - k, np.mean(observations[:k]), np.mean(observations[k:t]))
                          for k in range(w, t-w)]))

    monte_carlo = np.array([
            [memoized_test_statistic(t, window, bg.generate(max_observations))
             for t in range(2*window+1, max_observations, window)]
            for _ in range(n_tries)
        ])
    theory = np.array([np.sqrt(t) for t in range(2*window+1, max_observations, window)])
    return MonteCarloResult({"mean": monte_carlo.mean(axis=0), "std": monte_carlo.std(axis=0), "theory": theory},
                            out=out)


def correlation_coefficient(p, max_observation=None, window=None, n_tries=None, approx=True):
    if approx:
        return 0.5 - 12 * (p-0.5)**6
    mc_result = monte_carlo_constant(p, max_observation, n_tries, window)
    return np.mean((mc_result.mean / mc_result.theory)[int(len(mc_result.mean) / 2):])


def plot_correlation_coefficient_approximation(all_p=None, w=5, obs=500):
    if all_p is None:
        all_p = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    l = {}
    for p in all_p:
        mc_result = monte_carlo_constant(p, obs, 100, w)
        n = len(mc_result.mean)
        res = np.mean((mc_result.mean / mc_result.theory)[int(n / 2):])
        l[p] = res

    keys = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                     0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
    l = [correlation_coefficient(p, obs, w) for p in keys]
    plt.plot(keys, l)
    plt.plot(keys, [correlation_coefficient(k, approx=True) for k in keys])
    plt.show()


@cast_all_args(float)
def _dichotomy_search(fun, i, j, min_delta):
    k = (i + j) / 2
    try:
        if j-i < min_delta:
            return k
        if fun(k):
            return _dichotomy_search(fun, k, j, min_delta)
        return _dichotomy_search(fun, i, k, min_delta)
    except RuntimeError as e:
        print "Too many recursions, final (i, j)", i, j
        if str(e) == "maximum recursion depth exceeded while calling a Python object":
            return k
        raise e


# @timeit
def monte_carlo_bast_split(p1, p2, n, n_tries, precision, error_threshold):
    if error_threshold != 0.01:
        out = "best_split_%s_%s_n%s_tries%s_precision%s_err%s.csv" \
              % (str(p1), str(p2), str(n), str(n_tries), str(precision), str(error_threshold))
    else:
        out = "best_split_%s_%s_n%s_tries%s_precision%s.csv" % (str(p1), str(p2), str(n), str(n_tries), str(precision))
    if MonteCarloResult.exists(out):
        return MonteCarloResult(out=out)
    bg = BaseGenerator([p1, p2], [n, n])

    def wrapper(l, err):
        def fun(factor):
            return BinarySegmentationModel(error_threshold=err,
                                           gauss_threshold_function=
                                           lambda p, t: 1 + correlation_coefficient(p) * np.sqrt(t) * factor)\
                .segmentation(l, return_full_segmentation=False)
        return fun

    res = []
    for _ in range(n_tries):
        x = bg.generate(2*n)
        res.append(_dichotomy_search(wrapper(x, error_threshold), 0, 1, precision))
    return MonteCarloResult(res, ["all_values"], out=out)


if __name__ == "__main__":
    n_tries_ = 100
    precision_ = 2e-3
    max_observation_ = 1000
    all_errs_ = np.array([0.01, 0.02, 0.05, 0.1])

    color_map = dict(zip([0, 1, 2, 3, 4, 5],
                         ["#1F4B99", "#578FA1", "#BACBA1", "#F2BE75", "#CD7534", "#9E2B0E"]))

    d_values = {}
    start = time.time()
    for e_ind, error_threshold_ in enumerate(all_errs_):
        for p_ind, p1 in enumerate([0.01, 0.1, 0.2, 0.3, 0.4, 0.5]):
            for p2_ind, p2 in enumerate([0.01, 0.1, 0.2, 0.3, 0.4, 0.5]):
                mc_result = monte_carlo_bast_split(p1, p2, max_observation_, n_tries_, precision_, error_threshold_)
                l_ = mc_result.all_values
                d_values[error_threshold_, p1, p2] = np.percentile(l_, 95)

    for p1 in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for p2 in [0.1, 0.2, 0.3, 0.4, 0.5]:
            values = []
            for error_threshold_ in all_errs_:
                values.append(d_values[error_threshold_, p1, p2])
            plt.plot(all_errs_, values, 'g' if p1 == p2 else 'r', label=str(abs(p1-p2)))
            print(p1, p2, np.polyfit(all_errs_, values, 1))

    plt.plot(all_errs_, 0.4*all_errs_+0.01, 'b')

    # print "f(p1=%s, p2=%s, error=%s) = %s" \
    #       % (str(p1), str(p2), str(error_threshold_), str(np.percentile(l_, 95)))

    #     try:
    #         plt.hist(l_, bins=np.arange(0, 0.1, precision_), histtype="step", label=str(p2))
    #     except ValueError as e:
    #         if str(e) == "zero-size array to reduction operation minimum which has no identity":
    #             pass
    #         else:
    #             raise e
    plt.legend()
    plt.show()

    # p     | alpha | k
    # ------|-------|------
    # 0.2   | 0.1   | 0.035
    # 0.2   | 0.05  | 0.035
    # 0.2   | 0.03  | 0.025
    # 0.2   | 0.02  | 0.018
    # 0.2   | 0.01  | 0.01
