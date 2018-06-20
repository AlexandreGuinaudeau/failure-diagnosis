import os
import numpy as np
import time
from datetime import timedelta
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
from thesis import BaseGenerator, BinarySegmentationModel, HMMModel
from sklearn import linear_model


ALL_COLORS = ["#1F4B99", "#2B5E9C", "#38709E", "#48819F", "#5B92A1", "#71A3A2", "#8AB3A2",
              "#A7C3A2", "#C7D1A1", "#EBDDA0", "#FCD993", "#F5C57D", "#EDB269", "#E49F57",
              "#DA8C46", "#CF7937", "#C4662A", "#B8541E", "#AB4015", "#9E2B0E"]


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

    def __str__(self):
        return "<MonteCarloResult shape=%s tags=%s out=%s/>" % (str(self.arrays.shape), str(self.tags), str(self.out))

    @staticmethod
    def exists(out, force=False):
        if force:
            return False
        dir_path = os.path.join(os.getcwd(), "monte_carlo")
        out = os.path.join(dir_path, out)
        return os.path.exists(out)

    def __getattr__(self, item):
        if item in self.tags:
            return self.arrays[item]

    def save(self):
        if len(self.arrays) == 0:
            raise ValueError(str(self) + ": Array is of size 0")
        self.arrays.to_csv(self.out, **self.kwargs)

    def restore(self):
        return pd.read_csv(self.out)


# ### UTILS ### #

def cast_all_args(dtype, ignore_first=True):
    def decorator(f):
        def wrapper(self, *args):
            args = [dtype(i) for i in args]
            if not ignore_first:
                self = dtype(self)
            return f(self, *args)
        return wrapper
    return decorator


def timeit(f):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        print(time.time() - start)
        return result
    return wrapper


@cast_all_args(float, False)
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


@cast_all_args(float)
def _dichotomy_search(fun, i, j, min_delta):
    k = (i + j) / 2
    try:
        if j-i < min_delta:
            return k
        if fun(k, i, j):
            return _dichotomy_search(fun, k, j, min_delta)
        return _dichotomy_search(fun, i, k, min_delta)
    except RuntimeError as e:
        print "Too many recursions, final (i, j)", i, j
        if str(e) == "maximum recursion depth exceeded while calling a Python object":
            return k
        raise e
    except StopIteration:
        return k


@cast_all_args(float)
def test_statistic(n1, n2, p1, p2):
    p = (n1 * p1 + n2 * p2) / (n1 + n2)
    if p == 0 or p == 1:
        return 0
    return (p1 - p2) / np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))


# ### BINARY SEGMENTATION ### #
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


# @timeit
def monte_carlo_best_split(p1, p2, n, n_tries, precision, error_threshold):
    if error_threshold != 0.01:
        out = "best_split_%s_%s_n%s_tries%s_precision%s_err%s.csv" \
              % (str(p1), str(p2), str(n), str(n_tries), str(precision), str(error_threshold))
    else:
        out = "best_split_%s_%s_n%s_tries%s_precision%s.csv" % (str(p1), str(p2), str(n), str(n_tries), str(precision))
    if MonteCarloResult.exists(out):
        return MonteCarloResult(out=out)
    bg = BaseGenerator([p1, p2], [n, n])

    def wrapper(l, err):
        def fun(factor, **kwargs):
            return BinarySegmentationModel(error_threshold=err,
                                           gauss_threshold_function=
                                           lambda p, t: 1 + correlation_coefficient(p) * np.sqrt(t) * factor)\
                .has_change_point(l)
        return fun

    res = []
    for _ in range(n_tries):
        x = bg.generate(2*n)
        res.append(_dichotomy_search(wrapper(x, error_threshold), 0, 1, precision))
    return MonteCarloResult(res, ["all_values"], out=out)


def plot_cube(fun, x, y, label_prefix="", title=None, xlabel=None, ylabel=None):
    """
    Plots z=fun(x, y) with x as axis, for each value of y
    fun returns (actual, theory)
    """
    if len(x) == 1 or len(y) == 1:
        return
    plt.figure(figsize=(20, 10))
    all_colors = [ALL_COLORS[int(i*20/len(y))] for i in range(len(y))]
    for i, yi in enumerate(y):
        c = all_colors[i]
        plt.plot(x, [fun(xj, yi)[0] for xj in x], color=c, label="%s%.2f" % (label_prefix, yi))
        plt.plot(x, [fun(xj, yi)[1] for xj in x], color=c, linestyle='dashed')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def fit_polynom(proba, alpha, max_t, n_tries, fit_intercept=True, weight=False):
    X = []
    y = []
    weights = []
    for p in proba:
        for a in alpha:
            for t in max_t:
                out = "z_value_p%i_obs%i_tries%i.csv" % (int(100 * p), t, n_tries)
                X.append([(0.5-p)**2, a, np.sqrt(t)/100])
                y.append(MonteCarloResult(out=out).sorted_values[int((1-a) * n_tries)]/norm.ppf(1 - a / 2)
                         - (1-fit_intercept))
                weights.append(t)

    clf = linear_model.LinearRegression(fit_intercept=fit_intercept)
    clf.fit(X, y)
    score = 1 - clf.score(X, y, sample_weight=weights if weight else None)
    print "z = %.2f + %.2f|0.5-p|^2 + %.2fa + %.2fsqrt(t)/100 +/- %.2f" \
          % tuple([clf.intercept_ if fit_intercept else 1] + list(clf.coef_) + [score])
    return clf.intercept_, clf.coef_


# ### HMM MODEL ### #

def transition_probability_for_state(states, p, max_t, precision=0.1, n_tries=100, error_threshold=0.01, **kwargs):
    out = "best_transition_probability_s%i_p%s_n%s_tries%s_precision%s_err%s.csv" \
          % (states, str(p), str(max_t), str(n_tries), str(precision), str(error_threshold))
    if MonteCarloResult.exists(out):
        return MonteCarloResult(out=out)
    bg = BaseGenerator(p, max_t)

    def wrapper(l):
        def fun(factor, i, j):
            return not HMMModel(error_threshold=error_threshold,
                                proba_change_state=np.exp(factor),
                                states=states,
                                **kwargs)\
                .has_change_point(l)
        return fun

    res = []
    for _ in range(n_tries):
        x = bg.generate(max_t)
        res.append(np.exp(_dichotomy_search(wrapper(x), -3, 0, precision)))
    return MonteCarloResult(res, ["all_values"], out=out)


# ### BOTH MODELS ### #

def statistical_test_threshold(model_class, proba, alpha, max_t, n_tries=100, **kwargs):
    assert model_class in (BinarySegmentationModel, HMMModel)
    if model_class == BinarySegmentationModel:
        prefix = "z_value"
    else:
        prefix = "hmm_states"
    if isinstance(proba, float):
        proba = [proba]
    if isinstance(alpha, float):
        alpha = [alpha]
    if isinstance(max_t, int):
        max_t = [max_t]

    for a in alpha:
        assert np.floor(a*n_tries) == np.ceil(a*n_tries), "Increase n to have enough sensitivity for alpha=%f" % a

    res = {}
    start = time.time()
    i = 0
    model = model_class(**kwargs)
    for p in proba:
        for t in max_t:
            i += 1
            out = "%s_p%i_obs%i_tries%i.csv" % (prefix, int(100 * p), t, n_tries)
            if MonteCarloResult.exists(out):
                res[p, t] = MonteCarloResult(out=out)
                continue
            bg = BaseGenerator(p, t)
            l = [model.max_param(bg.generate(t)) for _ in range(n_tries)]
            res[p, t] = MonteCarloResult({"sorted_values": sorted(l)}, out=out)
            print pretty_print_progress(float(i)/(len(proba) * len(max_t)), start)

    def f(a_, p_, t_):
        return res[p_, t_].sorted_values[int((1-a_) * n_tries)]

    for a in alpha:
        plot_cube(lambda p, t: (f(a, p, t), BinarySegmentationModel.gauss_threshold(a, p, t)), proba, max_t,
                  label_prefix="Observations: ", title="alpha=%.2f" % a, xlabel="p", ylabel="t")
        plot_cube(lambda t, p: (f(a, p, t), BinarySegmentationModel.gauss_threshold(a, p, t)), max_t, proba,
                  label_prefix="Probability: ", title="alpha=%.2f" % a, xlabel="t", ylabel="p")
    for p in proba:
        plot_cube(lambda a, t: (f(a, p, t), BinarySegmentationModel.gauss_threshold(a, p, t)), alpha, max_t,
                  label_prefix="Observations: ", title="probability=%.2f" % p, xlabel="a", ylabel="t")
        plot_cube(lambda t, a: (f(a, p, t), BinarySegmentationModel.gauss_threshold(a, p, t)), max_t, alpha,
                  label_prefix="Error: ", title="probability=%.2f" % p, xlabel="t", ylabel="a")
    for t in max_t:
        plot_cube(lambda a, p: (f(a, p, t), BinarySegmentationModel.gauss_threshold(a, p, t)), alpha, proba,
                  label_prefix="Probability: ", title="observations=%i" % t, xlabel="a", ylabel="p")
        plot_cube(lambda p, a: (f(a, p, t), BinarySegmentationModel.gauss_threshold(a, p, t)), proba, alpha,
                  label_prefix="Error: ", title="observations=%i" % t, xlabel="p", ylabel="a")


def change_point_detection_probability(model_class, a, p1, t, p2, t_change, precision=0.01, states=None):
    assert model_class in (BinarySegmentationModel, HMMModel)
    if p1 == p2:
        return a
    if isinstance(t_change, float):
        assert 0 < t_change < 1
        t_change = int(t_change * t)
    assert 0 < precision < 1
    n_tries = int(1/precision)
    bg = BaseGenerator([p1, p2], [t_change, t-t_change])
    if model_class == BinarySegmentationModel:
        model = BinarySegmentationModel(a)
    else:
        model = HMMModel(a, states=states)
    return np.mean([model.has_change_point(bg.generate(t)) for _ in range(n_tries)])


def plot_model_sensitivity(model_class, a, p1, t, p2s, t_changes=(0.5,), precision=0.01):
    start = time.time()
    all_colors = [ALL_COLORS[int(i * 20 / len(t_changes))] for i in range(len(t_changes))]
    states = int(1 + 1/min([abs(p1-p2) for p2 in p2s if p1 != p2]))
    for j, t_change in enumerate(t_changes):
        if isinstance(t_change, float):
            t_change = int(t * t_change)
        y = []
        for i, p2 in enumerate(p2s):
            y.append(change_point_detection_probability(model_class, a, p1, t, p2, t_change, precision, states))
            print pretty_print_progress(float(i + j * len(p2s) + 1)/(len(p2s)*len(t_changes)), start)
        plt.plot(p2s, y, color=all_colors[j], label="Change after %i of %i observations" % (t_change, t))
    plt.legend()
    plt.title("Sensitivity of the Binary Segmentation Model")
    plt.show()


if __name__ == "__main__":
    # all_proba = [0.1, 0.2, 0.3]  # np.arange(0.05, 1, 0.05)
    # all_alpha = [0.2, 0.1, 0.05, 0.01]  # [0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01]
    # all_max_t = [500, 1000]  # [100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
    # n_tries_ = 100

    # statistical_test_threshold(HMMModel, all_proba, all_alpha, all_max_t, n_tries_, proba_change_state=0.5)
    # fit_polynom(all_proba, all_alpha, all_max_t, n_tries_, fit_intercept=False, weight=False)

    # plot_model_sensitivity(HMMModel, 0.05, 0.2, 1000, list(np.arange(0.01, 0.4, .01)) + list(np.arange(0.4, 1, 0.1)),
    #                        (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95))

    start = time.time()
    max_observation_ = 1000
    p_ = 0.2
    all_states_ = [11, 51, 101, 151, 201, 251, 301, 351, 401, 451, 501]
    n_tries_ = 100
    thresholds = [80, 95, 99]
    d_values = {}
    i = 0
    for states_ in all_states_:
        mc_result = transition_probability_for_state(states_, p_, max_observation_, n_tries=n_tries_)
        l_ = mc_result.all_values
        print states_, sorted(l_)
        for t in thresholds:
            i += 1
            d_values[states_, t] = np.percentile(l_, t)
            print pretty_print_progress(float(i)/(len(thresholds)*len(all_states_)), start)
    for t in thresholds:
        plt.plot(all_states_, [d_values[s, t] for s in all_states_], label="Confidence level: %i%%" % t)
    plt.legend()
    plt.show()

    raise Exception

    # n_tries_ = 100
    # precision_ = 2e-3
    # max_observation_ = 1000
    # all_errs_ = np.array([0.01, 0.02, 0.05, 0.1])
    #
    # color_map = dict(zip([0, 1, 2, 3, 4, 5],
    #                      ["#1F4B99", "#578FA1", "#BACBA1", "#F2BE75", "#CD7534", "#9E2B0E"]))
    #
    # d_values = {}
    # start = time.time()
    # for e_ind, error_threshold_ in enumerate(all_errs_):
    #     for p_ind, p1_ in enumerate([0.01, 0.1, 0.2, 0.3, 0.4, 0.5]):
    #         for p2_ind, p2_ in enumerate([0.01, 0.1, 0.2, 0.3, 0.4, 0.5]):
    #             mc_result = monte_carlo_best_split(p1_, p2_, max_observation_, n_tries_, precision_, error_threshold_)
    #             l_ = mc_result.all_values
    #             d_values[error_threshold_, p1_, p2_] = np.percentile(l_, 95)
    #
    # for p1_ in [0.1, 0.2, 0.3, 0.4, 0.5]:
    #     for p2_ in [0.1, 0.2, 0.3, 0.4, 0.5]:
    #         values = []
    #         for error_threshold_ in all_errs_:
    #             values.append(d_values[error_threshold_, p1, p2_])
    #         plt.plot(all_errs_, values, 'g' if p1_ == p2_ else 'r', label=str(abs(p1_-p2_)))
    #         print(p1_, p2_, np.polyfit(all_errs_, values, 1))
    #
    # plt.plot(all_errs_, 0.4*all_errs_+0.01, 'b')
    #
    # # print "f(p1=%s, p2=%s, error=%s) = %s" \
    # #       % (str(p1), str(p2), str(error_threshold_), str(np.percentile(l_, 95)))
    #
    # #     try:
    # #         plt.hist(l_, bins=np.arange(0, 0.1, precision_), histtype="step", label=str(p2))
    # #     except ValueError as e:
    # #         if str(e) == "zero-size array to reduction operation minimum which has no identity":
    # #             pass
    # #         else:
    # #             raise e
    # plt.legend()
    # plt.show()
    #
    # # p     | alpha | k
    # # ------|-------|------
    # # 0.2   | 0.1   | 0.035
    # # 0.2   | 0.05  | 0.035
    # # 0.2   | 0.03  | 0.025
    # # 0.2   | 0.02  | 0.018
    # # 0.2   | 0.01  | 0.01
