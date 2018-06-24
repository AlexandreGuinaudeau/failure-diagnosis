import os
import numpy as np
import time
from datetime import timedelta
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
from thesis import BaseGenerator, BinarySegmentationModel, HMMModel, get_all_colors
from sklearn import linear_model


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
        restored = False
        if arrays is not None:
            self.arrays = pd.DataFrame(data=arrays, columns=tags)
        else:
            assert out is not None
            self.arrays = self.restore()
            restored = True
        if tags is None:
            tags = self.arrays.columns
        self.tags = tags
        self.kwargs = dict(kwargs)
        if not restored:
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
    fun returns (monte_carlo, theory)
    """
    if len(x) == 1 or len(y) == 1:
        return
    plt.figure(figsize=(20, 10))
    all_colors =get_all_colors(len(y))
    for i, yi in enumerate(y):
        c = all_colors[i]
        plt.plot(x, [fun(xj, yi)[0] for xj in x], color=c, label="%s%.2f" % (label_prefix, yi))
        plt.plot(x, [fun(xj, yi)[1] for xj in x], color=c, linestyle='dashed')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def _plot_cube(f, x_label, y_label, z_label, score_label, x, y_array, z_array):
    plot_cube(f, y_array, z_array, label_prefix=z_label + ":",
              title="%s=%s" % (x_label, x), xlabel=y_label, ylabel=score_label)


def plot_all_cubes(fun_mc, fun_theory, x_label, y_label, z_label, score_label, x_array, y_array, z_array):
    for x in x_array:
        _plot_cube(lambda y, z: (fun_mc(x, y, z), fun_theory(x, y, z)),
                   x_label, y_label, z_label, score_label, x, y_array, z_array)
        _plot_cube(lambda z, y: (fun_mc(x, y, z), fun_theory(x, y, z)),
                   x_label, z_label, y_label, score_label, x, z_array, y_array)

    for y in y_array:
        _plot_cube(lambda z, x: (fun_mc(x, y, z), fun_theory(x, y, z)),
                   y_label, z_label, x_label, score_label, y, z_array, x_array)
        _plot_cube(lambda x, z: (fun_mc(x, y, z), fun_theory(x, y, z)),
                   y_label, x_label, z_label, score_label, y, x_array, z_array)

    for z in z_array:
        _plot_cube(lambda x, y: (fun_mc(x, y, z), fun_theory(x, y, z)),
                   z_label, x_label, y_label, score_label, z, x_array, y_array)
        _plot_cube(lambda y, x: (fun_mc(x, y, z), fun_theory(x, y, z)),
                   z_label, y_label, x_label, score_label, z, y_array, x_array)


def fit_polynom(proba, alpha, max_t, n_tries, fit_intercept=True, weight=False, model_class=BinarySegmentationModel):
    assert model_class in (BinarySegmentationModel, HMMModel)
    X = []
    y = []
    weights = []
    for p in proba:
        for a in alpha:
            for t in max_t:
                if model_class == BinarySegmentationModel:
                    out = "z_value_p%i_obs%i_tries%i.csv" % (int(100 * p), t, n_tries)
                    X.append([(0.5-p)**2, a, np.sqrt(t)/100])
                    y.append(MonteCarloResult(out=out).sorted_values[int((1-a) * n_tries)]/norm.ppf(1 - a / 2)
                             - (1-fit_intercept))
                    weights.append(t)
                else:
                    out = "best_transition_probability_s%i_p%s_n%s_tries%s_precision%s_err%s.csv" \
                          % (101, str(p), str(t), str(n_tries), str(0.001), str(0.01))
                    X.append([a, a*abs(0.5-p)])
                    y.append(np.percentile(MonteCarloResult(out=out).all_values*np.sqrt(t), int(100 * a)))
                    weights.append(t)

    clf = linear_model.LinearRegression(fit_intercept=fit_intercept)
    clf.fit(X, y)
    score = 1 - clf.score(X, y, sample_weight=weights if weight else None)
    if model_class == BinarySegmentationModel:
        print "z / norm.ppf(1 - a / 2) = %.2f + %.2f|0.5-p|^2 + %.2fa + %.2fsqrt(t)/100 +/- %.2f" \
              % tuple([clf.intercept_ if fit_intercept else 1] + list(clf.coef_) + [score])
    else:
        print "ratio * sqrt(t) = %.2f + %.2fa + %.2fa|0.5-p| +/- %.2f" \
              % tuple([clf.intercept_ if fit_intercept else 0] + list(clf.coef_) + [score])
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
                                change_state_ratio=pow(10, factor),
                                states=states,
                                **kwargs)\
                .has_change_point(l)
        return fun

    res = []
    for _ in range(n_tries):
        x = bg.generate(max_t)
        res.append(pow(10, _dichotomy_search(wrapper(x), -3, 1, precision)))
    return MonteCarloResult(res, ["all_values"], out=out)


def best_transition_probability(all_nstates, all_p, all_t, all_alpha, precision=0.001, n_tries=100, plot=True):
    i = 0
    d_values = {}
    start = time.time()
    for t in all_t:
        for p in all_p:
            for nstates in all_nstates:
                if not theory_is_significant(p, nstates, min(all_alpha), max(all_t)):
                    continue
                mc_result = transition_probability_for_state(nstates, p, t, precision=precision, n_tries=n_tries)
                l_ = mc_result.all_values
                for a in all_alpha:
                    if not theory_is_significant(p, nstates, a, t):
                        continue
                    d_values[p, nstates, t, a] = np.percentile(l_, int(100 * a))
                i += 1
                print pretty_print_progress(float(i) / (len(all_t)*len(all_p)*len(all_nstates)), start)

    if plot:
        plot_all_cubes(lambda p, t, a: d_values[p, 101, t, a] if (p, 101, t, a) in d_values.keys() else None,
                       lambda p, t, a: HMMModel.change_state_ratio_theory(p, a, t),
                       "Probability", "Observations", "Error", "Transition Probability Ratio",
                       all_p, all_t, all_alpha)
    return d_values


def theory_is_significant(p, n_states, alpha, max_observation):
    return p*n_states > 1 and max_observation > 1.5 * np.log(1 - alpha) / np.log(max(p, 1 - p))


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

    plot_all_cubes(f, BinarySegmentationModel.gauss_threshold,
                   "Error", "Probability", "Observations", "Z-value threshold",
                   alpha, proba, max_t)


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


def plot_model_sensitivity(model_class, a, p1, t, p2s, t_changes=(0.5,), precision=0.01, recompute=False, show=True,
                           **plot_kwargs):
    start = time.time()
    all_colors = get_all_colors(len(t_changes))
    for j, t_change in enumerate(t_changes):
        if isinstance(t_change, float):
            t_change = int(t * t_change)
        out = "%s_sensitivity_a%s_p_%s_t%s_change%s.csv" % (str(model_class), str(a), str(p1), str(t), str(t_change))
        if MonteCarloResult.exists(out, force=recompute):
            y = MonteCarloResult(out=out).all_values
        else:
            y = []
            for i, p2 in enumerate(p2s):
                y.append(change_point_detection_probability(model_class, a, p1, t, p2, t_change, precision, states=101))
                print pretty_print_progress(float(i + j * len(p2s) + 1)/(len(p2s)*len(t_changes)), start)
            y = MonteCarloResult({"all_values": y}, out=out).all_values
        plt.plot(p2s, y, color=all_colors[j], label="Change after %i of %i observations" % (t_change, t), **plot_kwargs)
    plt.legend()
    d_names = {BinarySegmentationModel: "Binary Segmentation", HMMModel: "Hidden Markov"}
    plt.title("Sensitivity of the %s Model" % d_names[model_class])
    if show:
        plt.show()


def plot_model_delay(model_class, a, p1, t, p2s, max_wait, n_tries=100, precision=1, recompute=False, show=True,
                     **plot_kwargs):
    start = time.time()
    all_colors = get_all_colors(len(p2s))
    i = 0
    for j, p2 in enumerate(p2s):
        out = "%s_delay_a%s_p1_%s_p2_%s_t%s_wait%s.csv" % (str(model_class), str(a), str(p1),  str(p2), str(t), str(max_wait))
        if MonteCarloResult.exists(out, force=recompute):
            y = MonteCarloResult(out=out).all_values
        else:
            y = []
            bg = BaseGenerator((p1, p2), (t, max_wait))

            def wrapper(l):
                def fun(k, i, j):
                    return not model_class(error_threshold=a).has_change_point(l[:int(k)])
                return fun

            for _ in range(n_tries):
                x = bg.generate(t+max_wait)
                y.append(int(_dichotomy_search(wrapper(x), 0, t+max_wait, precision)))
                i += 1
            print pretty_print_progress(float(i)/(len(p2s)*n_tries), start)
            y = MonteCarloResult({"all_values": y}, out=out).all_values
        y = np.sort(y)
        y = y[y < t+max_wait-1]
        plt.plot(sorted(y), np.arange(0, float(len(y))/n_tries, float(1)/n_tries), color=all_colors[j],
                 label="p2=%.2f" % p2, **plot_kwargs)
    plt.legend(bbox_to_anchor=(1.04, 1))
    d_names = {BinarySegmentationModel: "Binary Segmentation", HMMModel: "Hidden Markov"}
    plt.title("Delay of the %s Model" % d_names[model_class])
    if show:
        plt.show()


if __name__ == "__main__":
    # ### REPORT PLOTS ### #
    plt.figure(figsize=(20, 10))
    p_ = 0.2
    all_alpha_ = [0.2, 0.15, 0.1, 0.05, 0.01]
    T_ = 100
    n_states_ = 101
    precision_ = 0.001
    n_tries_ = 1000
    # n_states <-> rho: all_nstates_ = list(range(6, 102, 5)), all_alpha_ = [0.2, 0.15, 0.1, 0.05, 0.01], plt.title("Independence of N and rho"), plt.xlabel("Number of states N")
    # d_values = best_transition_probability(all_nstates_, [0.2], [T_], all_alpha_, precision=precision_, n_tries=n_tries_)
    # for i, alpha_ in enumerate(all_alpha_):
    #     plt.plot(all_nstates_, [d_values[p_, s, T_, alpha_] for s in all_nstates_], label="alpha=%.2f" % alpha_, color=colors[i])

    # alpha <-> rho: all_alpha_ = np.arange(0.01, 0.5, 0.02), all_p_ = [0.1, 0.2, 0.3, 0.4, 0.5], plt.title("Relationship between alpha and rho"), plt.xlabel("Level of confidence alpha")
    # d_values = best_transition_probability([n_states_], all_p_, [T_], all_alpha_, precision=precision_, n_tries=n_tries_, plot=False)
    # for i, p_ in enumerate(all_p_):
    #     plt.plot(all_alpha_, [d_values[p_, n_states_, T_, alpha_] for alpha_ in all_alpha_], label="p=%.2f" % p_, color=colors[i])

    # T <-> rho
    all_t_ = [25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000]
    colors = get_all_colors(len(all_alpha_))
    d_values = best_transition_probability([n_states_], [p_], all_t_, all_alpha_, precision=precision_,
                                           n_tries=n_tries_, plot=False)
    for i, alpha_ in enumerate(all_alpha_):
        plt.plot(all_t_, [d_values[p_, n_states_, T_, alpha_] if theory_is_significant(p_, n_states_, alpha_, T_) else None for T_ in all_t_], label="alpha=%.2f" % alpha_, color=colors[i])
        plt.plot(all_t_, [HMMModel.change_state_ratio_theory(0.4, alpha_, T_) for T_ in all_t_], color=colors[i], linestyle="dashed")
    plt.title("Relationship between T and rho")
    plt.xlabel("Number of observations T")
    plt.ylabel("Maximum transition probability ratio rho")
    plt.legend(loc="upper right")
    plt.show()

    # ### BinarySegmentation ###
    # all_proba = [0.1, 0.2, 0.3]  # np.arange(0.05, 1, 0.05)
    # all_alpha = [0.2, 0.1, 0.05, 0.01]  # [0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01]
    # all_max_t = [500, 1000]  # [100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
    # n_tries_ = 100

    # statistical_test_threshold(HMMModel, all_proba, all_alpha, all_max_t, n_tries_, proba_change_state=0.5)
    # fit_polynom(all_proba, all_alpha, all_max_t, n_tries_, fit_intercept=False, weight=False)

    # other_p_ = [0.01, 0.05] + list(np.arange(0.1, 1, .1)) + [1]

    # plot_model_sensitivity(HMMModel, 0.05, p_, 100, other_p_,
    #                        (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95), show=False)
    #
    # plot_model_sensitivity(BinarySegmentationModel, 0.05, p_, 100, other_p_,
    #                        (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95), linestyle="dashed")

    # plot_model_delay(HMMModel, 0.05, p_, 100, other_p_, 1000, show=False)
    # plot_model_delay(BinarySegmentationModel, 0.05, p_, 100, other_p_, 1000, linestyle="dashed")

    # ### HMM ###
    # start = time.time()
    # all_t_ = [25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000]
    # all_states_ = [101]  # list(range(11, 101, 10)) + list(range(101, 251, 50))
    # all_p_ = [0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5]
    # n_tries_ = 100
    # all_alpha_ = [0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01]
    # d_values = {}
    # i = 0

    # best_transition_probability(all_states_, all_p_, all_t_, all_alpha_)
    # fit_polynom(all_p_, all_alpha_, all_t_, n_tries_, fit_intercept=False, weight=True, model_class=HMMModel)
