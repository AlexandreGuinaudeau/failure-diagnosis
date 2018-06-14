import re
import os
import time
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors

CATEGORICAL = "CAT"
NUMERICAL = "NUM"
RESULTS_PATH = os.path.realpath(os.path.join(__file__, "..", "results.csv"))


class AssociationModel:
    def __init__(self, kind, n, p1=None, p2=None, n1=None, sigma=None, events=None, seed=None):
        self.kind = kind
        self.n = n
        assert kind in (CATEGORICAL, NUMERICAL)
        if kind == CATEGORICAL:
            assert p1 is not None
            assert p2 is not None
            assert n1 is not None
            self.p1 = p1
            self.p2 = p2
            self.n1 = n1
            self.n2 = n - n1
        else:
            assert sigma is not None
            assert events is not None
            self.sigma = sigma
            self.events = events
            self.num_events = len(events)
            self.length_numerical = max(2 * self.num_events, 10)

        if seed is not None:
            np.random.seed(seed)

    def __str__(self):
        return "<%s %s>" % (self.__class__.__name__, self.kind)

    def generate(self):
        if self.kind == CATEGORICAL:
            return np.array([r < self.p1 for r in np.random.rand(self.n1)]
                            + [r < self.p2 for r in np.random.rand(self.n2)])
        else:
            return self.sigma * np.random.randn(self.length_numerical, self.n)

    def score(self, x, y):
        raise NotImplementedError

    def best_candidate(self, rand, return_score=0):
        if self.kind == CATEGORICAL:
            y_matrix = np.tril(np.ones(self.n), 0)
            x_matrix = np.tile(rand, (self.n, 1))
            scores = [self.score(x_matrix[i], y_matrix[i]) for i in range(1, len(x_matrix) - 1)]
            candidate = np.argmax(scores)
        else:
            all_events = np.array(self.events)
            np.random.shuffle(all_events)
            other_events = np.array(list(set(np.arange(self.n)).difference(self.events)))
            np.random.shuffle(other_events)
            all_events = np.concatenate((all_events, other_events))

            y_matrix = np.zeros((self.length_numerical, self.n))
            for j in range(self.length_numerical):
                y_matrix[j][all_events[:j]] = 1
            y_matrix += rand
            x_matrix = np.zeros((1, self.n)).flatten()
            x_matrix[self.events] = 1
            x_matrix = np.tile(x_matrix, (self.length_numerical, 1))
            scores = [self.score(x_matrix[i], y_matrix[i]) for i in range(1, self.length_numerical)]
            candidate = 2*float(np.argmax(scores)+1-self.num_events)/self.num_events + 1
        if return_score == 1:
            return candidate, max(scores)
        elif return_score == 2:
            return candidate, scores
        return candidate


class PearsonModel(AssociationModel):
    def __init__(self, kind, n, p1=None, p2=None, n1=None, sigma=None, events=None, seed=None):
        AssociationModel.__init__(self, kind, n, p1=p1, p2=p2, n1=n1, sigma=sigma, events=events, seed=seed)

    def score(self, x, y):
        return abs(pearsonr(x, y)[0])


class NullHypothesisModel(AssociationModel):
    def __init__(self, kind, n, p1=None, p2=None, n1=None, sigma=None, events=None, seed=None):
        AssociationModel.__init__(self, kind, n, p1=p1, p2=p2, n1=n1, sigma=sigma, events=events, seed=seed)

    def score(self, x, y):
        return 1-abs(pearsonr(x, y)[1])
        # sigma = np.std(y)
        # mu_1 = np.mean(np.extract(x, y))
        # mu_2 = np.mean(np.extract(1-x, y))
        # return abs(mu_1-mu_2)/sigma * np.sqrt(len(y))


class SpearmanModel(AssociationModel):
    def __init__(self, kind, n, p1=None, p2=None, n1=None, sigma=None, events=None, seed=None):
        AssociationModel.__init__(self, kind, n, p1=p1, p2=p2, n1=n1, sigma=sigma, events=events, seed=seed)

    def score(self, x, y):
        return abs(spearmanr(x, y)[0])


class KendallModel(AssociationModel):
    def __init__(self, kind, n, p1=None, p2=None, n1=None, sigma=None, events=None, seed=None):
        AssociationModel.__init__(self, kind, n, p1=p1, p2=p2, n1=n1, sigma=sigma, events=events, seed=seed)

    def score(self, x, y):
        return abs(kendalltau(x, y)[0])


class MutualInformationModel(AssociationModel):
    def __init__(self, kind, n, p1=None, p2=None, n1=None, sigma=None, events=None, seed=None, n_neighbors=None):
        AssociationModel.__init__(self, kind, n, p1=p1, p2=p2, n1=n1, sigma=sigma, events=events, seed=seed)
        if n_neighbors is None:
            n_neighbors = int(np.sqrt(n))
        self.n_neighbors = n_neighbors
        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='ball_tree')

    def _epsilon(self, x):
        self.nn.fit(x)
        distances = self.nn.kneighbors(x)[0]
        dim = x.shape[1]
        return dim * np.mean(distances, axis=0)[-1]

    def _entropy(self, x):
        return digamma(self.n) - digamma(self.n_neighbors) + self._epsilon(x)

    def _joint_p(self, x, y, i, j):
        return len([k for k in range(len(x)) if x[k] == i and y[k] == j])

    def _p(self, x, i):
        return len([k for k in range(len(x)) if x[k] == i])

    def score(self, x, y):
        if self.kind == CATEGORICAL:
            z = np.array([x, y])
            unique, counts = np.unique(z, axis=1, return_counts=True)
            c_x = {}
            c_y = {}
            c_z = {}
            for k, tup in enumerate(unique.T):
                i, j = tup
                if i not in c_x.keys():
                    c_x[i] = 0
                if j not in c_y.keys():
                    c_y[j] = 0
                c_x[i] += counts[k]
                c_y[j] += counts[k]
                c_z[i, j] = counts[k]

            n = float(len(x))
            # for (i, j) in c_z.keys():
            #     print("p(x=%i)    \t= %f" % (i, c_x[i]/n))
            #     print("p(y=%i)    \t= %f" % (j, c_y[j]/n))
            #     print("p(x=%i)p(y=%i)= %f" % (i, j, c_x[i]/n * c_y[j] / n))
            #     print("p(x,y=%i,%i)\t= %f" % (i, j, c_z[i, j]/n))
            #     print("I(%i,%i)    \t= %f" % (i, j, c_z[i, j] * np.log(n*c_z[i, j]/float(c_x[i]*c_y[j]))/n))
            # raise Exception

            return sum([c_z[i, j] * np.log(n * c_z[i, j]/float(c_x[i]*c_y[j])) for (i, j) in c_z.keys()])/n
            # x_values = set(x)
            # y_values = set(y)
            # return sum([self._joint_p(x, y, i, j) * np.log(self._joint_p(x, y, i, j)/(self._p(x, i)*self._p(y, j)))
            #             for i in x_values for j in y_values])/len(x)
        else:
            return self._entropy(x.reshape(-1, 1)) + self._entropy(y.reshape(-1, 1)) \
                   - self._entropy(np.array([x, y]).transpose())


class MonteCarlo:
    def __init__(self, model, num_tries=1, seed=None):
        self.kind = model.kind
        self.model = model
        self.num_tries = num_tries
        if seed is not None:
            np.random.seed(seed)
        self.rand = None
        self.generate_random()
        self._start = None
        self._last_print = None

    def __str__(self):
        return "<MonteCarlo %s %s>" % (self.model, self.num_tries)

    @staticmethod
    def string_to_path(s):
        return "%s.png" % re.sub("\W", "", re.sub("\s", "_", s.lower()))

    @staticmethod
    def confidence_at_k(a, f):
        return a[int(len(a) * f)]

    def generate_random(self):
        self.rand = self.model.generate()

    def _log_start(self):
        self._start = time.time()
        self._last_print = self._start

    @staticmethod
    def _pretty_print_time(seconds):
        return str(timedelta(seconds=int(seconds)))

    def _pretty_print_progress(self, i, total, force=False):
        now = time.time()
        do_print = force or now - self._last_print > 1
        if do_print:
            self._last_print = now
            elapsed = now - self._start
            progress = float(i) / total
            print("Iteration %i/%i (%i%%)\tElapsed: %s\tExpected: %s"
                  % (i, total, progress*100, self._pretty_print_time(elapsed), self._pretty_print_time(elapsed / progress)))

    def _get_results(self, stats):
        d = self.model.__dict__
        ignore_pattern = "^(_.*|events|nn)"
        do_ignore = re.compile(ignore_pattern)
        columns = sorted(stats.keys()) + ["model", "num_tries"] + sorted(d.keys())
        columns = [k for k in columns if do_ignore.match(k) is None]
        data = [str(stats[k]) for k in sorted(stats.keys())] + [str(self.model.__class__), str(self.num_tries)] \
               + [str(d[k]) for k in sorted(d.keys()) if do_ignore.match(k) is None]
        return np.array(columns), np.array(data)

    def run(self, return_scores=None, show_plot=False, plot_path=None, confidence_levels=None):
        if plot_path is None:
            plot_path = self.string_to_path(str(self))
        if confidence_levels is None:
            confidence_levels = (0.5, 0.8, 0.9, 0.95, 0.99)
        if return_scores is None:
            return_scores = 0 if self.num_tries > 1 else 2

        self._log_start()
        result = []
        confidence = []
        scores = []
        for n in range(self.num_tries):
            self.generate_random()
            if return_scores == 2:
                candidate, scores = self.model.best_candidate(self.rand, return_score=return_scores)
            elif return_scores:
                candidate, score = self.model.best_candidate(self.rand, return_score=return_scores)
                confidence.append(score)
            else:
                candidate = self.model.best_candidate(self.rand, return_score=return_scores)
            result.append(candidate)
            self._pretty_print_progress(n+1, self.num_tries)

        plt.figure(figsize=(16, 8))
        if return_scores == 2:
            print scores
            plt.plot(scores)
        else:
            if self.kind == CATEGORICAL:
                plt.hist(result, bins=int(self.model.n / 10), range=(0, self.model.n), normed=True, color="#1F4B99",
                         linewidth=0)
                plt.xticks(np.arange(0, self.model.n, int(self.model.n/10)))
                plt.axvline(x=self.model.n1, color="#B3CFFF")
                plt.title("Optimal t' for p1=%i%%, p2=%i%% and t1=%i"
                          % (int(100*self.model.p1), int(100*self.model.p2), self.model.n1))
            else:
                plt.hist(result, bins=50, range=(0, 2),
                         normed=True, color="#1F4B99", linewidth=0)
                plt.title("Optimal y for %i events" % self.model.num_events)
                plt.ylabel("y/y*")
                plt.xticks(np.arange(0, 2, .2))

            plt.xlabel("Optimal t'")
            plt.ylabel("Frequency")

        plt.savefig(plot_path)
        if show_plot:
            plt.show()

        res = np.array(result)
        stats = {"mean": np.mean(res),
                 "median": np.median(res),
                 "std": np.std(res)}

        if self.kind == CATEGORICAL:
            res = sorted([abs(i - self.model.n1) for i in res])
        else:
            res = sorted([abs(i - self.model.num_events) for i in res])

        for f in confidence_levels:
            stats["<%f" % f] = self.confidence_at_k(res, f)

        columns, data = self._get_results(stats)
        print columns
        print data
        if not os.path.isfile(RESULTS_PATH):
            results = pd.DataFrame(data=data.reshape(1, -1), columns=columns)
        else:
            previous_results = pd.read_csv(RESULTS_PATH)
            new_results = pd.DataFrame(data=data.reshape(1, -1), columns=columns)
            results = pd.concat([previous_results, new_results], axis=0, ignore_index=True)

        results.to_csv(RESULTS_PATH, index=None)
        print results

        if return_scores:
            return stats, confidence, result
        return stats


if __name__ == "__main__":
    n_ = 1000
    n1_prop = 0.35
    p1_ = 0.2
    p2_ = 0.3

    sigma_ = 0.4
    # events_ = [0.45, 0.5, 0.52, 0.6, 0.7, 0.95]
    events_ = np.random.rand(int(n_*p1_))
    events_ = sorted(set([int(n_*f) for f in events_]))

    n1_ = int(n_*n1_prop)
    n2_ = n_ - n1_

    # num_tries_ = 1
    # m = MutualInformationModel(CATEGORICAL, n_, p1_, p2_, n1_)
    # mc = MonteCarlo(m, num_tries_, seed=0)
    # print(mc)
    # print(mc.run(show_plot=True))

    for num_tries_ in [10, 50, 100, 500]:
        model = MutualInformationModel
        # for model in [PearsonModel, SpearmanModel, KendallModel, NullHypothesisModel]:
        # model = PearsonModel
        # m = model(CATEGORICAL, n_, p1_, p2_, n1_)
        # MonteCarlo(m, num_tries_, seed=0).run()
        m = model(NUMERICAL, n_, sigma=sigma_, events=events_)
        MonteCarlo(m, num_tries_, seed=0).run()




