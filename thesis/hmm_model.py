import matplotlib.pyplot as plt
import numpy as np
from hmmlearn.base import _BaseHMM
from thesis import BaseModel, BaseGenerator


class BinaryHMM(_BaseHMM):
    def __init__(self, states=11, change_state_ratio=0.01, random_state=None, **kwargs):
        if isinstance(states, int):
            states = [i / float(states - 1) for i in range(0, states)]
        states = np.array(states)
        self._states = states
        n_states = len(states)
        self.states_dict = dict(zip(np.arange(n_states), states))

        proba_same_state = 1/((n_states-1) * change_state_ratio + 1)
        proba_change_state = proba_same_state * change_state_ratio
        proba_same_state = np.diag(np.array([proba_same_state] * n_states))
        proba_change_state = np.full((n_states, n_states), proba_change_state)
        proba_change_state -= np.diag(np.diag(proba_change_state))

        startprob_prior = np.array([1 / float(n_states)] * n_states)
        # if transition_weights is None:
        #     transitions = np.ones((n_states, n_states))
        # else:
        #     if len(transition_weights) == n_states - 1:
        #         transition_weights = [0] + list(transition_weights)
        #     transition_weights = np.array(transition_weights)
        #     assert len(transition_weights) == n_states, \
        #         "Expected %i weights, got %i" % (n_states, len(transition_weights))
        #     transitions = sum([np.diag([transition_weights[k]]*(n_states-k), k) for k in range(n_states)])
        #     transitions = transitions + transitions.transpose()
        # # Normalize
        # transitions -= np.diag(np.diag(transitions))  # Remove diagonal
        # transitions /= transitions.sum(axis=1)

        transmat_prior = proba_same_state + proba_change_state
        # transmat_prior = transmat_prior.transpose()
        super(BinaryHMM, self).__init__(n_components=n_states,
                                        startprob_prior=startprob_prior,
                                        transmat_prior=transmat_prior,
                                        random_state=random_state,
                                        **kwargs)
        self.startprob_ = np.array([1 / float(n_states)] * n_states)
        self.transmat_ = transmat_prior
        self._log_prob = {True: np.nan_to_num(np.array([np.log(p) for p in states])),
                          False: np.nan_to_num(np.array([np.log(1-p) for p in states]))}

    def probability_change_state(self, absolute):
        if absolute:
            return self.transmat_[0, 1]
        return self.transmat_[0, 1]/self.transmat_[0, 0]

    def _generate_sample_from_state(self, state, random_state=None):
        return np.random.rand() < self._states[state]

    def _compute_log_likelihood(self, X):
        X = X.astype(int)
        return np.multiply(X, self._log_prob[True]) + np.multiply(1-X, self._log_prob[False])

    def plot_example(self, x, z):
        t = np.arange(len(z))
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(t, z)
        ax1.set_ylim(0, self.n_components)
        ax2.plot(t, x.flatten(), 'o')
        ax2.set_ylim(0, 1)
        plt.show()
        return z


class HMMModel(BaseModel):
    def __init__(self, error_threshold=0.01, **kwargs):
        assert 0 < error_threshold <= 1
        super(HMMModel, self).__init__(error_threshold)
        self.kwargs = dict(kwargs)
        self._hmm = None
        self._change_state_ratio = None

    def hmm(self, x, kwargs=None):
        change_state_ratio = self.change_state_ratio_theory(np.mean(x), self.error_threshold, len(x))
        if change_state_ratio != self._change_state_ratio:
            if kwargs is None:
                kwargs = self.kwargs
            self._hmm = BinaryHMM(change_state_ratio=change_state_ratio, **kwargs)
        return self._hmm

    @classmethod
    def change_state_ratio_theory(cls, p, a, t):
        # ratio * sqrt(t) = 0.00 + 0.93a + 2.61a|0.5-p| +/- 0.30
        return (0.93 * a + 2.61 * a * abs(0.5 - p)) / np.sqrt(t)

    def run(self, x, return_full_prediction=True, kwargs=None):
        x = np.array(x)
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        pred = self.hmm(x, kwargs).predict(x)
        if return_full_prediction is False:
            return len(set(pred)) > 1
        return np.vectorize(self.hmm(x, kwargs).states_dict.__getitem__)(pred)

    def has_change_point(self, x, kwargs=None):
        return self.run(x, return_full_prediction=False, kwargs=kwargs)

    def max_param(self, x, min_states=1, max_states=1001, ):
        if max_states - min_states > 1:
            mean_states = (max_states + min_states) / 2
            kwargs = self.kwargs
            kwargs["states"] = mean_states
            if self.has_change_point(x, kwargs):
                return self.max_param(x, min_states, mean_states)
            return self.max_param(x, mean_states, max_states)
        return min_states


if __name__ == "__main__":
    n_components_ = 101
    proba_change_state_ = 0.1
    transition_weights_ = None  # [n_comp / float(n_components_) for n_comp in range(n_components_, 1, -1)]
    hmm = BinaryHMM(n_components_,
                    proba_change_state=proba_change_state_,
                    # proba_change_state=np.array([0.001]*7 + [0.1]*3),
                    transition_weights=transition_weights_)

    hmm_model = HMMModel(states=n_components_,
                         proba_change_state=proba_change_state_,
                         transition_weights=transition_weights_)
    p1_ = 0.2
    p2_ = 0.2
    n1_ = 80
    n_ = 1000
    n_tries_ = 100
    results = {}
    bg = BaseGenerator([p1_, p2_], [n1_, n_-n1_])
    for _ in range(n_tries_):
        x_ = bg.generate(n_)
        x_ = x_.reshape(-1, 1)
        z_ = hmm.predict(x_)
        n_change_points = np.sum(z_[1:] != z_[:-1])
        has_change_point = hmm_model.has_change_point(x_)
        if n_change_points not in results.keys():
            results[n_change_points] = 0
        results[n_change_points] += 1

    print results
    # hmm.plot_example(x_, z_)
