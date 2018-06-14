import numpy as np


class BaseGenerator(object):
    """
    A generator should be able to generate a sequence of events (one-hot represented)
    """
    def __init__(self, probabilities=(), durations=()):
        probabilities = self._cast_to_list(probabilities)
        durations = self._cast_to_list(durations)
        assert len(durations) == len(probabilities)
        self.durations = list(durations)
        self.probabilities = list(probabilities)
        self.iterations = 0
        self.current_duration = 0

    @staticmethod
    def _cast_to_list(int_or_iterable):
        try:
            len(int_or_iterable)
            return int_or_iterable
        except TypeError:
            return [int_or_iterable]

    def _update_current_duration(self):
        if self.durations[self.current_duration] is None:
            return
        total_duration = sum(self.durations[:self.current_duration+1])
        if self.iterations > total_duration:
            self.current_duration += 1

    def change_probability(self, new_probability):
        assert self.durations[-1] is None
        assert self.iterations >= sum(self.durations[:-1])
        self.durations[-1] = self.iterations - sum(self.durations[:-1])
        self.durations.append(None)
        self.probabilities.append(new_probability)

    def append_sequence(self, duration, probability):
        self.durations.append(duration)
        self.probabilities.append(probability)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        self.iterations += 1
        self._update_current_duration()
        if self.current_duration >= len(self.durations):
            raise StopIteration
        return int(np.random.rand() < self.probabilities[self.current_duration])

    def generate(self, n):
        return np.concatenate([self._generate(self.durations[i], self.probabilities[i], n - sum(self.durations[:-1]))
                               for i in range(len(self.durations))])

    @staticmethod
    def _generate(n, proba, default_value):
        if n is None:
            n = default_value
        return np.array(np.random.rand(n) < proba).astype(int)

    @staticmethod
    def to_one_hot(l, max_v):
        a = np.array([0] * (max_v + 1))
        a[l] = 1
        return a

    @staticmethod
    def from_one_hot(a):
        return np.where(a == 1)[0]
