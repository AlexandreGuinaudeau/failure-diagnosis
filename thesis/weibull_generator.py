import numpy as np
from thesis import BaseGenerator


class WeibullGenerator(BaseGenerator):
    def __init__(self, a, lambd=1):
        super(WeibullGenerator, self).__init__()
        self.a = a
        self.lambd = lambd

    def _next_delta(self):
        return int(np.ceil(self.lambd * np.random.weibull(self.a)))

    def change_points(self, n):
        l = [self._next_delta()]
        while sum(l) < n:
            l.append(self._next_delta())
            print self._next_delta()
        return l

    def run(self, n):
        change_points = self.change_points(n)
        res = []

        for c in change_points:
            v = np.random.rand()
            res.extend(np.random.rand(c) < v)

        return np.array(res[:n]).astype(int)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    wg = WeibullGenerator(3.5, 200)
    cp = wg.change_points(10000)
    ind = [0] + np.cumsum(cp)
    print np.mean(cp)
    plt.plot(ind, np.arange(len(ind)), 'o')
    plt.show()
