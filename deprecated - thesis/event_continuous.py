import numpy as np
from scipy.special import erf
from scipy.stats import pearsonr, spearmanr
from joint_probability import joint_information
import matplotlib.pyplot as plt


def _entropy(p, n):
    return n*(p*np.log(p) + (1-p)*np.log(1-p))


def null_hypothesis(idx, y):
    mu = y.mean()
    sigma = y.std()
    se = sigma/np.sqrt(len(idx))
    y_events = y[idx]
    z = (y_events.mean() - mu)/se
    return z


def _one_hot(idx, n, v=1):
    a = np.zeros(n)
    a[idx] = v
    return a


if __name__ == "__main__":
    n_ = 1000
    idx_ = np.array([100, 150, 200, 500, 600, 900])
    x_ = _one_hot(idx_, n_)
    v_ = 4

    y_ = np.random.normal(size=n_)
    y_best_ = y_ + _one_hot(idx_, n_, v_)
    y_rare_ = y_ + _one_hot(idx_[::2], n_, v_)
    y_frequent_ = y_ + _one_hot(idx_, n_) + _one_hot((idx_[::2] + idx_[1::2]) / 2, n_, v_)

    pearson = []
    spearman = []
    mutual_info = []
    for i, y in enumerate((y_rare_, y_best_, y_frequent_)):
        print(null_hypothesis(idx_, y))
        # pearson.append(pearsonr(x_, y)[0])
        # spearman.append(spearmanr(x_, y)[0])
        # mutual_info.append(joint_information(x_, y, random_noise=0))
        # plt.subplot(3, 1, i + 1)
        # plt.hist(y, bins=20)

    # for i, a in enumerate((pearson, spearman, mutual_info)):
    #     a /= max(a)
    #     a *= 100
    #     print(a)
    #
    # plt.show()

    # plt.subplot(4, 1, 1)
    # plt.plot(x_)
    # for i, y in enumerate((y_rare_, y_best_, y_frequent_)):
    #     plt.subplot(4, 1, i+2)
    #     plt.plot(y)
    # plt.show()

    # for i, y in enumerate((y_rare_, y_best_, y_frequent_)):
    #     plt.plot(x_, y, '+')
    #     plt.axis([-1, 5, -10, 10])
    #     plt.show()
