from scipy.stats import multivariate_normal
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors.unsupervised import NearestNeighbors
from scipy.special import digamma


def with_added_white_noise(x, noise_scale):
    return x + np.random.normal(scale=np.std(x)*noise_scale, size=x.shape)


def joint_information(x, y, n_neighbors=3, random_noise=0.3):
    n_samples = x.size

    if random_noise:
        x = with_added_white_noise(x, random_noise)
        y = with_added_white_noise(y, random_noise)

    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    xy = np.hstack((x, y))

    # Here we rely on NearestNeighbors to select the fastest algorithm.
    nn = NearestNeighbors(metric='chebyshev', n_neighbors=n_neighbors)

    nn.fit(xy)
    radius = nn.kneighbors()[0]
    radius = np.nextafter(radius[:, -1], 0)

    # Algorithm is selected explicitly to allow passing an array as radius
    # later (not all algorithms support this).
    nn.set_params(algorithm='kd_tree')

    nn.fit(x)
    ind = nn.radius_neighbors(radius=radius, return_distance=False)
    nx = np.array([i.size for i in ind])

    nn.fit(y)
    ind = nn.radius_neighbors(radius=radius, return_distance=False)
    ny = np.array([i.size for i in ind])

    mi = (digamma(n_samples) + digamma(n_neighbors) -
          np.mean(digamma(nx + 1)) - np.mean(digamma(ny + 1)))

    return max(0, mi)


def joint_information_matrix(X, y, nearest_neighbors=3):
    X = np.array(X).transpose()
    return np.array([joint_information(x, y, nearest_neighbors) for x in X])


if __name__ == "__main__":
    # mean = np.zeros(2)
    #
    # # Setup covariance matrix with correlation coeff. equal 0.5.
    # sigma_1 = 1
    # sigma_2 = 10
    # corr = 0.5
    # cov = np.array([
    #     [sigma_1 ** 2, corr * sigma_1 * sigma_2],
    #     [corr * sigma_1 * sigma_2, sigma_2 ** 2]
    # ])
    #
    # # True theoretical mutual information.
    # I_theory = (np.log(sigma_1) + np.log(sigma_2) -
    #             0.5 * np.log(np.linalg.det(cov)))
    #
    # Z = rng.multivariate_normal(mean, cov, size=1000)
    #
    # x, y = Z[:, 0], Z[:, 1]
    #
    # # Theory and computed values won't be very close, assert that the
    # # first figures after decimal point match.
    # for n_neighbors in [3, 5, 7]:
    #     I_computed = _compute_mi(x, y, False, False, n_neighbors)
    #     assert_almost_equal(I_computed, I_theory, 1)

    ts_ = pd.read_csv(os.path.realpath("data/products.csv"))
    ts_ = ts_[['product_%i' % i for i in range(1, 100)]]
    ts_ = ts_.apply(lambda x: (x-x.mean())/x.std(), axis=0)  # Normalize
    for j in range(1, 10):
        y = ts_['product_%i' % j]
        X = ts_[['product_%i' % i for i in range(1, 100)]]
        x = ts_['product_2']
        # score = joint_information(x, y)
        score = joint_information_matrix(X, y)

        print(score)
        order = np.argsort(score)
        print(order)
        plt.plot(score[order])
        plt.show()
        for i in order[-5:]:
            X[['product_%i' % j, 'product_%i' % (i+1)]].plot()
            # plt.plot(with_added_white_noise(X['product_1']))
            plt.title("%i: %f" % (i+1, score[i]))
            plt.show()

    # var_multivariate = multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]])
    # var_unique = multivariate_normal(mean=0, cov=1)
    # n = 100
    # # x = [var_unique.pdf(i/n) for i in range(-n, n)]
    # # y = [var_unique.pdf(i/n) for i in range(-n, n)]
    # # z = np.array([[var_multivariate.pdf([i/n, j/n]) for i in range(-n, n)] for j in range(-n, n)])
    #
    # joint_probability = 0
    # for i in range(-n, n):
    #     for j in range(-n, n):
    #         z = var_multivariate.pdf([i/n, j/n])
    #         joint_probability += z * np.log(z/(var_unique.pdf(i/n)*var_unique.pdf(j/n)))
    # # joint_probability += z * np.log(z / (x * y))
    # # joint_probability = z.sum()
    # print joint_probability
    # # plt.imshow(z, cmap='hot', interpolation='nearest')
    # # plt.show()
    # # print var_multivariate.pdf([.1, .1]), var_multivariate.pdf([.1, .5])
