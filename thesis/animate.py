import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from thesis import BaseGenerator



if __name__ == "__main__":
    probability = 0.2
    max_observations = 1000
    window = 5
    thresholds = (0.8, 0.9, 0.999)

    bg = BaseGenerator((probability, probability), (max_observations/2, max_observations/2))
    observations = bg.generate(max_observations)

    fig, ax = plt.subplots()

    x = np.arange(0, max_observations)
    line, = ax.plot(x, x)

    all_max = []
    all_t = np.arange(2 * window + 1, max_observations, window)
    z_values = []
    z_values_theory = {}
    for thresh in thresholds:
        # z_values[thresh] = []
        z_values_theory[thresh] = []
    ids = []

    def theoretical_z(alpha, p, t):
        # delta = np.ceil(np.log(alpha)/np.log(p))
        return norm.ppf(1 - alpha / 2) * np.sqrt(t/10)

    def animate(t):
        ind = np.arange(window, t - window, window)
        x = max([test_statistic(k, t - k, np.mean(observations[:k]), np.mean(observations[k:t])) for k in ind])
        # x = sorted(x)
        z_values.append(x)
        for thresh in thresholds:
            # z_values[thresh].append(x[int(thresh*len(x))])
            z_values_theory[thresh].append(theoretical_z(1-thresh, probability, t))
        ids.append(t)
        line.set_xdata([ids]*(len(thresholds)+1))
        line.set_ydata([z_values] + [z_values_theory[thresh] for thresh in thresholds])
        return line,


    # Init only required for blitting to give a clean slate.
    def init():
        # line.set_ydata(x)
        ax.set_ylim(-5, 20)
        return line,


    ani = animation.FuncAnimation(fig, animate, all_t, init_func=init, interval=25, blit=False, repeat=False)
    plt.show()

