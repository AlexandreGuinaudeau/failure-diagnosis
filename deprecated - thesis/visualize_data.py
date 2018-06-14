import os
import numpy as np
from scipy.special import erf
import pandas as pd
import matplotlib.pyplot as plt


def smooth(x, window_len=50):
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    w = np.hanning(window_len)

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y

data_dir = os.path.join(os.getcwd(), "stats")

# path1 = os.path.join(data_dir, "heartbeat1")
path2 = os.path.join(data_dir, "battery_statsbattery_temp_c.csv")

df_all = pd.read_csv(path2, header=None)

a = smooth(df_all[0])
local_min = pd.DataFrame(np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True])

switch = [x-25 for x in local_min[local_min[0]].index]

d = {}
for i in range(len(switch)-1):
    section = df_all.iloc[switch[i]:switch[i+1]]
    rate = int(1500/(switch[i + 1] - switch[i]))
    index = pd.date_range('1/1/2000', periods=switch[i+1]-switch[i], freq='%iS' % rate)
    if len(section) != len(index):
        continue
    section["ind"] = index
    section.set_index("ind", inplace=True)
    d[i] = section[[0]]
    d[i] = d[i].resample('15S').pad()[0][:92]

df = pd.DataFrame(d)

df_mean = df.mean(axis=1)
df_std = df.std(axis=1)
df_avg = []
cycle_lengths = []

for c in sorted(df.columns):
    df[c] = (df[c]-df_mean)/df_std
    df_avg.append(df[c].mean())
    cycle_lengths.append(float(switch[c + 1] - switch[c])/100)
    df[c] -= df_avg[-1]

df_error = df.apply(lambda x: np.log(1-np.abs(erf(x))))
df_avg = pd.DataFrame({0: df_avg, "cycles": cycle_lengths}, sorted(df.columns))
df_avg_error = df_avg[[0]].apply(lambda x: np.log(1-np.abs(erf(x))))

threshold = -2

outliers = set()

df_outlier_region = df_avg_error < threshold
df_outlier_region = list(df_outlier_region[df_outlier_region[0]].index)

df_outlier_in_region = (df_error < threshold).reset_index()

for i in df_outlier_region:
    outliers = outliers.union(set(range(switch[i], switch[i+1])))
for c in sorted(df.columns):
    df_outlier_local = df_outlier_in_region[c]
    df_outlier_local = list(df_outlier_in_region[df_outlier_local].index)
    df_outlier_local += switch[c]
    outliers = outliers.union(set(df_outlier_local))

df_all["outliers"] = df_all.loc[outliers]

ax = plt.subplot(2, 3, 1)
df_all.plot(ax=ax, legend=False)

ax = plt.subplot(2, 3, 3)
df.plot(ax=ax, legend=False)
ax = plt.subplot(2, 3, 2)
df_avg.plot(ax=ax, legend=False)
ax = plt.subplot(2, 3, 4)
df_stat = pd.DataFrame({"mean": df_mean, "lower": df_mean-df_std, "upper": df_mean+df_std})
df_stat.plot(ax=ax, legend=False)

ax = plt.subplot(2, 3, 5)
df_error.plot(ax=ax, legend=False)
ax = plt.subplot(2, 3, 6)
df_avg_error.plot(ax=ax, legend=False)

plt.show()

