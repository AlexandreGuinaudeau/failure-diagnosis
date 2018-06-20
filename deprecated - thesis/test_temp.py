import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def trendline(data, order=1):
    coeffs = np.polyfit(data.index.values, list(data), order)
    slope = coeffs[-2]
    return float(slope)


def seasonality(ts):
    # load data
    columns = ["product_%i" % (i + 1) for i in range(100)]
    ts['date'] = pd.to_datetime(ts['date'])
    ts["yearday"] = ts["date"].apply(lambda x: x.month * 100 + x.day)
    ts["weekday"] = ts["date"].apply(lambda x: x.weekday())
    ts.drop(columns="id", inplace=True)

    ts_yearday = ts[columns].rolling(7, center=True).mean()
    ts_yearday["date"] = ts["date"]
    ts_yearday["yearday"] = ts["yearday"]
    ts_yearday = ts_yearday.groupby("yearday").mean()

    ts_trend = ts.join(ts_yearday, on="yearday", rsuffix="_yearday")
    for c in columns:
        ts_trend[c] -= ts_trend[c + "_yearday"]
    ts_trend = ts_trend[columns + ["date", "weekday"]]

    ts_weekday = ts_trend.groupby("weekday").mean()
    ts_trend = ts_trend.join(ts_weekday, on="weekday", rsuffix="_weekday")
    for c in columns:
        ts_trend[c] -= ts_trend[c + "_weekday"]
    ts_trend = ts_trend[columns + ["date"]]

    ts = ts.join(ts_trend, rsuffix="_trend").drop(columns="date_trend")
    ts = ts.join(ts_yearday, on="yearday", rsuffix="_yearday")
    ts = ts.join(ts_weekday, on="weekday", rsuffix="_weekday")

    for i in range(1, 20):
        series_name = 'product_%i' % i
        plt.subplot("411")
        plt.plot(ts[[series_name]])
        plt.subplot("412")
        plt.plot(ts[[series_name + "_trend"]])
        plt.subplot("413")
        plt.plot(ts[[series_name + "_weekday"]])
        plt.subplot("414")
        plt.plot(ts[[series_name + '_yearday']])
        plt.show()

    return ts

    # series_name = 'product_1'
    # series = ts.set_index('date')[[series_name]]
    # trend, seasonal, residual = seasonal_analysis(ts)
    # df = pd.concat([
    #     trend.rename(columns={series_name: 'trend'}),
    #     seasonal.rename(columns={series_name: 'seasonal'}),
    #     residual.rename(columns={series_name: 'residual'})
    # ])
    # return ts


def seasonal_analysis(ts):
    ts["weekday"] = ts["date"].apply(lambda x: x.weekday())
    ts = ts.groupby("weekday").mean()
    return ts
    # decomposition = seasonal_decompose(ts)
    # return decomposition.trend, decomposition.seasonal, decomposition.resid


if __name__ == "__main__":
    import pystan
    import numpy as np
    import matplotlib.pyplot as plt

schools_code = """
data {
    int<lower=0> J; // number of schools
    real y[J]; // estimated treatment effects
    real<lower=0> sigma[J]; // s.e. of effect estimates
}
parameters {
    real mu;
    real<lower=0> tau;
    real eta[J];
}
transformed parameters {
    real theta[J];
    for (j in 1:J)
        theta[j] = mu + tau * eta[j];
}
model {
    eta ~ normal(0, 1);
    y ~ normal(theta, sigma);
}
"""

schools_dat = {'J': 8,
               'y': [28, 8, -3, 7, -1, 1, 18, 12],
               'sigma': [15, 10, 16, 11, 9, 11, 10, 18]}

sm = pystan.StanModel(model_code=schools_code)
fit = sm.sampling(data=schools_dat, iter=1000, chains=4)

print(fit)

eta = fit.extract(permuted=True)['eta']
np.mean(eta, axis=0)

# if matplotlib is installed (optional, not required), a visual summary and
# traceplot are available
fit.plot()
plt.show()
