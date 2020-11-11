import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.signal import wiener
from scipy.signal import detrend

data_loader = sm.datasets.sunspots.load_pandas()
sunspots = data_loader.data["SUNACTIVITY"].values
years = data_loader.data["YEAR"].values

plt.plot(years, sunspots, label="SUNACTIVITY")
plt.plot(years, medfilt(sunspots, 11), lw=2, label="Median")
plt.plot(years, wiener(sunspots, 11), '--', lw=2, label="Wiener")
plt.plot(years, detrend(sunspots), lw=3, label="Detrend")
plt.xlabel("YEAR")
plt.grid(True)
plt.legend()
plt.show()


# import numpy as np
# import statsmodels.api as sm
# import matplotlib.pyplot as plt
# from scipy.fftpack import rfft
# from scipy.fftpack import fftshift

# data_loader = sm.datasets.sunspots.load_pandas()
# sunspots = data_loader.data["SUNACTIVITY"].values

# transformed = fftshift(rfft(sunspots))

# plt.subplot(311)
# plt.plot(sunspots, label="Sunspots")
# plt.legend()
# plt.subplot(312)
# plt.plot(transformed ** 2, label="Power Spectrum")
# plt.legend()
# plt.subplot(313)
# # 相位谱可以为我们直观展示相位，即正弦函数的起始角
# plt.plot(np.angle(transformed), label="Phase Spectrum")
# plt.grid(True)
# plt.legend()
# plt.show()


# import statsmodels.api as sm
# import matplotlib.pyplot as plt
# from scipy.fftpack import rfft
# from scipy.fftpack import fftshift

# data_loader = sm.datasets.sunspots.load_pandas()
# sunspots = data_loader.data["SUNACTIVITY"].values

# t = np.linspace(-2 * np.pi, 2 * np.pi, len(sunspots))
# mid = np.ptp(sunspots)/2
# sine = mid + mid * np.sin(np.sin(t))

# # rfft函数对实值数据进行FFT
# # fftshift函数可以把零频分量（数据的平均值）移动到频谱中央
# sine_fft = np.abs(fftshift(rfft(sine)))
# print()
# print("最大振幅的相应索引: ", np.argsort(sine_fft)[-5:])

# transformed = np.abs(fftshift(rfft(sunspots)))
# print("频谱中峰值的相应索引: ", np.argsort(transformed)[-5:])

# plt.subplot(311)
# plt.plot(sunspots, label="Sunspots")
# plt.plot(sine, lw=2, label="Sine")
# plt.grid(True)
# plt.legend()
# plt.subplot(312)
# plt.plot(transformed, label="Transformed Sunspots")
# plt.grid(True)
# plt.legend()
# plt.subplot(313)
# plt.plot(sine_fft, lw=2, label="Transformed Sine")
# plt.grid(True)
# plt.legend()
# plt.show()

# from scipy.optimize import leastsq
# import statsmodels.api as sm
# import matplotlib.pyplot as plt
# import numpy as np


# def model(p, t):
#     C, p1, f1, phi1, p2, f2, phi2, p3, f3, phi3 = p
#     return C + p1 * np.sin(f1 * t + phi1) + p2 * np.sin(f2 * t + phi2) + p3 * np.sin(f3 * t + phi3)


# def error(p, y, t):
#     return y - model(p, t)


# def fit(y, t):
#     p0 = [y.mean(), 0, 2 * np.pi/11, 0, 0, 2 * np.pi/22, 0, 0, 2 * np.pi/100, 0]
#     params = leastsq(error, p0, args=(y, t))[0]
#     return params


# data_loader = sm.datasets.sunspots.load_pandas()
# sunspots = data_loader.data["SUNACTIVITY"].values
# years = data_loader.data["YEAR"].values

# cutoff = int(.9 * len(sunspots))
# params = fit(sunspots[:cutoff], years[:cutoff])
# print("Params", params)

# pred = model(params, years[cutoff:])
# actual = sunspots[cutoff:]
# print("均方根误差", np.sqrt(np.mean((actual - pred) ** 2)))
# print("绝对平均误差", np.mean(np.abs(actual - pred)))
# print("平均绝对百分比误差", 100 * np.mean(np.abs(actual - pred)/actual))
# mid = (actual + pred)/2
# print("对称平均绝对百分比误差", 100 * np.mean(np.abs(actual - pred)/mid))
# print("决定系数", 1 - ((actual - pred) ** 2).sum() / ((actual - actual.mean()) ** 2).sum())
# year_range = data_loader.data["YEAR"].values[cutoff:]
# plt.plot(year_range, actual, 'o', label="Sunspots")
# plt.plot(year_range, pred, 'x', label="Prediction")
# plt.grid(True)
# plt.xlabel("YEAR")
# plt.ylabel("SUNACTIVITY")
# plt.legend()
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# import statsmodels.api as sm

# data_loader = sm.datasets.sunspots.load_pandas()
# df = data_loader.data
# years = df["YEAR"].values.astype(int)
# df.index = pd.Index(sm.tsa.datetools.dates_from_range(str(years[0]),
#                                                       str(years[-1])))
# del df["YEAR"]

# model = sm.tsa.ARMA(df, (10, 1)).fit()
# prediction = model.predict('1975', str(years[-1]), dynamic=True)

# df['1975':].plot()
# prediction.plot(style='--', label='Prediction')
# plt.legend()
# plt.show()


# from scipy.optimize import leastsq
# import statsmodels.api as sm
# import matplotlib.pyplot as plt
# import numpy as np


# def model(p, x1, x10):
#     p1, p10 = p
#     return p1 * x1 + p10 * x10


# def error(p, data, x1, x10):
#     return data - model(p, x1, x10)


# def fit(data):
#     p0 = [.5, 0.5]
#     params = leastsq(error, p0, args=(data[10:], data[9:-1], data[:-10]))[0]
#     return params


# data_loader = sm.datasets.sunspots.load_pandas()
# sunspots = data_loader.data["SUNACTIVITY"].values
# # 90%的数据用于训练集
# cutoff = int(.9 * len(sunspots))
# params = fit(sunspots[:cutoff])
# print("Params", params)

# pred = params[0] * sunspots[cutoff-1:-1] + params[1] * sunspots[cutoff-10:-10]
# # 10%的数据用于测试验证
# actual = sunspots[cutoff:]
# print("均方根误差", np.sqrt(np.mean((actual - pred) ** 2)))
# print("平均绝对误差", np.mean(np.abs(actual - pred)))
# print("平均绝对百分比误差", 100 * np.mean(np.abs(actual - pred)/actual))
# mid = (actual + pred)/2
# print("对称平均绝对百分比误差", 100 * np.mean(np.abs(actual - pred)/mid))
# print("确定系数", 1 - ((actual - pred) ** 2).sum() / ((actual - actual.mean()) ** 2).sum())
# year_range = data_loader.data["YEAR"].values[cutoff:]
# plt.plot(year_range, actual, 'o', label="Sunspots")
# plt.plot(year_range, pred, 'x', label="Prediction")
# plt.grid(True)
# plt.xlabel("YEAR")
# plt.ylabel("SUNACTIVITY")
# plt.legend()
# plt.show()


# import numpy as np
# import statsmodels.api as sm
# import matplotlib.pyplot as plt
# from pandas.plotting import autocorrelation_plot

# data_loader = sm.datasets.sunspots.load_pandas()
# data = data_loader.data["SUNACTIVITY"].values
# y = data - np.mean(data)
# norm = np.sum(y ** 2)
# correlated = np.correlate(y, y, mode='full')/norm
# res = correlated[int(len(correlated)/2):]
# # 取得关联度最高值的索引
# print(np.argsort(res)[-5:])
# plt.plot(res)
# plt.grid(True)
# plt.xlabel("Lag")
# plt.ylabel("Autocorrelation")
# plt.show()
# autocorrelation_plot(data)
# plt.show()


# import statsmodels.api as sm
# import statsmodels.tsa.stattools as ts
# from tabulate import tabulate
# import numpy as np


# def calc_adf(title, x, y):
#     '''计算ADF统计量'''
#     # 普通最小二乘法
#     result = sm.OLS(x, y).fit()
#     result = list(ts.adfuller(result.resid))
#     result.insert(0, title)
#     return result
#     # return 't统计量: {0[0]}, t统计量的P值: {0[1]}, 延迟阶数: {0[2]},\
#     #         样本量: {0[3]}, t分布: {0[4]}\n'.format(adf)


# data_loader = sm.datasets.sunspots.load_pandas()
# data = data_loader.data.values
# N = len(data)

# t = np.linspace(-2 * np.pi, 2 * np.pi, N)
# sine = np.sin(np.sin(t))
# table = []
# table.append(calc_adf('计算正弦值与其自身的协整关系', sine, sine))
# noise = np.random.normal(0, .01, N)
# table.append(calc_adf('添加噪音后的正弦波信号', sine, sine + noise))
# cosine = 100 * np.cos(t) + 10
# table.append(calc_adf('生成一个幅值和偏移量更大的余弦波并混入噪音', sine, cosine + noise))
# table.append(calc_adf('正弦和太阳黑子之间的协整检验结果', sine, data))

# print(tabulate(table, headers=["计算项目", "t统计量", "t统计量的P值", "延迟阶数", "样本量",
#                                "t分布参考值", "其他"], tablefmt="github"))


# import matplotlib.pyplot as plt
# import statsmodels.api as sm
# import pandas as pd

# data_loader = sm.datasets.sunspots.load_pandas()
# df = data_loader.data.tail(150)
# df = pd.DataFrame({'SUNACTIVITY': df['SUNACTIVITY'].values}, index=df['YEAR'])
# ax = df.plot()


# def plot_window(wintype):
#     df2 = df.rolling(window=22, win_type=wintype, center=False, axis=0).mean()
#     df2.columns = [wintype]
#     df2.plot(ax=ax)


# plot_window('boxcar')    # 矩形窗
# plot_window('triang')    # 三角形窗
# plot_window('blackman')  # 钟形窗
# plot_window('hanning')   # 钟形窗
# plot_window('bartlett')  # 钟形窗
# plt.show()


# import matplotlib.pyplot as plt
# import statsmodels.api as sm
# import numpy as np


# def sma(arr, n):
#     # 简单移动平均使用的是等量加权策略
#     weights = np.ones(n) / n
#     # 计算两个序列的离散线性卷积
#     return np.convolve(weights, arr)[n-1:-n+1]


# def ema(arr, n):
#     # 指数移动平均使用的时数式递减加权策略
#     weights = np.exp(np.linspace(-1., 0., n))
#     weights /= weights.sum()
#     return np.convolve(weights, arr)[n-1:-n+1]


# data_loader = sm.datasets.sunspots.load_pandas()
# df = data_loader.data

# smav = sma(df["SUNACTIVITY"].values, 11)
# rm11 = df['SUNACTIVITY'].rolling(window=11).mean().values
# print('size of sma:{}, size of pandas rolling: {}'.format(smav.size, rm11.size))
# print(np.array_equal(smav, rm11))

# year_range = df["YEAR"].values
# plt.plot(year_range, df["SUNACTIVITY"].values, label="Original")
# plt.plot(year_range, rm11, label="SMA 11")
# plt.plot(year_range, df["SUNACTIVITY"].rolling(window=22).mean().values, label="SMA 22")
# plt.legend()
# plt.show()
