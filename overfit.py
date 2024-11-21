import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
matplotlib.rcParams['font.serif'] = ['KaiTi']

# 生成指数退化数据，加上不规则噪声
np.random.seed(42)
dots_num = 400
time = np.linspace(0, 10, dots_num)  # 时间轴，500个高频数据点
decay_data = np.exp(0.3 * time)  # 指数退化趋势
noise = 2 * np.random.randn(dots_num)  # 不规则噪声
high_freq_data = decay_data + noise  # 高频退化数据（加入噪声）
fake_low_freq_data = decay_data + ( 1 * np.random.randn(dots_num))

# 将高频数据前70%作为训练，后30%作为"真实"数据进行预测
train_high = high_freq_data[:int(dots_num*0.7)]
test_high = high_freq_data[int(dots_num*0.7):]

# 构造高频数据的"预测"结果，假设在测试区间表现不佳的过拟合效果
# pred_high_freq = train_high[-1] * np.ones_like(test_high)  # 过拟合导致预测保持训练尾部的水平
pred_high_freq = test_high - np.exp(np.linspace(0,2,len(test_high)))

# 低频采样（每10个点取1个），得到低频退化数据
low_freq_time = time[::5]
low_freq_data = fake_low_freq_data[::5]
low_dots_num  = len(low_freq_time)
# 将低频数据前70%作为训练，后30%作为"真实"数据进行预测
train_low = low_freq_data[:int(low_dots_num*0.7)]
test_low = low_freq_data[int(low_dots_num*0.7):]

# 构造低频数据的"预测"结果，假设在测试区间有更好的趋势保持效果
# pred_low_freq = np.exp(-0.05 * time[int(dots_num*0.7):])  # 用真实退化趋势预测，避免过拟合
pred_low_freq = test_high

# 绘图
plt.figure(figsize=(14, 6))

# 高频数据预测结果
plt.subplot(1, 2, 1)
# plt.plot(time, high_freq_data, label="High-Frequency Data (True)", color="blue", alpha=0.6)
plt.plot(time[:int(dots_num*0.7)], train_high, label="训练集", color="orange", alpha=0.8)
plt.plot(time[int(dots_num*0.7):], test_high, label="测试集", color="blue", linestyle='dashed')
plt.plot(time[int(dots_num*0.7):], pred_high_freq, label="预测", color="red", linestyle='--')
plt.title("原始数据")
plt.xlabel("时间")
plt.ylabel("退化数据")
plt.legend()

# 低频数据预测结果
plt.subplot(1, 2, 2)
# plt.plot(low_freq_time, low_freq_data, label="Low-Frequency Data (True)", color="blue", alpha=0.6)
plt.plot(low_freq_time[:int(low_dots_num*0.7)], train_low, label="训练集", color="orange", alpha=0.8)
plt.plot(time[int(dots_num*0.7):], pred_low_freq, label="测试集", color="blue", linestyle='dashed')
plt.plot(low_freq_time[int(low_dots_num*0.7):], test_low, label="预测", color="red", linestyle='--')
plt.title("稀疏数据")
plt.xlabel("时间")
plt.ylabel("退化数据")
plt.legend()

plt.tight_layout()
# plt.show()
plt.savefig("draw3.png")
plt.close
