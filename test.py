import matplotlib.pyplot as plt
import numpy as np

# 设置字体为楷体
plt.rcParams['font.family'] = 'KaiTi'

# 生成时间序列（假设从0到100秒）
time = np.linspace(0, 100, 1000)

# 生成带宽数据，使用正弦波模拟波动
bandwidth = 100 + 50 * np.sin(0.1 * time) + 20 * np.sin(0.5 * time) + np.random.normal(0, 5, len(time))

# 绘制图表
plt.figure(figsize=(10, 6))
plt.plot(time, bandwidth, color='blue')
plt.xlabel('时间', fontsize=16)
plt.ylabel('带宽', fontsize=16)
plt.show()