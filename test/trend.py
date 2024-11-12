
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('ETTh1.csv')  # 替换为您的数据文件路径
column_data = pd.to_datetime(df['OT'])
print(column_data)
def acfDataPlot(data):
    # 计算自相关函数
    acf_result = acf(column_data, nlags=20)

    # 设置 seaborn 风格
    sns.set(style='whitegrid')

    # 绘制图表
    plt.figure(figsize=(10, 6))
    plt.stem(acf_result)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation Function')

    # 保存图像文件
    plt.savefig('acf_plot.png')


def plot_lag_analysis(data):
    """
    绘制滞后性分析的ACF和PACF图
    参数:
    - data: 时间序列数据，可以是一个一维数组或列表
    """
    # 将数据转换为NumPy数组
    data = np.array(data)

    # 绘制ACF图
    plt.figure(figsize=(12, 4))
    ax1 = plt.subplot(121)
    plot_acf(data, ax=ax1)
    ax1.set_title('Autocorrelation Function (ACF)')

    # 绘制PACF图
    ax2 = plt.subplot(122)
    plot_pacf(data, ax=ax2)
    ax2.set_title('Partial Autocorrelation Function (PACF)')

    # 显示图形
    plt.tight_layout()
    plt.show()

plot_lag_analysis(column_data)

