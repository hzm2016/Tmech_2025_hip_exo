import numpy as np
from math import pi, tan, sin, cos

def butterworth_filter(data, cutoff, fs, order=4, filter_type='low'):
    """
    实现 Butterworth 滤波器
    :param data: 输入信号
    :param cutoff: 截止频率（Hz）
    :param fs: 采样率（Hz）
    :param order: 滤波器阶数
    :param filter_type: 滤波器类型，'low' 为低通，'high' 为高通
    :return: 滤波后的信号
    """
    # 归一化截止频率
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist

    # 预包络变换 (Pre-warping)
    omega = tan(pi * normalized_cutoff)

    # 计算滤波器系数 (Butterworth 特性)
    poles = []
    for k in range(order):
        angle = (2 * k + 1) * pi / (2 * order)
        poles.append(complex(-omega * sin(angle), omega * cos(angle)))

    # 系数计算 (差分方程)
    a = [1.0]
    for p in poles:
        a = np.convolve(a, [1.0, -2.0 * p.real, abs(p)**2])

    b = [omega**order]  # 滤波器增益

    # 数字化滤波器（双线性变换）
    if filter_type == 'high':
        a = a[::-1]
        b = [-b[0] if i % 2 else b[0] for i in range(len(a))]

    a = np.real(a).tolist()
    b = np.real(b).tolist()

    # 滤波计算（IIR Filter）
    y = np.zeros_like(data)
    x = np.array(data)

    for i in range(len(data)):
        y[i] = sum(b[j] * x[i - j] for j in range(len(b)) if i - j >= 0) \
             - sum(a[j] * y[i - j] for j in range(1, len(a)) if i - j >= 0)
    return y


# 测试代码
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 模拟输入信号
    fs = 1000  # 采样率
    t = np.linspace(0, 1, fs, endpoint=False)  # 时间
    input_signal = np.sin(2 * pi * 5 * t) + 0.5 * np.sin(2 * pi * 50 * t)  # 5 Hz + 50 Hz

    # Butterworth 滤波器应用
    cutoff = 50 # 截止频率 (Hz)
    filtered_signal = butterworth_filter(input_signal, cutoff, fs, order=4, filter_type='low')

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(t, input_signal, label='Original Signal')
    plt.plot(t, filtered_signal, label='Filtered Signal', color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Butterworth Low-pass Filter')
    plt.grid()
    plt.show()