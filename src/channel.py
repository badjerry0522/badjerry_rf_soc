import numpy as np
import matplotlib.pyplot as plt
import math
import random
import plt_funcs as loacl_plt
from scipy import signal
import consts

from rfsoc_book.helper_functions import symbol_gen, psd, \
frequency_plot, scatterplot, calculate_evm, awgn
import numpy as np

def add_phase(signal, phase_angle):
    """
    给复数信号添加相位角

    参数：
    signal: complex，要添加相位角的复数信号
    phase_angle: float，要添加的相位角（以弧度表示）

    返回：
    signal_with_phase: complex，添加相位角后的复数信号
    """
    return signal * np.exp(1j * phase_angle)

def digital_heterodyning_complex(signal, lo_frequency, fs):
    """
    复数信号的数字混频函数

    参数:
    signal (array_like): 输入信号数组，复数形式
    lo_frequency (float): 本振频率
    fs (float): 采样频率

    返回:
    mixed_signal (ndarray): 混频后的信号数组，复数形式
    """

    t = np.arange(len(signal)) / fs  # 时间序列
    lo_signal = np.exp(1j * 2 * np.pi * lo_frequency * t)  # 本振信号
    mixed_signal = signal * lo_signal  # 混频信号

    return mixed_signal

def AWGN_channel(tx_output_file,channel_output_file,SNR):
    txSig = np.loadtxt(tx_output_file,dtype = np.complex64)
    # Filter coefficients
    #ntap = 1
    #h = np.random.randn(ntap) + np.random.randn(ntap)*1j

    # Appy channel filter 
    #txSig_filt = np.convolve(txSig, h)

    channelSig = txSig

    real = txSig.real
    imag = txSig.imag


    #channelSig.real = 0.98 * real + 0.17 * imag
    #channelSig.imag = -1*0.17 * real - 0.98 * imag
    channelSig = awgn(txSig,SNR)
    #channelSig = digital_heterodyning_complex(channelSig,4e4,consts.FS)
    #loacl_plt.plot_all("channel_signal", channelSig,consts.FS)
    #channelSig = add_phase(channelSig, np.pi*1.7)
    np.savetxt(channel_output_file,channelSig,fmt="%.6f")