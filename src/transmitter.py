import numpy as np
import matplotlib.pyplot as plt
import math
import random
import consts
import time
from scipy import signal

from rfsoc_book.helper_functions import symbol_gen, psd, \
frequency_plot, scatterplot, calculate_evm, awgn
import data_func as df
import plt_funcs as pf




def interpolate_complex_signal_with_lowpass(input_signal, upsample_factor, fs):
    """
    使用低通滤波器对复数信号进行插值的函数

    参数:
    input_signal (array_like): 输入复数信号数组
    upsample_factor (int): 插值因子
    fs (float): 信号的采样率

    返回:
    interpolated_signal (ndarray): 插值后的复数信号数组
    t_upsampled (ndarray): 插值后的时间序列
    """

    # 生成插值后的时间序列
    t_upsampled = np.linspace(0, len(input_signal) / fs, len(input_signal) * upsample_factor)

    # 分别对实部和虚部进行插值
    interpolated_real = np.interp(t_upsampled, np.arange(len(input_signal)) / fs, np.real(input_signal))
    interpolated_imag = np.interp(t_upsampled, np.arange(len(input_signal)) / fs, np.imag(input_signal))

    # 设计低通滤波器
    cutoff_freq = 0.5 * fs / upsample_factor
    b, a = signal.butter(8, cutoff_freq / (fs / 2), 'low')

    # 对实部和虚部分别进行低通滤波
    interpolated_real = signal.filtfilt(b, a, interpolated_real)
    interpolated_imag = signal.filtfilt(b, a, interpolated_imag)

    # 合成复数信号
    interpolated_signal = interpolated_real + 1j * interpolated_imag

    return interpolated_signal, t_upsampled

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
    #lo_signal = np.cos(2 * np.pi * lo_frequency * t)
    mixed_signal = signal * lo_signal  # 混频信号

    return mixed_signal

def upsample(sig_in, upsample_rate):
    upsample_res = signal.resample(sig_in,upsample_rate*(len(sig_in)))
    return upsample_res

def spectrum_shift_right(sig_in, f_shift):
    """
    将输入信号在频谱上进行搬移

    参数:
    sig_in (ndarray): 输入信号（复数数组）
    shift (int): 频谱搬移量

    返回:
    ndarray: 搬移后的信号
    """
    shift = int(f_shift * len(sig_in) / consts.FS)

    # 对输入信号进行傅里叶变换
    sig_fft = np.fft.fft(sig_in)
    
    # 频谱搬移（向右移动）
    sig_shifted_fft = np.roll(sig_fft, shift)
    
    # 进行傅里叶逆变换
    sig_out = np.fft.ifft(sig_shifted_fft)
    
    return sig_out

def bpsk_mapping(value):
    return consts.bpsk_mapping_1div[value]

def qam16_mapping(value):
    # 16QAM调制符号映射表
    #print("value = ",value)
    return consts.QAM16_mapping_1div[value]

def qam4_mapping(value):
    return consts.QAM4_mapping_1div[value]

def QAM16_mod_single_ascii(data):
    # 将ASCII字符转换为整数
    ascii_value = ord(data)

    # 获取低8位中的高4位和低4位
    upper = (ascii_value >> 4) & 0x0F
    lower = ascii_value & 0x0F

    # 分别对upper和lower进行16QAM符号映射
    symb1 = qam16_mapping(upper)
    symb2 = qam16_mapping(lower)

    return symb1, symb2

def QAM16_mod_str(str_in):
    strlen = len(str_in)
    mod_res_len = len(str_in) *2
    mod_res = np.zeros(mod_res_len, dtype = np.complex64)

    for i in range(strlen):
       print("str_in[i]=",str_in[i])
       sym1,sym2=QAM16_mod_single_ascii(str_in[i])
       mod_res[i*2] = sym1
       mod_res[i*2+1] = sym2

    return mod_res

def QAM16_mod_ascii_file(file_in):
    str = df.read_ascii_file(file_in)
    return QAM16_mod_str(str)

def align_qam_symb(qam_sig):
    qam_sym_len = len(qam_sig)
    n_frames = math.ceil(qam_sym_len / consts.N_DATA / consts.N_OFDM)
    aligned_diff = (n_frames * consts.N_DATA * consts.N_OFDM) - qam_sym_len
    aligned_repeat = qam_sig[0:aligned_diff]
    qam_mod_aligned = np.zeros(n_frames * consts.N_DATA * consts.N_OFDM, dtype = np.complex64)
    qam_mod_aligned[0:qam_sym_len] = qam_sig
    qam_mod_aligned[qam_sym_len:qam_sym_len+aligned_diff] = aligned_repeat
    return n_frames, qam_mod_aligned

def add_cp(ofdm_symb,N,cp_len):
    
    #Extract CP
    cp = ofdm_symb[N-cp_len:N:1]
    
    # Concatenate CP and symbol 
    ofdm_symb_cp = np.concatenate((cp,ofdm_symb))
    
    return ofdm_symb_cp

def ofdm_mod(qam_sym):

    # check qam sym 
    #print("\n\n\n",qam_sym)

    sc_array = np.zeros(consts.N,np.complex64)
    sc_array[consts.OFDM_DATA_INDEX] = qam_sym
    sc_array[consts.OFDM_PILOT_INDEX] = consts.pilot_seq 

    #scatterplot(sc_array.real,sc_array.imag,ax=None)
    #plt.show()

    ofdm_mod = np.fft.ifft(sc_array,consts.N)

    ofdm_with_cp = add_cp(ofdm_mod,consts.N,consts.CP_LEN)

    return ofdm_with_cp


def Transmitter(ascii_file, output_file):
    #plot LLTF
    #pf.plot_all("LLTF_symb",consts.LLTF_symb,consts.SR)

    # 16QAM mod ascii file
    #qam_mod_res = QAM16_mod_ascii_file(ascii_file)

    #16QAM random array
    #qam_mod_res = np.zeros(consts.NSYM * consts.N_FRAME,dtype = np.complex64)
    #rand_qam_index = np.random.randint(0, 4, size=consts.NSYM* consts.N_FRAME)
    #for i in range(int((consts.NSYM * consts.N_FRAME)/2)):
    #    qam_mod_res[i] = qam4_mapping((rand_qam_index[i] ^ np.random.randint(0,4)))

    #for i in range(int((consts.NSYM * consts.N_FRAME)/2)):
    #    qam_mod_res[i+int((consts.NSYM * consts.N_FRAME)/2)] = -1*qam_mod_res[i]
        
    #df.save_array_to_file("D:/work/CA/simulation/dats/qam_mod_res.txt",qam_mod_res)
    qam_mod_res = np.zeros((consts.NSYM * consts.N_FRAME))
    qam_mod_res = np.loadtxt("D:/work/CA/simulation/dats/qam_mod_res.txt",dtype = np.complex64)
    # aligned qam signal to complete frame
    n_frames, qam_mod_aligned = align_qam_symb(qam_mod_res)

    

    align_qam_symb_sum = np.sum(qam_mod_aligned.real)
    print("align_qam_symb_sum = ",align_qam_symb_sum)
    #time.sleep(3)
    

    scatterplot(qam_mod_aligned.real,qam_mod_aligned.imag,ax=None)

    trans_res_len = consts.FRAME_LEN *n_frames
    trans_res = np.zeros(trans_res_len,dtype=np.complex64)
    
    print("n_frames = ",n_frames)
    # ofdm mod:
    for i in range(n_frames):
        # LLTF
        trans_res[i * consts.FRAME_LEN:i * consts.FRAME_LEN + consts.LTF_WITH_CP_LEN] = consts.LLTF

        # training seq
        training_seq_start_index = i * consts.FRAME_LEN + consts.LTF_WITH_CP_LEN
        trans_res[training_seq_start_index: training_seq_start_index + consts.TRAINING_SYMB_LEN] = consts.train_symbx2

        # payload
        ofdm_payload = np.zeros((consts.CP_LEN + consts.N)*consts.N_OFDM,dtype=np.complex64)
        
        for j in range(consts.N_OFDM):
            index = i*consts.N_OFDM*consts.N_DATA + j*consts.N_DATA
            ofdm_payload[j*(consts.CP_LEN+consts.N):(j+1)*(consts.CP_LEN + consts.N)] = ofdm_mod(qam_mod_aligned[index:index+consts.N_DATA])
        payload_start_index = i * consts.FRAME_LEN + consts.LTF_WITH_CP_LEN + consts.TRAINING_SYMB_LEN
        trans_res[payload_start_index:(i+1) * consts.FRAME_LEN] = ofdm_payload

        #store ofdm payload
        ofdm_payload_with_cp = np.zeros(consts.N * consts.N_OFDM, dtype = np.complex64)
        ofdm_payload_with_cp = ofdm_payload
        df.save_array_to_file(consts.tx_ofdm_payload_with_cp_file,ofdm_payload_with_cp)
    

    #up sample
        
    
    #trans_res = signal.resample_poly(trans_res,up=consts.SAMPLE_FACTOR,down=1)
    #trans_res = digital_heterodyning_complex(trans_res,consts.FREQ_SHIFT,512e6)
    #scatterplot(trans_res.real,trans_res.imag,ax=None)
    print("trans_res_len = ",len(trans_res))

    #trans_res = signal.resample_poly(trans_res,up=8,down=1)
    #trans_res = digital_heterodyning_complex(trans_res,1024e6,consts.FS)

    df.save_array_to_file(output_file,trans_res)
    #trans_res = upsample(trans_res, 8)
    #trans_res = upsample(trans_res, 2)


    #plot Transmitter res
    pf.plot_all("OFDM MOD RES imag",trans_res.imag, 512e6)
    pf.plot_all("OFDM MOD RES real",trans_res.real, 512e6)
    return