import numpy as np
import matplotlib.pyplot as plt
import math
import random
from scipy import signal
from rfsoc_book.helper_functions import symbol_gen, psd, \
frequency_plot, scatterplot, calculate_evm, awgn
import data_func as df
import plt_funcs as pf
import consts

#def dac_fir2():
#def dac_fir1():
#def dac_fir0():
#def adc_fir2():
#def adc_fir1():
#def adc_fir0():

def ranging(local_ofdm_payload, rx_ofdm_payload, no_frame,i):
    c=3 * 10 ** 8
    B=1.024 * 10 ** 9
    deltaR = c/(2*B)
    range = np.linspace(0,consts.N,consts.N)*deltaR
    rx_ofdm_payload_after_fft = np.fft.fft(rx_ofdm_payload,consts.N)
    local_ofdm_payload_after_fft = np.fft.fft(local_ofdm_payload,consts.N)

    csi = rx_ofdm_payload_after_fft/local_ofdm_payload_after_fft

    ranging_res = np.fft.ifft(csi,consts.N)
    ranging_res_abs = np.abs(ranging_res)
    ranging_res_dB = 20*np.log10(ranging_res_abs/max(ranging_res_abs))
    # plt.figure("range info")
    # plt.plot(range,ranging_res_dB)
    # plt.title([no_frame,'_',i])
    # plt.ylabel('Ambiguity (dB)')
    # plt.xlabel('Distance (m)')
    # plt.show()

    return ranging_res_abs


def estimate_phi(training_symb):

    

    train_symb_mod = consts.train_seq_ifft[0].real * consts.train_seq_ifft[0].real - consts.train_seq_ifft[0].imag * consts.train_seq_ifft[0].imag

    cos_phi = (training_symb.real * consts.train_seq_ifft[0].real + training_symb.imag * consts.train_seq_ifft[0].imag) / train_symb_mod
    sin_phi = -1 * (training_symb.imag * consts.train_seq_ifft[0].real + training_symb.real * consts.train_seq_ifft[0].imag) / train_symb_mod
    #phi_gain = np.sqrt(1/(cos_phi * cos_phi + sin_phi * sin_phi))

    cos_phi = cos_phi #* phi_gain
    cos_phi_aver = np.sum(cos_phi)/consts.N
    sin_phi = sin_phi #* phi_gain
    sin_phi_aver = np.sum(sin_phi)/consts.N
    print("cos_phi = ,",cos_phi,"\ncos_phi_aver = ",cos_phi_aver)
    print("sin_phi = ,",sin_phi,"\nsin_phi_aver = ",sin_phi_aver)

    #pf.plot_all("training_symb after samples",training_symb.real,512e6)
    #pf.plot_all("training_symb before sampels",consts.LTFsymb.real,512e6)
    #plt.show()

    return cos_phi,cos_phi_aver,sin_phi,sin_phi_aver

def phase_correct(input_sig, cos_phi, sin_phi):

    print("input_sig = ",input_sig)

    correct_phi_res_real  = cos_phi * (input_sig.real) + sin_phi * (input_sig.imag)
    correct_phi_res_imag  = cos_phi * (input_sig.imag) + sin_phi * (input_sig.real)

    return ((correct_phi_res_real + correct_phi_res_imag * 1j) / (cos_phi *cos_phi - sin_phi * sin_phi))

def estimate_frequency_offset(training_symb1,training_symb2):
    symb2= training_symb2
    symb1= training_symb1

    L = consts.N

    r = symb1 * np.conj(symb2)
    r_sum = np.sum(r)
    r_angle = np.angle(r_sum) / (np.pi * 2)

    freq_off = (r_angle* 512e6) /(L) 

    return r_angle,freq_off


def decimate_with_lowpass(input_signal, decimation_factor, fs):
    """
    使用低通滤波器进行降采样的函数

    参数:
    input_signal (array_like): 输入复数信号数组
    decimation_factor (int): 降采样因子
    fs (float): 信号的采样率

    返回:
    decimated_signal (ndarray): 降采样后的复数信号数组
    t_decimated (ndarray): 降采样后的时间序列
    """

    # 设计低通滤波器
    cutoff_freq = 0.5 * fs / decimation_factor
    b, a = signal.butter(8, cutoff_freq / (fs / 2), 'low')

    # 对输入信号进行低通滤波
    filtered_signal_real = signal.filtfilt(b, a, np.real(input_signal))
    filtered_signal_imag = signal.filtfilt(b, a, np.imag(input_signal))

    # 进行降采样
    decimated_signal_real = signal.decimate(filtered_signal_real, decimation_factor, ftype = "fir")
    decimated_signal_imag = signal.decimate(filtered_signal_imag, decimation_factor, ftype = "fir")

    # 生成降采样后的时间序列
    t_decimated = np.linspace(0, len(input_signal) / fs, len(decimated_signal_real))

    # 合成复数信号
    decimated_signal = decimated_signal_real + 1j * decimated_signal_imag

    return decimated_signal, t_decimated

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
    lo_signal = np.exp(-1j * 2 * np.pi * lo_frequency * t)  # 本振信号
    mixed_signal = signal * lo_signal  # 混频信号

    return mixed_signal

def downsample(sig_in, downsample_rate):
    downsample_res = signal.resample(sig_in,int(len(sig_in) / downsample_rate))
    return downsample_res

def spectrum_shift_left(sig_in, f_shift):
    """
    将输入信号在频谱上向左进行搬移

    参数:
    sig_in (ndarray): 输入信号（复数数组）
    f_shift (float): 频谱搬移量，单位为Hz
    fs (float): 原始采样率

    返回:
    ndarray: 搬移后的信号
    """
    shift = int(-f_shift * len(sig_in) / consts.FS)

    # 对输入信号进行傅里叶变换
    sig_fft = np.fft.fft(sig_in)
    
    # 频谱搬移（向左移动）
    sig_shifted_fft = np.roll(sig_fft, shift)
    
    # 进行傅里叶逆变换
    sig_out = np.fft.ifft(sig_shifted_fft)
    
    return sig_out

def channel_est_pilot(ofdm_symb):
    channel_est_res = np.zeros(consts.OFDM_DATA_BLOCK_NUM,dtype = np.complex64)
    recv_pilot = ofdm_symb[consts.OFDM_PILOT_INDEX]
    h = recv_pilot / (consts.pilot_seq)

    for i in range(5):
        channel_est_res[i] = (h[i] + h[i+1])/2
    
    

    for i in range(5,9):
        channel_est_res[i-1] = (h[i] + h[i+1])/2

    #print("channel_est_res = ",channel_est_res)

    return channel_est_res

def channel_est(LLTF_symb_1, LLTF_symb_2):

    LLTF_data_1 = LLTF_symb_1[consts.OFDM_PAYLOAD_INDEX]
    LLTF_data_2 = LLTF_symb_2[consts.OFDM_PAYLOAD_INDEX] 

    h_1 = LLTF_data_1 / (consts.train_seq[consts.OFDM_PAYLOAD_INDEX] * consts.TRAINING_GAIN)
    h_2 = LLTF_data_2 / (consts.train_seq[consts.OFDM_PAYLOAD_INDEX] * consts.TRAINING_GAIN)
    #h_1 = LLTF_data_1 / (consts.LTFsymb[consts.OFDM_DATA_INDEX])
    #h_2 = LLTF_data_2 / (consts.LTFsymb[consts.OFDM_DATA_INDEX])

    h_final = (h_1 + h_2) / 2
    #print(h_final)

    return h_final

# channel_equ: input: ofdm_symb[index] !!! the input has removed pilot 
def channel_equ(ofdm_symb, channel_est_res):
    channel_equ_res = ofdm_symb * (np.conj(channel_est_res)/(abs(channel_est_res)**2))
    return channel_equ_res

def channel_equ_pilot(ofdm_symb,channel_est_res):
    tmp = np.zeros(consts.N,dtype = np.complex64)
    for i in range(consts.OFDM_DATA_BLOCK_NUM):
        index_block = consts.OFDM_DATA_INDEX[i*4:i*4+4]
        tmp[index_block] = ofdm_symb[index_block] * (np.conj(channel_est_res[i])/(abs(channel_est_res[i])**2))
    #tmp[consts.OFDM_PILOT_INDEX] = ofdm_symb[consts.OFDM_PILOT_INDEX]
    return tmp[consts.OFDM_DATA_INDEX]

#ofdm_demod
def ofdm_demod(ofdm_rx,N,cp_len):
    # Remove CP 
    ofdm_u = ofdm_rx[cp_len:(N+cp_len)]
    #print(ofdm_u)
    # Perform FFT 
    data = np.fft.fft(ofdm_u,N)
    
    return data

def qam16_demodulation(symbol):

    # 计算输入符号与所有16QAM调制符号之间的距离
    distances = [np.abs(symbol - value) for value in consts.QAM16_mapping_2div.values()]

    # 找到距离最小的调制符号
    min_distance_index = np.argmin(distances)

    # 根据映射表找到对应的二进制数据
    demodulated_bits = list(consts.QAM16_mapping_2div.keys())[min_distance_index]

    # 将二进制数据转换为一个8位的无符号整数（np.uint8）
    result = np.uint8(demodulated_bits[0] << 2 | demodulated_bits[1])

    return result

def qam16_symb_to_uint8(qam_data):
    qam_demod_res = np.zeros(len(qam_data)//2,dtype=np.uint8)
    for i in range(len(qam_data)//2):
        symb1 = qam_data[i*2]
        symb2 = qam_data[i*2+1]

        upper = qam16_demodulation(symb1)
        lower = qam16_demodulation(symb2)

        tmp = (upper << 4) | lower
        qam_demod_res[i] = tmp
    return qam_demod_res



def Receiver(channel_output_file, rx_output_file):
    
    rx_din_sig = np.loadtxt(channel_output_file,dtype = np.complex64) 
    #rx_din_sig.imag = np.zeros(320000)

    pf.plot_all("sig before downsample",rx_din_sig.real,512e6)

    #rx_din_sig = digital_heterodyning_complex(rx_din_sig,1024e6,consts.FS)

    #real = rx_din_sig.real
    #imag = rx_din_sig.imag
    #rx_din_sig.real = 0.98 * real + 0.17 * imag
    #rx_din_sig.imag = -1*0.17 * real - 0.98 * imag 

    #rx_din_sig = signal.decimate(rx_din_sig,8,ftype="fir")

    
    #rx_din_sig = rx_din_sig *  np.exp(1j*0.9*np.pi)


    # test: 120000 - 160000
    #rx_din_sig = (rx_din_sig[20000:55000])
    pf.plot_all("sig after downsample real",rx_din_sig.real,512e6)
    pf.plot_all("sig after downsample imag",rx_din_sig.imag,512e6)
    
    #rx_din_sig = digital_heterodyning_complex(rx_din_sig,consts.FREQ_SHIFT,512e6)
    #rx_din_sig = signal.decimate(rx_din_sig,consts.SAMPLE_FACTOR,ftype="fir")

    #test_real = rx_din_sig.real[0:9770]
    #test_imag = rx_din_sig.imag[75:9770+75]

    #rx_din_sig = np.zeros(9770,dtype= np.complex64)
    #rx_din_sig.real = test_real
    #rx_din_sig.imag = test_imag

    #pf.plot_all("test_real",test_real,512e6)
    #pf.plot_all("test_imag",test_imag,512e6)

    #imag_peak = -96
    #real_peak = 640
    #sin_phi = (imag_peak) / np.sqrt(imag_peak * imag_peak + real_peak * real_peak)
    #cos_phi = (real_peak) / np.sqrt(imag_peak * imag_peak + real_peak * real_peak)
    #print("cos_phi = ",cos_phi,"\nsin_phi = ",sin_phi)

    #phase_correct_I = cos_phi * test_real + sin_phi * test_imag
    #phase_correct_Q = sin_phi * test_real - cos_phi * test_imag
#
    #test_sig = test_real +  1j*test_imag
#
    #pf.plot_all("phase_correct_I",phase_correct_I,512e6)
    #pf.plot_all("phase_correct_Q",phase_correct_Q,512e6)


    # analyse
    # iamg: 214  real 5318
    '''
    real_ts_res = signal.convolve(in1=rx_din_sig.real,in2 = np.conj(consts.LTF_matched_filter), mode = "same")
    real_ts_res = np.abs(real_ts_res) ** 2
    imag_ts_res = signal.convolve(in1=rx_din_sig.imag,in2 = np.conj(consts.LTF_matched_filter), mode = "same")
    imag_ts_res = np.abs(imag_ts_res) ** 2

    real_pks = signal.find_peaks(real_ts_res,height = 2e17, distance=consts.N)
    print("real_peaks = ",real_pks)

    imag_pks = signal.find_peaks(imag_ts_res,height = 2e17, distance=consts.N)
    print("iamg_peaks = ",imag_pks)

    pf.plot_dat("real_ts_res sync res",real_ts_res)
    pf.plot_dat("imag_ts_res sync res",imag_ts_res)

    offset = real_pks[0][0] - imag_pks[0][0]
    print("offset = ",offset)

    test_real = rx_din_sig.real[offset:consts.FRAME_LEN*10+offset]
    test_imag = rx_din_sig.imag[0:consts.FRAME_LEN*10]
    rx_din_sig = np.zeros(consts.FRAME_LEN*10,dtype= np.complex64)
    pf.plot_dat("real after offset",test_real)
    pf.plot_dat("imag after offset",test_imag)
    rx_din_sig.real = test_real
    rx_din_sig.imag = test_imag
    '''
    #plt.show()


    # time sync
    # not very good i think
    ts_res = signal.convolve(in1=rx_din_sig.real, in2 = np.conj(consts.LTF_matched_filter), mode = "same")
    ts_res = np.abs(ts_res) ** 2
    #ts_res = signal.convolve(in1 = rx_din_sig, in2 = np.conj(consts.head),mode = "same")
    pf.plot_dat("time sync res",ts_res)
    #plt.show()
    pks = signal.find_peaks(ts_res,height = consts.PEAK, distance=64)
    print(pks)

    plt.show()
    
  
    num_frame = int(len(pks[0]) // 2)
    rx_payload_after_fft = np.zeros((consts.N_DATA) * consts.N_OFDM * num_frame,np.complex64)

    print("num_frame = ",num_frame)

    for no_frame in range(num_frame):
        # get index
        LTF_seq_start_index = pks[0][no_frame*2+1] - 95
        if(LTF_seq_start_index + consts.FRAME_LEN - 32  > len(rx_din_sig)):
            break
        print("LTF_seq_start_index = ",LTF_seq_start_index)
        training_seq_start_index = pks[0][no_frame*2+1] + 32 + 1
        payload_start_index = training_seq_start_index + consts.TRAINING_SYMB_LEN

       

        # get LTF seq 
        LTF_symb1 = rx_din_sig[LTF_seq_start_index: LTF_seq_start_index + consts.LTF_SEQ_LEN]
        print("LTF_symb1_real = ",LTF_symb1.real)
        LTF_symb2 = rx_din_sig[LTF_seq_start_index + consts.LTF_SEQ_LEN: LTF_seq_start_index + 2*consts.LTF_SEQ_LEN]

        # est freq off
        r_angle,freq_off = estimate_frequency_offset(LTF_symb1,LTF_symb2)
        np.set_printoptions(suppress=True)
        print("freq_off = ",freq_off)

        # correct freq off
        payload = rx_din_sig [training_seq_start_index: training_seq_start_index + consts.TRAINING_SYMB_LEN + consts.PAYLOAD_LEN]
        #rx_din_sig [training_seq_start_index: training_seq_start_index + consts.TRAINING_SYMB_LEN + consts.PAYLOAD_LEN] = digital_heterodyning_complex(payload,freq_off,512e6)
        np.set_printoptions(suppress=True)
        print("r_angle = ",r_angle)


        # get trainning seq 
        training_symb1 = rx_din_sig[training_seq_start_index: training_seq_start_index + consts.CP_LEN + consts.N]
        training_symb2 = rx_din_sig[training_seq_start_index + consts.CP_LEN + consts.N: payload_start_index]
        df.save_array_to_file("D:/work/CA/simulation/dats/training_symb.txt",np.concatenate((training_symb1,training_symb2)))
        training_symb_all = np.concatenate((training_symb1,training_symb2))
        df.complex_file_to_IQ_file("D:/work/CA/simulation/dats/training_symb.txt",
                                   "D:/work/CA/simulation/dats/training_symb_I.txt",
                                   "D:/work/CA/simulation/dats/training_symb_Q.txt",
                                   "D:/work/CA/simulation/dats/training_symb_QI.txt")

        # est phase off
        cos_phi , cos_phi_aver, sin_phi,sin_phi_aver =  estimate_phi(training_symb2[consts.CP_LEN])
        #payload = rx_din_sig [training_seq_start_index: training_seq_start_index + consts.TRAINING_SYMB_LEN + consts.PAYLOAD_LEN]
        #rx_din_sig [training_seq_start_index: training_seq_start_index + consts.LLTF_SYMB_LEN + consts.PAYLOAD_LEN] = phase_correct(payload,cos_phi,sin_phi)
        
        

        


        #rx_din_sig = rx_din_sig * np.exp(-1j*np.pi*r_angle)

        #get trainning seq 
        training_symb1 = rx_din_sig[training_seq_start_index: training_seq_start_index + consts.CP_LEN + consts.N]
        training_symb2 = rx_din_sig[training_seq_start_index + consts.CP_LEN + consts.N: payload_start_index]

        # channel est using training seq
        channel_est_res = channel_est(ofdm_demod(training_symb1, consts.N, consts.CP_LEN),ofdm_demod(training_symb2, consts.N, consts.CP_LEN))

        

        #get payload
        payload_before_fft = rx_din_sig[payload_start_index:payload_start_index + (consts.CP_LEN + consts.N) * consts.N_OFDM]

        data_rx = np.zeros(consts.N_OFDM * consts.N_DATA,np.complex64)
        j = 0
        k = 0 
        ranging_all = np.zeros([consts.N_OFDM,consts.N])

        for i in range(consts.N_OFDM):
            # ranging
            ofdm_payload_no_cp = payload_before_fft[k + consts.CP_LEN:(k + consts.N + consts.CP_LEN)]
            local_ofdm_payload_no_cp = np.loadtxt(consts.tx_ofdm_payload_with_cp_file,dtype = np.complex64)
            local_ofdm_payload_no_cp_per_symb = local_ofdm_payload_no_cp[k + consts.CP_LEN:(k + consts.N + consts.CP_LEN)]
            ranging_res = ranging(local_ofdm_payload_no_cp_per_symb,ofdm_payload_no_cp,no_frame,i)
            ranging_all[i,:] = ranging_res
            # FFT
            rx_demod = ofdm_demod(payload_before_fft[k:(k + consts.N + consts.CP_LEN)],consts.N,consts.CP_LEN)
            # channel equ using training seq
            # rx_demod[consts.OFDM_PAYLOAD_INDEX] = channel_equ(rx_demod[consts.OFDM_PAYLOAD_INDEX],channel_est_res)

            # channel est using pilot 
            #channel_est_pilot_res = channel_est_pilot(rx_demod)
            # channel equ using pilot
            #channel_equ_res = channel_equ_pilot(rx_demod,channel_est_pilot_res)

            #data_rx[j:j+consts.N_DATA] = channel_equ_res
            data_rx[j:j+consts.N_DATA] = rx_demod[consts.OFDM_DATA_INDEX]

            j = j + consts.N_DATA
            k = k + consts.N + consts.CP_LEN 

        ranging_mean = np.mean(ranging_all,0)
        ranging_mean_dB = 10*np.log10(ranging_mean/max(ranging_mean))
        print('对一帧求均值:',np.size(ranging_mean_dB))
        plt.title([no_frame,'rangeEstimation'])
        plt.plot(consts.range,ranging_mean_dB)
        plt.ylabel('Ambiguity (dB)')
        plt.xlabel('Distance (m)')
        plt.show()
        

        print("data_rx len = ",len(data_rx))

        rx_payload_after_fft[no_frame*consts.N_DATA*consts.N_OFDM:(no_frame + 1)*consts.N_DATA*consts.N_OFDM] = data_rx #都用有效数据覆盖
        print("rx_payload_after_fft = ",len(rx_payload_after_fft))

    #for i in range (len(rx_payload_after_fft)):
    #    if(abs(rx_payload_after_fft[i].real) > 2e6):
    #        print("i= ",i)    

    pf.plot_constellation_peaks(rx_payload_after_fft) #传入若干组有效数据
    plt.show()
    
    #plot demod res 
    scatterplot(rx_payload_after_fft.real,rx_payload_after_fft.imag,ax=None)

    #demod qam
    qam_demod_res = qam16_symb_to_uint8(rx_payload_after_fft)
    #write demod res
    df.write_uint8_to_ascii(qam_demod_res,rx_output_file)
    return
