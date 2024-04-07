import numpy as np
import scipy.signal as signal
import preprocessed as prep
import matplotlib.pyplot as plt
# params
FS = 4096e6 # Sampling rate 
SR = 4096e6
#N = 4096 # No. of sub-carriers
N = 64
#N = 256
SAMPLE_FACTOR = 4
FREQ_SHIFT = 128e6
# file name
ascii_tx_file = "../dats/ascii_tx_input.txt"
tx_output_sig_file = "../dats/rtl_input/tx_output.txt"
tx_output_sig_I_file = "../dats/rtl_input/tx_Im_output.txt"
tx_output_sig_Q_file = "../dats/rtl_input/tx_Re_output.txt"
channel_output_sig_file = "../dats/channel_output.txt"
rx_output_ascii_file = "../dats/rx_output_ascii.txt"

tx_4096_samp = "../dats/tx_4096_samp.txt"
tx_ofdm_payload_with_cp_file = "../dats/ofdm_payload_with_cp.txt"

# rtl test input
tx_output_rtl_test_file = "../dats/rtl_input/rtl_test_input.txt"
tx_output_rtl_test_256bit_file = "../dats/rtl_input/rtl_test_input_256bit.txt"

# rtl test res
test_res_Q = "../dats/rtl_output/test_output_Q_file.txt"
test_res_I = "../dats/rtl_output/test_output_I_file.txt"
test_res_complex = "../dats/rtl_output/test_res_complex.txt"



# QAM16 mapping
QAM_GAIN = 65536
BPSK_GAIN = 32767
bpsk_mapping_1div = {
        0: -BPSK_GAIN,
        1: BPSK_GAIN
}

QAM4_mapping_1div = {
            0: (1 + 1j) *  QAM_GAIN,
            1: (-1 + 1j) * QAM_GAIN,
            2: (1 - 1j) *  QAM_GAIN, 
            3: (-1 - 1j) * QAM_GAIN
}
QAM16_mapping_1div = {
        0: ( 3 - 3j)*QAM_GAIN,
        1: ( 3 - 1j)*QAM_GAIN,
        2: ( 3 + 3j)*QAM_GAIN, 
        3: ( 3 + 1j)*QAM_GAIN,
        4: ( 1 - 3j)*QAM_GAIN, 
        5: ( 1 - 1j)*QAM_GAIN, 
        6: ( 1 + 3j)*QAM_GAIN, 
        7: ( 1 + 1j)*QAM_GAIN,
        8: (-3 - 3j)*QAM_GAIN, 
        9: (-3 - 1j)*QAM_GAIN, 
        10:(-3 + 3j)*QAM_GAIN, 
        11:(-3 + 1j)*QAM_GAIN,
        12:(-1 - 3j)*QAM_GAIN, 
        13:(-1 - 1j)*QAM_GAIN, 
        14:(-1 + 3j)*QAM_GAIN, 
        15:(-1 + 1j)*QAM_GAIN,
}
QAM16_mapping_2div = {
        (0, 0): (3- 3j)*QAM_GAIN,
        (0, 1): (3- 1j)*QAM_GAIN,
        (0, 2): (3+ 3j)*QAM_GAIN,
        (0, 3): (3+ 1j)*QAM_GAIN,
        (1, 0): (1- 3j)*QAM_GAIN,
        (1, 1): (1- 1j)*QAM_GAIN,
        (1, 2): (1+ 3j)*QAM_GAIN,
        (1, 3): (1+ 1j)*QAM_GAIN,
        (2, 0): (3- 3j)*QAM_GAIN,
        (2, 1): (3- 1j)*QAM_GAIN,
        (2, 2): (3+ 3j)*QAM_GAIN,
        (2, 3): (3+ 1j)*QAM_GAIN,
        (3, 0): (1- 3j)*QAM_GAIN,
        (3, 1): (1- 1j)*QAM_GAIN,
        (3, 2): (1+ 3j)*QAM_GAIN,
        (3, 3): (1+ 1j)*QAM_GAIN,
}

# m seq
head = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1,\
        0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0,\
        1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0,\
        1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1,\
        0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0,\
        1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1])
HEAD_LEN = 160
HEAD_GAIN = 16384
zero_before_head = np.zeros(33,dtype = np.int32)
head_symble = np.concatenate((zero_before_head, head)) * HEAD_GAIN


# LLTF
LLTF_GAIN = 30000
LTF_SEQ_LEN = 64
LTFseq = np.array([0,0,0,0,0,0,1,1,-1,\
                  -1,1,1,-1,1,-1,1,1,1,\
                  1,1,1,-1,-1,1,1,-1,1,\
                  -1,1,1,1,1,0,1,-1,-1,\
                  1,1,-1,1,-1,1,-1,-1,\
                  -1,-1,-1,1,1,-1,-1,1,\
                  -1,1,-1,1,1,1,1,0,0,0,\
                  0,0])
print(len(LTFseq))
LTFseq_inverted = np.array([0,0,\
                            0,0,0,1,1,1,1,-1,1,-1,\
                            1,-1,-1,1,1,-1,-1,-1,\
                            -1,-1,1,-1,1,-1,1,1\
                            -1,-1,1,0,1,1,1,1,-1,\
                            1,-1,1,1,-1,-1,1,1,1,\
                            1,1,1,-1,1,-1,1,1,-1,\
                            -1,1,1,0,0,0,0,0,0
                            ])
LTFseq_inverted_1 = np.array([0,0,\
                            0,0,0,1,1,1,1,-1,1,-1,\
                            1,-1,-1,1,1,-1,-1,-1,\
                            -1,-1,1,-1,1,-1,1,1\
                            -1,-1,1,0,1,1,1,1,-1,\
                            1,-1,1,1,1,-1,-1,1,1,1,\
                            1,1,1,-1,1,-1,1,1,-1,\
                            -1,1,1,0,0,0,0,0,0
                            ])
#plt.figure(1)
#plt.plot(signal.convolve(in1 = LTFseq, in2 = LTFseq_inverted_1,mode = "same"))
#plt.show()
print(len(LTFseq_inverted))
#LTF_symb1_real = np.array([26.79578968, -39.04881868, 58.96075908, -212.46020464, 350.58806485, -592.28104507, 17225.29003116, 15309.18920132, -15074.14591433, -18052.22007339, 18225.05238682, 14469.22154675, -14346.87680192, 14337.73294351, -14467.22323818, 14665.57709577, 17864.98541264, 15322.08360949, 17051.91922041, 16286.39341782, 16017.85227214, -15615.48316436, -17653.92616348, 17947.7538991, 14704.38555181, -14594.2817524, 14608.62310479, -14856.00083193, 15105.17036755, 17321.20609454, 15940.8532192, 16363.93783868, 661.44993441, 15232.29783694, -14686.67136296, -18619.37255373, 19081.59541463, 13424.90014646, -13244.29111047, 13104.36066531, -13074.26458088, 13061.39843108, -13141.06727675, -19644.94313876, -13262.85526004, -19490.42488742, -13412.50835127, 13355.33998127, 19434.42774771, -19440.65966137, -13407.33949831, 13434.61672887, -13498.62309154, 13640.24132622, -13767.40345033, 14026.12636165, 18575.04782396, 14586.84944373, 17874.64401532, -1119.45966227, 812.17538024, -559.83544502, 333.61115395, -128.80925472])



LTFcp = LTFseq[32:64]
# LTF matched filter
LTF_matched_filter = LTFseq_inverted * LLTF_GAIN
# Concatenate to form L-LTF
LLTF = np.concatenate((LTFcp, LTFseq, LTFseq)) * LLTF_GAIN





N_FRAME = 1

CP_LEN = N // 4 # CP length is 1/4 of symbol period
N_TRAINING_SEQ = 2
N_OFDM = 4 # No. of OFDM symbols per frame
#N_DATA = 3200 # No. of data sub-carriers
N_DATA = 52
#N_DATA = 200
#N_PILOT = 10
N_PILOT = 12
N_PAYLOAD = N_DATA + N_PILOT
NSYM = N_OFDM * N_DATA # No. of data symbols
PAYLOAD_LEN = N_OFDM * (CP_LEN + N)

# LLTF symb as training seq
#train_seq_real = np.ones(N) 
train_seq_real = LTFseq_inverted_1
#print(train_seq_real)
train_seq = train_seq_real

TRAINING_SYMB_LEN = 2*(CP_LEN + N)
TRAINING_GAIN = QAM_GAIN
train_seq_ifft = np.fft.ifft((train_seq * TRAINING_GAIN),N)
#LTFsymb = train_seq * TRAINING_GAIN
#LTFsymb = np.fft.ifft(np.fft.fftshift(LTFseq * LLTF_GAIN),N)
train_seq_ifft_inver = train_seq_ifft[::-1]
train_seq_cp = train_seq_ifft[N-CP_LEN:N]
train_symb = np.concatenate((train_seq_cp,train_seq_ifft))
train_symbx2 = np.concatenate((train_symb,train_symb))


ind_1 = np.arange(start=6, stop=32)
ind_2 = np.arange(start=33, stop=59)

#ind_1 = np.arange(start=16, stop=32)
#ind_2 = np.arange(start=43, stop=59)

#ind_1 = np.arange(start=348, stop=1953)
#ind_2 = np.arange(start=2143, stop=3748)

#ind_1 = np.arange(start=20, stop=124)
#ind_2 = np.arange(start=131, stop=235)
OFDM_PAYLOAD_INDEX = np.concatenate((ind_1,ind_2))
PILOT_GAIN = 32767

#OFDM_PILOT_INDEX = np.array([348,749,1150,1551,1952,2143,2544,2945,3346,3747])
#OFDM_PILOT_INDEX = np.array([20,46,72,98,124,131,157,183,209,235])
OFDM_PILOT_INDEX = np.array([0,1,2,3,4,5,32,59,60,61,62,63])
OFDM_DATA_BLOCK_NUM = 8
OFDM_DATA_INDEX = np.setdiff1d(OFDM_PAYLOAD_INDEX, OFDM_PILOT_INDEX)
print("DATA INDEX = ",OFDM_DATA_INDEX)


pilot_seq_real = np.array([4.242,4.242,4.242,4.242,\
                      4.242,4.242,4.242,4.242,\
                      4.242,4.242,4.242,4.242]) * PILOT_GAIN
pilot_seq = pilot_seq_real 


LTF_DATA_LEN = 64
LTF_CP_LEN = 32
LTF_WITH_CP_LEN = 160

FRAME_LEN = (CP_LEN + N ) * (N_OFDM + N_TRAINING_SEQ) + LTF_WITH_CP_LEN

MOD_SCHEME = '16-QAM' # Modulation scheme

# channel 
SNR = 30

# time sync 
#PEAK = 3e9
PEAK = 1.2e18


# distance Estimation
c = 3 * 10 ** 8
B = 1.024 * 10 ** 9
deltaR = c / (2 * B)
range = np.linspace(0,N,N) * c /(2 * B)

adc_fir2_coeff = np.array([
    5,0,-17,0,44,0,-96,0,187,0,-335,0,565,0,-906,
    0,1401,0,-2112,0,3145,0,-4723,0,7415,0,-13331,0,41526,
    65536,
    5,0,-17,0,44,0,-96,0,187,0,-335,0,565,0,-906,
    0,1401,0,-2112,0,3145,0,-4723,0,7415,0,-13331,0,41526
])

adc_fir1_coeff = np.array([
    -12,0,84,0,-337,1008,-2693,0,10142,
    16384,
    -12,0,84,0,-337,1008,-2693,0,10142
])

adc_fir0_coeff = np.array([
    -6,0,54,0,-254,0,1230,
    2048,
    -6,0,54,0,-254,0,1230
])




def m_sequence(length, feedback_taps):
    """
    Generate an M-sequence (Maximum Length Sequence) using a linear feedback shift register (LFSR).

    Args:
    - length (int): Length of the M-sequence to generate.
    - feedback_taps (list of int): Feedback taps for the LFSR.

    Returns:
    - sequence (list of int): Generated M-sequence.
    """

    # Initialize the LFSR with all zeros except the last bit set to 1
    lfsr = [0] * max(feedback_taps)
    lfsr[-1] = 1

    sequence = []

    for _ in range(length):
        # Generate the next bit of the sequence
        next_bit = sum(lfsr[tap - 1] for tap in feedback_taps) % 2

        # Map 0 to -1
        mapped_bit = 1 if next_bit == 1 else -1

        sequence.append(mapped_bit)
        lfsr = [next_bit] + lfsr[:-1]

    return sequence

