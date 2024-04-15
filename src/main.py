import transmitter as tx
import data_func as df
import consts
import matplotlib.pyplot as plt
import numpy as np
import channel
import receiver as rx
from scipy import signal
import plt_funcs as pf
import configs as cfgs
import sys
'''
def test_LTF_symb():
    matched_filter = np.flip(consts.LTFsymb)
    filtered_signal = signal.convolve(in1=consts.LLTF, in2 = matched_filter, mode = "valid")
    pf.plot_dat("time sync res" , filtered_signal)
    # have peak at 21 and 43
    return
'''


import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--EXE_MODE", help="exe mode", type=str)
    parser.add_argument("--N", help="N of FFT", type=int)
    parser.add_argument("--QAM_MODE",help="QAM mod",type=int)
    parser.add_argument("--N_OFDM", help="N of OFDM symb per frame", type=int)
    parser.add_argument("--PILOT_MODE", help="PILOT MODE", type=str)
    parser.add_argument("--N_FRAMES", help="frame num", type=int)
    parser.add_argument("--PEAK", help="peak", type=int)
    parser.add_argument("--SNR", help="snr", type=int)
    parser.add_argument("--TX_INPUT_QAM_FILE", help="TX_INPUT_QAM_FILE", type=str)
    parser.add_argument("--TX_OUTPUT_QAM_NO_PILOT_FILE", help="TX_OUTPUT_QAM_NO_PILOT_FILE", type=str)
    parser.add_argument("--TX_OUTPUT_QAM_PILOT_FILE", help="TX_OUTPUT_QAM_PILOT_FILE", type=str)
    parser.add_argument("--TX_OUTPUT_FILE", help="TX_OUTPUT_FILE", type=str)
    parser.add_argument("--CHANNEL_OUTPUT_FILE", help="CHANNEL_OUTPUT_FILE", type=str)
    parser.add_argument("--RTL_OUTPUT_I_FILE", help="RTL_OUTPUT_I_FILE", type=str)
    parser.add_argument("--RTL_OUTPUT_Q_FILE", help="RTL_OUTPUT_Q_FILE", type=str)
    parser.add_argument("--RTL_OUTPUT_COMPLEX_FILE", help="RTL_OUTPUT_COMPLEX_FILE", type=str)

    args = parser.parse_args()
    print("args: '{}'".format(args))
    return args

args = get_args()
ofdm_cfgs = cfgs.ofdm_config(args)
if(ofdm_cfgs.init_ok != 1):
    print("bad init")
    sys.exit()

ofdm_cfgs.display()

tx0 = tx.tx(ofdm_cfgs)
tx0.tx_run()



'''
#tx.Transmitter(consts.ascii_tx_file,consts.tx_output_sig_file)

# write for rtl test




#df.complex_file_to_IQ_file(consts.tx_output_sig_file,consts.tx_output_sig_I_file,
#                           consts.tx_output_sig_Q_file, consts.tx_output_rtl_test_file)
df.IQ_file_to_complex_file(consts.test_res_I,consts.test_res_Q,consts.test_res_complex)
#df.IQ_file_to_complex_file(consts.tx_output_sig_I_file,consts.tx_output_sig_Q_file,consts.test_res_complex)
#df.bit16_to_256bit(consts.tx_output_rtl_test_file,consts.tx_output_rtl_test_256bit_file)
#df.txt_file_to_coe_file(consts.tx_output_rtl_test_256bit_file,consts.tx_rom_input_file)

channel.AWGN_channel(consts.tx_output_sig_file,consts.channel_output_sig_file,consts.SNR)

#rx.Receiver(consts.channel_output_sig_file,consts.rx_output_ascii_file)
rx.Receiver(consts.test_res_complex,consts.rx_output_ascii_file)
plt.show()
'''