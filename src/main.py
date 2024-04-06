import transmitter as tx
import data_func as df
import consts
import matplotlib.pyplot as plt
import numpy as np
import channel
import receiver as rx
from scipy import signal
import plt_funcs as pf
'''
def test_LTF_symb():
    matched_filter = np.flip(consts.LTFsymb)
    filtered_signal = signal.convolve(in1=consts.LLTF, in2 = matched_filter, mode = "valid")
    pf.plot_dat("time sync res" , filtered_signal)
    # have peak at 21 and 43
    return
'''

tx.Transmitter(consts.ascii_tx_file,consts.tx_output_sig_file)

# write for rtl test

df.complex_file_to_IQ_file(consts.tx_output_sig_file,consts.tx_output_sig_I_file,
                           consts.tx_output_sig_Q_file, consts.tx_output_rtl_test_file)
#df.IQ_file_to_complex_file(consts.test_res_I,consts.test_res_Q,consts.test_res_complex)
#df.IQ_file_to_complex_file(consts.tx_output_sig_I_file,consts.tx_output_sig_Q_file,consts.test_res_complex)
#df.bit16_to_256bit(consts.tx_output_rtl_test_file,consts.tx_output_rtl_test_256bit_file)
#df.txt_file_to_coe_file(consts.tx_output_rtl_test_256bit_file,consts.tx_rom_input_file)

channel.AWGN_channel(consts.tx_output_sig_file,consts.channel_output_sig_file,consts.SNR)

rx.Receiver(consts.channel_output_sig_file,consts.rx_output_ascii_file)
#rx.Receiver(consts.test_res_complex,consts.rx_output_ascii_file)
plt.show()