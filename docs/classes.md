# configs.py

## cmd_args

用于处理、存储命令行输入参数。

| 参数名                           | 是否必须 | 传递方式                            | 功能                                         | 使用示例                                                     |
| -------------------------------- | -------- | ----------------------------------- | -------------------------------------------- | ------------------------------------------------------------ |
| EXE_MODE                         | 是       | --EXE_MODE=                         | 程序执行                                     | --EXE_MODE=GEN_RTL_INPUT<br/>--EXE_MODE=SIMU<br/>--EXE_MODE=DEMODE_RTL_OUTPUT |
| N                                | 是       | --N=                                | FFT点数                                      | --N=64<br/>--N=2048<br/>--N=4096                             |
| QAM_MODE                         | 否       | --QAM_MODE=                         | QAM调制阶数                                  | --QAM_MODE=4  (default)<br/>--QAM_MODE=16                    |
| N_OFDM                           | 是       | --N_OFDM=                           | 一帧中OFDM符号数                             | --N_OFDM=4                                                   |
| PILOT_MODE                       | 否       | --PILOT_MODE=                       | 导频模式                                     | --PILOT_MODE=NO_PILOT <br/>--PILOT_MODE=ZEROS （default)<br/>--PILOT_MODE=PILOT_MODE_1 |
| N_FRAMES                         | 否       | --N_FRAMES=                         | 帧数量                                       | --N_FRAMES=1(default)                                        |
| PEAK                             | 否       | --PEAK=                             | 时间同步帧强度                               | --PEAK=800,000,000,000,000 (deault)                          |
| SNR                              | 否       | --SNR=                              | 仿真用信噪比                                 | --SNR=30(default)                                            |
| TX_INPUT_<br/>QAM_FILE           | 否       | --TX_INPUT<br/>_QAM_FILE=           | 发射机输入QAM符号<br/>**带有导频！！！**     | --TX_INPUT_QAM_FILE=<br/>../dats/simu/tx_input_qam.txt (default) |
| TX_OUTPUT_QAM<br/>_NO_PILOT_FILE | 否       | --TX_OUTPUT_QAM<br/>_NO_PILOT_FILE= | 发射机随机生成QAM符号<br/>**不带导频！！！** | --TX_OUTPUT_QAM<br/>_NO_PILOT_FILE=<br/>../dats/rtl_input/tx_output<br/>_qam_no_pilot.txt (default) |
| TX_OUTPUT_QAM<br/>_PILOT_FILE    | 否       | --TX_OUTPUT_QAM<br/>_PILOT_FILE=    | 发射机随机生成QAM符号<br/>**带有导频！！！** | --TX_OUTPUT_QAM<br/>_PILOT_FILE=<br/>../dats/rtl_input/tx_output<br/>_qam_pilot.txt (default) |
| TX_OUTPUT_FILE                   | 否       | --TX_OUTPUT_FILE=                   | 送入RTL的OFDM信号                            | --TX_OUTPUT_FILE=<br/>../dats/rtl_test_input.txt (default)   |
| CHANNEL_OUTPUT_FILE              | 否       | --CHANNEL_OUTPUT_FILE=              | 仿真信道输出结果                             | --CHANNEL_OUTPUT_FILE=<br/>../dats/simu/channel_output.txt (default) |
| RTL_OUTPUT_I_FILE                | 否       | --RTL_OUTPUT_I_FILE=                | RTL接收I路信号                               | --RTL_OUTPUT_I_FILE=<br/>../dats/rtl_output/rtl_output_I.txt (default) |
| RTL_OUTPUT_Q_FILE                | 否       | --RTL_OUTPUT_Q_FILE=                | RTL接收Q路信号                               | --RTL_OUTPUT_Q_FILE=<br/>../dats/rtl_output/rtl_output_Q.txt (default) |
| RTL_OUTPUT_COMPLEX<br/>_FILE     | 否       | --RTL_OUTPUT_COMPLEX<br/>_FILE=     | RTL接收IQ信号                                | --RTL_OUTPUT_COMPLEX<br/>_FILE=../dats/rtl_output/<br/>rtl_output_complex.txt (default) |



# Transmitter.py

## TX

### 成员：

```python
ofdm_cfg:configs.ofdm_config = None
#gen_rand_qam_seq
RAND_SEED = 123
rand_qam_seq_len = None
rand_qam_seq_file = None
rand_qam_seq = None

#TX_qam_seq
guard_seq = None
pilot_seq = None
tx_qam_seq = None
tx_qam_seq_len = None
tx_qam_seq_file = None
    
#ofdm_seq
ofdm_symb_seq = None

#tx_frame_seq
tx_frame_seq = None
tx_frame_seq_len = None
tx_single_frame_seq = None
tx_single_frame_seq_file = None
tx_frame_seq_file = None
```



### 方法：

```python
def __init__(self,args_cfg):
    '''
    初始化，输入参数类型为cmd_args类。会生成pilot_seq 和 guard_seq
    '''
def gen_rand_qam_seq(self):
    '''
    生成随机的qam符号序列，其长度为 N_DATA_QAM_SYMB = N_OFDM * N_DATA
    '''
def gen_tx_qam_seq(self):
    '''
    将随机qam符号序列与导频和保护间隔组合
    '''
def ofdm_mod(self):
    '''
    ofdm调制，输出长度为 N*N_OFDM
    '''
def frame_assemble(self):
    '''
    帧组装与重复
    '''
def plt_single_frame(self):
    '''
    画图，一帧
	'''
def plt_frames(self):
    '''
    画图，多帧
    '''
def tx_run(self):
    '''
    tx类的调用接口，当EXE_MODE为DEMODE时，直接返回；当EXE_MODE为其他时，执行生成和调制
    '''
```



# consts