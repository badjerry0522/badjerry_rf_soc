@echo off
python ./src/main.py ^
--EXE_MODE=GEN_RTL_INPUT ^
--N=64 ^
--QAM_MODE=4 ^
--N_OFDM=4 ^
--PILOT_MODE=ZEROS ^
--N_FRAME=1 ^
--PEAK=800000000000000 ^
--SNR=30