#!/usr/bin/env bash
#source activate remind_proj

lr=2e-3
CONFIG=TDIUC_streaming
export PYTHONPATH=/home/qzhb/dorren/CL4VQA/REMIND

DATA_ORDER=iid
expt=${CONFIG}_${DATA_ORDER}_${lr}
QTYPE=-1 # or type name
MODE=limited_buffer
BUFFERSIZE=10000


CUDA_VISIBLE_DEVICES=0 nohup python -u vqa_trainer.py \
--config_name ${CONFIG} \
--expt_name ${expt} \
--data_order ${DATA_ORDER} \
--exstream \
--rehearsal_mode ${MODE} \
--max_buffer_size ${BUFFERSIZE} \
--only_qtype ${QTYPE} \
--lr ${lr} &> ../logs/${expt}.log &

#DATA_ORDER=qtype # or qtype
#expt=${CONFIG}_${DATA_ORDER}_${lr}

#CUDA_VISIBLE_DEVICES=0 python -u vqa_experiments/vqa_trainer.py \
#--config_name ${CONFIG} \
#--expt_name ${expt} \
#--stream_with_rehearsal \
#--data_order ${DATA_ORDER} \
#--lr ${lr}
