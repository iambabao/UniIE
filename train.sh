#!/usr/bin/env bash

set -e

SEED=2333

MODEL_TYPE="variant-b"
MODEL_NAME_OR_PATH="bert-base-cased"
MAX_SEQ_LENGTH=256
MAX_NUM_TOKENS=256
TASK_HIDDEN_SIZE=256

TASKS="conll04_re,ade_re,ace2005_re,conll03_ner,genia_ner"
DATA_DIR="data/formatted"
OUTPUT_DIR="checkpoints/kd/${TASKS}"
CACHE_DIR="${HOME}/003_downloads/cache_transformers"
LOG_DIR="log/kd/${TASKS}"

NUM_TRAIN_EPOCH=200
EARLY_STOP=-1
PER_DEVICE_TRAIN_BATCH_SIZE=64
PER_DEVICE_EVAL_BATCH_SIZE=64
LEARNING_RATE=5e-5
LOGGING_STEPS=2000
SAVE_STEPS=2000

CUDA_VISIBLE_DEVICES=0 python do_train_kd.py \
--tasks ${TASKS} \
--model_type ${MODEL_TYPE} \
--model_name_or_path ${MODEL_NAME_OR_PATH} \
--max_seq_length ${MAX_SEQ_LENGTH} \
--max_num_tokens ${MAX_NUM_TOKENS} \
--task_hidden_size ${TASK_HIDDEN_SIZE} \
--data_dir ${DATA_DIR} \
--output_dir ${OUTPUT_DIR} \
--cache_dir ${CACHE_DIR} \
--log_dir ${LOG_DIR} \
--do_train \
--do_eval \
--evaluate_during_training \
--save_best \
--overwrite_output_dir \
--num_train_epochs ${NUM_TRAIN_EPOCH} \
--early_stop ${EARLY_STOP} \
--per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
--per_device_eval_batch_size ${PER_DEVICE_EVAL_BATCH_SIZE} \
--learning_rate ${LEARNING_RATE} \
--logging_steps ${LOGGING_STEPS} \
--save_steps ${SAVE_STEPS} \
--seed ${SEED}
