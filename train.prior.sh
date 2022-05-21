#!/usr/bin/env bash

set -e

MODEL_TYPE="variant-single"
MODEL_NAME_OR_PATH="bert-base-cased"
MAX_SEQ_LENGTH=256
MAX_NUM_TOKENS=256
TASK_HIDDEN_SIZE=256
PER_DEVICE_BATCH_SIZE=32

DATA_DIR="data/formatted"
CACHE_DIR="${HOME}/003_downloads/cache_transformers"

for TASKS in "genia_ner" "ace2005_re" "conll03_ner" "ade_re" "conll04_re"; do
for ROLE in "train"; do

PRETRAINED_MODEL="checkpoints/${TASKS}/${MODEL_TYPE}_bert-base-cased_256_cased_256_5.0e-05"
OUTPUT_DIR="data/prior/${TASKS}"

CUDA_VISIBLE_DEVICES=3 python do_prior.py \
--tasks ${TASKS} \
--role ${ROLE} \
--model_type ${MODEL_TYPE} \
--model_name_or_path ${MODEL_NAME_OR_PATH} \
--pretrained_model ${PRETRAINED_MODEL} \
--max_seq_length ${MAX_SEQ_LENGTH} \
--max_num_tokens ${MAX_NUM_TOKENS} \
--task_hidden_size ${TASK_HIDDEN_SIZE} \
--data_dir ${DATA_DIR} \
--output_dir ${OUTPUT_DIR} \
--cache_dir ${CACHE_DIR} \
--per_device_batch_size ${PER_DEVICE_BATCH_SIZE}

done
done
