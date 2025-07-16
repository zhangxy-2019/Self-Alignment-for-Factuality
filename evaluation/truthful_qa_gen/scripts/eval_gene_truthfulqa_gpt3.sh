#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

DATA_DIR1=./test_gene_decode/run_results_llama_7b_temp1.0_top_p0.9_top_k10_vllm_infer.json
CONFIG=gpt3.config.json
OUTPUT_DIR1=./dpo_llama2_7b_generation_epoch_5_truthfulqa_train_max_len_4096_lr_e_6_epoch_self_eval/checkpoint-epoch4/test_gene_decode/gpt3_eval09.json

python3 tfqa_gpt3_rating.py \
    --input-file ${DATA_DIR1} \
    --gpt3-config ${CONFIG} \
    --output-file ${OUTPUT_DIR1} 
