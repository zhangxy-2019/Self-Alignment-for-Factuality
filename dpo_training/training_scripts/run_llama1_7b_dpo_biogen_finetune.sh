#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
ACTOR_MODEL_PATH=/data/huggingface_models/llama/hf_models/7B
CRITIC_MODEL_PATH=/data/huggingface_models/llama/hf_models/7B
# ACTOR_MODEL_PATH=/data/huggingface_models/Llama-2-7b-hf
# CRITIC_MODEL_PATH=/data/huggingface_models/Llama-2-7b-hf
ACTOR_ZERO_STAGE=3
CRITIC_ZERO_STAGE=3
# OUTPUT1=./dpo_llama2_7b_wiki_ood_epoch_10_max_seq_len_2048_commonsenseqa_epoch
# OUTPUT2=./dpo_llama2_7b_wiki_ood_epoch_10_max_seq_len_2048_openbookqa_epoch
# OUTPUT1=./dpo_llama1_7b_generation_biogen_lr_e_6_epoch_percent05_new
OUTPUT2=./dpo_llama1_7b_generation_biogen_lr_e_6_epoch_percent05_naive_pture_new
# OUTPUT3=./dpo_llama1_7b_generation_biogen_lr_e_6_epoch_percent025
# if [ "$OUTPUT" == "" ]; then
#     OUTPUT1=./dpo_llama2_7b_train_epoch_10_max_seq_len_2048_trained_truthfulqa_epoch
# fi
# if [ "$ACTOR_ZERO_STAGE" == "" ]; then
#     ACTOR_ZERO_STAGE=3
# fi
# if [ "$CRITIC_ZERO_STAGE" == "" ]; then
#     CRITIC_ZERO_STAGE=3
# fi
# mkdir -p $OUTPUT1
mkdir -p $OUTPUT2
# mkdir -p $OUTPUT3

Num_Padding_at_Beginning=1 # this is model related

Actor_Lr=9.65e-6
Critic_Lr=5e-6

# deepspeed --master_port 12350 main_biogen_dpo_generation_epoch.py \
#    --data_path /user/svetzhang/dschat_v1/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning_w_rewards/dpo_llama1_7b_wiki_max_seq_len_4096_lr_e7_step2500_tf_pretrain/checkpoint-step12500/tf_eval_biogen/records_biogen_llama1_7b_dpo_generation_train_len50.json \
#    --data_eval_path /user/svetzhang/dschat_v1/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning_w_rewards/dpo_llama1_7b_wiki_max_seq_len_4096_lr_e7_step2500_tf_pretrain/checkpoint-step12500/tf_eval_biogen/records_biogen_llama1_7b_dpo_generation_train_len50.json \
#    --actor_model_name_or_path $ACTOR_MODEL_PATH \
#    --critic_model_name_or_path $CRITIC_MODEL_PATH \
#    --num_padding_at_beginning 1 \
#    --per_device_generation_batch_size 1 \
#    --per_device_training_batch_size 1 \
#    --generation_batches 1 \
#    --ppo_epochs 1 \
#    --max_answer_seq_len 600 \
#    --max_prompt_seq_len 1600 \
#    --actor_learning_rate ${Actor_Lr} \
#    --critic_learning_rate ${Critic_Lr} \
#    --actor_weight_decay 0.1 \
#    --critic_weight_decay 0.1 \
#    --num_train_epochs 5 \
#    --lr_scheduler_type cosine \
#    --gradient_accumulation_steps 8 \
#    --actor_gradient_checkpointing \
#    --critic_gradient_checkpointing \
#    --offload_reference_model \
#    --disable_actor_dropout \
#    --num_warmup_steps 100 \
#    --deepspeed --seed 1234 \
#    --normalize_reward \
#    --actor_zero_stage $ACTOR_ZERO_STAGE \
#    --critic_zero_stage $CRITIC_ZERO_STAGE \
#    --actor_lora_dim 64 \
#    --critic_lora_dim 64 \
#    --critic_lora_module_name "layers." \
#    --actor_lora_module_name "layers." \
#    --save_steps 2500 \
#    --beta 0.1 \
#    --percentage 0.5 \
#    --num_negative_example 3 \
#    --output_dir $OUTPUT1 \
#     &> $OUTPUT1/training.log


deepspeed --master_port 12350 main_biogen_dpo_generation_epoch.py \
   --data_path /user/svetzhang/cali_data0826/tf_data/biogen/llama_7b/naive_ptrue/biogen_tf/records_biogen_llama1_7b_dpo_generation_train_len50.json \
   --data_eval_path /user/svetzhang/cali_data0826/tf_data/biogen/llama_7b/naive_ptrue/biogen_tf/records_biogen_llama1_7b_dpo_generation_train_len50.json \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 1 \
   --per_device_training_batch_size 1 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 600 \
   --max_prompt_seq_len 1600 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 5 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 8 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload_reference_model \
   --disable_actor_dropout \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   --normalize_reward \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --critic_zero_stage $CRITIC_ZERO_STAGE \
   --actor_lora_dim 64 \
   --critic_lora_dim 64 \
   --critic_lora_module_name "layers." \
   --actor_lora_module_name "layers." \
   --save_steps 2500 \
   --beta 0.1 \
   --percentage 0.5 \
   --num_negative_example 3 \
   --output_dir $OUTPUT2 \
    &> $OUTPUT2/training.log

# deepspeed --master_port 12350 main_biogen_dpo_generation_epoch.py \
#    --data_path /user/svetzhang/dschat_v1/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning_w_rewards/dpo_llama1_7b_wiki_max_seq_len_2048_lr_e7_step5000/checkpoint-step12500/biogen_tf/records_biogen_llama1_7b_dpo_generation_train_len50.json \
#    --data_eval_path /user/svetzhang/dschat_v1/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning_w_rewards/dpo_llama1_7b_wiki_max_seq_len_2048_lr_e7_step5000/checkpoint-step12500/biogen_tf/records_biogen_llama1_7b_dpo_generation_train_len50.json \
#    --actor_model_name_or_path $ACTOR_MODEL_PATH \
#    --critic_model_name_or_path $CRITIC_MODEL_PATH \
#    --num_padding_at_beginning 1 \
#    --per_device_generation_batch_size 1 \
#    --per_device_training_batch_size 1 \
#    --generation_batches 1 \
#    --ppo_epochs 1 \
#    --max_answer_seq_len 600 \
#    --max_prompt_seq_len 1600 \
#    --actor_learning_rate ${Actor_Lr} \
#    --critic_learning_rate ${Critic_Lr} \
#    --actor_weight_decay 0.1 \
#    --critic_weight_decay 0.1 \
#    --num_train_epochs 5 \
#    --lr_scheduler_type cosine \
#    --gradient_accumulation_steps 8 \
#    --actor_gradient_checkpointing \
#    --critic_gradient_checkpointing \
#    --offload_reference_model \
#    --disable_actor_dropout \
#    --num_warmup_steps 100 \
#    --deepspeed --seed 1234 \
#    --normalize_reward \
#    --actor_zero_stage $ACTOR_ZERO_STAGE \
#    --critic_zero_stage $CRITIC_ZERO_STAGE \
#    --actor_lora_dim 64 \
#    --critic_lora_dim 64 \
#    --critic_lora_module_name "layers." \
#    --actor_lora_module_name "layers." \
#    --save_steps 2500 \
#    --beta 0.1 \
#    --percentage 0.15 \
#    --num_negative_example 3 \
#    --output_dir $OUTPUT2 \
#     &> $OUTPUT2/training.log

# deepspeed --master_port 12350 main_biogen_dpo_generation_epoch.py \
#    --data_path /user/svetzhang/dschat_v1/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning_w_rewards/dpo_llama1_7b_wiki_max_seq_len_2048_lr_e7_step5000/checkpoint-step12500/biogen_tf/records_biogen_llama1_7b_dpo_generation_train_len50.json \
#    --data_eval_path /user/svetzhang/dschat_v1/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning_w_rewards/dpo_llama1_7b_wiki_max_seq_len_2048_lr_e7_step5000/checkpoint-step12500/biogen_tf/records_biogen_llama1_7b_dpo_generation_train_len50.json \
#    --actor_model_name_or_path $ACTOR_MODEL_PATH \
#    --critic_model_name_or_path $CRITIC_MODEL_PATH \
#    --num_padding_at_beginning 1 \
#    --per_device_generation_batch_size 1 \
#    --per_device_training_batch_size 1 \
#    --generation_batches 1 \
#    --ppo_epochs 1 \
#    --max_answer_seq_len 600 \
#    --max_prompt_seq_len 1600 \
#    --actor_learning_rate ${Actor_Lr} \
#    --critic_learning_rate ${Critic_Lr} \
#    --actor_weight_decay 0.1 \
#    --critic_weight_decay 0.1 \
#    --num_train_epochs 5 \
#    --lr_scheduler_type cosine \
#    --gradient_accumulation_steps 8 \
#    --actor_gradient_checkpointing \
#    --critic_gradient_checkpointing \
#    --offload_reference_model \
#    --disable_actor_dropout \
#    --num_warmup_steps 100 \
#    --deepspeed --seed 1234 \
#    --normalize_reward \
#    --actor_zero_stage $ACTOR_ZERO_STAGE \
#    --critic_zero_stage $CRITIC_ZERO_STAGE \
#    --actor_lora_dim 64 \
#    --critic_lora_dim 64 \
#    --critic_lora_module_name "layers." \
#    --actor_lora_module_name "layers." \
#    --save_steps 2500 \
#    --beta 0.1 \
#    --percentage 0.25 \
#    --num_negative_example 3 \
#    --output_dir $OUTPUT3 \
#     &> $OUTPUT3/training.log
