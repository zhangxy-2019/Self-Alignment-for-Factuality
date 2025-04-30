#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
ACTOR_MODEL_PATH=/data/huggingface_models/Llama-2-7b-hf
CRITIC_MODEL_PATH=/data/huggingface_models/Llama-2-7b-hf
# ACTOR_MODEL_PATH=/data/huggingface_models/llama/hf_models/7B
# CRITIC_MODEL_PATH=/data/huggingface_models/llama/hf_models/7B
ACTOR_ZERO_STAGE=3
CRITIC_ZERO_STAGE=3
# OUTPUT1=./dpo_llama2_7b_truthfulqa_mc_tf_format_max_seq_len_4096_lr_e6_epoch_max
OUTPUT1=./dpo_llama2_7b_truthfulqa_mc_tf_format_max_seq_len_4096_lr_e6_epoch_sat_new
# OUTPUT2=./dpo_llama2_7b_truthfulqa_mc_tf_format_max_seq_len_4096_lr_e6_epoch_usc_check
# OUTPUT3=./dpo_llama2_7b_truthfulqa_mc_tf_format_max_seq_len_4096_lr_e6_epoch_sc_check
# OUTPUT3=./dpo_llama2_7b_truthfulqa_mc_tf_format_max_seq_len_4096_lr_e6_epoch_naive_ptrue

# if [ "$OUTPUT" == "" ]; then
#     OUTPUT=./dpo_llama1_7b_truthfulqa_mc_tf_format_max_seq_len_2048_lr_e6_epoch
# fi
# if [ "$ACTOR_ZERO_STAGE" == "" ]; then
#     ACTOR_ZERO_STAGE=3
# fi
# if [ "$CRITIC_ZERO_STAGE" == "" ]; then
#     CRITIC_ZERO_STAGE=3
# fi
mkdir -p $OUTPUT1
# mkdir -p $OUTPUT2
# # mkdir -p $OUTPUT2
# mkdir -p $OUTPUT3
Num_Padding_at_Beginning=1 # this is model related

Actor_Lr=9.65e-6
# Actor_Lr=9.65e-7
# Actor_Lr2=9.65e-8
Critic_Lr=5e-6

deepspeed --master_port 12351 main_dpo_tf.py \
   --data_path /user/svetzhang/dschat_v1/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning_w_rewards/dpo_llama2_7b_bigbench_max_seq_len_4096_lr_e7_step2500_tf_pretrain/checkpoint-step7500/train/mc_train_tf_judge234.json \
   --data_eval_path /user/svetzhang/dschat_v1/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning_w_rewards/dpo_llama2_7b_bigbench_max_seq_len_4096_lr_e7_step2500_tf_pretrain/checkpoint-step7500/train/mc_train_tf_judge234.json \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 1 \
   --per_device_training_batch_size 1 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 1 \
   --max_prompt_seq_len 4096 \
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
   --save_steps 2500 \
   --beta 0.1 \
   --few_shot_prompts \
   --instruction_prompts \
   --output_dir $OUTPUT1 \
    &> $OUTPUT1/training.log

# # mkdir -p $OUTPUT3

# mkdir -p $OUTPUT2

# deepspeed --master_port 12351 main_dpo_tf.py \
#    --data_path /user/svetzhang/dschat_v1/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/llama2_7b_bigbench_70b_ik_file_pretrain_5_shot_tf_lre_7_sft_step/checkpoint-epoch0-step5000/truthfulqa/train/tf/usc_mc_train_tf_format230.json \
#    --data_eval_path /user/svetzhang/dschat_v1/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/llama2_7b_bigbench_70b_ik_file_pretrain_5_shot_tf_lre_7_sft_step/checkpoint-epoch0-step5000/truthfulqa/train/tf/usc_mc_train_tf_format230.json \
#    --actor_model_name_or_path $ACTOR_MODEL_PATH \
#    --critic_model_name_or_path $CRITIC_MODEL_PATH \
#    --num_padding_at_beginning 1 \
#    --per_device_generation_batch_size 1 \
#    --per_device_training_batch_size 1 \
#    --generation_batches 1 \
#    --ppo_epochs 1 \
#    --max_answer_seq_len 1 \
#    --max_prompt_seq_len 4096 \
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
#    --save_steps 2500 \
#    --beta 0.1 \
#    --few_shot_prompts \
#    --instruction_prompts \
#    --output_dir $OUTPUT2 \
#     &> $OUTPUT2/training.log

# deepspeed --master_port 12351 main_dpo_tf.py \
#    --data_path /user/svetzhang/dschat_v1/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/llama2_7b_bigbench_train_5_shot_pretrain_7b_tf_lre_8_sft_step/checkpoint-epoch2-step12500/truthfulqa/train/mc/sc_mc_train_tf_format.json \
#    --data_eval_path /user/svetzhang/dschat_v1/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/llama2_7b_bigbench_train_5_shot_pretrain_7b_tf_lre_8_sft_step/checkpoint-epoch2-step12500/truthfulqa/train/mc/sc_mc_train_tf_format.json \
#    --actor_model_name_or_path $ACTOR_MODEL_PATH \
#    --critic_model_name_or_path $CRITIC_MODEL_PATH \
#    --num_padding_at_beginning 1 \
#    --per_device_generation_batch_size 1 \
#    --per_device_training_batch_size 1 \
#    --generation_batches 1 \
#    --ppo_epochs 1 \
#    --max_answer_seq_len 1 \
#    --max_prompt_seq_len 4096 \
#    --actor_learning_rate ${Actor_Lr} \
#    --critic_learning_rate ${Critic_Lr} \
#    --actor_weight_decay 0.1 \
#    --critic_weight_decay 0.1 \
#    --num_train_epochs 10 \
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
#    --save_steps 2500 \
#    --beta 0.1 \
#    --few_shot_prompts \
#    --instruction_prompts \
#    --output_dir $OUTPUT3 \
#     &> $OUTPUT3/training.log

# deepspeed --master_port 12351 main_dpo_tf.py \
#    --data_path /user/svetzhang/dschat_v1/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/llama2_7b_bigbench_train_5_shot_pretrain_7b_tf_lre_8_sft_step/checkpoint-epoch3-step15000/truthfulqa/train/tf/mc_train_tf_judge234.json \
#    --data_eval_path /user/svetzhang/dschat_v1/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/llama2_7b_bigbench_train_5_shot_pretrain_7b_tf_lre_8_sft_step/checkpoint-epoch3-step15000/truthfulqa/train/tf/mc_train_tf_judge234.json \
#    --actor_model_name_or_path $ACTOR_MODEL_PATH \
#    --critic_model_name_or_path $CRITIC_MODEL_PATH \
#    --num_padding_at_beginning 1 \
#    --per_device_generation_batch_size 1 \
#    --per_device_training_batch_size 1 \
#    --generation_batches 1 \
#    --ppo_epochs 1 \
#    --max_answer_seq_len 1 \
#    --max_prompt_seq_len 4096 \
#    --actor_learning_rate ${Actor_Lr3} \
#    --critic_learning_rate ${Critic_Lr} \
#    --actor_weight_decay 0.1 \
#    --critic_weight_decay 0.1 \
#    --num_train_epochs 10 \
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
#    --save_steps 250 \
#    --beta 0.1 \
#    --few_shot_prompts \
#    --instruction_prompts \
#    --output_dir $OUTPUT1 \
#     &> $OUTPUT1/training.log
