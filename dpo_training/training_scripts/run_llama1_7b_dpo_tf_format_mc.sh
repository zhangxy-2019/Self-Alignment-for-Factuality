#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
ACTOR_MODEL_PATH=/data/huggingface_models/llama/hf_models/7B
CRITIC_MODEL_PATH=/data/huggingface_models/llama/hf_models/7B
ACTOR_ZERO_STAGE=3
CRITIC_ZERO_STAGE=3
OUTPUT1=./dpo_llama1_7b_truthfulqa_mc_tf_format_max_seq_len_2048_lr_e6_naivepture_epoch5
# OUTPUT2=./dpo_llama1_7b_truthfulqa_mc_tf_format_max_seq_len_2048_lr_e6_step_combine_bigbench
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

Num_Padding_at_Beginning=1 # this is model related

Actor_Lr=9.65e-6
Critic_Lr=5e-6

deepspeed --master_port 12351 main_dpo_tf.py \
   --data_path /user/svetzhang/cali_data0826/tf_data/truthfulqa/train/llama_7b_naive_pture/mc_train_tf_judge234.json \
   --data_eval_path /user/svetzhang/cali_data0826/tf_data/truthfulqa/train/llama_7b_naive_pture/mc_train_tf_judge234.json \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 1 \
   --per_device_training_batch_size 1 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 1 \
   --max_prompt_seq_len 2048 \
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


# deepspeed --master_port 12351 main_dpo_tf_step.py \
#    --data_path /user/svetzhang/dschat_v1/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/llama_7b_bigbench_pretrain_5_shot_tf_lre_7_sft_step/checkpoint-epoch2-step10000/truthfulqa/train/tf/mc_train_com_bigbench_tf_judge.json \
#    --data_eval_path /user/svetzhang/dschat_v1/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/llama_7b_bigbench_pretrain_5_shot_tf_lre_7_sft_step/checkpoint-epoch2-step10000/truthfulqa/train/tf/mc_train_com_bigbench_tf_judge.json \
#    --actor_model_name_or_path $ACTOR_MODEL_PATH \
#    --critic_model_name_or_path $CRITIC_MODEL_PATH \
#    --num_padding_at_beginning 1 \
#    --per_device_generation_batch_size 1 \
#    --per_device_training_batch_size 1 \
#    --generation_batches 1 \
#    --ppo_epochs 1 \
#    --max_answer_seq_len 1 \
#    --max_prompt_seq_len 2048 \
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
#    --save_steps 500 \
#    --beta 0.1 \
#    --few_shot_prompts \
#    --instruction_prompts \
#    --output_dir $OUTPUT2 \
#     &> $OUTPUT2/training.log
