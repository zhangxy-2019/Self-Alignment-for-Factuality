#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
ACTOR_MODEL_PATH=/data/huggingface_models/Llama-2-7b-hf
CRITIC_MODEL_PATH=/data/huggingface_models/Llama-2-7b-hf
ACTOR_ZERO_STAGE=3
CRITIC_ZERO_STAGE=3
OUTPUT=""
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./dpo_llama2_7b_wiki_max_seq_len_4096_lr_e7_step2500_gene_tf_latest
fi
if [ "$ACTOR_ZERO_STAGE" == "" ]; then
    ACTOR_ZERO_STAGE=3
fi
if [ "$CRITIC_ZERO_STAGE" == "" ]; then
    CRITIC_ZERO_STAGE=3
fi
mkdir -p $OUTPUT

Num_Padding_at_Beginning=1 # this is model related
# /user/svetzhang/dschat_v1/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning_w_rewards

Actor_Lr=9.65e-7
Critic_Lr=5e-6
deepspeed --master_port 12351 main_dpo_ik_tf_step.py \
   --data_path /user/svetzhang/cali_data0826/mc_ik/wiki/ik_tf_short/gene_tf_rest/gene_tf_5_shot_1500710_gene_tf_rest_sft.json \
   --data_eval_path /user/svetzhang/cali_data0826/mc_ik/wiki/ik_tf_short/gene_tf_rest/gene_tf_5_shot_1500710_gene_tf_rest_sft.json \
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
   --num_train_epochs 10 \
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
   --output_dir $OUTPUT \
    &> $OUTPUT/training.log

# deepspeed --master_port 12351 main_dpo_tf_step.py \
#    --data_path /user/svetzhang/cali_data0826/mc_ik/wiki/data/train/ik_tf_5_shot_2705567_train_tf_dpo.json \
#    --data_eval_path /user/svetzhang/cali_data0826/mc_ik/wiki/data/train/ik_tf_5_shot_2705567_train_tf_dpo.json \
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
#    --output_dir $OUTPUT \
#     &> $OUTPUT/training.log
