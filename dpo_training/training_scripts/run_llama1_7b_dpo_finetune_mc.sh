#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
ACTOR_MODEL_PATH=/data/huggingface_models/llama/hf_models/7B
CRITIC_MODEL_PATH=/data/huggingface_models/llama/hf_models/7B
ACTOR_ZERO_STAGE=3
CRITIC_ZERO_STAGE=3
# OUTPUT1=./dpo_llama1_7b_gene_truthfulqa_train_epoch_5_max_seq_len_2048_single_max
# OUTPUT2=./dpo_llama1_7b_gene_truthfulqa_train_epoch_5_max_seq_len_2048_max
# OUTPUT3=./dpo_llama1_7b_gene_truthfulqa_train_epoch_5_max_seq_len_2048_self_consist
# OUTPUT4=./dpo_llama1_7b_mc_truthfulqa_train_epoch_10_max_seq_len_2048_epoch
OUTPUT5=./dpo_llama1_7b_mc_truthfulqa_train_epoch_10_max_seq_len_2048_combine_bigbench_step
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
# mkdir -p $OUTPUT2
# mkdir -p $OUTPUT3
# mkdir -p $OUTPUT4
mkdir -p $OUTPUT5

Num_Padding_at_Beginning=1 # this is model related

Actor_Lr=9.65e-6
Critic_Lr=5e-6

# deepspeed --master_port 12350 main_dpo_mc_epoch.py \
#    --data_path /user/svetzhang/dschat_v1/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/llama_7b_bigbench_pretrain_5_shot_tf_lre_7_sft_step/checkpoint-epoch2-step10000/truthfulqa/train/tf/len_41_dpo_rlhf_labeled_mc.json \
#    --data_eval_path /user/svetzhang/dschat_v1/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/llama_7b_bigbench_pretrain_5_shot_tf_lre_7_sft_step/checkpoint-epoch2-step10000/truthfulqa/train/tf/len_41_dpo_rlhf_labeled_mc.json \
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
#    --actor_lora_dim 64 \
#    --critic_lora_dim 64 \
#    --critic_lora_module_name "layers." \
#    --actor_lora_module_name "layers." \
#    --save_steps 2500 \
#    --beta 0.1 \
#    --num_negative_example 3 \
#    --output_dir $OUTPUT4 \
#     &> $OUTPUT4/training.log


deepspeed --master_port 12350 main_dpo_mc_step.py \
   --data_path /user/svetzhang/dschat_v1/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/llama_7b_bigbench_pretrain_5_shot_tf_lre_7_sft_step/checkpoint-epoch2-step10000/truthfulqa/train/tf/dpo_rlhf_bigbench_test_truthfulqa_train_labeled_mc.json \
   --data_eval_path /user/svetzhang/dschat_v1/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/llama_7b_bigbench_pretrain_5_shot_tf_lre_7_sft_step/checkpoint-epoch2-step10000/truthfulqa/train/tf/dpo_rlhf_bigbench_test_truthfulqa_train_labeled_mc.json \
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
   --actor_lora_dim 64 \
   --critic_lora_dim 64 \
   --critic_lora_module_name "layers." \
   --actor_lora_module_name "layers." \
   --save_steps 250 \
   --beta 0.1 \
   --num_negative_example 3 \
   --output_dir $OUTPUT5 \
    &> $OUTPUT5/training.log



# deepspeed --master_port 12350 main_dpo_generation_epoch.py \
#    --data_path /user/svetzhang/dschat_v1/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/llama_7b_bigbench_pretrain_5_shot_tf_lre_7_sft_step/checkpoint-epoch2-step10000/truthfulqa/train/decode_tf/gene_dpo_train_41_single_max_data.json \
#    --data_eval_path /user/svetzhang/dschat_v1/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/llama_7b_bigbench_pretrain_5_shot_tf_lre_7_sft_step/checkpoint-epoch2-step10000/truthfulqa/train/decode_tf/gene_dpo_train_41_single_max_data.json \
#    --actor_model_name_or_path $ACTOR_MODEL_PATH \
#    --critic_model_name_or_path $CRITIC_MODEL_PATH \
#    --num_padding_at_beginning 1 \
#    --per_device_generation_batch_size 1 \
#    --per_device_training_batch_size 1 \
#    --generation_batches 1 \
#    --ppo_epochs 1 \
#    --max_answer_seq_len 50 \
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
#    --actor_lora_dim 64 \
#    --critic_lora_dim 64 \
#    --critic_lora_module_name "layers." \
#    --actor_lora_module_name "layers." \
#    --save_steps 2500 \
#    --beta 0.1 \
#    --num_negative_example 3 \
#    --output_dir $OUTPUT1 \
#     &> $OUTPUT1/training.log



# deepspeed --master_port 12350 main_dpo_generation_epoch.py \
#    --data_path /user/svetzhang/dschat_v1/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/llama_7b_bigbench_pretrain_5_shot_tf_lre_7_sft_step/checkpoint-epoch2-step10000/truthfulqa/train/decode_tf/gene_dpo_train_41_sc_data.json \
#    --data_eval_path /user/svetzhang/dschat_v1/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/llama_7b_bigbench_pretrain_5_shot_tf_lre_7_sft_step/checkpoint-epoch2-step10000/truthfulqa/train/decode_tf/gene_dpo_train_41_sc_data.json \
#    --actor_model_name_or_path $ACTOR_MODEL_PATH \
#    --critic_model_name_or_path $CRITIC_MODEL_PATH \
#    --num_padding_at_beginning 1 \
#    --per_device_generation_batch_size 1 \
#    --per_device_training_batch_size 1 \
#    --generation_batches 1 \
#    --ppo_epochs 1 \
#    --max_answer_seq_len 50 \
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
#    --actor_lora_dim 64 \
#    --critic_lora_dim 64 \
#    --critic_lora_module_name "layers." \
#    --actor_lora_module_name "layers." \
#    --save_steps 2500 \
#    --beta 0.1 \
#    --num_negative_example 3 \
#    --output_dir $OUTPUT3 \
#     &> $OUTPUT3/training.log



# deepspeed --master_port 12350 main_dpo_generation_epoch.py \
#    --data_path /user/svetzhang/dschat_v1/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/llama_7b_bigbench_pretrain_5_shot_tf_lre_7_sft_step/checkpoint-epoch2-step10000/truthfulqa/train/decode_tf/gene_dpo_train_41max_data.json \
#    --data_eval_path /user/svetzhang/dschat_v1/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/llama_7b_bigbench_pretrain_5_shot_tf_lre_7_sft_step/checkpoint-epoch2-step10000/truthfulqa/train/decode_tf/gene_dpo_train_41max_data.json \
#    --actor_model_name_or_path $ACTOR_MODEL_PATH \
#    --critic_model_name_or_path $CRITIC_MODEL_PATH \
#    --num_padding_at_beginning 1 \
#    --per_device_generation_batch_size 1 \
#    --per_device_training_batch_size 1 \
#    --generation_batches 1 \
#    --ppo_epochs 1 \
#    --max_answer_seq_len 50 \
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
#    --actor_lora_dim 64 \
#    --critic_lora_dim 64 \
#    --critic_lora_module_name "layers." \
#    --actor_lora_module_name "layers." \
#    --save_steps 2500 \
#    --beta 0.1 \
#    --num_negative_example 3 \
#    --output_dir $OUTPUT2 \
#     &> $OUTPUT2/training.log

# deepspeed --master_port 12350 main_dpo_v3.py \
#    --data_path /user/svetzhang/dschat_v1/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning_w_rewards/dpo_llama2_7b_wiki_max_seq_len_4096_lr_e7_step5000/checkpoint-step7500/commonsenseqa/len_1221_dpo_rlhf_labeled_mc.json \
#    --data_eval_path /user/svetzhang/dschat_v1/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning_w_rewards/dpo_llama2_7b_wiki_max_seq_len_4096_lr_e7_step5000/checkpoint-step7500/commonsenseqa/len_1221_dpo_rlhf_labeled_mc.json \
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
#    --save_steps 2500 \
#    --beta 0.1 \
#    --num_negative_example 3 \
#    --output_dir $OUTPUT1 \
#     &> $OUTPUT1/training.log


# deepspeed --master_port 12350 main_dpo_v3.py \
#    --data_path /user/svetzhang/dschat_v1/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning_w_rewards/dpo_llama2_7b_wiki_max_seq_len_4096_lr_e7_step5000/checkpoint-step7500/openbookqa/len_500_dpo_rlhf_labeled_mc.json \
#    --data_eval_path /user/svetzhang/dschat_v1/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning_w_rewards/dpo_llama2_7b_wiki_max_seq_len_4096_lr_e7_step5000/checkpoint-step7500/openbookqa/len_500_dpo_rlhf_labeled_mc.json \
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
#    --actor_lora_dim 64 \
#    --critic_lora_dim 64 \
#    --critic_lora_module_name "layers." \
#    --actor_lora_module_name "layers." \
#    --save_steps 2500 \
#    --beta 0.1 \
#    --num_negative_example 3 \
#    --output_dir $OUTPUT2 \
#     &> $OUTPUT2/training.log
