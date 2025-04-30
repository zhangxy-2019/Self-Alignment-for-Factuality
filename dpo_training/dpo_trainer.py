# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
import torch.nn.functional as F
import sys
import os
import time
import random
random.seed(42)
import copy
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import torch.nn as nn
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from transformers import DataCollator, PreTrainedModel
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils_dpo import pad_to_length
from utils.utils import print_rank_0


def print_all_ranks(tag, value, rank):
    world_size = torch.distributed.get_world_size()
    all_tensor = torch.zeros(world_size, dtype=torch.float32).cuda()
    all_tensor[rank] = value
    torch.distributed.all_reduce(all_tensor, op=torch.distributed.ReduceOp.SUM)
    print_rank_0(f'{tag} {all_tensor}', rank)


def get_model_norm(model):
    with torch.no_grad():
        total = 0.0
        for param in model.parameters():
            should_gather = hasattr(
                param,
                'ds_id') and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
            with deepspeed.zero.GatheredParameters(param,
                                                   enabled=should_gather):
                total += float(param.float().norm())

    return total


def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


class DeepSpeedPPOTrainer():

    def __init__(self, rlhf_engine, args):
        self.rlhf_engine = rlhf_engine
        self.actor_model = self.rlhf_engine.actor
        # self.critic_model = self.rlhf_engine.critic
        self.ref_model = self.rlhf_engine.ref
        # self.reward_model = self.rlhf_engine.reward
        self.tokenizer = self.rlhf_engine.tokenizer
        self.args = args
        self.max_answer_seq_len = args.max_answer_seq_len
        self.end_of_conversation_token_id = self.tokenizer(
            args.end_of_conversation_token)['input_ids'][-1]
        self.z3_enabled = args.actor_zero_stage == 3
        # print("z3 enabled:\n", self.z3_enabled)

        # Those value can be changed
        self.beta = 0.5
        self.kl_ctl = 0.1
        self.clip_reward_value = 5
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        self.lam = 0.95
        self.generate_time = 0.0
        self.normalize_reward = args.normalize_reward
        self.is_encoder_decoder = False
        self.label_pad_token_id = -100
        self.padding_value = 0
        

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_free: bool = False,
    ):
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
            reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if reference_free:
            ref_logratios = 0

        logits = pi_logratios - ref_logratios

        losses = -F.logsigmoid(self.beta * logits)
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        # if not self.is_encoder_decoder:
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != self.label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == self.label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def concatenated_inputs(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}

        if self.is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k or self.is_encoder_decoder else self.padding_value
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k or self.is_encoder_decoder else self.padding_value
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                )

        if self.is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1)
            concatenated_batch["concatenated_attention_mask"] = batch["prompt_attention_mask"].repeat(2, 1)

        return concatenated_batch

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(batch)
        # print(f"concate_batch: {concatenated_batch} ")
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            **model_kwargs,
        ).logits.to(torch.float32)

        all_logps = self._get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)
        with torch.no_grad():
            (
                reference_chosen_logps,
                reference_rejected_logps,
                _,
                _,
            ) = self.concatenated_forward(self.ref_model, batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().mean()

        return losses.mean(), metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        loss, metrics = self.get_batch_metrics(model, inputs, train_eval="train")

        # # force log the metrics
        # if self.accelerator.is_main_process:
        #     self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        policy_output = model.generate(
            batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            max_length=self.max_length,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        reference_output = self.ref_model.generate(
            batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            max_length=self.max_length,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        policy_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(reference_output, self.max_length, self.tokenizer.pad_token_id)
        reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded

    def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score,
                        action_mask):

        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        rewards = kl_divergence_estimate
        start = prompts.shape[1] - 1
        ends = start + action_mask[:, start:].sum(1) + 1
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value,
                                  self.clip_reward_value)
        batch_size = log_probs.shape[0]
        for j in range(batch_size):
            rewards[j, start:ends[j]][-1] += reward_clip[j]
            ## only sample one token ##
        return rewards

    def train_ppo(self, inputs):
        
        self.train()
        # print(f"inputs: {inputs}")
        actor_loss = self.compute_loss(model=self.actor_model, inputs=inputs)

        print(f"actor loss: {actor_loss}") # nan
        self.actor_model.backward(actor_loss)
        # "Exception: Current loss scale already at minimum - cannot decrease scale anymore." is a feature: when DeepSpeed detects that you are keeping getting overflow and already at lowest possible loss scale, it becomes meaningless to continue the training and users need to tune hyperparmeters by themselves.

        if not self.args.align_overflow:
            self.actor_model.step()

        if self.args.align_overflow:
            actor_overflow = self.actor_model.optimizer.check_overflow(
                external=True)
            # critic_overflow = self.critic_model.optimizer.check_overflow(
            #     external=True)

            rank = torch.distributed.get_rank()
            if actor_overflow and not critic_overflow:
                self.critic_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: actor overflow, skipping both actor and critic steps",
                    rank)
            elif not actor_overflow and critic_overflow:
                self.actor_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: critic overflow, skipping both actor and critic steps",
                    rank)
            elif actor_overflow and critic_overflow:
                print_rank_0(
                    "OVERFLOW: actor and critic overflow, skipping both actor and critic steps",
                    rank)
            self.actor_model.step()

        # self.critic_model.step()

        # return actor_loss, critic_loss
        return actor_loss, 0

    def get_overflow(self):
        actor_overflow = self.actor_model.optimizer.overflow
        # critic_overflow = self.critic_model.optimizer.overflow

        return actor_overflow

    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        ## policy gradient loss
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                             1.0 + self.cliprange)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        return pg_loss

    def critic_loss_fn(self, values, old_values, returns, mask):
        ## value loss
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        vf_loss = 0.5 * torch.sum(
            torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        return vf_loss

    def get_advantages_and_returns(self, values, rewards, start):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        for t in reversed(range(start, length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]
        return advantages.detach(), returns

    def get_advantages(self, rewards, start):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        ## set value = const
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        for t in reversed(range(start, length)):
            # nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        # returns = advantages + values[:, start:]
        return advantages.detach()

    def _validate_training_mode(self):
        assert self.actor_model.module.training
        # assert self.critic_model.module.training

    def _validate_evaluation_mode(self):
        assert not self.actor_model.module.training
        # assert not self.critic_model.module.training
        assert not self.ref_model.module.training
        # assert not self.reward_model.module.training

    def train(self):
        self.actor_model.train()
        # self.critic_model.train()

    def eval(self):
        self.actor_model.eval()
        # self.critic_model.eval()
        # self.reward_model.eval()
        self.ref_model.eval()

    def dump_model_norms(self, tag):
        actor_model_norm = get_model_norm(self.actor_model)
        ref_model_norm = get_model_norm(self.ref_model)
        # critic_model_norm = get_model_norm(self.critic_model)
        # reward_model_norm = get_model_norm(self.reward_model)
        print_all_ranks(f'{tag} global_actor_model_norm', actor_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_ref_model_norm', ref_model_norm,
                        self.args.local_rank)
        # print_all_ranks(f'{tag} global_critic_model_norm', critic_model_norm,
        #                 self.args.local_rank)
        # print_all_ranks(f'{tag} global_reward_model_norm', reward_model_norm,
        #                 self.args.local_rank)


class DeepSpeedPPOTrainerUnsupervised(DeepSpeedPPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_unsupervised(self, inputs, unsup_coef):
        # Train the unsupervised model here
        self._validate_training_mode()

        outputs = self.actor_model(**inputs, use_cache=False)
        loss = outputs.loss
        self.actor_model.backward(unsup_coef * loss)
        self.actor_model.step()

        return loss
