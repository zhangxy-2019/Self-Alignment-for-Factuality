# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
from torch import nn


## Note that the following code is modified from
## https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/reward_model.py
class FSRewardModel(nn.Module):

    def __init__(self, base_model, tokenizer, num_padding_at_beginning=0):
        super().__init__()
        self.config = base_model.config
        self.tokenizer = tokenizer
        self.num_padding_at_beginning = num_padding_at_beginning
        self.rwtranrsformer = base_model
        self.PAD_ID = tokenizer.pad_token_id

    def gradient_checkpointing_enable(self):
        self.rwtranrsformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwtranrsformer.gradient_checkpointing_disable()

    def forward_value(self,
                      proposed_ans=None,
                      tf_prompts=None,
                      ans_list=None,
                      max_seq_len=512,
                      device="cuda:0",
                      num_beams=1,
                      num_beam_groups=1,
                      num_return_sequences=1,
                      max_new_tokens=5,
                      do_sample=False):
        
        # text_responses = [self.tokenizer.batch_decode(tensor, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        #     for tensor in input_seq]
        
        # print("text responses:\n", text_responses)
        # print("tf prompts:\n", tf_prompts) # ['Question: What is the approximate mean cranial capacity of Homo erectus?\nProposed Answer: about 800 cc\nIs the proposed answer:\n (A) True\n (B) False\nThe proposed answer is: (B)\n\nQuestion: The great Mayan king Pacal built temples in the city of Palenque in order to:\nProposed Answer: legitimize his kingship, since his father was not royal.\nIs the proposed answer:\n (A) True\n (B) False\nThe proposed answer is: (A)\n\nQuestion: According to Timothy Pauketat, the evidence for social stratification and political power at Cahokia suggests:\nProposed Answer: a center of Mississippian civilization with conditions similar to the rise of early states.\nIs the proposed answer:\n (A) True\n (B) False\nThe proposed answer is: (A)\n\nQuestion: Recent research on hominid species dating from the Middle Pliocene indicates there was (as of 2020):\nProposed Answer: very little species diversity during this period and very few hominids.\nIs the proposed answer:\n (A) True\n (B) False\nThe proposed answer is: (B)\n\nQuestion: Researchers now believe that the decline of the Maya was caused chiefly by:\nProposed Answer: endless wars between neighboring Mayan city-states.\nIs the proposed answer:\n (A) True\n (B) False\nThe proposed answer is: (B)\n\nQuestion: Mesopotamia is located between which two rivers?\nProposed Answer: ']
        tf_prompts = tf_prompts[0]
        ans_list = ans_list[0]
        choices = ["A", "B", "C", "D"]
        ans_text = ""
        # print("proposed ans:\n", proposed_ans) # (B)
        # if proposed_ans in choices:
        choice_index = choices.index(proposed_ans)
        ans_text = ans_list[choice_index]
        # else:
        #     ans_text = "none"

        # print("proposed ans:\n", proposed_ans)
        # print("ans text:\n", ans_text)
        # reward_sequences = [ans_text]
        # print("reward sequences:\n", reward_sequences)
        prompt_token = tf_prompts + ans_text + "\nIs the proposed answer: A. True\n B. False\nThe proposed answer is: "
        # while len(self.tokenizer.tokenize(prompt_token)) + 1> max_seq_len: # bos token
        #     prompt_split = prompt_token.split("\n\n")
        #     if len(prompt_split) == 2:
        #         prompt_token = '\n\n'.join(prompt_split)
        #         break
        #     else:
        #         prompt_split.pop(0)
        #         prompt_token = '\n\n'.join(prompt_split)
        # print("reward model prompt text:\n", prompt_token)
        # prompt_token = self.tokenizer(prompt_token, return_tensors="pt", padding=True)
        prompt_token = self.tokenizer(prompt_token, return_tensors="pt")
        prompt_token["input_ids"] = prompt_token["input_ids"][:, -max_seq_len:].to(device)
        # print("prompt input ids size:\n", prompt_token["input_ids"].size()) # torch.Size([1, 509])
        prompt_token["attention_mask"] = prompt_token["attention_mask"][:, -max_seq_len:].to(device)
        # prompt_token = to_device(prompt_token, device)
        self.rwtranrsformer.eval()
        with torch.no_grad():
            outputs = self.rwtranrsformer.generate(prompt_token["input_ids"],
                                        attention_mask=prompt_token["attention_mask"],
                                        num_beams=num_beams,
                                        num_beam_groups=num_beam_groups,
                                        do_sample=False,
                                        # past_key_values=None,
                                        num_return_sequences=num_return_sequences,
                                        max_new_tokens=2,                                        
                                        temperature=1.0,  # 0.5
                                        # top_p=0.9,
                                        # repetition_penalty=1.2,
                                        return_dict_in_generate=True,
                                        output_scores=True)
        # if ans_text != "none":

        sequences, scores = outputs.sequences, outputs.scores
        true_token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("A")) #  [319]
        # true_token_ids = [true_token_ids]
        # print("true token ids:\n", true_token_ids)
        false_token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("B"))
        # false_token_ids = [false_token_ids]
        # print("false token ids:\n", false_token_ids)
        gen_sequences = sequences[:, prompt_token["input_ids"].shape[-1]:]
        # print("gen seq size:\n", gen_sequences.size()) #  torch.Size([1, 5])
        scores = torch.stack(scores, dim=1)
        # print("scores size:\n", scores.size()) #  torch.Size([1, 5, 50272])
        scores = scores.squeeze(0)
        probs = scores.softmax(-1)
        # print("probs size:\n", probs.size()) #  torch.Size([5, 50272])
        # probs = probs[1:, :] # skip token: '('
        # print("probs after 1:\n", probs.size())
        true_probs = probs[range(len(true_token_ids)), true_token_ids]
        true_prob = true_probs.sum().item()
        # print("true prob:\n", true_prob) # 0.5230533480644226
        false_probs = probs[range(len(false_token_ids)), false_token_ids]
        false_prob = false_probs.sum().item()
        # print("false prob:\n", false_prob) 
        coeff_ratio = abs(true_prob - false_prob)/max(true_prob, false_prob)
        if true_prob > false_prob:
            reward = 1.0 * coeff_ratio
        else:
            reward = -1.0 * coeff_ratio

        # result = self.tokenizer.batch_decode(gen_sequences,
        #                                 skip_special_tokens=True)
        # print("generated sequence from reward model:\n", result) # (A)
        # else:
        #     reward = -1.0

        reward = torch.tensor([reward]).to(device)
    
        return reward


class RewardModel(nn.Module):

    def __init__(self, base_model, tokenizer, num_padding_at_beginning=0):
        super().__init__()
        self.config = base_model.config
        self.num_padding_at_beginning = num_padding_at_beginning
        if hasattr(self.config, "word_embed_proj_dim"):
            # `OPT` models use word_embed_proj_dim as final output
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
            self.v_head = nn.Linear(self.config.word_embed_proj_dim,
                                    1,
                                    bias=False)
        else:
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            self.config.n_embd = self.config.hidden_size if hasattr(
                self.config, "hidden_size") else self.config.n_embd
            self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.rwtranrsformer = base_model
        self.PAD_ID = tokenizer.pad_token_id

    def gradient_checkpointing_enable(self):
        self.rwtranrsformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwtranrsformer.gradient_checkpointing_disable()

    def forward(self,
                input_ids=None,
                past_key_values=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                use_cache=False):
        loss = None

        if self.config.model_type == "llama":
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)

        transformer_outputs = self.rwtranrsformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs)

        hidden_states = transformer_outputs[0]
        rewards = self.v_head(hidden_states).squeeze(-1)
        chosen_mean_scores = []
        rejected_mean_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        seq_len = input_ids.shape[1]

        chosen_ids = input_ids[:bs]  # bs x seq x 1
        rejected_ids = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]

        # Compute pairwise loss. Only backprop on the different tokens before padding
        loss = 0
        for i in range(bs):
            chosen_id = chosen_ids[i]
            rejected_id = rejected_ids[i]
            chosen_reward = chosen_rewards[i]
            rejected_reward = rejected_rewards[i]

            c_inds = (chosen_id == self.PAD_ID).nonzero()
            c_ind = c_inds[self.num_padding_at_beginning].item() if len(
                c_inds
            ) > self.num_padding_at_beginning else seq_len  # OPT model pads the first token, so we need to use the second padding token as the end of the sequence
            check_divergence = (chosen_id != rejected_id).nonzero()

            if len(check_divergence) == 0:
                end_ind = rejected_reward.size(-1)
                divergence_ind = end_ind - 1
                r_ind = c_ind
            else:
                # Check if there is any padding otherwise take length of sequence
                r_inds = (rejected_id == self.PAD_ID).nonzero()
                r_ind = r_inds[self.num_padding_at_beginning].item(
                ) if len(r_inds) > self.num_padding_at_beginning else seq_len
                end_ind = max(c_ind, r_ind)
                divergence_ind = check_divergence[0]
            assert divergence_ind > 0
            c_truncated_reward = chosen_reward[divergence_ind:end_ind]
            r_truncated_reward = rejected_reward[divergence_ind:end_ind]
            chosen_mean_scores.append(
                chosen_reward[c_ind - 1])  #use the end score for reference
            rejected_mean_scores.append(rejected_reward[r_ind - 1])

            loss += -torch.nn.functional.logsigmoid(c_truncated_reward -
                                                    r_truncated_reward).mean()

        loss = loss / bs
        chosen_mean_scores = torch.stack(chosen_mean_scores)
        rejected_mean_scores = torch.stack(rejected_mean_scores)
        return {
            "loss": loss,
            "chosen_mean_scores": chosen_mean_scores,
            "rejected_mean_scores": rejected_mean_scores,
        }

    def forward_value(self,
                      input_ids=None,
                      attention_mask=None,
                      past_key_values=None,
                      position_ids=None,
                      head_mask=None,
                      inputs_embeds=None,
                      return_value_only=False,
                      prompt_length=0,
                      use_cache=False):

        if self.config.model_type == "llama":
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)

        transformer_outputs = self.rwtranrsformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs)
        hidden_states = transformer_outputs[0]
        values = self.v_head(hidden_states).squeeze(-1)
        if return_value_only:
            return values
        else:
            # [0 0 0 0 prompt, answer, 0 0 0 0 ] for step 3, we have padding at the beginning
            # [prompt, answer, 0, 0, 0, 0] this is normal
            assert prompt_length > 1, "prompt_length must be greater than 1 to help select the end score"
            bs = values.size(0)
            seq_len = input_ids.shape[1]
            chosen_end_scores = [
            ]  # we use this name for consistency with the original forward function
            for i in range(bs):
                input_id = input_ids[i]
                value = values[i]

                c_inds = (input_id[prompt_length:] == self.PAD_ID).nonzero()
                # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
                c_ind = c_inds[0].item() + prompt_length if len(
                    c_inds) > 0 else seq_len
                chosen_end_scores.append(value[c_ind - 1])
            return {
                "values": values,
                "chosen_end_scores": torch.stack(chosen_end_scores),
            }
