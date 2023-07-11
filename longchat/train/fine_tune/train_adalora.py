# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional, Sequence

import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, AdaLoraConfig

from longchat.conversation import get_default_conv_template, SeparatorStyle

from longchat.train.monkey_patch.llama_condense_monkey_patch import (
    replace_llama_with_condense,
)

replace_llama_with_condense(ratio=2)

from longchat.train.monkey_patch.llama_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)

replace_llama_attn_with_flash_attn()
# replace_llama_attn_with_xformer()
import os

os.environ["WANDB_PROJECT"] = "open-llama-sft"

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    num_data: int = field(
        default=-1, metadata={"help": "Number of training data to use."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    report_to: str = field(default="wandb")


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = get_default_conv_template("vicuna").copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        # if source[0]["from"] not in roles.keys() or roles[source[0]["from"]] != conv.roles[0]:
        if source["system_prompt"]:
            conv.system = source["system_prompt"]
        source = source["conversations"]
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        skipped = 0
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            if role != conv.roles[(j - skipped) % 2]:
                print("skipping misaligned rounds")
                skipped += 1
                continue
            # assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
        print(conversations)
    #        assert False

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID

            # rank0_print(tokenizer.decode(target[cur_len+instruction_len:cur_len+round_len]))

            cur_len += round_len
        target[cur_len:] = IGNORE_TOKEN_ID

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                rank0_print(
                    f"WARNING: tokenization mismatch " f"{cur_len} vs. {total_len}"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, num_data: int
    ):
        super(SupervisedDataset, self).__init__()
        rank0_print("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))

        if num_data != -1:
            list_data_dict = list_data_dict[:num_data]
        rank0_print("Formatting inputs...")
        sources = [example["vicuna_format"] for example in list_data_dict]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, num_data: int
    ):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))
        print(len(list_data_dict))
        if num_data != -1:
            list_data_dict = list_data_dict[:num_data]
        print(len(list_data_dict))
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        data_dict = preprocess([e["vicuna_format"] for e in sources], self.tokenizer)
        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0],
                labels=data_dict["labels"][0],
                attention_mask=data_dict["attention_mask"][0],
            )
        return data_dict


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    train_dataset = dataset_cls(
        tokenizer=tokenizer, data_path=data_args.data_path, num_data=data_args.num_data
    )
    return dict(train_dataset=train_dataset, eval_dataset=None)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {(trainable_params/1e9):.2f}B || all params: {(all_param/1e9):.2f}B || trainable%: {100 * trainable_params / all_param}"
    )


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    model = transformers.LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        load_in_8bit=True,
        device_map="auto",
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    config = AdaLoraConfig(
        target_r=4,
        init_r=8,
        tinit=500,
        tfinal=1500,
        deltaT=10,
        beta1=0.85,
        beta2=0.85,
        target_modules=["q_proj", "v_proj", "k_proj"],
        lora_dropout=0.05
        # orth_reg_weight,
        # rank_pattern
    )

    # config = LoraConfig(
    #    r=256, #attention heads
    #    lora_alpha=32, #alpha scaling
    #    target_modules=["q_proj", "v_proj"], #if you know the
    #    lora_dropout=0.05,
    #    bias="none",
    #    task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
    # )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    tokenizer.pad_token = tokenizer.unk_token

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # import os
    # os.environ["WANDB_DISABLED"] = "true"
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
