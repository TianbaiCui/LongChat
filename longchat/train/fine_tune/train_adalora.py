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
from typing import Dict, Optional

import transformers
from transformers import Trainer
from transformers.data.data_collator import DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, AdaLoraConfig


from longchat.train.custom_data.datamodule import (
    LazySupervisedDataset,
    SupervisedDataset,
    train_val_dataset,
)

from longchat.train.utils.io_utils import (
    print_trainable_parameters,
    safe_save_model_for_hf_trainer,
)

from longchat.train.monkey_patch.llama_condense_monkey_patch import (
    replace_llama_with_condense,
)

replace_llama_with_condense(ratio=1.5)

from longchat.train.monkey_patch.llama_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)

replace_llama_attn_with_flash_attn()
# replace_llama_attn_with_xformer()
import os

os.environ["WANDB_PROJECT"] = "open-llama-sft"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default="", metadata={"help": "Path to the training data."})
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
    remove_unused_columns: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=True)
    dataloader_num_workers: int = field(default=8)


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    dataset = dataset_cls(
        tokenizer=tokenizer, data_path=data_args.data_path, num_data=data_args.num_data
    )
    train_dataset, eval_dataset = train_val_dataset(dataset, val_split=0.02)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)  # type: ignore
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

    #    config = LoraConfig(
    #        r=16,  # attention heads
    #        lora_alpha=32,  # alpha scaling
    #        target_modules=["q_proj", "v_proj"],  # if you know the
    #        lora_dropout=0.05,
    #        bias="none",
    #        task_type="CAUSAL_LM",  # set this for CLM or Seq2Seq
    #    )

    model = get_peft_model(model, config)  # type: ignore
    print_trainable_parameters(model)
    tokenizer.pad_token = tokenizer.unk_token
    model.config.use_cache = False
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()  # type: ignore

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # import os
    # os.environ["WANDB_DISABLED"] = "true"
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=DataCollatorWithPadding,
        **data_module,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
