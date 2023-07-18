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

import os, sys
import yaml
import argparse
from transformers import HfArgumentParser
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

from dataclasses import dataclass, field
import torch
import pathlib
from typing import Dict, Optional

import evaluate
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    LlamaForCausalLM,
    LlamaTokenizer,
)
from transformers import Trainer
from transformers.data.data_collator import DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, AdaLoraConfig


from longchat.train.custom_data.datamodule import (
    VicunaFormatDataset,
    DialogueDataCollator,
    train_val_dataset,
    IGNORE_TOKEN_ID,
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

os.environ["WANDB_PROJECT"] = "open-llama-sft"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="openlm-research/open_llama_13b")


@dataclass
class DataArguments:
    data_path: str = field(default="", metadata={"help": "Path to the training data."})
    num_data: int = field(
        default=-1, metadata={"help": "Number of training data to use."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainArguments(TrainingArguments):
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
    dataloader_num_workers: int = field(default=12)


def default_preprocess(eval_pred, ignote_negative_labels=True):
    preds, labels = eval_pred.predictions, eval_pred.label_ids
    preds = preds[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()
    if not ignote_negative_labels:
        return preds, labels

    mask = labels != IGNORE_TOKEN_ID
    return preds[mask], labels[mask]


# def get_metrics(conf, tokenizer):
#     # the reason behind using a list is that we might want to extend the list of our
#     # metrics in the future for more thorough evaluation
#     metrics, preprocess_fns = [evaluate.load("accuracy")], [default_preprocess]
#
#     # if any(dataset in QA_DATASETS for dataset in conf.datasets):
#     #     raise ValueError("TODO")
#     #     metrics.append(evaluate.load("squad_v2"))
#     #     preprocess_fns.append(preprocess_qa)
#     # if any(dataset in SUMMARIZATION_DATASETS for dataset in conf.datasets):
#     #     raise ValueError("TODO")
#     #     metrics.append(evaluate.load("rouge"))
#     #     preprocess_fns.append(
#     #         partial(preprocess_summarization, tokenizer, ignore_pad_token_for_loss=conf.ignore_pad_token_for_loss)
#     #     )
#
#     return metrics, preprocess_fns


def compute_metrics(eval_pred):
    out = {}
    metrics, preprocess_fns = [evaluate.load("accuracy")], [default_preprocess]
    for metric, preprocess_fn in zip(metrics, preprocess_fns):
        preds, labels = preprocess_fn(eval_pred)
        out = dict(**out, **metric.compute(predictions=preds, references=labels))

    return out


def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def train():
    global local_rank
    #    parser = argparse.ArgumentParser()
    #    parser.add_argument(
    #        "--config_file", type=str, default=None, help="Path to the config file"
    #    )
    #    parsed_args, unknown = parser.parse_known_args()
    #
    hf_parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainArguments)  # type: ignore
    )
    #
    #    model_args, data_args, training_args = hf_parser.parse_args_into_dataclasses()
    #    local_rank = training_args.local_rank
    #    if parsed_args.config_file is not None:
    #        # Load config file
    #        with open(parsed_args.config_file, "r") as f:
    #            config = yaml.safe_load(f)
    #        # Use config file values only if the corresponding command-line argument is not set
    #        for k, v in config.items():
    #            if (
    #                getattr(model_args, k, None) is None
    #                and getattr(data_args, k, None) is None
    #                and getattr(training_args, k, None) is None
    #            ):
    #                sys.argv += ["--{}={}".format(k, v)]
    #
    #        # Remove the '--config_file' argument and its value
    #    if "--config_file" in sys.argv:
    #        config_file_index = sys.argv.index("--config_file")
    #        del sys.argv[config_file_index : config_file_index + 2]

    model_args, data_args, training_args = hf_parser.parse_args_into_dataclasses()

    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        # load_in_8bit=True,
        torch_dtype=torch.bfloat16,
        # device_map="auto",
    )
    tokenizer = LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        add_eos_token=True,
        use_fast=False,
    )
    # config = AdaLoraConfig(
    #     target_r=4,
    #     init_r=8,
    #     tinit=500,
    #     tfinal=1500,
    #     deltaT=10,
    #     beta1=0.85,
    #     beta2=0.85,
    #     target_modules=["q_proj", "v_proj", "k_proj"],
    #     lora_dropout=0.05
    #     # orth_reg_weight,
    #     # rank_pattern
    # )

    config = LoraConfig(
        r=16,  # attention heads
        lora_alpha=16,  # alpha scaling
        target_modules=["q_proj", "v_proj"],  # if you know the
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",  # set this for CLM or Seq2Seq
    )

    model = get_peft_model(model, config)  # type: ignore
    print_trainable_parameters(model)
    tokenizer.pad_token = tokenizer.unk_token
    model.config.use_cache = False
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()  # type: ignore

    dataset = VicunaFormatDataset(
        tokenizer=tokenizer, data_path=data_args.data_path, num_data=data_args.num_data
    )
    train_dataset, eval_dataset = train_val_dataset(dataset, val_split=0.032)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=DialogueDataCollator(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
            padding="longest",
            max_length=training_args.model_max_length,
        ),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    # Save states. Weights might be a placeholder in zero3 and need a gather
    # state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), None)
    # if training_args.local_rank == 0:
    #    model.save_pretrained(training_args.output_dir, state_dict=state_dict)

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
