import json
from typing import Dict, Union, Optional
from pathlib import Path
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
import transformers
from transformers import BatchEncoding
from transformers.trainer_pt_utils import LabelSmoother
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    PreTrainedTokenizerBase,
    TruncationStrategy,
)
from longchat.conversation import get_default_conv_template, SeparatorStyle

LOCAL_RANK = None
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def rank0_print(*args):
    """Print only on rank 0"""
    if LOCAL_RANK == 0:
        print(*args)


def default_preprocess(eval_pred, ignote_negative_labels=True):
    """Default preprocessing function for evaluation."""
    preds, labels = eval_pred.predictions, eval_pred.label_ids

    if not ignote_negative_labels:
        return preds, labels

    mask = labels > 0
    return preds[mask], labels[mask]


def train_val_dataset(dataset, val_split=0.1) -> tuple[Dataset, Dataset | None]:
    """Split dataset into train and validation sets."""
    if val_split == 0:
        return dataset, None

    train_idx, val_idx = train_test_split(
        list(range(len(dataset))), test_size=val_split, random_state=42, shuffle=True
    )
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


class VicunaFormatDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, data_path: str | Path, tokenizer: PreTrainedTokenizerBase, num_data: int
    ):
        super().__init__()
        self.tokenizer = tokenizer

        rank0_print("Loading data...")
        data_path = Path(data_path).expanduser()
        # list_data_dict = json.load(open(data_path, "r"))
        list_data_dict = []
        with open(data_path, "r", encoding="utf-8") as file:
            for line in file:
                list_data_dict.append(json.loads(line)["vicuna_format"])
        if num_data != -1:
            import random

            random.seed(42)
            list_data_dict = random.choices(list_data_dict, k=num_data)
        print(f"Total number of data: {len(list_data_dict)}")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict

    def process_one(
        self,
        source,
        tokenizer: PreTrainedTokenizerBase,
    ):
        """Preprocess data for supervised fine-tuning."""
        conv = get_default_conv_template("vicuna").copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        source["system_prompt"] = source.get("system_prompt", "")
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
            conv.append_message(role, sentence["value"])
        del skipped
        # Tokenize conversation
        tokenized_text = tokenizer(
            conv.get_prompt(),
            padding=False,
            truncation=TruncationStrategy.LONGEST_FIRST,
        )

        assert conv.sep_style == SeparatorStyle.TWO

        # Mask targets
        sep_tokens = [
            tokenizer.encode(conv.sep + role + ": ", add_special_tokens=False)
            for role in conv.roles[::-1]
        ]
        window_sizes = [len(sep_token) for sep_token in sep_tokens]
        label_mask = [0 for _ in tokenized_text.input_ids]

        turn = 0
        for i in range(window_sizes[0], len(tokenized_text.input_ids)):
            if (
                tokenized_text.input_ids[i - window_sizes[turn % 2] : i]
                == sep_tokens[turn % 2]
            ):
                if turn % 2 == 1:
                    label_mask[i - window_sizes[1] : i] = [0] * window_sizes[1]
                turn += 1
            label_mask[i] = turn % 2

        return tokenized_text, label_mask

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> tuple[BatchEncoding, list]:
        return self.process_one(self.list_data_dict[i], self.tokenizer)


@dataclass
class DialogueDataCollator:
    """
    Expects a list of texts corresponding to a sequence of
    [question, answer, question, answer, ...] pairs.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        # flatten_messages = []
        # label_masks = []
        # last_indices = []
        flatten_messages, label_masks = zip(*features)
        # for flatten_message, label_mask in features:
        #    flatten_messages.append(flatten_message)
        #    label_masks.append(label_mask)
        # last_indices.append(len(flatten_message.input_ids) - 1)
        # last_indices = torch.tensor(last_indices)
        # packing
        batch = self.tokenizer.pad(
            flatten_messages,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        dim = batch.input_ids.shape[-1]

        label_masks = torch.stack(
            [F.pad(torch.tensor(x), (0, dim - len(x)), value=0) for x in label_masks]
        ).bool()
        # labels are shifted in the LlamaForCausalLM forward method: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L772
        targets = batch.input_ids.clone()
        # targets = torch.roll(batch.input_ids, -1, -1)
        # targets[targets == self.tokenizer.bos_token_id] = IGNORE_TOKEN_ID
        targets = torch.where(
            label_masks, targets, torch.full_like(targets, IGNORE_TOKEN_ID)
        )
        # targets[range(len(targets)), last_indices] = (
        #     self.tokenizer.eos_token_id
        #     if self.tokenizer.eos_token_id is not None
        #     else IGNORE_TOKEN_ID
        # )
        batch["labels"] = targets

        return batch
