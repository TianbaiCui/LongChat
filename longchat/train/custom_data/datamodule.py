import json
from typing import Dict, Union, Optional
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
import transformers
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


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess data for supervised fine-tuning."""
    conv = get_default_conv_template("vicuna").copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for source in sources:
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


def train_val_dataset(dataset, val_split=0.1) -> tuple[Dataset, Dataset | None]:
    """Split dataset into train and validation sets."""
    if val_split == 0:
        return dataset, None

    train_idx, val_idx = train_test_split(
        list(range(len(dataset))), test_size=val_split, random_state=42, shuffle=True
    )
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, num_data: int
    ):
        super().__init__()
        rank0_print("Loading data...")
        # list_data_dict = json.load(open(data_path, "r"))
        list_data_dict = []
        with open(data_path, "r", encoding="utf-8") as file:
            for line in file:
                list_data_dict.append(json.loads(line))
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
        super().__init__()
        self.tokenizer = tokenizer

        rank0_print("Loading data...")
        # list_data_dict = json.load(open(data_path, "r"))
        list_data_dict = []
        with open(data_path, "r", encoding="utf-8") as file:
            for line in file:
                list_data_dict.append(json.loads(line))
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


class VicunaFormatDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, data_path: str, tokenizer: PreTrainedTokenizerBase, num_data: int
    ):
        super().__init__()
        self.tokenizer = tokenizer

        rank0_print("Loading data...")
        # list_data_dict = json.load(open(data_path, "r"))
        list_data_dict = []
        with open(data_path, "r", encoding="utf-8") as file:
            for line in file:
                list_data_dict.append(json.loads(line)["vicuna_format"])
        if num_data != -1:
            list_data_dict = list_data_dict[:num_data]
        print(f"Total number of data: {len(list_data_dict)}")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, str]:
        return self.list_data_dict[i]


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

    def process_one(
        self,
        source,
        tokenizer: PreTrainedTokenizerBase,
    ):
        """Preprocess data for supervised fine-tuning."""
        conv = get_default_conv_template("vicuna").copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
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
            max_length=self.max_length,
            truncation=TruncationStrategy.LONGEST_FIRST,
        )

        assert conv.sep_style == SeparatorStyle.TWO

        # Mask targets
        sep_tokens = [
            tokenizer.encode(conv.sep + role + ": ", add_special_tokens=False)
            for role in conv.roles
        ]
        window_sizes = [len(sep_token) for sep_token in sep_tokens]
        label_mask = [0 for _ in tokenized_text.input_ids]

        turn = 0
        for i in range(window_sizes[0], len(tokenized_text.input_ids)):
            if (
                tokenized_text.input_ids[i - window_sizes[turn % 2] : i]
                == sep_tokens[turn % 2]
            ):
                turn += 1
            label_mask[i] = turn % 2

        return tokenized_text, label_mask

    def __call__(self, features):
        flatten_messages = []
        label_masks = []
        for messages in features:
            flatten_message, label_mask = self.process_one(messages, self.tokenizer)
            flatten_messages.append(flatten_message)
            label_masks.append(label_mask)
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
        targets = torch.roll(batch.input_ids, -1, -1)
        targets = torch.where(
            label_masks, targets, torch.full_like(targets, IGNORE_TOKEN_ID)
        )
        batch["labels"] = targets

        return batch
