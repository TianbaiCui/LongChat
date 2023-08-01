import json
import pytest
from pathlib import Path
import torch
from transformers import LlamaTokenizer
from transformers.tokenization_utils_base import TruncationStrategy

from longchat.conversation import get_default_conv_template, SeparatorStyle
from longchat.train.custom_data.datamodule import VicunaFormatDataset

TEST_CASES_DIR = Path(__file__).parent / "test_cases"


class TestProcessOne:
    test_cases_file = TEST_CASES_DIR / "conversation_vicuna.json"
    list_data_dict = []
    with open(test_cases_file, "r", encoding="utf-8") as file:
        for line in file:
            list_data_dict.append(json.loads(line)["vicuna_format"])

    tokenizer = LlamaTokenizer.from_pretrained(
        "openlm-research/open_llama_13b",
        model_max_length=2048,
        padding_side="right",
        # add_eos_token=True,
        use_fast=False,
        # legacy=False,
    )

    @pytest.fixture
    def test_cases(self):
        list_data_dict = []
        with open(self.test_cases_file, "r", encoding="utf-8") as file:
            for line in file:
                list_data_dict.append(json.loads(line)["vicuna_format"])
        return list_data_dict

    @pytest.fixture
    def data_processor(self):
        return VicunaFormatDataset(
            str(self.test_cases_file), self.tokenizer, num_data=-1
        )  # assume the DataProcessor takes a tokenizer as an argument

    @pytest.mark.parametrize("test_case", list_data_dict)
    def test_process_one(self, test_case, data_processor):
        # Given
        conv = get_default_conv_template("vicuna").copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        if test_case["system_prompt"]:
            conv.system = test_case["system_prompt"]
        source = test_case["conversations"]
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
        original_text = conv.get_prompt()

        tokenized_text, label_mask = data_processor.process_one(
            test_case, self.tokenizer
        )
        input_ids = torch.tensor(tokenized_text["input_ids"])
        label_mask = torch.tensor(label_mask)
        targets = torch.where(
            label_mask.bool(), input_ids, torch.full_like(input_ids, -100)
        )
        target_text = self.tokenizer.decode(
            targets[targets != -100], skip_bos_tokens=True
        )

        print(f"target_text: {target_text}")
        ## tests:
        assert len(target_text.split(self.tokenizer.eos_token)) == len(
            original_text.split(self.tokenizer.eos_token)
        )
        assert (
            len(target_text.split(self.tokenizer.eos_token))
            == len([s["value"] for s in source if s["from"] == "gpt"]) + 1
        )
        true_text = " ".join([s["value"] for s in source if s["from"] == "gpt"])
        true_text = self.tokenizer.decode(
            self.tokenizer.encode(true_text), skip_special_tokens=True
        )
        masked_text = self.tokenizer.decode(
            targets[targets != -100], skip_special_tokens=True
        )
        assert true_text == masked_text, print(
            f"true_text: {true_text}\nmasked_text: {masked_text}"
        )
