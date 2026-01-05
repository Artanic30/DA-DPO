import os
import random

import torch.distributed
from collections import defaultdict
from itertools import combinations
import datasets
import torch.distributed
from trl.trainer import DPOTrainer
from trl.trainer.utils import DPODataCollatorWithPadding
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, Sequence
import torch

import transformers
import sys
sys.path.append('/public/home/qiult/projects/BPO-main')

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token

from PIL import Image
import re
from tqdm import tqdm
import random
import torch.nn.functional as F
from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

from torch.utils.data import Sampler


replace_llama_attn_with_flash_attn()

def debug_point():
    import torch.distributed as dist
    if dist.get_rank() == 0:
        import ipdb
        ipdb.set_trace()
    dist.barrier()

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    dataset_size: int = field(
        default=-1,
    )
    filter_size: int = field(
        default=350,
    )
    test_size: float = 0.05
    subset_percent: float = -1

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    group_by_clip_delta: bool = field(default=False)
    beta: float = field(default=0.1)
    generate_during_eval: bool = field(default=False)
    group_by_clip_delta_noise: float = field(default=0.0)
    db_alpha: float = field(default=1.0)
    group_by_clip_delta_curriculum: bool = field(default=False)
    by_cluster: bool = field(default=False)



def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
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
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation




def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources
    for source in sources:
        for sentence in source:
            pattern = r"<img>.*?<\/img>"
            # Remove the matched pattern from the string
            sentence['value'] = re.sub(pattern, DEFAULT_IMAGE_TOKEN + '\n', sentence['value'])
            sentence['value'] = sentence['value'].strip()
            if "mmtag" in conversation_lib.default_conversation.version:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
            if len(sentence['value']) == 0:
                sentence['value'] = " "

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    # sources,
    source,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    # for i, source in enumerate(sources):
    if roles[source[0]["from"]] != conv.roles[0]:
        # Skip the first one if it is not from human
        source = source[1:]

    conv.messages = []
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        assert role == conv.roles[j % 2], f"{i}"
        conv.append_message(role, sentence["value"])

    # Tokenize conversations

    input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, return_tensors='pt')

    targets = [input_ids.clone()]
    instructions = []
    conversations = [conv.get_prompt()]
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_ids = tokenizer_image_token(parts[0], tokenizer)
                instruction_len = len(instruction_ids) - 2
                instructions.append(instruction_ids)
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_ids = tokenizer(parts[0]).input_ids
                instructions.append(instruction_ids)
                instruction_len = len(instruction_ids.input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    input_dict = dict(
        input_ids=input_ids,
        labels=targets[0],
        attention_mask = torch.ones_like(input_ids)
    )
    instruction_dict = dict(
        input_ids=instructions[0],
        labels=instructions[0],
        attention_mask = [1] * len(instructions[0])
    )

    return input_dict, instruction_dict


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)




def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    prompts = []
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)
        targets.append(source[:tokenized_lens[0]])

    return dict(prompt_id = prompts, input_ids=input_ids, labels=targets)


def read_jsonl(file_path):
    """Read a JSONL file and return a list of dictionaries."""
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def make_conv(prompt, answer):
    return [
        {
            "from": "human",
            "value": prompt,
        },
        {
            "from": "gpt",
            "value": answer,
        },
    ]


@dataclass
class LLaVADPODataCollator(DPODataCollatorWithPadding):
    def __init__(self, data_args, *args, **kwargs):
        super(LLaVADPODataCollator, self).__init__(*args, **kwargs)
        self.data_args = data_args

    def tokenize_batch_element(
        self,
        prompt: str,
        chosen: dict,
        rejected: dict,
        # chosen_clip_score: float,
        # rejected_clip_score: float
    ) -> Dict:
        """Tokenize a single batch element.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
            in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}

        chosen, chosen_clip_score = chosen['text'], chosen['clip_score']
        rejected, rejected_clip_score = rejected['text'], rejected['clip_score']
        chosen_conv = make_conv(prompt, chosen)
        rejected_conv = make_conv(prompt, rejected)
        # processing image
        if "img" in prompt:
            # Define the regular expression pattern
            pattern = r"<img>(.*?)<\/img>"

            # Find all matches of the pattern in the string
            matches = re.findall(pattern, prompt)
            image_file = matches[0]
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(self.data_args.image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            # replacing img tokens
            chosen_conv,  rejected_conv = preprocess_multimodal([copy.deepcopy(chosen_conv), copy.deepcopy(rejected_conv)], data_args=self.data_args)
            # self.data_args None)
        # tokenize
        chosen_conv_dict, prompt_dict = preprocess(
            chosen_conv,
            self.tokenizer,
            has_image=True)
        rejected_conv_dict, _ = preprocess(
            rejected_conv,
            self.tokenizer,
            has_image=True)

        for k, toks in {
            "chosen": chosen_conv_dict,
            "rejected": rejected_conv_dict,
            "prompt": prompt_dict,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}_{type_key}"] = tokens
        batch["images"] = image
        batch["chosen_clip_score"] = chosen_clip_score
        batch["rejected_clip_score"] = rejected_clip_score

        return batch

    def collate(self, batch):
        # first, pad everything to the same length
        padded_batch = super(LLaVADPODataCollator, self).collate(batch)
        images = torch.stack([b["images"] for b in batch])
        padded_batch.update({"images": images})
        return padded_batch

def prompt_format(prompt, img_path):
    out = []
    out.append(f"<img>{img_path}</img>")
    out.append(prompt.strip())
    return "".join(out)


def bpo_paired_dataset(ds, local_rank, data_args):
    def set_format(sample):
        prompt = sample["prompt"].replace('<image>', '')
        img_path = sample["image"]
        sample["prompt"] = prompt_format(prompt, img_path)
        if 'cluster_id' not in sample:
            sample['cluster_id'] = -1
        return sample

    ds = ds.map(set_format)
    # format prompt
    if local_rank > 0 and torch.distributed.is_initialized():
        torch.distributed.barrier()

    print(f"original length {len(ds)}")
    ds = ds.filter(lambda example: all(len(comp["response"].split()) <= data_args.filter_size for comp in example["completions"]))
    print(f"filtered length {len(ds)}")

    if local_rank == 0 and torch.distributed.is_initialized():
        torch.distributed.barrier()

    # make comparison pairs from completion list
    if local_rank > 0 and torch.distributed.is_initialized():
        torch.distributed.barrier()

    def make_batch_pairs(sample):
        converted_sample = defaultdict(list)

        for sample_idx, (prompt, image, completions, cluster_id) in enumerate(zip(sample["prompt"], sample["image"], sample["completions"], sample["cluster_id"])):
            for comp_idx1, comp_idx2 in combinations(range(len(completions)), 2):
                avg_score1, avg_score2 = completions[comp_idx1]["score"], completions[comp_idx2]["score"]
                # get chosen and rejected responses
                if avg_score1 > avg_score2:
                    chosen = completions[comp_idx1]["response"]
                    chosen_clip_score = completions[comp_idx1]["clip_score"]
                    rejected = completions[comp_idx2]["response"]
                    rejected_clip_score = completions[comp_idx2]["clip_score"]

                elif avg_score2 > avg_score1:
                    chosen = completions[comp_idx2]["response"]
                    chosen_clip_score = completions[comp_idx2]["clip_score"]
                    rejected = completions[comp_idx1]["response"]
                    rejected_clip_score = completions[comp_idx1]["clip_score"]
                else:
                    continue
                converted_sample["prompt"].append(prompt)
                # trick for saving clip score
                converted_sample["chosen"].append({
                    'text': chosen,
                    'clip_score': chosen_clip_score,
                    'cluster_id': cluster_id
                })
                converted_sample["rejected"].append({
                    'text': rejected,
                    'clip_score': rejected_clip_score,
                    'cluster_id': cluster_id
                })

        return converted_sample

    ds = ds.map(
        make_batch_pairs,
        batched=True,
        remove_columns=set(ds.column_names) - set(
            ["prompt", "chosen", "rejected", "cluster_id"]),
    )

    if local_rank == 0 and torch.distributed.is_initialized():
        torch.distributed.barrier()

    return ds


class CustomDPOTrainer(DPOTrainer):
    def __init__(self, *args, **kwargs):
        self.db_alpha = kwargs["db_alpha"]
        print(f'db_alpha: {self.db_alpha}')
        del kwargs["db_alpha"]
        super(CustomDPOTrainer, self).__init__(*args, **kwargs)

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

        if not self.is_encoder_decoder:
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

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_free: bool = False,
        chosen_clip_score: list = None,
        rejected_clip_score: list = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
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
        device = policy_chosen_logps.device
        chosen_clip_score = torch.tensor(chosen_clip_score).to(device)
        rejected_clip_score = torch.tensor(rejected_clip_score).to(device)
        # norm_chosen_clip_score = 2 * chosen_clip_score / (chosen_clip_score + rejected_clip_score)
        # norm_rejected_clip_score = 2 * rejected_clip_score / (chosen_clip_score + rejected_clip_score)


        # pi_logratios = norm_chosen_clip_score * policy_chosen_logps - norm_rejected_clip_score * policy_rejected_logps
        # ref_logratios = norm_chosen_clip_score * reference_chosen_logps - norm_rejected_clip_score * reference_rejected_logps

        pi_logratios =  policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if reference_free:
            ref_logratios = 0

        logits = pi_logratios - ref_logratios

        clip_weight = ((chosen_clip_score - rejected_clip_score).abs() + 1) * self.db_alpha

        losses = -F.logsigmoid(self.beta * clip_weight * logits)

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards, clip_weight


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
            chosen_clip_score,
            rejected_clip_score
        ) = self.concatenated_forward(model, batch)
        with torch.no_grad():
            if self.ref_model is None:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                        _,
                        _,
                    ) = self.concatenated_forward(self.model, batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, batch)

        losses, chosen_rewards, rejected_rewards, clip_weight = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            chosen_clip_score=chosen_clip_score,
            rejected_clip_score=rejected_clip_score
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
        metrics[f"{prefix}clip_weight/delta"] = clip_weight.detach().cpu().mean()


        return losses.mean(), metrics

    def concatenated_forward(
        self, model, batch):
        chosen_clip_score = batch['chosen_clip_score']
        rejected_clip_score = batch['rejected_clip_score']

        concatenated_batch = self.concatenated_inputs(batch)
        len_chosen = batch["chosen_labels"].shape[0]
        images = batch["images"].repeat(2, 1,1,1)
        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )
        outputs, all_labels = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            labels=concatenated_batch["concatenated_labels"],
            images=images,
            return_label=True,
            **model_kwargs,
        )
        all_logits = outputs.logits.to(torch.float32)

        all_logps = self._get_batch_logps(
            all_logits,
            all_labels,
            average_log_prob=False,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_clip_score,rejected_clip_score)


    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None:
            return None

        if self.args.group_by_clip_delta:
            clip_score_delta = []
            cluster_ids = []
            for c_id, pos, neg in zip(self.train_dataset.data[1], self.train_dataset.data[2], self.train_dataset.data[3]):
                pos_score = pos.as_py()['clip_score']
                neg_score = neg.as_py()['clip_score']
                distortion = torch.empty(1).uniform_(1 - self.args.group_by_clip_delta_noise, 1 + self.args.group_by_clip_delta_noise).item()

                clip_score_delta.append((pos_score - neg_score) * distortion)
                cluster_ids.append(str(c_id))


            if self.args.group_by_clip_delta_curriculum:

                return ClipScoreGroupedSampler(
                    self.args.train_batch_size,
                    world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                    clip_score_delta=clip_score_delta,
                    cluster_ids=cluster_ids,
                    by_cluster=self.args.by_cluster,
                    curriculum=True
                )
            else:
                return ClipScoreGroupedSampler(
                    self.args.train_batch_size,
                    world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                    clip_score_delta=clip_score_delta,
                )
        else:
            return super()._get_train_sampler()


class ClipScoreGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        clip_score_delta: Optional[List[float]] = None,
        curriculum: bool = False,
        cluster_ids=None,
        generator=None,
        by_cluster=False
    ):
        if clip_score_delta is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.curriculum = curriculum
        self.clip_score_delta = clip_score_delta
        self.generator = generator
        self.cluster_ids = cluster_ids
        self.by_cluster = by_cluster

    def __len__(self):
        return len(self.clip_score_delta)

    def __iter__(self):
        if self.by_cluster:
            indices = get_clip_score_delta_grouped_indices_clustered(self.clip_score_delta, self.batch_size, self.world_size,
                                                          generator=self.generator, cluster_ids=self.cluster_ids)
        else:

            indices = get_clip_score_delta_grouped_indices(self.clip_score_delta, self.batch_size, self.world_size,
                                                           self.curriculum,
                                                           generator=self.generator)


        return iter(indices)


def get_clip_score_delta_grouped_indices_clustered(clip_score_delta, batch_size, world_size, cluster_ids=None,
                                                generator=None):
    # 初始化 cluster2deltas
    cluster2deltas = {}
    for idx, (delta, cluster_id) in enumerate(zip(clip_score_delta, cluster_ids)):
        if cluster_id not in cluster2deltas:
            cluster2deltas[cluster_id] = []
        cluster2deltas[cluster_id].append([idx, delta])

    # 对每个 cluster 内部按 delta 排序并生成 megabatches
    all_cluster_megabatches = []
    for cluster_id in sorted(cluster2deltas.keys()):
        cluster2deltas[cluster_id].sort(key=lambda x: x[1])  # 按 delta 排序
        sorted_indices = torch.tensor([_[0] for _ in cluster2deltas[cluster_id]])

        # 按 megabatch_size 分割
        megabatch_size = world_size * batch_size
        cluster_megabatches = [sorted_indices[i: i + megabatch_size].tolist() for i in
                               range(0, len(sorted_indices), megabatch_size)]

        # 如果启用 curriculum，按照 delta 平均值调整顺序
        deltas = []
        for batch_idx in cluster_megabatches:
            deltas.append(sum(clip_score_delta[b_id] for b_id in batch_idx) / len(batch_idx))

        deltas = F.normalize(torch.tensor(deltas), p=2)

        # Normalize to get probabilities (adding a small value to avoid division by zero)
        probabilities = deltas / (deltas.sum() + 1e-8)

        # Use torch.multinomial to sample megabatch_indices based on these probabilities
        megabatch_indices = torch.multinomial(probabilities, len(cluster_megabatches), replacement=False,
                                              generator=generator)

        # Re-arrange megabatches based on the sampled indices
        cluster_megabatches = [cluster_megabatches[i] for i in megabatch_indices]
        print(f'Curriculum applied for cluster {cluster_id}')

        all_cluster_megabatches.append(cluster_megabatches)

    # 交叉合并各个 cluster 的 megabatches
    merged_megabatches = []
    max_megabatch_len = max(len(cluster_mb) for cluster_mb in all_cluster_megabatches)

    for i in range(max_megabatch_len):
        for cluster_megabatches in all_cluster_megabatches:
            if i < len(cluster_megabatches):
                merged_megabatches.append(cluster_megabatches[i])

    return [i for megabatch in merged_megabatches for i in megabatch]


def get_clip_score_delta_grouped_indices(clip_score_delta, batch_size, world_size, cluster_ids=None, curriculum=False, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    idx2delta = [[_, clip_score_delta[_]] for _ in range(len(clip_score_delta))]
    idx2delta.sort(key=lambda x: x[1])
    sorted_indices = torch.tensor([_[0] for _ in idx2delta])
    # indices = torch.randperm(len(clip_score_delta), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [sorted_indices[i : i + megabatch_size].tolist() for i in range(0, len(clip_score_delta), megabatch_size)]
    if not curriculum:
        megabatch_indices = torch.randperm(len(megabatches), generator=generator)
        megabatches = [megabatches[i] for i in megabatch_indices]
    else:
        deltas = []
        for batch_idx in megabatches:
            deltas.append(sum(clip_score_delta[b_id] for b_id in batch_idx) / len(batch_idx))

        deltas = F.normalize(torch.tensor(deltas), p=2)

        # Normalize to get probabilities (adding a small value to avoid division by zero)
        probabilities = deltas / deltas.sum()

        # Use torch.multinomial to sample megabatch_indices based on these probabilities
        megabatch_indices = torch.multinomial(probabilities, len(megabatches), replacement=False, generator=generator)

        # Re-arrange megabatches based on the sampled indices
        megabatches = [megabatches[i] for i in megabatch_indices]
        print(f'use curricumlum')

    return [i for megabatch in megabatches for i in megabatch]



def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    (
        model_args, data_args, training_args
    ) = parser.parse_args_into_dataclasses()
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    bnb_model_from_pretrained_args = {}

    local_rank = training_args.local_rank
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:

        model = LlavaLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)
    else:
        model.model.requires_grad_(True)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'phi' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)


    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    dataset = datasets.load_dataset('json', data_files=data_args.data_path, split="train")

    def sample_data(dataset, sample_percentage):
        ids = list(range(len(dataset)))
        indices = torch.randperm(len(ids))

        sampled_indices = indices[:int(sample_percentage * len(indices))]

        sampled_data = dataset.select(sorted(sampled_indices))
        # sampled_data = dataset.select(sorted(random.sample(ids, int(sample_percentage * len(ids)))))
        return sampled_data
    if data_args.subset_percent > 0:
        dataset = sample_data(dataset, data_args.subset_percent)
    train_dataset = dataset
    train_dataset = bpo_paired_dataset(train_dataset, local_rank, data_args)
    collator = LLaVADPODataCollator(data_args=data_args, tokenizer=tokenizer, max_length=1024)

    print(f"rank {local_rank} train length {len(train_dataset)}")
    trainer = CustomDPOTrainer(
        model,
        args=training_args,
        beta=training_args.beta,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        max_length=training_args.model_max_length,
        db_alpha=training_args.db_alpha
    )
    print(f'trainer loaded')

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
