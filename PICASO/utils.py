import os
import random
import numpy as np
import pandas as pd
import logging

import torch
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)

class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 src_ant,
                 so_title,
                 so_api,
                 target,
                 ):
        self.idx = idx
        self.src_ant = src_ant
        self.so_title = so_title
        self.so_api = so_api
        self.target = target


def read_examples(filename):
    """Read examples from filename."""
    examples = []
    
    df = pd.read_csv(filename)
    df = df.fillna("")

    ant = df['annotation'].astype("string").tolist()
    so_title = df['so_title'].astype("string").tolist()

    df['so_api'] = df['so_api'].apply(lambda x: ' '.join(x.split(',')))
    so_api = df['so_api'].astype("string").tolist()

    df['target_api'] = df['target_api'].apply(lambda x: x.replace('.', ' . '))
    api_seq = df['target_api'].astype("string").tolist()
    
    for i in range(len(ant)):
        examples.append(
                Example(
                    idx=i,
                    src_ant=ant[i].lower(),
                    so_title=so_title[i].lower(),
                    so_api=so_api[i].lower(),
                    target=api_seq[i].lower(),
                )
            )

    return examples


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 src_ant_ids,
                 so_title_ids,
                 so_api_ids,
                 target_ids,
                 src_ant_mask,
                 so_title_mask,
                 so_api_mask,
                 target_mask,
                 ):
        self.example_id = example_id
        self.src_ant_ids = src_ant_ids
        self.so_title_ids = so_title_ids
        self.so_api_ids = so_api_ids
        self.target_ids = target_ids
        self.src_ant_mask = src_ant_mask
        self.so_title_mask = so_title_mask
        self.so_api_mask = so_api_mask
        self.target_mask = target_mask


def convert_examples_to_features(examples, tokenizer, max_source_length,\
    max_target_length, stage=None):
    features = []
    for example_index, example in enumerate(tqdm(examples, desc='convert examples to features...')):
        # annotation
        src_ant_tokens = tokenizer.tokenize(example.src_ant)[:max_source_length - 2]
        src_ant_tokens = [tokenizer.cls_token] + src_ant_tokens + [tokenizer.sep_token]
        src_ant_ids = tokenizer.convert_tokens_to_ids(src_ant_tokens)
        src_ant_mask = [1] * (len(src_ant_tokens))
        padding_length = max_source_length - len(src_ant_ids)
        src_ant_ids += [tokenizer.pad_token_id] * padding_length
        src_ant_mask += [0] * padding_length

        # so_title
        so_title_tokens = tokenizer.tokenize(example.so_title)[:max_source_length - 2]
        so_title_tokens = [tokenizer.cls_token] + so_title_tokens + [tokenizer.sep_token]
        so_title_ids = tokenizer.convert_tokens_to_ids(so_title_tokens)
        so_title_mask = [1] * (len(so_title_tokens))
        padding_length = max_source_length - len(so_title_ids)
        so_title_ids += [tokenizer.pad_token_id] * padding_length
        so_title_mask += [0] * padding_length

        # so_api
        so_api_tokens = tokenizer.tokenize(example.so_api)[:max_source_length - 2]
        so_api_tokens = [tokenizer.cls_token] + so_api_tokens + [tokenizer.sep_token]
        so_api_ids = tokenizer.convert_tokens_to_ids(so_api_tokens)
        so_api_mask = [1] * (len(so_api_tokens))
        padding_length = max_source_length - len(so_api_ids)
        so_api_ids += [tokenizer.pad_token_id] * padding_length
        so_api_mask += [0] * padding_length

        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:max_target_length - 2]
            
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length

        features.append(
            InputFeatures(
                example_index,
                src_ant_ids,
                so_title_ids,
                so_api_ids,
                target_ids,
                src_ant_mask,
                so_title_mask,
                so_api_mask,
                target_mask,
            )
        )
    return features


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def set_logger(log_path):
    """ e.g., logging.info """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s', datefmt = '%F %A %T'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)