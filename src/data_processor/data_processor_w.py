# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/7/26
@Desc               : 
@Last modified by   : Bao
@Last modified date : 2021/5/8
"""

import os
import copy
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
from scipy.sparse import coo_matrix, vstack
from torch.utils.data import TensorDataset

from src.utils import read_file, read_json_lines

logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, context, nodes, edges, trigger, task):
        self.guid = guid
        self.context = context
        self.nodes = nodes
        self.edges = edges
        self.trigger = trigger
        self.task = task

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    def __init__(
            self, guid, task_id, input_ids,
            attention_mask=None, token_type_ids=None, length=None,
            start_labels=None, end_labels=None, labels=None,
    ):
        self.guid = guid
        self.task_id = task_id
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.length = length[0]
        self.start_labels = start_labels
        self.end_labels = end_labels
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def convert_examples_to_features(examples, tokenizer, max_seq_length, task2id, task2schema):
    features = []
    for (ex_index, example) in enumerate(tqdm(examples, desc="Converting Examples")):
        task_id = task2id[example.task]
        label2id = task2schema[example.task]["label2id"]
        num_node_types, num_edge_types = task2schema[example.task]["num_types"]
        encoded = {"guid": example.guid, "task_id": task_id}

        if example.trigger is not None:
            prefix = "{}: The trigger of event {} is {}".format(
                example.task, example.trigger['type'], example.trigger['text']
            )
        else:
            prefix = example.task
        input_text = (prefix, example.context)
        encoded.update(tokenizer.encode_plus(
            input_text,
            truncation="longest_first",
            max_length=max_seq_length,
            return_length=True,
            return_offsets_mapping=True,
        ))
        tokenizer.pad(encoded, padding="max_length", max_length=max_seq_length)

        char2token = []
        offset = encoded["input_ids"].index(tokenizer.sep_token_id) if isinstance(input_text, tuple) else 0
        for char_index in range(len(example.context)):
            for token_index, (start, end) in enumerate(encoded["offset_mapping"][offset:]):
                if char_index < start:
                    char2token.append(token_index - 1 + offset)
                    break
                elif start <= char_index < end:
                    char2token.append(token_index + offset)
                    break

        start_labels, end_labels = [0] * max_seq_length, [0] * max_seq_length
        labels = [[0] * max_seq_length for _ in range(max_seq_length)]
        for node in example.nodes:
            start, end = node["start"], node["end"]
            if start >= len(char2token) or end > len(char2token):
                logger.warning("({}) Node {} out of range and will be skipped.".format(ex_index, node))
                continue
            start, end = char2token[start], char2token[end - 1]
            start_labels[start], end_labels[end] = 1, 1
            if labels[start][end] == 0:
                labels[start][end] = label2id["node: {}".format(node["type"])]
            else:
                logger.warning("({}) Node conflict at ({}, {}) and will be skipped.".format(ex_index, start, end))
        for edge in example.edges:
            head = example.nodes[edge["head"]]
            head_start, head_end = head["start"], head["end"]
            if head_start >= len(char2token) or head_end > len(char2token):
                logger.warning("({}) Head {} out of range and will be skipped.".format(ex_index, head))
                continue
            tail = example.nodes[edge["tail"]]
            tail_start, tail_end = tail["start"], tail["end"]
            if tail_start >= len(char2token) or tail_end > len(char2token):
                logger.warning("({}) Tail {} out of range and will be skipped.".format(ex_index, tail))
                continue
            head_start, head_end = char2token[head_start], char2token[head_end - 1]
            tail_start, tail_end = char2token[tail_start], char2token[tail_end - 1]
            if head_start <= tail_start:
                reverse = False
            else:
                reverse = True
                head_start, head_end, tail_start, tail_end = tail_start, tail_end, head_start, head_end
            if labels[head_start][tail_start] == 0 and labels[head_end][tail_end] == 0:
                if not reverse:
                    labels[head_start][tail_start] = label2id["edge: {}".format(edge["type"])]
                    labels[head_end][tail_end] = label2id["edge: {}".format(edge["type"])]
                else:
                    labels[head_start][tail_start] = label2id["reverse edge: {}".format(edge["type"])]
                    labels[head_end][tail_end] = label2id["reverse edge: {}".format(edge["type"])]
            elif labels[head_start][tail_start] != 0:
                logger.warning(
                    "({}) Edge conflict at ({}, {}) and will be skipped.".format(ex_index, head_start, tail_start)
                )
            else:
                logger.warning(
                    "({}) Edge conflict at ({}, {}) and will be skipped.".format(ex_index, head_end, tail_end)
                )
        encoded["start_labels"], encoded["end_labels"] = start_labels, end_labels
        encoded["labels"] = coo_matrix(labels).reshape(1, max_seq_length * max_seq_length)

        del encoded["offset_mapping"]
        features.append(InputFeatures(**encoded))

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: {}".format(encoded["guid"]))
            logger.info("input ids: {}".format(encoded["input_ids"]))
            logger.info("input text: {}".format(input_text))
            for node in example.nodes:
                start, end = node["start"], node["end"]
                logger.info("golden node: {}".format(example.context[start:end]))
            for edge in example.edges:
                head, tail = example.nodes[edge["head"]], example.nodes[edge["tail"]]
                logger.info("golden edge: {} --> {}".format(head["text"], tail["text"]))
            for start in range(max_seq_length):
                for end in range(max_seq_length):
                    if 0 < labels[start][end] <= num_node_types:
                        logger.info("labeled node: {}".format(tokenizer.decode(encoded["input_ids"][start:end + 1])))
                    elif labels[start][end] > num_node_types:
                        if (labels[start][end] - num_node_types) % 2 == 1:
                            logger.info("labeled edge: {} --> {}".format(
                                tokenizer.decode(encoded["input_ids"][start]),
                                tokenizer.decode(encoded["input_ids"][end]),
                            ))
                        else:
                            logger.info("labeled edge: {} --> {}".format(
                                tokenizer.decode(encoded["input_ids"][end]),
                                tokenizer.decode(encoded["input_ids"][start]),
                            ))

    return features


class DataProcessorW:
    def __init__(
            self,
            model_type,
            model_name_or_path,
            max_seq_length,
            tasks,
            data_dir="",
            cache_dir="cache-w",
            overwrite_cache=False,
    ):
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.tasks = tasks

        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.overwrite_cache = overwrite_cache

        self.id2task = {}
        self.task2id = {}
        self.task2schema = {}
        for index, task in enumerate(tasks):
            self.id2task[index] = task
            self.task2id[task] = index
            id2label, label2id = {}, {}
            for line in read_file(os.path.join(data_dir, task, 'schema_node.txt')):
                label = "node: {}".format(line.strip())
                id2label[len(id2label) + 1] = label
                label2id[label] = len(label2id) + 1
            num_node_types = len(id2label)
            for line in read_file(os.path.join(data_dir, task, 'schema_edge.txt')):
                label = "edge: {}".format(line.strip())
                id2label[len(id2label) + 1] = label
                label2id[label] = len(label2id) + 1
                label = "reverse edge: {}".format(line.strip())
                id2label[len(id2label) + 1] = label
                label2id[label] = len(label2id) + 1
            num_edge_types = len(id2label) - num_node_types
            self.task2schema[task] = {
                "id2label": id2label,
                "label2id": label2id,
                "num_types": (num_node_types, num_edge_types),
            }

    def load_and_cache_data(self, tokenizer, role, suffix=None):
        if suffix is not None:
            role = "{}_{}".format(role, suffix)

        all_examples, all_features, counter = [], [], {}
        for task in self.tasks:
            data_dir = os.path.join(self.data_dir, task)
            cache_dir = os.path.join(data_dir, self.cache_dir)
            os.makedirs(cache_dir, exist_ok=True)

            cached_examples = os.path.join(cache_dir, "cached_example_{}".format(role))
            if os.path.exists(cached_examples) and not self.overwrite_cache:
                logger.info("Loading examples from cached file {}".format(cached_examples))
                examples = torch.load(cached_examples)
            else:
                examples = []
                for entry in tqdm(
                    list(read_json_lines(os.path.join(data_dir, "data_{}.json".format(role)))), desc="Loading Examples"
                ):
                    entry["guid"] = "{}-{}".format(task, len(examples))
                    examples.append(InputExample(**entry))
                logger.info("Saving examples into cached file {}".format(cached_examples))
                torch.save(examples, cached_examples)
            all_examples.extend(examples)

            cached_features = os.path.join(
                cache_dir,
                "cached_feature_{}_{}_{}".format(
                    role,
                    list(filter(None, self.model_name_or_path.split("/"))).pop(),
                    self.max_seq_length,
                ),
            )
            if os.path.exists(cached_features) and not self.overwrite_cache:
                logger.info("Loading features from cached file {}".format(cached_features))
                features = torch.load(cached_features)
            else:
                features = convert_examples_to_features(
                    examples, tokenizer, self.max_seq_length, self.task2id, self.task2schema,
                )
                logger.info("Saving features into cached file {}".format(cached_features))
                torch.save(features, cached_features)
            all_features.extend(features)

            counter[task] = len(features)

        for task, cnt in counter.items():
            logger.info("{}: {}".format(task, cnt))

        return all_examples, self._create_tensor_dataset(all_features)

    def _create_tensor_dataset(self, features):
        all_task_id = torch.tensor([f.task_id for f in features], dtype=torch.long)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use token_type_ids
        if self.model_type in ["bert", "xlnet", "albert"]:
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        else:
            all_token_type_ids = torch.tensor([[0] * self.max_seq_length for _ in features], dtype=torch.long)
        all_length = torch.tensor([f.length for f in features], dtype=torch.long)
        all_start_labels = torch.tensor([f.start_labels for f in features], dtype=torch.long)
        all_end_labels = torch.tensor([f.end_labels for f in features], dtype=torch.long)
        all_labels = vstack([f.labels for f in features])
        all_labels = torch.sparse_coo_tensor(
            torch.tensor(np.vstack([all_labels.row, all_labels.col]), dtype=torch.long),
            torch.tensor(all_labels.data, dtype=torch.long),
            size=all_labels.shape,
            dtype=torch.long,
        )

        dataset = TensorDataset(
            all_task_id, all_input_ids, all_attention_mask, all_token_type_ids, all_length,
            all_start_labels, all_end_labels, all_labels,
        )

        return dataset


def run_test():
    from transformers import AutoTokenizer
    from src.utils import init_logger

    init_logger(logging.INFO)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', use_fast=True)
    processor = DataProcessorW(
        'bert',
        'bert-base-cased',
        max_seq_length=256,
        tasks=['ade_re'],
        data_dir='../../data/processed',
        overwrite_cache=True,
    )
    processor.load_and_cache_data(tokenizer, 'test')


if __name__ == '__main__':
    run_test()
