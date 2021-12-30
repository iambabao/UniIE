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
from scipy.sparse import coo_matrix

from src.utils import read_file, read_json_lines

logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, context, tokens, nodes, edges, trigger, task):
        self.guid = guid
        self.context = context
        self.tokens = tokens
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
            self, guid, task, input_ids, attention_mask=None, token_type_ids=None,
            token_mapping=None, length=None, start_labels=None, end_labels=None, labels=None, prior=None,
    ):
        self.guid = guid
        self.task = task
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.token_mapping = token_mapping
        self.length = length
        self.start_labels = start_labels
        self.end_labels = end_labels
        self.labels = labels
        self.prior = prior

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def convert_examples_to_features(examples, tokenizer, max_seq_length, max_num_tokens, task2schema):
    counter = 0
    oor_counter = 0
    conflict_counter = 0
    features = []
    for (ex_index, example) in enumerate(tqdm(examples, desc="Converting Examples")):
        label2id = task2schema[example.task]["label2id"]
        encoded = {"guid": example.guid, "task": example.task}

        if example.trigger is not None:
            prefix = "The task is {} and the trigger of event {} is {}".format(
                example.task, example.trigger["type"], example.trigger["text"]
            )
        else:
            prefix = "The task is {}".format(example.task)
        input_text = (prefix, example.context)
        encoded.update(tokenizer.encode_plus(
            input_text,
            padding="max_length",
            truncation="longest_first",
            max_length=max_seq_length,
            return_offsets_mapping=True,
        ))

        token_mapping = []
        offset = encoded["input_ids"].index(tokenizer.sep_token_id) if isinstance(input_text, tuple) else 0
        for _, char_start, char_end in example.tokens:
            token_start, token_end = -1, -1
            for i, (start, end) in enumerate(encoded["offset_mapping"][offset:]):
                if token_start == -1:
                    if start <= char_start < end:
                        token_start = i + offset
                if token_end == -1:
                    if char_end < end:
                        token_end = i + offset
                    elif char_end == end:
                        token_end = i + offset + 1
            token_start = 0 if token_start == -1 else token_start
            token_end = 1 if token_end == -1 else token_end
            token_mapping.append((token_start, token_end))
        token_mapping = token_mapping[:max_num_tokens] + [(0, 1)] * (max_num_tokens - len(example.tokens))

        start_labels = np.zeros([max_num_tokens], dtype=np.int32)
        end_labels = np.zeros([max_num_tokens], dtype=np.int32)
        labels = np.zeros([max_num_tokens, max_num_tokens], dtype=np.int32)
        for node in example.nodes:
            counter += 1
            start, end = node["start"], node["end"]
            if start >= max_num_tokens or end > max_num_tokens:
                oor_counter += 1
                logger.warning("({}) Node {} out of range and will be skipped.".format(ex_index, node))
                continue
            start_labels[start], end_labels[end - 1] = 1, 1
            if labels[start][end - 1] == 0:
                labels[start][end - 1] = label2id["node: {}".format(node["type"])]
            else:
                conflict_counter += 1
                logger.warning("({}) Node conflict at ({}, {}) and will be skipped.".format(ex_index, start, end))
        for edge in example.edges:
            counter += 1
            head = example.nodes[edge["head"]]
            head_start, head_end = head["start"], head["end"]
            if head_start >= max_num_tokens or head_end > max_num_tokens:
                oor_counter += 1
                logger.warning("({}) Head {} out of range and will be skipped.".format(ex_index, head))
                continue
            tail = example.nodes[edge["tail"]]
            tail_start, tail_end = tail["start"], tail["end"]
            if tail_start >= max_num_tokens or tail_end > max_num_tokens:
                oor_counter += 1
                logger.warning("({}) Tail {} out of range and will be skipped.".format(ex_index, tail))
                continue
            if head_start <= tail_start:
                reverse = False
            else:
                reverse = True
                head_start, head_end, tail_start, tail_end = tail_start, tail_end, head_start, head_end
            # TODO: single edge
            if labels[head_start][tail_start] == 0:
                if not reverse:
                    labels[head_start][tail_start] = label2id["edge: {}".format(edge["type"])]
                else:
                    labels[head_start][tail_start] = label2id["reverse edge: {}".format(edge["type"])]
            else:
                conflict_counter += 1
                logger.warning("({}) Edge {} conflict and will be skipped.".format(ex_index, edge))
            # TODO: double edge
            # if labels[head_start][tail_start] == 0 and labels[head_end - 1][tail_end - 1] == 0:
            #     if not reverse:
            #         labels[head_start][tail_start] = label2id["edge: {}".format(edge["type"])]
            #         labels[head_end - 1][tail_end - 1] = label2id["edge: {}".format(edge["type"])]
            #     else:
            #         labels[head_start][tail_start] = label2id["reverse edge: {}".format(edge["type"])]
            #         labels[head_end - 1][tail_end - 1] = label2id["reverse edge: {}".format(edge["type"])]
            # elif labels[head_start][tail_start] != 0:
            #     conflict_counter += 1
            #     logger.warning("({}) Edge {} conflict at start and will be skipped.".format(ex_index, edge))
            # else:
            #     conflict_counter += 1
            #     logger.warning("({}) Edge {} conflict at end and will be skipped.".format(ex_index, edge))

        del encoded["offset_mapping"]
        encoded["token_mapping"] = token_mapping
        encoded["length"] = min(max_num_tokens, len(example.tokens))
        encoded["start_labels"], encoded["end_labels"] = start_labels, end_labels
        encoded["labels"] = coo_matrix(labels)
        features.append(InputFeatures(**encoded))

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: {}".format(encoded["guid"]))
            logger.info("input ids: {}".format(encoded["input_ids"]))
            logger.info("input text: {}".format(input_text))
            logger.info("input tokens: {}".format(
                [tokenizer.decode(encoded["input_ids"][_[0]:_[1]]) for _ in token_mapping[:encoded["length"]]]
            ))
            for node in example.nodes:
                logger.info("golden node: {} ({})".format(node["text"], node["type"]))
            for edge in example.edges:
                head, tail = example.nodes[edge["head"]], example.nodes[edge["tail"]]
                logger.info("golden edge: {} --> {} ({})".format(head["text"], tail["text"], edge["type"]))
            logger.info("labels: {}".format(labels[:encoded["length"], :encoded["length"]]))

    return features, counter, oor_counter, conflict_counter


class DataProcessorKD:
    def __init__(
            self,
            tasks,
            model_type,
            model_name_or_path,
            max_seq_length,
            max_num_tokens,
            do_lower_case=False,
            data_dir="",
            cache_dir="cache",
            overwrite_cache=False,
    ):
        self.tasks = tasks
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.max_num_tokens = max_num_tokens
        self.do_lower_case = do_lower_case

        self.data_dir = data_dir
        self.cache_dir = "{}_{}".format(cache_dir, "uncased" if do_lower_case else "cased")
        self.overwrite_cache = overwrite_cache

        self.task2id = {}
        self.id2task = {}
        self.task2schema = {}
        for task in tasks:
            self.task2id[task] = len(self.task2id)
            self.id2task[len(self.id2task)] = task
            id2label, label2id = {}, {}
            for line in read_file(os.path.join(data_dir, task, "schema_node.txt")):
                label = "node: {}".format(line.strip())
                id2label[len(id2label) + 1] = label
                label2id[label] = len(label2id) + 1
            num_node_types = len(id2label)
            for line in read_file(os.path.join(data_dir, task, "schema_edge.txt")):
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
                "cached_feature_{}_{}_{}_{}".format(
                    role,
                    list(filter(None, self.model_name_or_path.split("/"))).pop(),
                    self.max_seq_length,
                    self.max_num_tokens
                ),
            )
            if os.path.exists(cached_features) and not self.overwrite_cache:
                logger.info("Loading features from cached file {}".format(cached_features))
                features, u, v, w = torch.load(cached_features)
            else:
                features, u, v, w = convert_examples_to_features(
                    examples, tokenizer, self.max_seq_length, self.max_num_tokens, self.task2schema,
                )
                logger.info("Saving features into cached file {}".format(cached_features))
                torch.save((features, u, v, w), cached_features)

            if role == "train" and self.max_seq_length == 256 and self.max_num_tokens == 256:
                for index, line in tqdm(
                    enumerate(read_file("data/prior/{}/prior_train_256_256.txt".format(task))), desc="Loading prior",
                ):
                    prior = eval(line)
                    features[index].prior = prior
            all_features.extend(features)

            counter[task] = (len(features), u, v, w)

        for task, (x, u, v, w) in counter.items():
            logger.info("{}: {} samples with {} items ({} out of range, {} conflicts)".format(task, x, u, v, w))

        return all_examples, all_features


def run_test():
    from transformers import AutoTokenizer
    from src.utils import init_logger

    init_logger(logging.INFO)
    tasks = ['ace2005_re']
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', use_fast=True)
    processor = DataProcessorKD(
        tasks,
        'bert',
        'bert-base-cased',
        max_seq_length=256,
        max_num_tokens=256,
        data_dir='../../data/formatted',
        overwrite_cache=True,
    )
    processor.load_and_cache_data(tokenizer, 'train')


if __name__ == '__main__':
    run_test()
