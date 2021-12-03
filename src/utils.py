# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/1/1
@Desc               :
@Last modified by   : Bao
@Last modified date : 2021/5/8
"""

import json
import random
import logging
import numpy as np
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


def init_logger(level, filename=None, mode='a', encoding='utf-8'):
    logging_config = {
        'format': '%(asctime)s - %(levelname)s - %(name)s:\t%(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S',
        'level': level,
        'handlers': [logging.StreamHandler()]
    }
    if filename:
        logging_config['handlers'].append(logging.FileHandler(filename, mode, encoding))
    logging.basicConfig(**logging_config)


def log_title(title, sep='='):
    return sep * 50 + '  {}  '.format(title) + sep * 50


def read_file(filename, mode='r', encoding='utf-8', skip=0):
    with open(filename, mode, encoding=encoding) as fin:
        for line in fin:
            if skip > 0:
                skip -= 1
                continue
            yield line


def save_file(data, filename, mode='w', encoding='utf-8', skip=0):
    with open(filename, mode, encoding=encoding) as fout:
        for line in data:
            if skip > 0:
                skip -= 1
                continue
            print(line, file=fout)


def read_json(filename, mode='r', encoding='utf-8'):
    with open(filename, mode, encoding=encoding) as fin:
        return json.load(fin)


def save_json(data, filename, mode='w', encoding='utf-8'):
    with open(filename, mode, encoding=encoding) as fout:
        json.dump(data, fout, ensure_ascii=False, indent=4)


def read_json_lines(filename, mode='r', encoding='utf-8', skip=0):
    with open(filename, mode, encoding=encoding) as fin:
        for line in fin:
            if skip > 0:
                skip -= 1
                continue
            yield json.loads(line)


def save_json_lines(data, filename, mode='w', encoding='utf-8', skip=0):
    with open(filename, mode, encoding=encoding) as fout:
        for line in data:
            if skip > 0:
                skip -= 1
                continue
            print(json.dumps(line, ensure_ascii=False), file=fout)


def read_txt_dict(filename, sep=None, mode='r', encoding='utf-8', skip=0):
    key_2_id = dict()
    with open(filename, mode, encoding=encoding) as fin:
        for line in fin:
            if skip > 0:
                skip -= 1
                continue
            key, _id = line.strip().split(sep)
            key_2_id[key] = _id
    id_2_key = dict(zip(key_2_id.values(), key_2_id.keys()))

    return key_2_id, id_2_key


def save_txt_dict(key_2_id, filename, sep=None, mode='w', encoding='utf-8', skip=0):
    with open(filename, mode, encoding=encoding) as fout:
        for key, value in key_2_id.items():
            if skip > 0:
                skip -= 1
                continue
            if sep:
                print('{} {}'.format(key, value), file=fout)
            else:
                print('{}{}{}'.format(key, sep, value), file=fout)


def read_json_dict(filename, mode='r', encoding='utf-8'):
    with open(filename, mode, encoding=encoding) as fin:
        key_2_id = json.load(fin)
        id_2_key = dict(zip(key_2_id.values(), key_2_id.keys()))

    return key_2_id, id_2_key


def save_json_dict(data, filename, mode='w', encoding='utf-8'):
    with open(filename, mode, encoding=encoding) as fout:
        json.dump(data, fout, ensure_ascii=False, indent=4)


def pad_list(item_list, pad, max_len):
    item_list = item_list[:max_len]
    return item_list + [pad] * (max_len - len(item_list))


def pad_batch(data_batch, pad, max_len=None):
    if max_len is None:
        max_len = len(max(data_batch, key=len))
    return [pad_list(data, pad, max_len) for data in data_batch]


def convert_item(item, convert_dict, unk):
    return convert_dict[item] if item in convert_dict else unk


def convert_list(item_list, convert_dict, pad, unk, max_len=None):
    item_list = [convert_item(item, convert_dict, unk) for item in item_list]
    if max_len is not None:
        item_list = pad_list(item_list, pad, max_len)

    return item_list


def make_batch_iter(data, batch_size, shuffle):
    data_size = len(data)
    num_batches = (data_size + batch_size - 1) // batch_size

    if shuffle:
        random.shuffle(data)
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min(data_size, (i + 1) * batch_size)
        yield data[start_index:end_index]


# ====================
def generate_outputs(predicted, task_id, length, id2task, task2schema):
    outputs = []
    for flags, t_id, max_len in zip(predicted, task_id, length):
        task = id2task[t_id]
        id2label = task2schema[task]["id2label"]
        num_node_types, num_edge_types = task2schema[task]["num_types"]

        nodes = []
        start2nodes, end2nodes = defaultdict(list), defaultdict(list)
        for i in range(max_len):
            for j in range(i, max_len):
                if 0 < flags[i][j] <= num_node_types:
                    node = {"start": i, "end": j + 1, "type": id2label[flags[i][j]][6:]}
                    nodes.append(node)
                    start2nodes[i].append(node)
                    end2nodes[j].append(node)
        start_edges, end_edges = [], []
        for i in range(max_len):
            for j in range(i, max_len):
                if num_node_types < flags[i][j]:
                    for node_i in start2nodes[i]:
                        for node_j in start2nodes[j]:
                            if node_i == node_j:
                                continue
                            if (flags[i][j] - num_node_types) % 2 == 1:
                                start_edges.append({"head": node_i, "tail": node_j, "type": id2label[flags[i][j]][6:]})
                            else:  # reverse edges
                                start_edges.append({"head": node_j, "tail": node_i, "type": id2label[flags[i][j]][14:]})
                    for node_i in end2nodes[i]:
                        for node_j in end2nodes[j]:
                            if node_i == node_j:
                                continue
                            if (flags[i][j] - num_node_types) % 2 == 1:
                                end_edges.append({"head": node_i, "tail": node_j, "type": id2label[flags[i][j]][6:]})
                            else:
                                end_edges.append({"head": node_j, "tail": node_i, "type": id2label[flags[i][j]][14:]})
        edges = [item for item in start_edges if item in end_edges]
        outputs.append({"predicted_nodes": nodes, "predicted_edges": edges})
    return outputs


def generate_outputs_plus(predicted, predicted_start, predicted_end, task_id, length, id2task, task2schema):
    outputs = []
    for flags, s_flags, e_flags, t_id, max_len in zip(predicted, predicted_start, predicted_end, task_id, length):
        task = id2task[t_id]
        id2label = task2schema[task]["id2label"]
        num_node_types, num_edge_types = task2schema[task]["num_types"]

        starts, ends = [], []
        for index, (s, e) in enumerate(zip(s_flags, e_flags)):
            if s: starts.append(index)
            if e: ends.append(index)

        nodes = []
        start2nodes, end2nodes = defaultdict(list), defaultdict(list)
        for i in range(max_len):
            for j in range(i, max_len):
                if i not in starts or j not in ends:
                    continue
                if 0 < flags[i][j] <= num_node_types:
                    node = {"start": i, "end": j + 1, "type": id2label[flags[i][j]][6:]}
                    nodes.append(node)
                    start2nodes[i].append(node)
                    end2nodes[j].append(node)
        start_edges, end_edges = [], []
        for i in range(max_len):
            for j in range(i, max_len):
                if num_node_types < flags[i][j]:
                    for node_i in start2nodes[i]:
                        for node_j in start2nodes[j]:
                            if node_i == node_j:
                                continue
                            if (flags[i][j] - num_node_types) % 2 == 1:
                                start_edges.append({"head": node_i, "tail": node_j, "type": id2label[flags[i][j]][6:]})
                            else:  # reverse edges
                                start_edges.append({"head": node_j, "tail": node_i, "type": id2label[flags[i][j]][14:]})
                    for node_i in end2nodes[i]:
                        for node_j in end2nodes[j]:
                            if node_i == node_j:
                                continue
                            if (flags[i][j] - num_node_types) % 2 == 1:
                                end_edges.append({"head": node_i, "tail": node_j, "type": id2label[flags[i][j]][6:]})
                            else:
                                end_edges.append({"head": node_j, "tail": node_i, "type": id2label[flags[i][j]][14:]})
        edges = [item for item in start_edges if item in end_edges]
        outputs.append({"predicted_nodes": nodes, "predicted_edges": edges})
    return outputs


def generate_outputs_uni_hard(predicted, task_id, length, id2task, task2schema):
    outputs = []
    for flags, t_id, max_length in zip(predicted, task_id, length):
        task = id2task[t_id]
        id2label = task2schema[task]["id2label"]
        num_node_types, num_edge_types = task2schema[task]["num_types"]

        nodes = []
        i = 0
        while i < max_length:
            if 0 < flags[i][i] <= num_node_types:
                j = i + 1
                while j < max_length and flags[i][i] == flags[i][j]:
                    j += 1
                if np.all(flags[i:j,i:j] == flags[i][i]):
                    node = {"start": i, "end": j, "type": id2label[flags[i][i]][6:]}
                    nodes.append(node)
                    i = j
                else:
                    i = i + 1
            else:
                i = i + 1

        edges = []
        for head in nodes:
            for tail in nodes:
                if head == tail:
                    continue
                h_start, h_end = head["start"], head["end"]
                t_start, t_end = tail["start"], tail["end"]
                if num_node_types < flags[h_start][t_start]:
                    if np.all(flags[h_start:h_end, t_start:t_end] == flags[h_start][t_start]):
                        edges.append({"head": head, "tail": tail, "type": id2label[flags[h_start][t_start]][6:]})

        outputs.append({"predicted_nodes": nodes, "predicted_edges": edges})
    return outputs


def generate_outputs_uni_soft(scores, task_id, length, id2task, task2schema):
    outputs = []
    for score, t_id, max_len in zip(scores, task_id, length):
        task = id2task[t_id]
        id2label = task2schema[task]["id2label"]
        num_node_types, num_edge_types = task2schema[task]["num_types"]
        entity_labels = np.array(range(1, num_node_types + 1), dtype=np.int32)
        relation_labels = np.array(range(num_node_types + 1, num_node_types + num_edge_types + 1), dtype=np.int32)
        threshold = task2schema[task].get("threshold", 1.0)  # TODO

        joint_score = score[:max_len, :max_len, :]
        joint_score[..., entity_labels] = \
            (joint_score[..., entity_labels] + joint_score[..., entity_labels].transpose((1, 0, 2))) / 2

        feature_score_row = joint_score.reshape(max_len, -1)
        feature_score_col = joint_score.transpose((1, 0, 2)).reshape(max_len, -1)
        positions = (
            (
                np.linalg.norm(feature_score_row[0:max_len - 1] - feature_score_row[1:max_len], axis=1) +
                np.linalg.norm(feature_score_col[0:max_len - 1] - feature_score_col[1:max_len], axis=1)
            ) / 2 > threshold
        ).nonzero()[0]
        if len(positions) > 0:
            spans = [(0, positions[0].item() + 1), (positions[-1].item() + 1, max_len)] + \
                    [(positions[_].item() + 1, positions[_ + 1].item() + 1) for _ in range(len(positions) - 1)]
        else:
            spans = [(0, max_len)]

        nodes = []
        for start, end in spans:
            mean_score = np.mean(joint_score[start:end, start:end, :], axis=(0, 1))
            if np.max(mean_score[entity_labels]) > mean_score[0]:
                predicted_label = id2label[entity_labels[np.argmax(mean_score[entity_labels])]][6:]
                node = {"start": start, "end": end, "type": predicted_label}
                nodes.append(node)

        edges = []
        for head in nodes:
            for tail in nodes:
                if head == tail:
                    continue
                h_start, h_end = head["start"], head["end"]
                t_start, t_end = tail["start"], tail["end"]
                mean_score = np.mean(joint_score[h_start:h_end, t_start:t_end, :], axis=(0, 1))
                if np.max(mean_score[relation_labels]) > mean_score[0]:
                    predicted_label = id2label[relation_labels[np.argmax(mean_score[relation_labels])]][6:]
                    edges.append({"head": head, "tail": tail, "type": predicted_label})
        outputs.append({"predicted_nodes": nodes, "predicted_edges": edges})
    return outputs


def refine_outputs(examples, outputs):
    refined_outputs = []
    for example, entry in zip(examples, outputs):
        golden_nodes = [{"start": node["start"], "end": node["end"], "type": node["type"]} for node in example.nodes]
        golden_edges = [
            {"head": golden_nodes[edge["head"]], "tail": golden_nodes[edge["tail"]], "type": edge["type"]}
            for edge in example.edges
        ]
        refined_outputs.append({
            "context": example.context,
            "golden_nodes": golden_nodes,
            "golden_edges": golden_edges,
            "predicted_nodes": entry["predicted_nodes"],
            "predicted_edges": entry["predicted_edges"],
            "task": example.task,
        })
    return refined_outputs


def compute_metrics(outputs):
    results = {}
    for entry in outputs:
        task = entry["task"]
        if task not in results:
            results[task] = Counter()
        golden_nodes = entry['golden_nodes']
        predicted_nodes = entry['predicted_nodes']
        results[task]['golden_nodes'] += len(golden_nodes)
        results[task]['predicted_nodes'] += len(predicted_nodes)
        results[task]['matched_nodes'] += len([_ for _ in golden_nodes if _ in predicted_nodes])

        golden_edges = entry['golden_edges']
        predicted_edges = entry['predicted_edges']
        results[task]['golden_edges'] += len(golden_edges)
        results[task]['predicted_edges'] += len(predicted_edges)
        results[task]['matched_edges'] += len([_ for _ in golden_edges if _ in predicted_edges])

    node_scores = []
    edge_scores = []
    for task in results:
        results[task]['node_p'], results[task]['node_r'], results[task]['node_f'] = get_prf(
            golden=results[task]['golden_nodes'],
            predicted=results[task]['predicted_nodes'],
            matched=results[task]['matched_nodes'],
        )
        node_scores.append(results[task]['node_f'])
        results[task]['edge_p'], results[task]['edge_r'], results[task]['edge_f'] = get_prf(
            golden=results[task]['golden_edges'],
            predicted=results[task]['predicted_edges'],
            matched=results[task]['matched_edges'],
        )
        edge_scores.append(results[task]['edge_f'])

    results['node_score'] = sum(node_scores) / len(node_scores)
    results['edge_score'] = sum(edge_scores) / len(edge_scores)
    results['score'] = (results['node_score'] + results['edge_score']) / 2

    return results


def get_prf(golden, predicted, matched):
    if golden == 0:
        if predicted == 0:
            precision = recall = f1 = 1.0
        else:
            precision = recall = f1 = 0.0
    else:
        if matched == 0 or predicted == 0:
            precision = recall = f1 = 0.0
        else:
            precision = matched / predicted
            recall = matched / golden
            f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1
