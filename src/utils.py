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
from collections import Counter

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
        for i in range(max_len):
            for j in range(i, max_len):
                predicted = flags[i][j]
                if 0 < predicted <= num_node_types:
                    node = {"start": i, "end": j + 1, "type": id2label[predicted][6:]}
                    nodes.append(node)

        # TODO: single edge
        # edges = []
        # for head in nodes:
        #     for tail in nodes:
        #         if head == tail: continue
        #         head_start, tail_start = head["start"], tail["start"]
        #         if head_start > tail_start: continue
        #
        #         predicted = flags[head_start][tail_start]
        #         if num_node_types < predicted:
        #             if (predicted - num_node_types) % 2 == 1:
        #                 edges.append({"head": head, "tail": tail, "type": id2label[predicted][6:]})
        #             else:  # reverse edges
        #                 edges.append({"head": tail, "tail": head, "type": id2label[predicted][14:]})

        # TODO:double edge
        edges = []
        for head in nodes:
            for tail in nodes:
                if head == tail: continue
                head_start, head_end = head["start"], head["end"]
                tail_start, tail_end = tail["start"], tail["end"]
                if head_start > tail_start: continue
                if flags[head_start][tail_start] != flags[head_end - 1][tail_end - 1]: continue

                predicted = flags[head_start][tail_start]
                if num_node_types < predicted:
                    if (predicted - num_node_types) % 2 == 1:
                        edges.append({"head": head, "tail": tail, "type": id2label[predicted][6:]})
                    else:  # reverse edges
                        edges.append({"head": tail, "tail": head, "type": id2label[predicted][14:]})

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
