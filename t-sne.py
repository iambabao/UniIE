# -*- coding:utf-8  -*-

"""
@Author             : Bao
@Date               : 2021/12/18
@Desc               :
@Last modified by   : Bao
@Last modified date : 2021/12/18
"""


import logging
import os
import torch
import tensorflow as tf
from tqdm import tqdm
from tensorboard.plugins import projector
from transformers import AutoTokenizer, AutoConfig, AutoModel

from src.utils import init_logger, read_json_lines, save_file

logger = logging.getLogger(__name__)


def encode_data(data_dir, tasks, model_name_or_path, output_dir, no_cuda=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    config = AutoConfig.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path, config=config)
    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    model.to(device)

    embedding = []
    metadata = ["Dataset\tLength\t#Nodes\t#Edges"]
    for task in tasks:
        for role in ['valid', 'test']:
            filename = os.path.join(data_dir, task, 'data_{}.json'.format(role))
            for entry in tqdm(read_json_lines(filename), desc=task):
                prefix = 'The task is {}'.format(task)
                encoded = tokenizer.encode_plus(prefix, entry['context'], return_tensors='pt')
                for key, value in encoded.items():
                    encoded[key] = value.to(device)
                outputs = model(**encoded)
                embedding.append(outputs['pooler_output'][0].detach().cpu().tolist())

                length = len(entry['tokens']) // 10
                length = str(length) if length < 10 else '>10'
                num_nodes = len(entry['nodes'])
                num_nodes = str(num_nodes) if num_nodes < 10 else '>10'
                num_edges = len(entry['edges'])
                num_edges = str(num_edges) if num_edges < 10 else '>10'
                metadata.append('\t'.join([task, length, num_nodes, num_edges]))

    os.makedirs(output_dir, exist_ok=True)
    torch.save(embedding, os.path.join(output_dir, 'embedding.bin'))
    save_file(metadata, os.path.join(output_dir, 'metadata.tsv'))


def visualize(root_dir):
    var = torch.load(os.path.join(root_dir, 'embedding.bin'))
    var = tf.Variable(var, name='embedding')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver([var])
        saver.save(sess, os.path.join(root_dir, 'embedding.ckpt'))

        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = var.name
        embedding.metadata_path = 'metadata.tsv'

        writer = tf.summary.FileWriter(root_dir)
        projector.visualize_embeddings(writer, config)


def main():
    init_logger(logging.INFO)

    tasks = ['conll04_re', 'ade_re', 'ace2005_re', 'conll03_ner', 'genia_ner']

    # encode_data(
    #     'data/formatted',
    #     tasks,
    #     'bert-base-cased',
    #     'data/temp/raw',
    # )
    # visualize('data/temp/raw')

    encode_data(
        'data/formatted',
        tasks,
        'checkpoints/conll04_re,ade_re,ace2005_re,conll03_ner,genia_ner/1220/variant-a_bert-base-cased_256_cased_256_5.0e-05/checkpoint-best',
        'data/temp/tuned_a',
    )
    visualize('data/temp/tuned_a')
    encode_data(
        'data/formatted',
        tasks,
        'checkpoints/conll04_re,ade_re,ace2005_re,conll03_ner,genia_ner/1220/variant-b_bert-base-cased_256_cased_256_5.0e-05/checkpoint-best',
        'data/temp/tuned_b',
    )
    visualize('data/temp/tuned_b')
    encode_data(
        'data/formatted',
        tasks,
        'checkpoints/conll04_re,ade_re,ace2005_re,conll03_ner,genia_ner/1220/variant-c_bert-base-cased_256_cased_256_5.0e-05/checkpoint-best',
        'data/temp/tuned_c',
    )
    visualize('data/temp/tuned_c')


if __name__ == '__main__':
    main()
