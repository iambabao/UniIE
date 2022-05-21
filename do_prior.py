# -*- coding: utf-8 -*-

"""
@Author             : huggingface
@Date               : 2020/7/26
@Desc               :
@Last modified by   : Bao
@Last modified date : 2021/5/14
"""

import argparse
import logging
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
AutoConfig,
AutoTokenizer,
)

from src.data_processor import DataProcessor
from src.models import *
from src.utils import init_logger

logger = logging.getLogger(__name__)
MODEL_MAPPING = {
    "variant-single": VariantSingle,
}


def pruning(outputs):
    logits = outputs["task_logits"]
    mask = outputs["position_mask"]

    for u, v in zip(logits, mask):
        yield torch.softmax(u[v], dim=-1).detach().cpu().tolist()


def get_prior(args, data_processor, model, tokenizer):
    os.makedirs(args.output_dir, exist_ok=True)

    args.batch_size = args.per_device_batch_size * max(1, args.n_gpu)
    examples, dataset = data_processor.load_and_cache_data(tokenizer, args.role, args.suffix)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)

    prior_file = "prior_{}_{}_{}.txt".format(args.role, args.max_seq_length, args.max_num_tokens)
    with open(os.path.join(args.output_dir, prior_file), "w", encoding="utf-8") as fp:
        for batch in tqdm(dataloader, desc="{}: {}".format(args.tasks, args.role)):
            model.eval()
            with torch.no_grad():
                inputs = {
                    "task_id": batch[0].to(args.device),
                    "input_ids": batch[1].to(args.device),
                    "attention_mask": batch[2].to(args.device),
                    "token_mapping": batch[4].to(args.device),
                    "length": batch[5].to(args.device),
                }
                # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use token_type_ids
                if args.model_type in ["bert", "xlnet", "albert"]:
                    inputs["token_type_ids"] = batch[3].to(args.device)

                outputs = model(inputs)

                for prior in pruning(outputs):
                    print(prior, file=fp)


def main():
    parser = argparse.ArgumentParser()

    # Datasets parameters
    parser.add_argument("--tasks", required=True, type=str, help="")
    parser.add_argument("--suffix", default=None, type=str, help="")
    parser.add_argument("--role", required=True, type=str, help="")

    # Model hyper parameters
    parser.add_argument("--model_type", required=True, type=str, help="")
    parser.add_argument("--model_name_or_path", required=True, type=str, help="")
    parser.add_argument("--pretrained_model", default=None, type=str, help="")
    parser.add_argument("--max_seq_length", default=128, type=int, help="")
    parser.add_argument("--max_num_tokens", default=128, type=int, help="")
    parser.add_argument("--do_lower_case", action="store_true", help="")
    parser.add_argument("--task_hidden_size", default=128, type=int, help="")

    # Directory parameters
    parser.add_argument("--data_dir", type=str, required=True, help="")
    parser.add_argument("--output_dir", type=str, required=True, help="")
    parser.add_argument("--cache_dir", default="", type=str, help="")

    # Running parameters
    parser.add_argument("--per_device_batch_size", default=8, type=int, help="")

    # Other parameters
    parser.add_argument("--no_cuda", action="store_true", help="")
    args = parser.parse_args()

    # Setup CUDA, GPU training
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    init_logger(logging.INFO)

    # Parse tasks
    args.tasks = sorted(args.tasks.split(","))

    # Load config, tokenizer and pretrained model
    data_processor = DataProcessor(
        args.tasks,
        args.model_type,
        args.model_name_or_path,
        args.max_seq_length,
        args.max_num_tokens,
        do_lower_case=args.do_lower_case,
        data_dir=args.data_dir,
    )
    config = AutoConfig.from_pretrained(
        args.pretrained_model,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = MODEL_MAPPING[args.model_type].from_pretrained(
        args.pretrained_model,
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model.to(args.device)

    logger.info("Parameters %s", args)
    logger.info("Config %s", config)

    get_prior(args, data_processor, model, tokenizer)


if __name__ == "__main__":
    main()
