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
import glob
import json
import torch
import torch.distributed as dist
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (
set_seed,
AutoConfig,
AutoTokenizer,
WEIGHTS_NAME,
AdamW,
get_linear_schedule_with_warmup,
)

from src.data_processor import DataProcessorKD as DataProcessor
from src.models import *
from src.utils import (
init_logger,
save_json,
save_json_lines,
generate_outputs,
refine_outputs,
compute_metrics,
)

logger = logging.getLogger(__name__)
MODEL_MAPPING = {
    "variant-a": VariantA,
    "variant-b": VariantB,
    "variant-c": VariantC,
}


def convert_batch(args, batch, task2id):
    outputs = OrderedDict()

    outputs["task_id"] = torch.tensor([task2id[f.task] for f in batch], dtype=torch.long).to(args.device)
    outputs["input_ids"] = torch.tensor([f.input_ids for f in batch], dtype=torch.long).to(args.device)
    outputs["attention_mask"] = torch.tensor([f.attention_mask for f in batch], dtype=torch.long).to(args.device)
    # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use token_type_ids
    if args.model_type in ["bert", "xlnet", "albert"]:
        outputs["token_type_ids"] = torch.tensor([f.token_type_ids for f in batch], dtype=torch.long).to(args.device)
    outputs["token_mapping"] = torch.tensor([f.token_mapping for f in batch], dtype=torch.long).to(args.device)
    outputs["length"] = torch.tensor([f.length for f in batch], dtype=torch.long).to(args.device)
    outputs["start_labels"] = torch.tensor([f.start_labels for f in batch], dtype=torch.long).to(args.device)
    outputs["end_labels"] = torch.tensor([f.end_labels for f in batch], dtype=torch.long).to(args.device)
    outputs["labels"] = torch.tensor([f.labels.todense().tolist() for f in batch], dtype=torch.long).to(args.device)
    if hasattr(batch[0], "prior") and batch[0].prior is not None:
        outputs["prior"] = [torch.tensor(f.prior, dtype=torch.float).to(args.device) for f in batch]

    return outputs


def save_checkpoint(output_dir, model, tokenizer, args):
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Saving model checkpoint to {}".format(output_dir))
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(args, os.path.join(output_dir, "training_args.bin"))


def train(args, data_processor, model, tokenizer, role):
    args.train_batch_size = args.per_device_train_batch_size * max(1, args.n_gpu)
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    _, train_dataset = data_processor.load_and_cache_data(tokenizer, role, args.suffix)
    if args.local_rank == 0:
        torch.distributed.barrier()
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=lambda x: x,
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total,
    )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("Num examples = %d", len(train_dataset))
    logger.info("Num Epochs = %d", args.num_train_epochs)
    logger.info("Instantaneous batch size per GPU = %d", args.per_device_train_batch_size)
    logger.info(
        "Total train batch size (w. parallel, accumulation & distributed) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("Total optimization steps = %d", t_total)

    global_step = 0
    training_loss = 0.0
    early_stop_flag = 0
    current_score, best_score = -float("inf"), -float("inf")
    set_seed(args.seed)
    model.zero_grad()
    for _ in range(int(args.num_train_epochs)):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()

            inputs = convert_batch(args, batch, data_processor.task2id)
            outputs = model(inputs)
            loss = outputs["loss"]
            start_loss, end_loss, kd_loss = outputs["start_loss"], outputs["end_loss"], outputs["kd_loss"]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            training_loss += loss.item()
            description = "Global step: {:>6d}, Loss: {:>.4f} (start: {:>.4f}, end: {:>.4f}, kd: {:>.4f})".format(
                global_step, loss.item(), start_loss.item(), end_loss.item(), kd_loss.item()
            )
            epoch_iterator.set_description(description)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0]:
                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        logger.info(description)
                        if args.evaluate_during_training:
                            results = evaluate(
                                args,
                                data_processor,
                                model,
                                tokenizer,
                                role="valid",
                                prefix=str(global_step)
                            )
                            current_score, best_score = results["score"], max(best_score, results["score"])
                            if current_score >= best_score:
                                early_stop_flag = 0
                                output_dir = os.path.join(args.output_dir, "checkpoint-best")
                                save_checkpoint(output_dir, model, tokenizer, args)
                            else:
                                early_stop_flag += 1
                                save_checkpoint(args.output_dir, model, tokenizer, args)
                            if 0 < args.early_stop < early_stop_flag:
                                break

                    if not args.save_best and args.save_steps > 0 and global_step % args.save_steps == 0:
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                        save_checkpoint(output_dir, model, tokenizer, args)
                if args.local_rank != -1:
                    torch.distributed.barrier()  # wait for evaluation if needed

            if 0 < args.early_stop < early_stop_flag or 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.early_stop < early_stop_flag or 0 < args.max_steps < global_step:
            break

    if args.local_rank in [-1, 0]:
        save_checkpoint(args.output_dir, model, tokenizer, args)

    return global_step, training_loss / global_step


def evaluate(args, data_processor, model, tokenizer, role, prefix=""):
    if prefix == "":
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(prefix))
    os.makedirs(output_dir, exist_ok=True)

    args.eval_batch_size = args.per_device_eval_batch_size * max(1, args.n_gpu)
    examples, dataset = data_processor.load_and_cache_data(tokenizer, role, args.suffix)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(
        dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=lambda x: x,
    )

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("Num examples = %d", len(dataset))
    logger.info("Batch size = %d", args.eval_batch_size)

    eval_outputs = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        with torch.no_grad():
            inputs = convert_batch(args, batch, data_processor.task2id)
            outputs = model(inputs)

            logits = outputs["task_logits"]
            predicted = torch.stack([torch.argmax(_, dim=-1) for _ in logits], dim=0)
            eval_outputs.extend(generate_outputs(
                predicted.detach().cpu().numpy(),
                inputs["task_id"].detach().cpu().numpy(),
                inputs["length"].detach().cpu().numpy(),
                data_processor.id2task,
                data_processor.task2schema,
            ))

    eval_outputs = refine_outputs(examples, eval_outputs)
    eval_outputs_file = os.path.join(output_dir, "{}_outputs.json".format(role))
    save_json_lines(eval_outputs, eval_outputs_file)

    eval_results = compute_metrics(eval_outputs)
    eval_results_file = os.path.join(output_dir, "{}_results.txt".format(role))
    save_json(eval_results, eval_results_file)
    logger.info("***** Eval results {} *****".format(prefix))
    logger.info(json.dumps(eval_results, ensure_ascii=False, indent=4))

    return eval_results


def main():
    parser = argparse.ArgumentParser()

    # Datasets parameters
    parser.add_argument("--tasks", required=True, type=str, help="")
    parser.add_argument("--suffix", default=None, type=str, help="")

    # Model hyper parameters
    parser.add_argument("--model_type", required=True, type=str, help="Model type")
    parser.add_argument(
        "--model_name_or_path",
        required=True,
        type=str,
        help="Path to model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--pretrained_model",
        default=None,
        type=str,
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument("--max_num_tokens", default=128, type=int, help="The maximum total input tokens.")
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument("--task_hidden_size", default=128, type=int, help="The hidden size of task specific layer.")

    # Directory parameters
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--log_dir",
        default=None,
        type=str,
        help="The log directory where the running details will be written.",
    )
    parser.add_argument(
        "--log_file",
        default=None,
        type=str,
        help="The log file where the running details will be written. This will overwrite --log_dir.",
    )

    # Training parameters
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the eval set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument("--save_best", action="store_true", help="Whether to save the best model .")
    parser.add_argument(
        "--per_device_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training."
    )
    parser.add_argument(
        "--per_device_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument("--early_stop", default=0, type=int, help="Early stop strategy.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )

    # Other parameters
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup log dir
    if args.local_rank in [-1, 0] and args.log_file is None and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        args.log_file = os.path.join(
            args.log_dir,
            "{}_{}_{}_{}_{:03d}_{:.1e}.txt".format(
                args.model_type,
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                args.max_seq_length,
                "uncased" if args.do_lower_case else "cased",
                args.task_hidden_size,
                args.learning_rate,
            ),
        )
    else:
        args.log_file = None
    init_logger(logging.INFO if args.local_rank in [-1, 0] else logging.WARNING, args.log_file)
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )

    # Set seed
    set_seed(args.seed)

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
        overwrite_cache=args.overwrite_cache,
    )
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if args.local_rank == 0:
        torch.distributed.barrier()
    config.num_tasks = len(args.tasks)
    config.task2labels = {task_name: schema["label2id"] for task_name, schema in data_processor.task2schema.items()}
    config.task_hidden_size = args.task_hidden_size

    logger.info("Training/evaluation parameters %s", args)
    logger.info("Training/evaluation config %s", config)

    # Training
    if args.do_train:
        args.output_dir = os.path.join(
            args.output_dir,
            "{}_{}_{}_{}_{:03d}_{:.1e}".format(
                args.model_type,
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                args.max_seq_length,
                "uncased" if args.do_lower_case else "cased",
                args.task_hidden_size,
                args.learning_rate,
            ),
        )
        if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                    args.output_dir
                )
            )

        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()
        model = MODEL_MAPPING[args.model_type].from_pretrained(
            args.pretrained_model if args.pretrained_model else args.model_name_or_path,
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        if args.local_rank == 0:
            torch.distributed.barrier()
        model.to(args.device)
        for n, p in model.named_parameters():
            logger.info("{} (size: {} requires_grad: {})".format(n, p.size(), p.requires_grad))

        global_step, training_loss = train(args, data_processor, model, tokenizer, role="train")
        logger.info("global_step = %s, average loss = %s", global_step, training_loss)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.local_rank in [-1, 0] and args.do_eval:
        checkpoints = [args.output_dir]
        if os.path.exists(os.path.join(args.output_dir, "checkpoint-best")):
            checkpoints.append(os.path.join(args.output_dir, "checkpoint-best"))
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            if checkpoint != args.output_dir and os.path.split(checkpoint)[-1].startswith("checkpoint-"):
                global_step = checkpoint.split("-")[-1]
            else:
                global_step = ""

            # Reload the model
            try:
                logger.info("Reload model from {}".format(checkpoint))
                config = AutoConfig.from_pretrained(checkpoint)
                model = MODEL_MAPPING[args.model_type].from_pretrained(checkpoint, config=config)
                model.to(args.device)
            except OSError:
                logger.warning("Model not found in {}".format(checkpoint))
                continue

            # Evaluate
            result = evaluate(args, data_processor, model, tokenizer, role="test", prefix=global_step)

            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

        logger.info("***** Final results *****")
        logger.info(json.dumps(results, ensure_ascii=False, indent=4))
        save_json(results, os.path.join(args.output_dir, "all_results.json"))

    if args.local_rank != -1:
        torch.distributed.barrier()
    logger.warning("Rank {} done!".format(args.local_rank))


if __name__ == "__main__":
    main()
