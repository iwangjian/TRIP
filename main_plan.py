# -*- coding: utf-8 -*-
import argparse
import os
import sys
import logging
import json
import numpy as np
import random
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from model.model_trip import TRIP
from utils.dataset import PlanDataset
from utils.data_utils import get_tokenizer, convert_ids_to_tokens
from utils.data_utils import get_eval_output, get_plan_path
from utils.data_collator import PlanCollator
from utils.trainer import Trainer

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
    handlers = [
        logging.StreamHandler(sys.stdout)
    ]
)

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"])
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--use_gpu', type=str2bool, default="True")

    # data config
    parser.add_argument('--train_data', type=str, default=None)
    parser.add_argument('--dev_data', type=str, default=None)
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--bert_dir', type=str, default="config/bert-base-chinese")
    parser.add_argument('--cache_dir', type=str, default="caches/plan/")
    parser.add_argument('--log_dir', type=str, default="logs/plan/")
    
    # training args
    parser.add_argument('--load_checkpoint', type=str, default=None)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--log_steps', type=int, default=200)
    parser.add_argument('--validate_steps', type=int, default=1000)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--turn_type_size', type=int, default=16)
    parser.add_argument('--num_negatives', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warm_up_ratio', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--max_position_embeddings', type=int, default=512)
    parser.add_argument('--share_embedding', type=str2bool, default="False")
    parser.add_argument('--scale_embedding', type=str2bool, default="True")
    parser.add_argument('--decoder_layers', type=int, default=6)
    parser.add_argument('--decoder_attention_heads', type=int, default=8)
    parser.add_argument('--activation_function', type=str, default="gelu")
    parser.add_argument('--decoder_ffn_dim', type=int, default=3072)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--decoder_layerdrop', type=float, default=0.1)
    parser.add_argument('--activation_dropout', type=float, default=0.1)
    parser.add_argument('--attention_dropout', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--init_std', type=float, default=0.02)

    # decoding args
    parser.add_argument('--infer_checkpoint', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default="outputs/plan/")
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--max_dec_len', type=int, default=80)
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--lambda_scale', type=float, default=0.1)
    parser.add_argument('--min_length', type=int, default=1)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--diversity_penalty', type=float, default=0.0)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0)
    
    return parser.parse_args()

def str2bool(v):
    if v.lower() in ('true', 'yes', 't', 'y', '1'):
        return True
    elif v.lower() in ('false',' no', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")

def print_args(args):
    print("=============== Args ===============")
    for k in vars(args):
        print("%s: %s" % (k, vars(args)[k]))

def set_seed(args):
    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)

def run_train(args):
    logging.info("=============== Training ===============")
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer, num_added_tokens, token_id_dict = get_tokenizer(config_dir=args.bert_dir)
    args.vocab_size = len(tokenizer)
    args.pad_token_id = token_id_dict["pad_token_id"]
    args.bos_token_id = token_id_dict["bos_token_id"]
    args.eos_token_id = token_id_dict["eos_token_id"]
    logging.info("{}: Add {} additional special tokens.".format(type(tokenizer).__name__, num_added_tokens))
    
    # define dataset
    train_dataset = PlanDataset(data_path=args.train_data, data_partition="train",
        tokenizer=tokenizer, cache_dir=args.cache_dir, max_seq_len=args.max_seq_len, 
        turn_type_size=args.turn_type_size, num_negatives=args.num_negatives)
    dev_dataset = PlanDataset(data_path=args.dev_data, data_partition="dev",
        tokenizer=tokenizer, cache_dir=args.cache_dir, max_seq_len=args.max_seq_len, 
        turn_type_size=args.turn_type_size, num_negatives=args.num_negatives)

    # create dataloader
    collator = PlanCollator(device=device, padding_idx=args.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator.custom_collate)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size // 2, shuffle=False, collate_fn=collator.custom_collate)

    # build model
    if args.load_checkpoint is not None:
        model = torch.load(args.load_checkpoint)
    else:
        model = TRIP(args=args)
    model.to(device)
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Total parameters: {}\tTrainable parameters: {}".format(total_num, trainable_num))
    
    # build trainer and execute model training
    trainer = Trainer(model=model, train_loader=train_loader, dev_loader=dev_loader,
        log_dir=args.log_dir, log_steps=args.log_steps, validate_steps=args.validate_steps, 
        num_epochs=args.num_epochs, lr=args.lr, warm_up_ratio=args.warm_up_ratio,
        weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm
    )
    trainer.train()


def run_test(args):
    logging.info("=============== Testing ===============")
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    if args.infer_checkpoint is not None:
        model_path = os.path.join(args.log_dir, args.infer_checkpoint)
    else:
        model_path = os.path.join(args.log_dir, "best_model.bin")
    model = torch.load(model_path)
    logging.info("Model loaded from [{}]".format(model_path))
    model.to(device)
    model.eval()

    tokenizer, _, token_id_dict = get_tokenizer(config_dir=args.bert_dir)
    args.pad_token_id = token_id_dict["pad_token_id"]

    # load data
    data_partition = "test"
    if args.test_data.endswith("test_unseen.json"):
        data_partition = "test_unseen"
    elif args.test_data.endswith("test_seen.json"):
        data_partition = "test_seen"

    test_dataset = PlanDataset(data_path=args.test_data, data_partition=data_partition,
        tokenizer=tokenizer, cache_dir=args.cache_dir, max_seq_len=args.max_seq_len,
        turn_type_size=args.turn_type_size, is_test=True)
    collator = PlanCollator(device=device, padding_idx=args.pad_token_id)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn=collator.custom_collate)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    output_prefix = model_path.split('/')[-1].replace(".bin", "_%s.txt" % data_partition)
    output_path = os.path.join(args.output_dir, output_prefix)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for inputs in tqdm(test_loader):
            with torch.no_grad():
                outputs = model.generate(args, inputs)
                # post-process
                backward_plans = convert_ids_to_tokens(outputs["backward_plan"], tokenizer)
                beam_backward_plans = convert_ids_to_tokens(outputs["beam_backward_plan"], tokenizer)
                #beam_forward_plans = convert_ids_to_tokens(outputs["beam_forward_plan"], tokenizer)
                assert len(backward_plans) == len(beam_backward_plans)
                
                for bp, beam_bp in zip(backward_plans, beam_backward_plans):
                    action, topic = get_eval_output(bp, backward=True)
                    plan_path = get_plan_path(bp, backward=True)
                    plan = {
                        "action": action,
                        "topic": topic,
                        "plan_path": plan_path,
                        "backward_plan": bp
                    }
                    line = json.dumps(plan, ensure_ascii=False)
                    f.write(line + "\n")
                    f.flush()
    logging.info("Saved output to [{}]".format(output_path))


if __name__ == "__main__":
    args = parse_config()
    print_args(args)
    set_seed(args)
    
    if args.mode == "train":
        run_train(args)
    elif args.mode == "test":
        run_test(args)
    else:
        exit("Please specify the \"mode\" parameter!")
