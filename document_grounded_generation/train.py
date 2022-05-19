# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import os
import math
import logging
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data as data

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from transformers import (AdamW, GPT2LMHeadModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME)

from utils import get_dataset, make_logdir

SPECIAL_TOKENS = ["<bos>", "<eos>", "<doc>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<speaker1>', '<speaker2>','<doc>']}
MODEL_INPUTS = ["input_ids", "lm_labels" "token_type_ids"]

logger = logging.getLogger(__file__)


def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -100] * (max_l - len(x)) for x in dataset[name]]
    return dataset


def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

def build_input_from_segments(dialog, tokenizer, valid=False, baseline=False, baseline_nodoc=False): 
    bos, eos, doc, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    instance = {}
    sequence = [[doc] +  dialog[0] + [eos]]
    label_sequence = [[-100] + [-100]*len(dialog[0]) + [-100] ]
    del dialog[0]
    sequence = list(chain(*sequence))
    if valid:
        label_sequence = list(chain(*label_sequence))

    instance["token_type_ids"] = [doc for _ in sequence]

    if baseline:
        sequence=sequence[:-512]
        label_sequence=label_sequence[:-512]
        instance["token_type_ids"] = instance["token_type_ids"][:-512]

    if baseline_nodoc:
        sequence=[]
        label_sequence=[]
        instance["token_type_ids"] = []
    
    for i in range(len(dialog)):
        sequence =  [sequence] + [[speaker2 if i%2==0 else speaker1]]
        if valid:
            label_sequence = [label_sequence] + [[-100]]
        
        sequence[-1].extend(dialog[i])
        if valid:
            label_sequence[-1].extend(dialog[i])

            
        if i %2==0:
            instance["token_type_ids"] += [speaker2] *(len(dialog[i])+1)
        else:
            instance["token_type_ids"] += [speaker1] *(len(dialog[i])+1)

        sequence = list(chain(*sequence))
        if valid:
            label_sequence = list(chain(*label_sequence))

        if baseline and i==0:
            sequence=sequence[:-512]
            label_sequence=label_sequence[:-512]
            instance["token_type_ids"] = instance["token_type_ids"][:-512]

    instance["input_ids"] = sequence
    if valid:
        instance['lm_labels'] = label_sequence
    else:
        instance['lm_labels'] = sequence

    return instance


class DoGDataset(data.Dataset):
    def __init__(self, data, max_len, baseline):

        self.data=data
        self.max_len = max_len
        self.baseline = baseline

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, idx):
        

        inputs, token_ids, lm_labels = self.data['input_ids'][idx], self.data['token_type_ids'][idx], self.data['lm_labels'][idx]
        inputs_, token_ids_, lm_labels_ = [], [], []
        while len(inputs)>0:
            inputs_.append(torch.LongTensor(inputs[-self.max_len:]))
            token_ids_.append(torch.LongTensor(token_ids[-self.max_len:]))
            lm_labels_.append(torch.LongTensor(lm_labels[-self.max_len:]))
            inputs = inputs[:-self.max_len]
            token_ids = token_ids[:-self.max_len]
            lm_labels = lm_labels[:-self.max_len]

        return inputs_[::-1], token_ids_[::-1], lm_labels_[::-1]
        

        

def get_data_loaders(args, tokenizer):
    """ Prepare the dataset for training and evaluation """
    chat_dataset={}
    chat_dataset['train'] = get_dataset(tokenizer, args.dataset_path, args.dataset_cache,'train', args.script, args.script_doc)
    chat_dataset['valid'] = get_dataset(tokenizer, args.dataset_path, args.dataset_cache,'valid', args.script, args.script_doc)

    
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    for dataset_name, dataset in chat_dataset.items():
        for dialog in dataset:
            if dataset_name=='valid':
                instance = build_input_from_segments(dialog, tokenizer, valid=True, baseline=args.baseline, baseline_nodoc=args.baseline_nodoc)
            else:
                instance = build_input_from_segments(dialog, tokenizer, valid=False, baseline=args.baseline, baseline_nodoc=args.baseline_nodoc)
            for input_name, input_array in instance.items():
                datasets[dataset_name][input_name].append(input_array)

    logger.info("Build data loaders")


    train_dataset = DoGDataset(datasets['train'], max_len=args.max_len, baseline=args.baseline)
    valid_dataset = DoGDataset(datasets['valid'], max_len=args.max_len, baseline=args.baseline)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.valid_batch_size, shuffle=False)
    
    return train_loader, valid_loader


def train():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument('--script', action='store_true')
    parser.add_argument('--script_doc', action='store_true')
    parser.add_argument('--baseline_nodoc', action='store_true')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument("--model_checkpoint", type=str, default="openai-gpt", help="Path, url or short name of the model")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=1, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument('--kl_regularizer', action='store_true')
    parser.add_argument('--kl_m', type=float, default=.00001)
    parser.add_argument('--lr_ltm', type=float, default=.0)
    parser.add_argument('--name', type=str, default="")

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))

    logger.info("Prepare tokenizer, pretrained model and optimizer.")
    tokenizer_class = GPT2Tokenizer
    tokenizer = tokenizer_class.from_pretrained('gpt2')


    model_class = GPT2LMHeadModel
    model = model_class.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    # Add special tokens if they are not already added
    add_special_tokens_(model, tokenizer)
    if args.lr_ltm==0:
        optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    else:
        
        optimizer_grouped_parameters = [
                {"params": [p for n, p in model.named_parameters() if 'long_term_attention' not in n],},
                {"params": [p for n, p in model.named_parameters() if 'long_term_attention' in n],'lr': args.lr_ltm}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=True)
        
    logger.info("Prepare datasets")
    train_loader, val_loader = get_data_loaders(args, tokenizer)

    # Training function and trainer
    def update(engine, batch):
        model.train()
        input_ids, token_type_ids, lm_labels = batch[0], batch[1], batch[2]
        #print(engine.state.epoch)
        #if engine.state.epoch>1:
        for i in range(len(input_ids)):
            if input_ids[i].size(-1)>1:
                if i==1:
                    new_doc=True
                else:
                    new_doc=False
                if i==len(input_ids)-1:
                    last_doc=True
                else:
                    last_doc=False
                    
                if args.kl_regularizer:
                    outputs, kl_loss = model(input_ids[i].to(args.device), token_type_ids=token_type_ids[i].to(args.device),
                                labels=lm_labels[i].to(args.device), new_doc=new_doc, last_doc=last_doc)
                else:
                    outputs = model(input_ids[i].to(args.device), token_type_ids=token_type_ids[i].to(args.device),
                                labels=lm_labels[i].to(args.device), new_doc=new_doc, last_doc=last_doc)

                lm_loss = outputs['loss']
                
                if args.kl_regularizer and kl_loss is not None:
                    loss = (lm_loss+args.kl_m*kl_loss.mean()) / args.gradient_accumulation_steps #/ len(input_ids)
                else:
                    loss = (lm_loss) / args.gradient_accumulation_steps #/ len(input_ids)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        #else:
        #    lm_loss=torch.tensor(0)
        
        #for param_group in optimizer.param_groups:
        #    print(param_group['lr'])
        return (lm_loss / args.gradient_accumulation_steps).item()
    
    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        lm_logits_flat_shifted=[]
        with torch.no_grad():
            input_ids, token_type_ids, lm_labels = batch[0], batch[1], batch[2]

            for i in range(len(input_ids)):
                if input_ids[i].size(-1)>1:
                    if i==1:
                        new_doc=True
                    else:
                        new_doc=False
                    if i==len(input_ids)-1:
                        last_doc=True
                    else:
                        last_doc=False
                    if args.kl_regularizer:
                        output,_ = model(input_ids[i].to(args.device), token_type_ids=token_type_ids[i].to(args.device),
                                    new_doc=new_doc, last_doc=last_doc)
                    else:
                        output = model(input_ids[i].to(args.device), token_type_ids=token_type_ids[i].to(args.device),
                                    new_doc=new_doc, last_doc=last_doc)
                    lm_logits=output['logits']
                    if len(lm_logits_flat_shifted)==0:
                        lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1)).cpu()
                        lm_labels_flat_shifted = lm_labels[i][..., 1:].contiguous().view(-1).cpu() #to(args.device)
                    else:
                        lm_logits_flat_shifted = torch.cat([lm_logits_flat_shifted, 
                                                            lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1)).cpu()],0)
                        lm_labels_flat_shifted = torch.cat([lm_labels_flat_shifted,
                                                            lm_labels[i][..., 1:].contiguous().view(-1).cpu()],0)
                        
        return lm_logits_flat_shifted.unsqueeze(0), lm_labels_flat_shifted.unsqueeze(0)
    
    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))


    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)],param_group_index=0)
    if args.lr_ltm!=0:
        scheduler_ltm = PiecewiseLinear(optimizer, "lr", [(0, args.lr_ltm), (args.n_epochs * len(train_loader), 0.0)],param_group_index=-1)
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler_ltm)

    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-100), output_transform=lambda x: (x[0][0], x[1][0])),
               "accuracy": Accuracy(output_transform=lambda x: (x[0][0], x[1][0]))}
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args),
                    "average_accuracy": MetricsLambda(average_distributed_scalar, metrics["accuracy"], args)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        log_dir = make_logdir(args.name)
        #tb_logger = TensorboardLogger(log_dir)

        #tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        #tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        #tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), global_step_transform=global_step_from_engine(trainer)), event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(log_dir, 'checkpoint', save_interval=1, n_saved=3)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" takes care of distributed encapsulation

        torch.save(args, log_dir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(log_dir, CONFIG_NAME))
        tokenizer.save_pretrained(log_dir)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(os.path.join(log_dir, checkpoint_handler._saved[-1][1]), os.path.join(log_dir, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        #tb_logger.close()

if __name__ == "__main__":
    train()
