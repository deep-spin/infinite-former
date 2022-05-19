# coding: utf-8
import argparse
import time
import math
import os, sys
import itertools

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from data_utils import get_lm_corpus
from mem_transformer import MemTransformerLM
from utils.exp_utils import create_exp_dir
from utils.data_parallel import BalancedDataParallel

from itertools import islice, chain, repeat


parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--dataset', type=str, default='repeat_sequence')
parser.add_argument('--n_layer', type=int, default=6,
                    help='number of total layers')
parser.add_argument('--n_head', type=int, default=4,
                    help='number of heads')
parser.add_argument('--d_head', type=int, default=20,
                    help='head dimension')
parser.add_argument('--d_embed', type=int, default=-1,
                    help='embedding dimension')
parser.add_argument('--d_model', type=int, default=40,
                    help='model dimension')
parser.add_argument('--d_inner', type=int, default=80,
                    help='inner dimension in FF')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='global dropout rate')
parser.add_argument('--dropatt', type=float, default=0.0,
                    help='attention probability dropout rate')
parser.add_argument('--init', default='normal', type=str,
                    help='parameter initializer to use.')
parser.add_argument('--emb_init', default='normal', type=str,
                    help='parameter initializer to use.')
parser.add_argument('--init_range', type=float, default=0.1,
                    help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--emb_init_range', type=float, default=0.01,
                    help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--init_std', type=float, default=0.02,
                    help='parameters initialized by N(0, init_std)')
parser.add_argument('--proj_init_std', type=float, default=0.01,
                    help='parameters initialized by N(0, init_std)')
parser.add_argument('--optim', default='adam', type=str,
                    choices=['adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--lr', type=float, default=0.00025,
                    help='initial learning rate (0.00025|5 for adam|sgd)')
parser.add_argument('--mom', type=float, default=0.0,
                    help='momentum for sgd')
parser.add_argument('--scheduler', default='cosine', type=str,
                    choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant'],
                    help='lr scheduler to use.')
parser.add_argument('--warmup_step', type=int, default=0,
                    help='upper epoch limit')
parser.add_argument('--decay_rate', type=float, default=0.5,
                    help='decay factor when ReduceLROnPlateau is used')
parser.add_argument('--lr_min', type=float, default=0.0,
                    help='minimum learning rate during annealing')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--clip_nonemb', action='store_true',
                    help='only clip the gradient of non-embedding params')
parser.add_argument('--max_step', type=int, default=20000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch size')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='split batch into chunks to save memory')
parser.add_argument('--tgt_len', type=int, default=50,
                    help='number of tokens to predict')
parser.add_argument('--eval_tgt_len', type=int, default=50,
                    help='number of tokens to predict for evaluation')
parser.add_argument('--ext_len', type=int, default=0,
                    help='length of the extended context')
parser.add_argument('--mem_len', type=int, default=50,
                    help='length of the retained previous heads')
parser.add_argument('--not_tied', action='store_true',
                    help='do not tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--adaptive', action='store_true',
                    help='use adaptive softmax')
parser.add_argument('--div_val', type=int, default=1,
                    help='divident value for adapative input and softmax')
parser.add_argument('--pre_lnorm', action='store_true',
                    help='apply LayerNorm to the input instead of the output')
parser.add_argument('--varlen', action='store_true',
                    help='use variable length')
parser.add_argument('--multi_gpu', action='store_true',
                    help='use multiple GPU')
parser.add_argument('--log-interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--eval-interval', type=int, default=600,
                    help='evaluation interval')
parser.add_argument('--work_dir', default='LM-TFM', type=str,
                    help='experiment directory.')
parser.add_argument('--restart', action='store_true',
                    help='restart training from the saved checkpoint')
parser.add_argument('--restart_dir', type=str, default='',
                    help='restart dir')
parser.add_argument('--debug', action='store_true',
                    help='run in debug mode (do not create exp dir)')
parser.add_argument('--same_length', action='store_true',
                    help='use the same attn length for all tokens')
parser.add_argument('--attn_type', type=int, default=0,
                    help='attention type. 0 for ours, 1 for Shaw et al,'
                    '2 for Vaswani et al, 3 for Al Rfou et al.')
parser.add_argument('--clamp_len', type=int, default=-1,
                    help='use the same pos embeddings after clamp_len')
parser.add_argument('--eta_min', type=float, default=0.0,
                    help='min learning rate for cosine scheduler')
parser.add_argument('--gpu0_bsz', type=int, default=-1,
                    help='batch size on gpu 0')
parser.add_argument('--max_eval_steps', type=int, default=-1,
                    help='max eval steps')
parser.add_argument('--sample_softmax', type=int, default=-1,
                    help='number of samples in sampled softmax')
parser.add_argument('--patience', type=int, default=0,
                    help='patience')
parser.add_argument('--finetune_v2', action='store_true',
                    help='finetune v2')
parser.add_argument('--finetune_v3', action='store_true',
                    help='finetune v3')
parser.add_argument('--fp16', action='store_true',
                    help='Run in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument('--static-loss-scale', type=float, default=1,
                    help='Static loss scale, positive power of 2 values can '
                    'improve fp16 convergence.')
parser.add_argument('--dynamic-loss-scale', action='store_true',
                    help='Use dynamic loss scaling.  If supplied, this argument'
                    ' supersedes --static-loss-scale.')
parser.add_argument('--long_term_attention', action='store_true')
parser.add_argument('--long_term_attention_basis', type=int)
parser.add_argument('--long_term_attention_norm', type=str)
parser.add_argument('--infinite_memory', action='store_true')
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--eval', action='store_true')
parser.add_argument('--augment_len', type=int)
parser.add_argument('--mask', action='store_true')
parser.add_argument('--mask_type', type=str)
parser.add_argument('--kl_regularizer', action='store_true')
parser.add_argument('--sigma_0', type=float)
parser.add_argument('--mu_0', type=float)
parser.add_argument('--kl_m', type=float)
args = parser.parse_args()
args.tied = not args.not_tied

if args.d_embed < 0:
    args.d_embed = args.d_model

assert args.ext_len >= 0, 'extended context length must be non-negative'
assert args.batch_size % args.batch_chunk == 0

work_dir_eval = os.path.join(args.work_dir, args.name)
#args.work_dir = os.path.join(args.work_dir, args.dataset)
args.work_dir = os.path.join(args.work_dir, args.name)
logging = create_exp_dir(args.work_dir,
    scripts_to_save=['train.py', 'mem_transformer.py', 'long_term_attention.py'], debug=args.debug)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')
    else:
        torch.cuda.manual_seed_all(args.seed)

# Validate `--fp16` option
if args.fp16:
    if not args.cuda:
        print('WARNING: --fp16 requires --cuda, ignoring --fp16 option')
        args.fp16 = False
    else:
        try:
            from apex.fp16_utils import FP16_Optimizer
        except:
            print('WARNING: apex not installed, ignoring --fp16 option')
            args.fp16 = False

device = torch.device('cuda' if args.cuda else 'cpu')




###############################################################################
# Load data
###############################################################################
def load_sequences(data_type, vocab, tgt_len, pad, sep, batch_size=1):
    def chunk(it, size):
        it = iter(it)
        sentinel = ()
        return iter(lambda: tuple(islice(it, size)), sentinel)

    if batch_size>1:
        sequences=[]
        targets=[]
        sequence_batch=[]
        target_batch=[]
        with open(args.dataset+'_'+data_type+'.txt', 'r') as f:
            for line in f:
                line=line.replace('\n','')
                seq=[]
                tgt=[]
                aux=False
                aux_=False
                for i in line.split(' '):
                    if i!='':
                        seq.append(int(i))
                        if aux and aux_:
                            tgt.append(int(i))
                        elif aux_:
                            tgt.append(pad)
                        else:
                            aux_=True
                        if int(i)==sep:
                            aux=True

                sequence = list(chunk(seq[:-1],tgt_len))
                target = list(chunk(tgt,tgt_len))

                sequence_batch.append(sequence)
                target_batch.append(target)
                
                if len(sequence_batch)==batch_size:
                    sequences.append(sequence_batch)
                    targets.append(target_batch)
                    sequence_batch=[]
                    target_batch=[]
        
        return sequences, targets, vocab

    else:
        sequences=[]
        targets=[]
        with open(args.dataset+'_'+data_type+'.txt', 'r') as f:
            for line in f:
                line=line.replace('\n','')
                seq=[]
                tgt=[]
                aux=False
                aux_=False
                for i in line.split(' '):
                    if i!='':
                        seq.append(int(i))
                        if aux and aux_:
                            tgt.append(int(i))
                        elif aux_:
                            tgt.append(pad)
                        else:
                            aux_=True
                        if int(i)==sep:
                            aux=True

                sequence = list(chunk(seq[:-1],tgt_len))
                target = list(chunk(tgt,tgt_len))
                sequences.append(sequence)
                targets.append(target)

        for i in sequences:
            for v in i:
                for c in v:
                    if c not in vocab:
                        vocab.append(c)
        
        return sequences, targets, vocab

vocab = []
train_seq, train_tgt, vocab = load_sequences('train', vocab, args.tgt_len, pad=21, sep=20, batch_size=args.batch_size)
valid_seq, valid_tgt, vocab = load_sequences('valid', vocab, args.tgt_len, pad=21, sep=20, batch_size=1)
test_seq, test_tgt, vocab = load_sequences('test', vocab, args.tgt_len, pad=21, sep=20, batch_size=1)


vocab = torch.arange(0,22,1).tolist()
ntokens = len(vocab)
args.n_token = ntokens

###############################################################################
# Build the model
###############################################################################
def init_weight(weight):
    if args.init == 'uniform':
        nn.init.uniform_(weight, -args.init_range, args.init_range)
    elif args.init == 'normal':
        nn.init.normal_(weight, 0.0, args.init_std)

def init_bias(bias):
    nn.init.constant_(bias, 0.0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, args.proj_init_std)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.cluster_weight)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, args.proj_init_std)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, args.init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('TransformerLM') != -1:
        if hasattr(m, 'r_emb'):
            init_weight(m.r_emb)
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias)
        if hasattr(m, 'r_bias'):
            init_bias(m.r_bias)

def update_dropout(m):
    classname = m.__class__.__name__
    if classname.find('Dropout') != -1:
        if hasattr(m, 'p'):
            m.p = args.dropout

def update_dropatt(m):
    if hasattr(m, 'dropatt'):
        m.dropatt.p = args.dropatt

if args.eval:
    with open(os.path.join(work_dir_eval, 'model.pt'), 'rb') as f:
        model = torch.load(f)
    model.backward_compatible()
    model = model.to(device)
elif args.restart:
    with open(os.path.join(args.restart_dir, 'model.pt'), 'rb') as f:
        model = torch.load(f)
    if not args.fp16:
        model = model.float()
    model.apply(update_dropout)
    model.apply(update_dropatt)
else:
    model = MemTransformerLM(ntokens, args.n_layer, args.n_head, args.d_model,
        args.d_head, args.d_inner, args.dropout, args.dropatt,
        tie_weight=args.tied, d_embed=args.d_embed, div_val=args.div_val,
        pre_lnorm=args.pre_lnorm, tgt_len=args.tgt_len, eval_tgt_len=args.eval_tgt_len,
        ext_len=args.ext_len, mem_len=args.mem_len,
        same_length=args.same_length, attn_type=args.attn_type,
        clamp_len=args.clamp_len, sample_softmax=args.sample_softmax, 
        continuous_attention=args.continuous_attention, long_term_attention=args.long_term_attention,
        long_term_attention_basis=args.long_term_attention_basis, long_term_attention_norm=args.long_term_attention_norm,
        nfinite_memory=args.infinite_memory,augment_len=args.augment_len,
        mask=args.mask, mask_type=args.mask_type,kl_regularizer=args.kl_regularizer,
        sigma_0=args.sigma_0, mu_0=args.mu_0,)

    model.apply(weights_init)
    model.word_emb.apply(weights_init) # ensure embedding init is not overridden by out_layer in case of weight sharing
args.n_all_param = sum([p.nelement() for p in model.parameters()])
args.n_nonemb_param = sum([p.nelement() for p in model.layers.parameters()])

if args.fp16:
    model = model.half()

if args.multi_gpu:
    model = model.to(device)
    if args.gpu0_bsz >= 0:
        para_model = BalancedDataParallel(args.gpu0_bsz // args.batch_chunk,
                                          model, dim=1).to(device)
    else:
        para_model = nn.DataParallel(model, dim=1).to(device)
else:
    para_model = model.to(device)

#### optimizer
if args.optim.lower() == 'sgd':
    if args.sample_softmax > 0:
        dense_params, sparse_params = [], []
        for param in model.parameters():
            if param.size() == model.word_emb.weight.size():
                sparse_params.append(param)
            else:
                dense_params.append(param)
        optimizer_sparse = optim.SGD(sparse_params, lr=args.lr * 2)
        optimizer = optim.SGD(dense_params, lr=args.lr, momentum=args.mom)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
            momentum=args.mom)
elif args.optim.lower() == 'adam':
    if args.sample_softmax > 0:
        dense_params, sparse_params = [], []
        for param in model.parameters():
            if param.size() == model.word_emb.weight.size():
                sparse_params.append(param)
            else:
                dense_params.append(param)
        optimizer_sparse = optim.SparseAdam(sparse_params, lr=args.lr)
        optimizer = optim.Adam(dense_params, lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optim.lower() == 'adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)

#### scheduler
if args.scheduler == 'cosine':
    # here we do not set eta_min to lr_min to be backward compatible
    # because in previous versions eta_min is default to 0
    # rather than the default value of lr_min 1e-6
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
        args.max_step, eta_min=args.eta_min) # should use eta_min arg
    if args.sample_softmax > 0:
        scheduler_sparse = optim.lr_scheduler.CosineAnnealingLR(optimizer_sparse,
            args.max_step, eta_min=args.eta_min) # should use eta_min arg
elif args.scheduler == 'inv_sqrt':
    # originally used for Transformer (in Attention is all you need)
    def lr_lambda(step):
        # return a multiplier instead of a learning rate
        if step == 0 and args.warmup_step == 0:
            return 1.
        else:
            return 1. / (step ** 0.5) if step > args.warmup_step \
                   else step / (args.warmup_step ** 1.5)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
elif args.scheduler == 'dev_perf':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        factor=args.decay_rate, patience=args.patience, min_lr=args.lr_min)
    if args.sample_softmax > 0:
        scheduler_sparse = optim.lr_scheduler.ReduceLROnPlateau(optimizer_sparse,
            factor=args.decay_rate, patience=args.patience, min_lr=args.lr_min)
elif args.scheduler == 'constant':
    pass

if args.cuda and args.fp16:
    # If args.dynamic_loss_scale is False, static_loss_scale will be used.
    # If args.dynamic_loss_scale is True, it will take precedence over static_loss_scale.
    optimizer = FP16_Optimizer(optimizer,
                               static_loss_scale = args.static_loss_scale,
                               dynamic_loss_scale = args.dynamic_loss_scale,
                               dynamic_loss_args = {'init_scale': 2 ** 16})

if args.restart:
    if os.path.exists(os.path.join(args.restart_dir, 'optimizer.pt')):
        with open(os.path.join(args.restart_dir, 'optimizer.pt'), 'rb') as f:
            opt_state_dict = torch.load(f)
            optimizer.load_state_dict(opt_state_dict)
    else:
        print('Optimizer was not saved. Start from scratch.')

logging('=' * 100)
for k, v in args.__dict__.items():
    logging('    - {} : {}'.format(k, v))
logging('=' * 100)
logging('#params = {}'.format(args.n_all_param))
logging('#non emb params = {}'.format(args.n_nonemb_param))


###############################################################################
# Training code
###############################################################################

def evaluate(eval_iter, eval_tgt):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    pad=21
    # If the model does not use memory at all, make the ext_len longer.
    # Otherwise, make the mem_len longer and keep the ext_len the same.
    if args.mem_len == 0:
        model.reset_length(args.eval_tgt_len,
            args.ext_len+args.tgt_len-args.eval_tgt_len, args.mem_len)
    else:
        model.reset_length(args.eval_tgt_len,
            args.ext_len, args.mem_len+args.tgt_len-args.eval_tgt_len)

    # Evaluation
    total_len, total_loss, right, mem_len, mem_right = 0, 0., 0, 0, 0
    with torch.no_grad():
        mems = tuple()
        for v in range(len(eval_iter)):
            mems = tuple()

            for j in range(len(eval_iter[v])):
                d = eval_iter[v][j]
                tgt = eval_tgt[v][j]

                if j == len(eval_iter[v])-1:
                    final=True
                else:
                    final=False

                if not args.kl_regularizer:
                    ret, pred_hid, target = model(torch.tensor(d).cuda().unsqueeze(1), 
                                            torch.tensor(tgt).cuda().unsqueeze(1), *mems, doc_final=final)
                else:
                    ret, pred_hid, target,_ = model(torch.tensor(d).cuda().unsqueeze(1), 
                                            torch.tensor(tgt).cuda().unsqueeze(1), *mems, doc_final=final)

                loss, mems = ret[0], ret[1:]
                seq_len = len(tgt) - tgt.count(pad)
                total_loss += seq_len * loss.float().item()
                total_len += seq_len

                pred = torch.softmax(pred_hid,-1)
                pred = pred.argmax(-1)

                for i in range(len(target)):
                    if target[i] != pad:
                        if target[i]==pred[i]:
                            right+=1
                        
    # Switch back to the training mode
    model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
    model.train()

    return right / total_len


def train(train_iter, train_tgt, va_iter, valid_tgt):
    # Turn on training mode which enables dropout.
    pad=21
    global train_step, train_loss, best_acc, eval_start_time, log_start_time, kl_loss
    model.train()

    mems = tuple()

    breaking=False
    while True:
        if breaking:
            break
        for batch in range(len(train_iter)):
            model.zero_grad()
            mems = tuple()

            right, total_len, mem_len, mem_right = 0,0,0,0
            train_step += 1

            if args.batch_size>1:
                steps=len(train_iter[batch][0])
            else:
                steps=len(train_iter[batch])

            for j in range(steps):

                if j == steps-1:
                    final=True
                else:
                    final=False

                if not final:
                    with torch.no_grad():

                        if args.batch_size>1:
                            d = [train_iter[batch][v][j] for v in range(len(train_iter[batch]))]
                            tgt = [train_tgt[batch][v][j] for v in range(len(train_iter[batch]))]

                            if not args.kl_regularizer:
                                ret, pred_hid, target = para_model(torch.tensor(d).permute(1,0).cuda(), torch.tensor(tgt).permute(1,0).cuda(), *mems, doc_final=final)
                            else:
                                ret, pred_hid, target, kl_reg = para_model(torch.tensor(d).permute(1,0).cuda(), torch.tensor(tgt).permute(1,0).cuda(), *mems, doc_final=final)
                        else:
                            d = train_iter[batch][j]
                            tgt = train_tgt[batch][j]
                            ret, pred_hid, target = para_model(torch.tensor(d).cuda().unsqueeze(1), torch.tensor(tgt).cuda().unsqueeze(1), *mems, doc_final=final)

                else:
                    if args.batch_size>1:
                            d = [train_iter[batch][v][j] for v in range(len(train_iter[batch]))]
                            tgt = [train_tgt[batch][v][j] for v in range(len(train_iter[batch]))]

                            if not args.kl_regularizer:
                                ret, pred_hid, target = para_model(torch.tensor(d).permute(1,0).cuda(), torch.tensor(tgt).permute(1,0).cuda(), *mems, doc_final=final)
                            else:
                                ret, pred_hid, target, kl_reg = para_model(torch.tensor(d).permute(1,0).cuda(), torch.tensor(tgt).permute(1,0).cuda(), *mems, doc_final=final)
                    else:
                        d = train_iter[batch][j]
                        tgt = train_tgt[batch][j]
                        ret, pred_hid, target = para_model(torch.tensor(d).cuda().unsqueeze(1), torch.tensor(tgt).cuda().unsqueeze(1), *mems, doc_final=final)



                loss, mems = ret[0], ret[1:]

                if final:
                    if args.kl_regularizer and kl_reg is not None:
                        (loss+args.kl_m*kl_reg.float().mean().type_as(loss)).backward()
                        kl_loss+=kl_reg.float().mean().item()
                    else:
                        loss.backward()

                train_loss += loss.float().item()

                pred = torch.softmax(pred_hid,-1)
                pred = pred.argmax(-1)

                for i in range(len(target)):
                    if target[i] != pad:
                        if target[i]==pred[i]:
                            right+=1
                        
                        total_len += 1
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

                optimizer.step()
                if args.sample_softmax > 0:
                    optimizer_sparse.step()

                # step-wise learning rate annealing
                
                if args.scheduler in ['cosine', 'constant', 'dev_perf']:
                    # linear warmup stage
                    if train_step < args.warmup_step:
                        curr_lr = args.lr * train_step / args.warmup_step
                        optimizer.param_groups[0]['lr'] = curr_lr
                        if args.sample_softmax > 0:
                            optimizer_sparse.param_groups[0]['lr'] = curr_lr * 2
                    elif train_step%(60*4)==0:
                        if args.scheduler == 'cosine':
                            scheduler.step(train_step)
                            if args.sample_softmax > 0:
                                scheduler_sparse.step(train_step)
                elif args.scheduler == 'inv_sqrt':
                    scheduler.step(train_step)

            if train_step % args.log_interval == 0:
                cur_loss = train_loss / args.log_interval
                cur_kl_loss = kl_loss / args.log_interval
                elapsed = time.time() - log_start_time
                if args.kl_regularizer:
                    log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.3g} ' \
                          '| ms/batch {:5.2f} | loss {:5.2f} | kl {:5.2f} '.format(
                    epoch, train_step, batch+1, optimizer.param_groups[0]['lr'],
                    elapsed * 1000 / args.log_interval, cur_loss, cur_kl_loss)
                else:
                    log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.3g} ' \
                          '| ms/batch {:5.2f} | loss {:5.2f}'.format(
                    epoch, train_step, batch+1, optimizer.param_groups[0]['lr'],
                    elapsed * 1000 / args.log_interval, cur_loss)
                if total_len>0:
                    log_str += ' | acc {:9.3f}'.format(right/total_len)
                    
                logging(log_str)
                train_loss = 0
                kl_loss = 0
                log_start_time = time.time()

            if train_step % args.eval_interval == 0:
                acc = evaluate(va_iter, valid_tgt)
                logging('-' * 100)
                log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' .format(
                    train_step // args.eval_interval, train_step,
                    (time.time() - eval_start_time))
                
                log_str += ' | valid acc {:9.3f}'.format(acc)
                
                logging(log_str)
                logging('-' * 100)
                # Save the model if the validation loss is the best we've seen so far.
                if not best_acc or acc > best_acc:
                    if not args.debug:
                        with open(os.path.join(args.work_dir, 'model.pt'), 'wb') as f:
                            torch.save(model, f)
                        with open(os.path.join(args.work_dir, 'optimizer.pt'), 'wb') as f:
                            torch.save(optimizer.state_dict(), f)
                    best_acc = acc

                # dev-performance based learning rate annealing
                if args.scheduler == 'dev_perf':
                    scheduler.step(val_loss)
                    if args.sample_softmax > 0:
                        scheduler_sparse.step(val_loss)

                eval_start_time = time.time()

            if train_step == args.max_step:
                breaking=True
                break

# Loop over epochs.
train_step = 0
train_loss = 0
kl_loss = 0
best_acc = None

log_start_time = time.time()
eval_start_time = time.time()

# At any point you can hit Ctrl + C to break out of training early.
if args.eval:
    acc = evaluate(test_seq, test_tgt)
else:
    try:
        for epoch in itertools.count(start=1):
            train(train_seq, train_tgt, valid_seq, valid_tgt)
            if train_step == args.max_step:
                logging('-' * 100)
                logging('End of training')
                break
    except KeyboardInterrupt:
        logging('-' * 100)
        logging('Exiting from training early')

# Load the best saved model.
with open(os.path.join(args.work_dir, 'model.pt'), 'rb') as f:
    model = torch.load(f)
para_model = model.to(device)

# Run on test data.
test_acc = evaluate(test_seq, test_tgt)

logging('=' * 100)
logging('| End of training | test acc {:9.3f}'.format(test_acc))
logging('=' * 100)
