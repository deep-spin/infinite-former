import random
import logging
from pprint import pformat
from collections import defaultdict, Counter
from functools import partial
from tqdm import trange
from argparse import ArgumentParser
from itertools import chain
import string

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data as data
from transformers import (GPT2LMHeadModel, GPT2Tokenizer)

from train import SPECIAL_TOKENS, add_special_tokens_
from utils import get_dataset

from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar

import re
import math
import numpy as np

from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
import nltk.translate.meteor_score as meteor_score
try:
    from gensim.models import KeyedVectors
except ImportError:
    from gensim.models import Word2Vec as KeyedVectors

import pickle

logger = logging.getLogger(__file__)


class Embedding(object):
    def __init__(self):
        self.m = KeyedVectors.load('glove.6B.300d.model.bin', mmap='r')
        try:
            self.unk = self.m.vectors.mean(axis=0)
        except AttributeError:
            self.unk = self.m.syn0.mean(axis=0)

    @property
    def w2v(self):
        return np.concatenate((self.m.syn0, self.unk[None,:]), axis=0)

    def __getitem__(self, key):
        try:
            return self.m.vocab[key].index
        except KeyError:
            return len(self.m.syn0)

    def vec(self, key):
        try:
            vectors = self.m.vectors
        except AttributeError:
            vectors = self.m.syn0
        try:
            return vectors[self.m.vocab[key].index]
        except KeyError:
            return self.unk


def eval_emb_metrics(hypothesis, references, emb=None, metrics_to_omit=None):
    from sklearn.metrics.pairwise import cosine_similarity
    from nltk.tokenize import word_tokenize
    import numpy as np
    if emb is None:
        emb = Embedding()

    if metrics_to_omit is None:
        metrics_to_omit = set()
    else:
        if 'EmbeddingAverageCosineSimilairty' in metrics_to_omit:
            metrics_to_omit.remove('EmbeddingAverageCosineSimilairty')
            metrics_to_omit.add('EmbeddingAverageCosineSimilarity')

    emb_hyps = []
    avg_emb_hyps = []
    extreme_emb_hyps = []
    for hyp in hypothesis:
        embs = [emb.vec(word) for word in word_tokenize(hyp)]

        avg_emb = np.sum(embs, axis=0) / np.linalg.norm(np.sum(embs, axis=0))
        if not np.any(np.isnan(avg_emb)):

            maxemb = np.max(embs, axis=0)
            minemb = np.min(embs, axis=0)
            extreme_emb = list(map(lambda x, y: x if ((x>y or x<-y) and y>0) or ((x<y or x>-y) and y<0) else y, maxemb, minemb))

            emb_hyps.append(embs)
            avg_emb_hyps.append(avg_emb)
            extreme_emb_hyps.append(extreme_emb)

    emb_refs = []
    avg_emb_refs = []
    extreme_emb_refs = []
    for refsource in references:
        emb_refsource = []
        avg_emb_refsource = []
        extreme_emb_refsource = []
        for ref in refsource:
            embs = [emb.vec(word) for word in word_tokenize(ref)]
            avg_emb = np.sum(embs, axis=0) / np.linalg.norm(np.sum(embs, axis=0))
            if not np.any(np.isnan(avg_emb)):

                maxemb = np.max(embs, axis=0)
                minemb = np.min(embs, axis=0)
                extreme_emb = list(map(lambda x, y: x if ((x>y or x<-y) and y>0) or ((x<y or x>-y) and y<0) else y, maxemb, minemb))

                emb_refsource.append(embs)
                avg_emb_refsource.append(avg_emb)
                extreme_emb_refsource.append(extreme_emb)
        emb_refs.append(emb_refsource)
        avg_emb_refs.append(avg_emb_refsource)
        extreme_emb_refs.append(extreme_emb_refsource)

    rval = []
    if 'EmbeddingAverageCosineSimilarity' not in metrics_to_omit:
        cos_similarity = list(map(lambda refv: cosine_similarity(refv, avg_emb_hyps).diagonal(), avg_emb_refs))
        cos_similarity = np.max(cos_similarity, axis=0).mean()
        rval.append("EmbeddingAverageCosineSimilarity: %0.6f" % (cos_similarity))
        # For backwards compatibility with an old typo before Nov 20, 2019.
        rval.append("EmbeddingAverageCosineSimilairty: %0.6f" % (cos_similarity))

    if 'VectorExtremaCosineSimilarity' not in metrics_to_omit:
        cos_similarity = list(map(lambda refv: cosine_similarity(refv, extreme_emb_hyps).diagonal(), extreme_emb_refs))
        cos_similarity = np.max(cos_similarity, axis=0).mean()
        rval.append("VectorExtremaCosineSimilarity: %0.6f" % (cos_similarity))

    if 'GreedyMatchingScore' not in metrics_to_omit:
        scores = []
        for emb_refsource in emb_refs:
            score_source = []
            for emb_ref, emb_hyp in zip(emb_refsource, emb_hyps):
                simi_matrix = cosine_similarity(emb_ref, emb_hyp)
                dir1 = simi_matrix.max(axis=0).mean()
                dir2 = simi_matrix.max(axis=1).mean()
                score_source.append((dir1 + dir2) / 2)
            scores.append(score_source)
        scores = np.max(scores, axis=0).mean()
        rval.append("GreedyMatchingScore: %0.6f" % (scores))

    rval = "\n".join(rval)
    return rval



def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()

def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


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
    def __init__(self, data, max_len):

        self.data=data
        self.max_len = max_len

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
    chat_dataset['test'] = get_dataset(tokenizer, args.dataset_path, args.dataset_cache,'test', args.script, args.script_doc)

    
    datasets = {"test": defaultdict(list)}
    for dataset_name, dataset in chat_dataset.items():
        for dialog in dataset:
            instance = build_input_from_segments(dialog, tokenizer, valid=True, baseline=args.baseline, baseline_nodoc=args.baseline_nodoc)
            for input_name, input_array in instance.items():
                datasets[dataset_name][input_name].append(input_array)

    logger.info("Build data loaders")


    test_dataset = DoGDataset(datasets['test'], max_len=512)

    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    
    return test_loader



def build_input_from_segments_(dialog, tokenizer, min_length=2, baseline=False, baseline_nodoc=False): 
    bos, eos, doc, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    instance = {'input_ids':[],'lm_labels':[],'token_type_ids':[]}
    
    doc_sequence = [[doc] +  dialog[0] + [eos]]
    
    del dialog[0]
    doc_sequence = list(chain(*doc_sequence))
    doc_token_types = [doc for _ in doc_sequence]


    if baseline_nodoc:
        doc_sequence=[]
        doc_token_types=[]

    if baseline:
        doc_sequence=doc_sequence[:-512]
        doc_token_types=doc_token_types[:-512]


    new=True
    for i in range(len(dialog)-1):
        if new:
            sequence = doc_sequence + [speaker2 if i%2==0 else speaker1]
            token_types = doc_token_types.copy() + [speaker2 if i%2==0 else speaker1]
        
        sequence.extend(dialog[i])

        if i %2==0:
            token_types += [speaker2] *(len(dialog[i]))
        else:
            token_types += [speaker1] *(len(dialog[i]))

        if i>=min_length:
            doc_sequence=sequence
            doc_token_types=token_types

            instance["input_ids"].append(sequence)
            instance['lm_labels'].append(dialog[i+1])
            instance['token_type_ids'].append(token_types)
        else:
            new=False

    return instance

class DoGDataset_(data.Dataset):
    def __init__(self, data, max_len, max_utt_len):

        self.data=data
        self.max_len = max_len
        self.max_utt_len = max_utt_len

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, idx):
        
        inputs, token_ids, lm_labels = self.data['input_ids'][idx], self.data['token_type_ids'][idx], self.data['lm_labels'][idx]
        inputs_, token_ids_ = [], []

        i=0
        while len(inputs)>0:
            if i==0:
                max_len=self.max_len-(self.max_utt_len+1)
            else:
                max_len=self.max_len
            inputs_.append(torch.LongTensor(inputs[-max_len:]))
            token_ids_.append(torch.LongTensor(token_ids[-max_len:]))
            inputs = inputs[:-max_len]
            token_ids = token_ids[:-max_len]

            i+=1

        return inputs_[::-1], token_ids_[::-1], lm_labels
        
        

def get_data_loaders_(args, tokenizer):
    """ Prepare the dataset for training and evaluation """
    chat_dataset={}
    chat_dataset['test'] = get_dataset(tokenizer, args.dataset_path, args.dataset_cache,'test', args.script, args.script_doc)

    
    datasets = {"test": defaultdict(list)}
    for dataset_name, dataset in chat_dataset.items():
        for dialog in dataset:
            instance = build_input_from_segments_(dialog, tokenizer, min_length=args.min_history, baseline=args.baseline, baseline_nodoc=args.baseline_nodoc)
            for input_name, input_array in instance.items():
                for i in range(len(input_array)):
                    datasets[dataset_name][input_name].append(input_array[i])

    logger.info("Build data loaders")

    test_dataset = DoGDataset_(datasets['test'], max_len=512, max_utt_len=args.max_length)

    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    
    return test_loader



def compute_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_bleu_score(predictions, references):
    bleu=BLEU()
    return bleu.corpus_score(predictions,references)

def compute_rouge_score(prediction, ground_truth):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    return scorer.score(prediction, ground_truth)

def compute_meteor_score(prediction, ground_truth):
    return meteor_score.meteor_score([ground_truth], prediction)

parser = ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
parser.add_argument('--script', action='store_true')
parser.add_argument('--script_doc', action='store_true')
parser.add_argument("--model", type=str, default="gpt2")  # anything besides gpt2 will load openai-gpt
parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
parser.add_argument("--min_history", type=int, default=2, help="Number of previous utterances to keep in history")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
parser.add_argument("--max_length", type=int, default=50, help="Maximum length of the output utterances")
parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
parser.add_argument("--eval_type", type=str, default='ppl')
parser.add_argument("--seed", type=int, default=11, help="Seed")
parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
parser.add_argument("--temperature", type=float, default=1, help="Sampling softmax temperature")
parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
parser.add_argument('--baseline_nodoc', action='store_true')
parser.add_argument('--baseline', action='store_true')

args = parser.parse_args()

torch.manual_seed(args.seed)

logging.basicConfig(level=logging.INFO)
logger.info("Arguments: %s", pformat(args))


logger.info("Prepare tokenizer, pretrained model and optimizer.")
tokenizer_class = GPT2Tokenizer
tokenizer = tokenizer_class.from_pretrained('gpt2')

model_class = GPT2LMHeadModel
model = model_class.from_pretrained(args.model_checkpoint)
model.to(args.device)

add_special_tokens_(model, tokenizer)


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
                
                output = model(input_ids[i].to(args.device), token_type_ids=token_type_ids[i].to(args.device),new_doc=new_doc, last_doc=last_doc)
                
                if isinstance(output, tuple):
                    lm_logits = output[0]['logits']
                else:
                    lm_logits = output['logits']
                
                if len(lm_logits_flat_shifted)==0:
                    lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1)).cpu()
                    lm_labels_flat_shifted = lm_labels[i][..., 1:].contiguous().view(-1).cpu() #to(args.device)
                else:
                    lm_logits_flat_shifted = torch.cat([lm_logits_flat_shifted, 
                                                        lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1)).cpu()],0)
                    lm_labels_flat_shifted = torch.cat([lm_labels_flat_shifted,
                                                        lm_labels[i][..., 1:].contiguous().view(-1).cpu()],0)
                
    return lm_logits_flat_shifted.unsqueeze(0), lm_labels_flat_shifted.unsqueeze(0)
    

def sample_sequence(dialog, tokenizer, model, args, current_output=None):
    model.eval()
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []


    input_ids=dialog[0]
    token_type_ids=dialog[1]
    labels=dialog[2]

    for i in range(len(input_ids)-1):

        if i==0 or i==1:
            logits = model(input_ids[i].to(args.device), token_type_ids=token_type_ids[i].to(args.device), new_doc=True)
        else:
            logits = model(input_ids[i].to(args.device), token_type_ids=token_type_ids[i].to(args.device), new_doc=False)

    if token_type_ids[-1][-1,-1]==50260:
        token_id=50261
    else:
        token_id=50260

    for j in range(args.max_length):
        if len(current_output)!=0:
            inputs=torch.cat([input_ids[-1].to(args.device), torch.LongTensor([current_output]).to(args.device)],-1)
        else:
            inputs=input_ids[-1].to(args.device)

        if not args.baseline_nodoc and not args.baseline:
            if j==0 and (i==0 or i==1):
                logits = model(inputs, new_doc=True, generating=False)
            elif j==0: 
                logits = model(inputs, new_doc=False, generating=False)
            else:
                logits = model(inputs, new_doc=False, generating=True, last_doc=True)
        else:
            logits = model(inputs)
        
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]['logits']
        else:
            logits = logits['logits']
            

        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = torch.softmax(logits, dim=-1)
        prev = torch.multinomial(probs, 1)

        if j < args.min_length and prev.item() in special_tokens_ids:
            loops=0
            while prev.item() in special_tokens_ids:
                loops+=1
                if probs.max().item() == 1:
                    break  # avoid infinitely looping over special token
                elif loops>10:
                    break  # avoid infinitely looping over special token

                prev = torch.multinomial(probs, num_samples=1)
        if prev.item() in special_tokens_ids and prev.item()!=50260 and prev.item()!=50261:
            break
        current_output.append(prev.item())

    return current_output, torch.cat(labels).tolist()[:-1]



if args.eval_type=='ppl':

    logger.info("Prepare datasets")
    test_loader = get_data_loaders(args, tokenizer)

    evaluator = Engine(inference)
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-100), output_transform=lambda x: (x[0][0], x[1][0]))}
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    
    for name, metric in metrics.items():
        metric.attach(evaluator, name)
    pbar = ProgressBar(persist=True)
    
    evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Test: %s" % pformat(evaluator.state.metrics)))

    evaluator.run(test_loader)
    

else:
    logger.info("Prepare datasets")
    test_loader = get_data_loaders_(args, tokenizer)

    saving=False
    if saving:
        test_refs=[]
        test_outs=[]
        test_context=[]

    outputs=[]
    references=[]
    i=0
    for data in test_loader:
        
        output, labels = sample_sequence(data, tokenizer, model, args)
        outputs.append(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(output)).replace('<speaker1>','').replace('<speaker2>',''))
        references.append(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(labels)))

        if saving:
            cont=[]
            for k in range(len(data[0])):
                cont.append(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(data[0][k].squeeze().data.tolist())))

            test_context.append(cont)
            test_outs.append(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(output)))
            test_refs.append(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(labels)))


        if i%500==0:
            print(i, len(test_loader))
        
        #if i==500:
        #    break
        
        i+=1

    if saving:
        with open('./atts_infinite_transformer_sm/context','wb') as f:
            pickle.dump(test_context,f)

        with open('./atts_infinite_transformer_sm/outputs','wb') as f:
            pickle.dump(test_outs,f)

        with open('./atts_infinite_transformer_sm/refs','wb') as f:
            pickle.dump(test_refs,f)


    f1_scores=[]
    rouge1_scores=[]
    rougel_scores=[]
    meteor_scores=[]
    for i in range(len(outputs)):
        f1_scores.append(compute_f1_score(outputs[i],references[i]))
        rouge1_scores.append(compute_rouge_score(outputs[i],references[i])['rouge1'][0])
        rougel_scores.append(compute_rouge_score(outputs[i],references[i])['rougeL'][0])
        meteor_scores.append(compute_meteor_score(outputs[i],references[i]))
 

    references=[references]
    bleu_score = compute_bleu_score(outputs,references)

    emb=Embedding()
    emb_scores=eval_emb_metrics(outputs,references)   


    print('f1 score: ', np.array(f1_scores).mean())
    print('bleu score: ', bleu_score)
    print('rouge1 score: ', np.array(rouge1_scores).mean())
    print('rougeL score: ', np.array(rougel_scores).mean())
    print('meteor score: ', np.array(meteor_scores).mean())
    print('emb metrics: ', emb_scores)
