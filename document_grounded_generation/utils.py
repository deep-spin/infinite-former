# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from datetime import datetime
import json
import logging
import os
import tarfile
import tempfile
import socket

import torch

from transformers import cached_path


logger = logging.getLogger(__file__)

def get_dataset(tokenizer, dataset_path, dataset_cache, partition, script=False, script_doc=False):
    if script:
        dataset_cache += '_' +  partition + 'script_cached'
    elif script_doc:
        dataset_cache += '_' +  partition + 'script_doc_cached'
    else:
        dataset_cache += '_' +  partition + '_cached'

    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Loading dataset from %s", dataset_path)
        if script:
            dataset_file = cached_path(dataset_path+partition+'_script')
        elif script_doc:
            dataset_file = cached_path(dataset_path+partition+'_script_doc')
        else:
            dataset_file = cached_path(dataset_path+partition)
        with open(dataset_file, "r") as f:
            lines = f.readlines()

        if not script and not script_doc:
            dataset=[]
            chat=[]
            for line in lines:
                if 'Document:' in line:
                    dataset.append([line.replace('Document:','').replace('\n','')])
                elif 'Dialog:' in line:
                    chat=[line.replace('Dialog:','').replace('\n','')]
                elif line=='\n':
                    dataset[-1].extend(chat)
                    chat=[]
                else:
                    chat.extend([line.replace('\n','')])
        else:
            dataset=[]
            chat=[]
            for line in lines:
                if 'Document:' in line:
                    aux_document=True
                    document=line.replace('Document:','').replace('\n','')
                elif 'Dialog:' in line:
                    dataset.append([document])
                    aux_document=False
                    chat=[line.replace('Dialog:','').replace('\n','')]
                elif line=='\n':
                    if aux_document:
                        document+=line
                    else:
                        dataset[-1].extend(chat)
                        chat=[]
                else:
                    if aux_document:
                        document+=line
                    else:
                        chat.extend([line.replace('\n','')])

        logger.info("Tokenize and encode the dataset")
        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            return list(tokenize(o) for o in obj)

        dataset = tokenize(dataset)
        torch.save(dataset, dataset_cache)
    return dataset


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def make_logdir(model_name: str):
    """Create unique path to save results and checkpoints, e.g. runs/Sep22_19-45-59_gpu-7_gpt2"""
    # Code copied from ignite repo
    logdir = os.path.join('runs', model_name)
    return logdir
