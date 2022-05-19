#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python3 train.py \
        --cuda \
        --data ../data/ \
        --dataset ../data_sort_8000 \
        --n_layer 3 \
        --d_model 300 \
        --n_head 6 \
        --d_head 50 \
        --d_inner 300 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.0002 \
        --warmup_step 0 \
        --max_step 20000 \
        --tgt_len 1024 \
        --mem_len 1024 \
        --eval_tgt_len 1024 \
        --batch_size 8 \
        --gpu0_bsz 8 \
        --continuous \
        --long_term_attention \
        --long_term_attention_norm='softmax' \
        --long_term_attention_basis 512 \
        --affines \
        --augment \
        --augment_len 1024 \
        --infinite_memory \
        --mask \
        --mask_type 'cnn' \
        --kl_regularizer  \
        --kl_m .000001 \
        --sigma_0 .05 \
        --name infty_former \
        --work_dir ./sort_8000 \
        ${@:2}
    echo 'unknown argment 1'
fi
