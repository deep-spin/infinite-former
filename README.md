# Infinite-former
Implementation of the Infinite-former model.

## Sorting
To perform experiments on the sorting task, you first need to generate the data. 
You can do that as (you will need to select the vocabulary size and the sequence length on the file ./sorting/generate_data.py):
```
python ./sorting/generate_data.py
```

Then, you simply need to run (changing the sequence length on the script run_sort_inftyformer.sh):
```
bash run_sort_inftyformer.sh train
```

## Fine-tuning GPT-2

To fine-tune the GPT-2 with the long-term memory, you first need to install the Transformers library as:
```
pip install --editable ./finetune_gpt2 
```
Then, to fine-tune the model run the command:
```
python ./finetune_gpt2/examples/language-modeling/run_clm.py 	\
	--model_name_or_path=gpt2 \
	--model_type=gpt2 \
	--config_name=infinite_memory_transformer_mask_kl_sticky_mem \
	--per_device_eval_batch_size=1 \
	--per_device_train_batch_size=1 \
	--train_file=</path/to/train/data/file> \
	--validation_file=</path/to/val/data/file> \
	--do_train \
	--do_eval \
	--block_size=512 \
	--output_dir=</path/to/output/dir>
	--kl_regularizer \
	--kl_m=.000001
```
To evaluate the model do:
```
python ./finetune_gpt2/examples/language-modeling/run_clm.py \
	--model_name_or_path=</path/to/output/dir>
	--model_type=gpt2 \
	--config_name=infinite_memory_transformer_mask_kl \ 
	--per_device_eval_batch_size=1 \
	--validation_file=</path/to/test/data/file> \
	--do_eval \
	--block_size=512 \
	--output_dir=</path/to/output/dir> 
```


# Citation

    @inproceedings{martins2022infinite,
      author    = {Martins, Pedro Henrique and Marinho, Zita and  Martins, Andr{\'e} FT},
      title     = {$\infty $-former: Infinite Memory Transformer},
      booktitle = {Proc. ACL},
      year      = {2022}
    }