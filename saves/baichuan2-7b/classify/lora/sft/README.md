---
license: other
library_name: peft
tags:
- llama-factory
- lora
- generated_from_trainer
base_model: /data/workspace/models/Baichuan2-7B-Chat
model-index:
- name: sft
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sft

This model is a fine-tuned version of [/data/workspace/models/Baichuan2-7B-Chat](https://huggingface.co//data/workspace/models/Baichuan2-7B-Chat) on the classify_train dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1016

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 16
- total_train_batch_size: 16
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 0.1
- num_epochs: 6.0
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- PEFT 0.11.1
- Transformers 4.41.1
- Pytorch 2.3.0+cu121
- Datasets 2.14.7
- Tokenizers 0.19.1