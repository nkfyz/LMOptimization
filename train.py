import os
import math
import json
import wandb
import torch
import logging
from tqdm.auto import tqdm
from datetime import datetime
from utils import get_model_parameters, set_seed, get_logger, move_to_cuda

import transformers
from transformers import (
    EsmConfig,
    EsmTokenizer,
    EsmForMaskedLM,
    DataCollatorForLanguageModeling,
    get_scheduler,
)

from transformers.trainer_utils import get_last_checkpoint
import datasets
from datasets import load_dataset, DatasetDict, load_from_disk
from torch.utils.data import DataLoader
import evaluate

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device
import torch.distributed as dist

os.environ['OMP_NUM_THREADS'] = '8'

run_name = f'AbFM_{datetime.now().strftime("%Y_%m_%d_%H_%M")}'

abfm_config = {
    'model_parameters': {
        'num_hidden_layers': 12,
        'num_attention_heads': 20,
        'hidden_size': 480,
        'intermediate_size': 1920,
        'hidden_dropout_prob': 0.0,
        'attention_probs_dropout_prob': 0.0,
        'max_position_embeddings': 1026,
        'position_embedding_type': 'rotary',
        'initializer_range': 0.02,
        'layer_norm_eps': 1e-05,
        'emb_layer_norm_before': False,
        'token_dropout': True,
        'use_cache': True,
        'classifier_dropout': None,
        'hidden_act': 'gelu'
    },
    
    'vocab_parameters': {
        'vocab_size': 33,
        'mask_token_id': 32,
        'pad_token_id': 1
    },

    'tokenizer_parameters': {
        'padding': False,
        'truncation': True,
        'return_special_tokens_mask': True
    },

    'tokenization_parameters': {
        'load_saved_dataset': True,
        'save_tokenized_dataset': True,
        'overwrite_cache': True,
        'preprocessing_num_workers': 64,
        'max_seq_length' : 256, 
        'max_all_samples': None,
        'max_train_samples': None,
        'max_eval_samples': None,
        'max_test_samples': None,
        'streaming': False,
        'remove_empty_lines': False,
        'mlm_probability': 0.15
    },

    'training_parameters': {
        'do_train': True,
        'do_eval': True,
        'learning_rate': 5e-5,
        'lr_scheduler_type': 'cosine',
        'per_device_train_batch_size': 64,
        'per_device_eval_batch_size': 64,
        'auto_find_batch_size': False,
        'num_train_epochs': 1,
        'max_steps': 30,
        'warmup_ratio': 0.01,
        'weight_decay': 0.01,
        'gradient_accumulation_steps': 1,
        'adam_epsilon': 1e-6,
        'adam_beta1': 0.9,
        'adam_beta2': 0.98,
        'adam_epsilon': 1e-08, 
        'fp16': True,
        'resume_from_checkpoint': False,
        'evaluation_strategy': 'steps',
        'prediction_loss_only': False,
        'save_strategy': 'steps',
        'seed': 42
    },
    'input_parameters': {
        'dataset_file': './data/mini_seqs_clu.txt',
        'vocab_file': './vocab.txt'
    },

    'output_parameters': {
        'save_steps': 1000000,
        'eval_steps': 1000000,
        'save_total_limit': None,
        'output_dir': f'./checkpoints/{run_name}',
        'logging_dir': f'./logs/{run_name}',
        'logging_steps': 1000,
        'logging_strategy': 'steps',
        'overwrite_output_dir': False,
        'logging_first_step': True,
        'report_to': 'wandb'
    },

    'cache_parameters': {
        'cache_dir': '/data/cache',
    },
    
    'colossal_parameters': {
        'plugin': 'low_level_zero'
        # 'plugin': 'torch_ddp_fp16'
        # 'plugin': 'gemini'
        # 'plugin': 'hybrid_parallel'
    }
}

os.makedirs(abfm_config['output_parameters']['output_dir'], exist_ok=True)
logger = get_logger(__name__,
                    output_file=os.path.join(abfm_config['output_parameters']['output_dir'], 'logger_out.log'))
if (seed := abfm_config['training_parameters']['seed']) is not None:
    set_seed(seed)

with open(os.path.join(abfm_config['output_parameters']['output_dir'], 'parameters.json'), 'w') as f:
    json.dump(abfm_config, f, indent=2)

colossalai.launch_from_torch(config={}, seed=abfm_config['training_parameters']['seed'])
coordinator = DistCoordinator()

LEARNING_RATE = abfm_config['training_parameters']['learning_rate'] * coordinator.world_size 

def log_on_master(info):
    if coordinator.is_master():
        logger.info(info)

booster_kwargs = {}
if abfm_config['colossal_parameters']['plugin'] == "torch_ddp_fp16":
    booster_kwargs["mixed_precision"] = "fp16"
if abfm_config['colossal_parameters']['plugin'].startswith("torch_ddp"):
    plugin = TorchDDPPlugin()
elif abfm_config['colossal_parameters']['plugin'] == "gemini":
    plugin = GeminiPlugin(initial_scale=2**5)
elif abfm_config['colossal_parameters']['plugin'] == "low_level_zero":
    plugin = LowLevelZeroPlugin(initial_scale=2**5)
elif abfm_config['colossal_parameters']['plugin'] == "hybrid_parallel":
    plugin = HybridParallelPlugin(
        tp_size=1,
        pp_size=2,
        num_microbatches=None,
        pp_style="interleaved",
        num_model_chunks=2,
        microbatch_size=16,
        enable_all_optimization=True,
        zero_stage=1,
        precision="fp16",
        initial_scale=1,
    )

booster = Booster(plugin=plugin, **booster_kwargs)

model_config = EsmConfig(
    **abfm_config['model_parameters'],
    **abfm_config['vocab_parameters'],
)

model = EsmForMaskedLM(model_config)
model_size = get_model_parameters(model)
log_on_master(f'Number of model parameters is {model_size/1e6:.2f}M')

tokenizer = EsmTokenizer(abfm_config['input_parameters']['vocab_file'], unk_token='<unk>', cls_token='<cls>', pad_token='<pad>', mask_token='<mask>', eos_token='<eos>')

dataset_filename = os.path.splitext(os.path.basename(abfm_config['input_parameters']['dataset_file']))[0]
if abfm_config['tokenization_parameters']['load_saved_dataset'] and os.path.exists(dataset_cache_dir := os.path.join(abfm_config['cache_parameters']['cache_dir'], f'tokenized_dataset_{dataset_filename}')):
    log_on_master(f'Loading from cached dataset: {dataset_cache_dir}')
    tokenized_dataset = load_from_disk(dataset_cache_dir)
else:
    data_files = {
        'train': [abfm_config['input_parameters']['dataset_file']],
    }

    log_on_master(f'Loading dataset from file: {abfm_config["input_parameters"]["dataset_file"]}')
    dataset = load_dataset('text', data_files=data_files)
    if (max_all_samples := abfm_config['tokenization_parameters']['max_all_samples']) is not None:
        max_all_samples = min(len(dataset['train']), max_all_samples)
        dataset['train'] = dataset['train'].select(range(max_all_samples))
    log_on_master(f'Number of total sequences: {len(dataset["train"])}')
    train_testeval_dataset = dataset["train"].train_test_split(test_size=0.1, shuffle=True, seed=abfm_config['training_parameters']['seed'])
    test_eval_dataset = train_testeval_dataset['test'].train_test_split(test_size=0.5, shuffle=True, seed=abfm_config['training_parameters']['seed'])
    dataset = DatasetDict({
        'train': train_testeval_dataset['train'],
        'eval': test_eval_dataset['train'],
        'test': test_eval_dataset['test']})
    if (max_train_samples := abfm_config['tokenization_parameters']['max_train_samples']) is not None:
        max_train_samples = min(len(dataset['train']), max_train_samples)
        dataset['train'] = dataset['train'].select(range(max_train_samples))
    if (max_eval_samples := abfm_config['tokenization_parameters']['max_eval_samples']) is not None:
        max_eval_samples = min(len(dataset['eval']), max_eval_samples)
        dataset['eval'] = dataset['eval'].select(range(max_eval_samples))
    if (max_test_samples := abfm_config['tokenization_parameters']['max_test_samples']) is not None:
        max_test_samples = min(len(dataset['test']), max_test_samples)
        dataset['test'] = dataset['test'].select(range(max_test_samples))

    max_seq_length = min(abfm_config['tokenization_parameters']['max_seq_length'], tokenizer.model_max_length)
    no_streaming_kwargs = {} if abfm_config['tokenization_parameters']['streaming'] else {
        'num_proc': abfm_config['tokenization_parameters']['preprocessing_num_workers'],
        'load_from_cache_file': not abfm_config['tokenization_parameters']['overwrite_cache']
    }

    tokenized_dataset = dataset.map(
        lambda x: tokenizer(
            [line for line in x['text'] if len(line) > 0 and not line.isspace()] if abfm_config['tokenization_parameters']['remove_empty_lines'] else x['text'],
            max_length=max_seq_length,
            **abfm_config['tokenizer_parameters']
        ),
        batched=True,
        remove_columns=['text'],
        **no_streaming_kwargs
    )
    dataset.set_format(type='torch')

    if abfm_config['tokenization_parameters']['save_tokenized_dataset']:
        tokenized_dataset.save_to_disk(os.path.join(abfm_config['cache_parameters']['cache_dir'], f'tokenized_dataset_{dataset_filename}'))

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=abfm_config['tokenization_parameters']['mlm_probability']
)

train_dataloader = plugin.prepare_dataloader(tokenized_dataset['train'], shuffle=True, collate_fn=data_collator, batch_size=abfm_config['training_parameters']['per_device_train_batch_size'], drop_last=True)
eval_dataloader = plugin.prepare_dataloader(tokenized_dataset['eval'], collate_fn=data_collator, batch_size=abfm_config['training_parameters']['per_device_eval_batch_size'], drop_last=False)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {
        'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay': abfm_config['training_parameters']['weight_decay'],
    },
    {
        'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0,
    },
]
optimizer = HybridAdam(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=1e-8)

overrode_max_train_steps = False
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / abfm_config['training_parameters']['gradient_accumulation_steps'])
if abfm_config['training_parameters']['max_steps'] is None:
    abfm_config['training_parameters']['max_steps'] = abfm_config['training_parameters']['num_train_epochs'] * num_update_steps_per_epoch
    overrode_max_train_steps = True

lr_scheduler = get_scheduler(
    name=abfm_config['training_parameters']['lr_scheduler_type'],
    optimizer=optimizer,
    num_warmup_steps=math.ceil(abfm_config['training_parameters']['max_steps'] * abfm_config['training_parameters']['warmup_ratio'] * abfm_config['training_parameters']['gradient_accumulation_steps']),
    num_training_steps=abfm_config['training_parameters']['max_steps'] * abfm_config['training_parameters']['gradient_accumulation_steps'],
)

def _criterion(outputs, inputs):
    loss = outputs.loss
    return loss

model, optimizer, _criterion, _, lr_scheduler = booster.boost(
    model, optimizer, criterion=_criterion, lr_scheduler=lr_scheduler
)

num_update_steps_per_epoch = math.ceil(len(train_dataloader) / abfm_config['training_parameters']['gradient_accumulation_steps'])
if overrode_max_train_steps:
    abfm_config['training_parameters']['max_steps'] = abfm_config['training_parameters']['num_train_epochs'] * num_update_steps_per_epoch
abfm_config['training_parameters']['num_train_epochs'] = math.ceil(abfm_config['training_parameters']['max_steps'] / num_update_steps_per_epoch)

log_on_master('*** Train ***')
log_on_master(f'Num train examples = {len(tokenized_dataset["train"])}')
log_on_master(f'Num Epochs = {abfm_config["training_parameters"]["num_train_epochs"]}')
log_on_master(f'Instantaneous batch size per device = {abfm_config["training_parameters"]["per_device_train_batch_size"]}')
log_on_master(f'Gradient Accumulation steps = {abfm_config["training_parameters"]["gradient_accumulation_steps"]}')
log_on_master(f'Total optimization steps = {abfm_config["training_parameters"]["max_steps"]}')

use_pipeline = isinstance(booster.plugin, HybridParallelPlugin) and booster.plugin.pp_size > 1
is_pp_last_device = use_pipeline and booster.plugin.stage_manager.is_last_stage(ignore_chunk=True)
print_flag = (not use_pipeline and coordinator.is_master()) or (use_pipeline and is_pp_last_device)

progress_bar = tqdm(range(abfm_config['training_parameters']['max_steps']), disable=not print_flag)
completed_steps = 0
starting_epoch = 0

if checkpoint_path := abfm_config['training_parameters']['resume_from_checkpoint']:
    if 'step' not in checkpoint_path:
        dirs = [f.name for f in os.scandir(checkpoint_path) if f.is_dir()]
        dirs.sort(key=os.path.getctime)
        path = dirs[-1]
        checkpoint_path = path
    path = os.path.basename(checkpoint_path)

    log_on_master(f'Resumed from checkpoint: {checkpoint_path}')
    booster.load_model(model, checkpoint_path)
    booster.load_optimizer(optimizer, checkpoint_path)
    booster.load_lr_scheduler(lr_scheduler, checkpoint_path)

    training_difference = os.path.splitext(path)[0]
    if 'epoch' in training_difference:
        starting_epoch = int(training_difference.replace('epoch_', '')) + 1
        resume_step = None
        completed_steps = starting_epoch * num_update_steps_per_epoch
    else:
        resume_step = int(training_difference.replace("step_", "")) * abfm_config['training_parameters']['gradient_accumulation_steps']
        starting_epoch = resume_step // len(train_dataloader)
        completed_steps = resume_step // abfm_config['training_parameters']['gradient_accumulation_steps']
        resume_step -= starting_epoch * len(train_dataloader)

progress_bar.update(completed_steps)


LAUNCH_PROFILE = False
LAUNCH_TIMER = False

if LAUNCH_PROFILE: 
    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        # schedule=torch.profiler.schedule(wait=3, warmup=1, active=1750),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/bert_comm_optimize'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True)
    prof.start()

if LAUNCH_TIMER:
    import time
    train_start = time.time()

for epoch in range(starting_epoch, abfm_config['training_parameters']['num_train_epochs']):
    model.train()
    optimizer.zero_grad()
    train_dataloader_iter = iter(train_dataloader)
    if abfm_config['training_parameters']['resume_from_checkpoint'] and epoch == starting_epoch and resume_step is not None:
        for _ in range(resume_step - 1):
            next(train_dataloader_iter)
    for _ in range(len(train_dataloader_iter)):
        if LAUNCH_PROFILE: 
            prof.step()
        if use_pipeline:
            outputs = booster.execute_pipeline(
                train_dataloader_iter, model, _criterion, optimizer, return_loss=True, return_outputs=True
            )
            if is_pp_last_device:
                 loss = outputs["loss"]
                 progress_bar.set_postfix({"loss": loss.item()})
        else:
            data = next(train_dataloader_iter)
            data = move_to_cuda(data)
            outputs = model(**data)
            loss = _criterion(outputs, None)
            booster.backward(loss, optimizer)
            progress_bar.set_postfix({"loss": loss.item()})
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        completed_steps += 1

        if isinstance(abfm_config['output_parameters']['save_steps'], int) and completed_steps % abfm_config['output_parameters']['save_steps'] == 0:
            save_dir = os.path.join(abfm_config['output_parameters']['output_dir'], f'step_{completed_steps}')
            booster.save_model(model, save_dir)
            booster.save_optimizer(optimizer, save_dir)
            booster.save_lr_scheduler(lr_scheduler, save_dir)
        if completed_steps % abfm_config['output_parameters']['logging_steps'] == 0:
            log_on_master(f'step {completed_steps}, train loss {loss.item()}')
        if completed_steps >= abfm_config['training_parameters']['max_steps']:
            break

    if abfm_config['output_parameters']['save_steps'] == 'epoch':
        save_dir = os.path.join(abfm_config['output_parameters']['output_dir'], f'epoch_{epoch}')
        booster.save_model(model, save_dir)
        booster.save_optimizer(optimizer, save_dir)
        booster.save_lr_scheduler(lr_scheduler, save_dir)

    if abfm_config['training_parameters']['do_eval']:
        metric = evaluate.load('accuracy', run_name, process_id=coordinator.rank, num_process=coordinator.world_size)
        model.eval()
        accum_loss = torch.zeros(1, device=get_current_device())
        for batch in eval_dataloader:
            batch = move_to_cuda(batch)
            labels = batch['labels'].reshape(-1)
            mask = labels != -100 
            labels = labels[mask]
            if use_pipeline:
                pg_mesh = booster.plugin.pg_mesh
                pp_group = booster.plugin.pp_group
                current_pp_group_ranks = pg_mesh.get_ranks_in_group(pp_group)
                current_rank = dist.get_rank()
                batch = iter([batch])
                outputs = booster.execute_pipeline(batch, model, criterion, return_loss=True, return_outputs=True)
                if is_pp_last_device:
                    logits = outputs["outputs"]["logits"]
                    val_loss = outputs["loss"]
                    accum_loss.add_(val_loss)

                    preds = torch.argmax(logits, axis=1)
                    preds = preds.reshape(-1)
                    preds = preds[mask]

                    dist.broadcast_object_list([preds, val_loss], src=current_pp_group_ranks[-1], group=pp_group)

                    metric.add_batch(predictions=preds, references=labels)
                elif current_rank in current_pp_group_ranks:
                    object_list = [None, None]
                    dist.broadcast_object_list(object_list, src=current_pp_group_ranks[-1], group=pp_group)

                    metric.add_batch(predictions=object_list[0].to(get_current_device()), references=labels)
                    accum_loss.add_(object_list[1].to(get_current_device()))
            else:
                with torch.no_grad():
                    outputs = model(**batch)
                val_loss, logits = outputs[:2]
                accum_loss.add_(val_loss)

                preds = torch.argmax(logits, axis=-1)
                preds = preds.reshape(-1)
                preds = preds[mask]

                metric.add_batch(predictions=preds, references=labels)

        eval_results = metric.compute()

        dist.all_reduce(accum_loss.div_(len(eval_dataloader)))

        if coordinator.is_master() and eval_results is not None:
            eval_results['loss'] = accum_loss.item() / coordinator.world_size
            try:
                perplexity = math.exp(eval_results['loss'])
            except OverflowError:
                perplexity = float('inf')
        if coordinator.is_master():
            logger.info(f'epoch {epoch}: perplexity: {perplexity}')
            with open(os.path.join(abfm_config['output_parameters']['output_dir'], f'eval_epoch{epoch}.json'), 'w') as f:
                json.dump(eval_results, f, indent=2)

if LAUNCH_PROFILE: 
    prof.stop()
    
if LAUNCH_TIMER:
    train_end = time.time()
    print(f"Training time: {train_end - train_start}s")

save_dir = os.path.join(abfm_config['output_parameters']['output_dir'], 'final')
log_on_master(f'Saving final model to dir {save_dir}')
booster.save_model(model, save_dir)

if coordinator.is_master():
    tokenizer.save_pretrained(abfm_config['output_parameters']['output_dir'])
