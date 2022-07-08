import time
from typing import Dict, Any, List
from pathlib import Path
from datetime import date
import re 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau

from baselines.models_factory import ModelsFactory
from datasets.datasets_factory import DatasetsFactory
from datasets.dst_datasets import collate_fn

from models.label_smoothing import *
from models.optimize import * 

from tqdm import tqdm 
from functools import partial 
import pdb 
import json 
import pickle as pkl 

def save_checkpoint(model: nn.Module, model_name: str, dev_loss: float, checkpoint_dir: str, preds, epoch) -> None:
    current_date = date.today().strftime("%d-%m-%y")

    # create checkpoint folder if it doesn't exist
    checkpoint_path = Path(checkpoint_dir) 
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # save the model to dict
    checkpoint_file = checkpoint_path / f"ep{epoch}_{current_date}_{dev_loss}.pth"
    torch.save(model.state_dict(), checkpoint_file)
    print(f"Saved best model so far to {checkpoint_file}")

    pred_file = checkpoint_path / f"ep{epoch}_{current_date}_{dev_loss}.pred"
    json.dump(preds, open(pred_file, 'w'), indent=4)

# acc by string matching (raw) 
def get_raw_state_acc(preds, in_vocab, vocab, sample, all_preds, config, mode='seq'):
    
    gt = sample['state_raw']
    gt_raw = sample['state_raw'] 
    _, indices = preds.max(dim=-1)
    indices = indices.cpu().numpy()
    matches = 0 
    frame_pattern = re.compile('<frame\d+>')
    for i in range(len(preds)):
        state = indices[i]
        decoded_state = []
        state_in = sample['state_in'][i]
        prior_value = None
        curr_obj = None 
        for j in range(len(state)):
            if vocab is not None and state[j] not in vocab: 
                pdb.set_trace()
            value =  vocab[state[j]]
            if value in ['<pad>', '<frame>', '<unk>', '<usr>', '<sys>', '<context>', '<state>',
                    '<response>', '<end_response>', 'none', 'dontcare']:
                continue 
            if value == '<end_state>':
                break 
            if mode =='class': 
                if state_in[j][0] == 0: # pad token 
                    break
                if j == 0 or j == 1:
                    if frame_pattern.match(value):
                        decoded_state.append(value) #frame 
                elif not frame_pattern.match(value):
                    obj = in_vocab[state_in[j][0].item()]
                    slot = in_vocab[state_in[j][1].item()]
                    if obj != curr_obj: 
                        decoded_state.append(obj)
                        curr_obj = obj 
                    decoded_state.append(slot) 
                    decoded_state.append(value) 
                continue 
            elif value != prior_value:
                if value == 'snitch': 
                    value = 'spl'
                decoded_state.append(value)
                prior_value = value
        gt_state = ' '.join(gt[i][1:-1])
        decoded_state = ' '.join(decoded_state)
        if gt_state == decoded_state: 
            matches += 1 
        gt_raw_state = ' '.join(gt_raw[i][1:-1])
        vid = sample['vid'][i]
        turn_idx = sample['turn_idx'][i]
        gt_segment = sample['gt_temporal'][i] 
        if vid not in all_preds: all_preds[vid] = {}
        if turn_idx not in all_preds[vid]: all_preds[vid][turn_idx] = {} 
        all_preds[vid][turn_idx]['gt_state'] = gt_state
        all_preds[vid][turn_idx]['pred_state'] = decoded_state
        all_preds[vid][turn_idx]['gt_raw_state'] = gt_raw_state
        all_preds[vid][turn_idx]['gt_segment'] = gt_segment
    acc = matches/len(preds)
    return acc, all_preds 

def self_supervised_loss_compute(output, sample, criterion, mode):
    if mode == 'bb':
        labels = sample['obj_bb']
        mask = sample['bb_mask']
        norm = mask.sum()*4
    elif mode == 'resnext':
        labels = sample['resnext']
        mask = sample['resnext_mask']
        norm = mask.sum()*2048 

    loss = criterion(output, labels)
    loss = loss * mask[:,:,None]
    loss = loss.sum()/norm
    return loss 

def loss_compute(model_name: str, model: nn.Module, compute_device: torch.device, data_loader: data.DataLoader, 
    criterion, bb_criterion, resnext_criterion, 
    vocab, state_vocab, res_vocab, config) -> Any:
    model.eval()
    model.to(compute_device)

    with torch.no_grad():
        total_state_loss = 0
        total_samples = 0 
        total_state_acc = 0 
        total_bb_loss = 0 
        total_resnext_loss = 0 
        output_preds = {} 
        for batch_idx, sample in tqdm(enumerate(data_loader), total=len(data_loader), ncols=0):
            for k,v in sample.items():
                if type(v) == torch.Tensor:
                    sample[k] = v.to(compute_device) 
           
            if config['state_prediction']: 
                state_output = model(sample, mode='state')
                state_labels = sample['state_out']
                state_loss = criterion(state_output.reshape(-1, state_output.shape[-1]), state_labels.reshape(-1).long(), 'state')
                total_state_loss += state_loss.item()
                state_acc, output_preds = get_raw_state_acc(state_output, vocab, state_vocab, 
                    sample, output_preds, config, mode='seq')
                total_state_acc += state_acc
            
            if config['mask_bb'] and config['mask_resnext']:
                bb_output, resnext_output = model.decode_masked_video(sample, mode='bb_resnext')
                bb_loss = self_supervised_loss_compute(bb_output, sample, bb_criterion, mode='bb')
                resnext_loss = self_supervised_loss_compute(resnext_output, sample, resnext_criterion, mode='resnext')
                total_bb_loss += bb_loss.item()
                total_resnext_loss += resnext_loss.item()

            elif config['mask_bb']: 
                bb_output = model.decode_masked_video(sample, mode='bb') 
                bb_loss = self_supervised_loss_compute(bb_output, sample, bb_criterion, mode='bb')
                total_bb_loss += bb_loss.item() 

            elif config['mask_resnext']: 
                resnext_output = model.decode_masked_video(sample, mode='resnext')
                resnext_loss = self_supervised_loss_compute(resnext_output, sample, resnext_criterion, mode='resnext')
                total_resnext_loss += resnext_loss.item()

        avg_state_loss = total_state_loss / len(data_loader)
        avg_state_acc = total_state_acc / len(data_loader) 
        avg_bb_loss = total_bb_loss / len(data_loader) 
        avg_resnext_loss = total_resnext_loss / len(data_loader)

        return avg_state_loss, avg_state_acc, output_preds, avg_bb_loss, avg_resnext_loss 

def training_dst_main(model_name: str, train_config: Dict[str, Any], model_config: Dict[str, int]):

    # create train and dev datasets using the files specified in the training configuration
    train_video_ft_dir = train_config["train_video_ft_dir"]
    train_video_labels_dir = train_config["train_video_labels_dir"]
    train_dial_dir = train_config["train_dial_dir"]

    dev_video_ft_dir = train_config["dev_video_ft_dir"]
    dev_video_labels_dir = train_config["dev_video_labels_dir"]
    dev_dial_dir = train_config["dev_dial_dir"]

    num_dials = train_config['num_dials']

    train_video_resnext_dir = train_config['train_video_resnext_dir']
    dev_video_resnext_dir = train_config['dev_video_resnext_dir']

    train_dataset: data.Dataset = DatasetsFactory.get_training_dst_dataset(model_name, train_video_ft_dir, train_video_labels_dir,
        train_dial_dir, train_video_resnext_dir, train_config)
    dev_dataset: data.Dataset = DatasetsFactory.get_training_dst_dataset(model_name, dev_video_ft_dir, dev_video_labels_dir, 
        dev_dial_dir, dev_video_resnext_dir, train_config, 
        vocab=train_dataset.vocab, state_vocab=train_dataset.state_vocab, 
        res_vocab=train_dataset.res_vocab)

    # save vocab 
    saved_vocab = {
            'all': train_dataset.vocab,
            'state': train_dataset.state_vocab,
            'res': train_dataset.res_vocab
            }
    exp_path = Path(train_config['checkpoints_path']) 
    exp_path.mkdir(parents=True, exist_ok=True)
    vocab_file = exp_path / "vocab.pkl"
    pkl.dump(saved_vocab, open(vocab_file, 'wb'))

    # save config files 
    config_file = exp_path / "train_config.json"
    json.dump(train_config, open(config_file, 'w'), indent=4)
    config_file = exp_path / "model_config.json"
    json.dump(model_config, open(config_file, 'w'), indent=4) 
    # save training loss logs
    log_file = exp_path / "log.csv" 
    with open(log_file, 'w') as f: 
        f.write('epoch, split, step, state_loss, state_acc, bb_loss, resnext_loss\n') 

    # training hyper parameters and configuration
    batch_size = train_config["batch_size"]
    num_workers = train_config["num_workers"]
    num_epochs = train_config["num_epochs"]
    learning_rate = train_config["learning_rate"]
    print_batch_step = train_config["print_step"]
    inference_batch_size = train_config["inference_batch_size"]
    checkpoints_path = train_config["checkpoints_path"]
    device = torch.device(train_config["device"])

    # model, loss and optimizer
    model: nn.Module = ModelsFactory.get_model(model_name, model_config, 
        train_dataset.vocab, train_dataset.state_vocab, train_dataset.res_vocab, train_config)

    model_opt = NoamOpt(model_config['word_embedding_dim'], 1, train_config['warmup_steps'],
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    
    if train_config['label_smoothing']:
        criterion =  LabelSmoothing(len(train_dataset.state_vocab), len(train_dataset.res_vocab), 
            padding_idx=0, smoothing=0.1).cuda()
    else:
        pad_idx = 0 
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).cuda() 
    bb_criterion = nn.L1Loss(reduction='none') 
    resnext_criterion = nn.L1Loss(reduction='none')
    
    # create data loaders
    train_config_dict = {"batch_size": batch_size, "num_workers": num_workers, "shuffle": True}
    inference_config_dict = {"batch_size": inference_batch_size, "num_workers": num_workers}
    training_loader = data.DataLoader(train_dataset, 
            collate_fn=partial(collate_fn, vocab=train_dataset.vocab, config=train_config,
            ), **train_config_dict)
    dev_loader = data.DataLoader(dev_dataset, 
            collate_fn=partial(collate_fn, vocab=train_dataset.vocab, config=train_config,
            ),**inference_config_dict)

    # Start training
    model = model.to(device)
    min_dev_loss: float = 1e10
    train_start_time = time.time()

    stop_count = 0 

    for epoch in range(num_epochs):
        model.train(mode=True)
        epoch_num = epoch + 1

        # loss statistics
        batches_running_state_loss = 0
        batches_running_state_acc = 0 
        batches_running_bb_loss = 0 
        batches_running_resnext_loss = 0 

        for batch_idx, sample in tqdm(enumerate(training_loader, 1), total=len(training_loader), position=0, leave=True, ncols=0):
            
            for k,v in sample.items(): 
                if type(v) == torch.Tensor: 
                    sample[k] = v.to(device) 

            model_opt.optimizer.zero_grad()
            loss = 0 

            if train_config['state_prediction']: 
                state_output = model(sample, mode='state')
                state_labels = sample['state_out']
                state_loss = criterion(state_output.reshape(-1, state_output.shape[-1]), state_labels.reshape(-1).long(), 'state')
                state_acc, state_preds = get_raw_state_acc(state_output, train_dataset.reverse_vocab, 
                    train_dataset.reverse_state_vocab, sample, {}, train_config, mode='seq')
                loss += state_loss 
                batches_running_state_loss += state_loss.item()
                batches_running_state_acc += state_acc

            if train_config['mask_bb'] and train_config['mask_resnext']:
                bb_output, resnext_output = model.decode_masked_video(sample, mode='bb_resnext')
                bb_loss = self_supervised_loss_compute(bb_output, sample, bb_criterion, mode='bb')
                loss += bb_loss
                batches_running_bb_loss += bb_loss.item()
                resnext_loss = self_supervised_loss_compute(resnext_output, sample, resnext_criterion, mode='resnext')
                loss += resnext_loss
                batches_running_resnext_loss += resnext_loss.item()

            elif train_config['mask_bb'] or train_config['track_bb']: 
                bb_output = model.decode_masked_video(sample, mode='bb')
                bb_loss = self_supervised_loss_compute(bb_output, sample, bb_criterion, mode='bb') 
                loss += bb_loss
                batches_running_bb_loss += bb_loss.item()

            elif train_config['mask_resnext']: 
                resnext_output = model.decode_masked_video(sample, mode='resnext')
                resnext_loss = self_supervised_loss_compute(resnext_output, sample, resnext_criterion, mode='resnext')
                loss += resnext_loss 
                batches_running_resnext_loss += resnext_loss.item() 

            loss.backward()
            model_opt.step()

            # print inter epoch statistics
            if batch_idx % print_batch_step == 0:
                
                average_running_state_loss = batches_running_state_loss / print_batch_step
                average_running_state_acc = batches_running_state_acc / print_batch_step 
                average_running_bb_loss = batches_running_bb_loss / print_batch_step 
                average_running_resnext_loss = batches_running_resnext_loss / print_batch_step 

                with open(log_file, 'a') as f: 
                    f.write('{},{},{},{},{},{},{}\n'.format(epoch, 'train', batch_idx, 
                    average_running_state_loss, average_running_state_acc, average_running_bb_loss, average_running_resnext_loss))
                
                batches_running_state_loss = 0
                batches_running_state_acc = 0
                batches_running_bb_loss = 0 
                batches_running_resnext_loss = 0 

        dev_state_loss, dev_state_acc, dev_preds, dev_bb_loss, dev_resnext_loss = loss_compute(model_name, model, device, 
            dev_loader, criterion, bb_criterion, resnext_criterion, 
            train_dataset.reverse_vocab, train_dataset.reverse_state_vocab, train_dataset.reverse_res_vocab, 
            train_config)

        print("Experiment path: {}".format(checkpoints_path))
        print("Epoch {} Dev Set: State Loss {:.4f} Acc {:4f} BB Loss {:4f} ResNext Loss {:4f}".format(
            epoch_num, dev_state_loss, dev_state_acc, dev_bb_loss, dev_resnext_loss))

        with open(log_file, 'a') as f: 
            f.write('{},{},{},{},{},{},{}\n'.format(epoch, 'valid', -1, 
            dev_state_loss, dev_state_acc, dev_bb_loss, dev_resnext_loss))
       
        dev_loss = 0 
        if train_config['state_prediction']:
            dev_loss += dev_state_loss

        # check if it is the best performing model so far and save it
        if dev_loss < min_dev_loss:
            min_dev_loss = dev_loss
            save_checkpoint(model, model_name, round(min_dev_loss, 5), checkpoints_path, dev_preds, epoch)
            stop_count = 0 
        else:
            stop_count += 1

        if stop_count == train_config['stop_count']: 
            print("Model has not improved for the last {} epochs. stop straining now".format(stop_count))
            break 
