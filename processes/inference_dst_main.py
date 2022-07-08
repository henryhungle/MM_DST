import json
import numpy as np
import pickle
from typing import List, Dict
from pathlib import Path
import re 
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils import data
import torch.nn.functional as F 
from baselines.models_factory import ModelsFactory 
from datasets.datasets_factory import DatasetsFactory
from datasets.dst_datasets import collate_fn 
import json 
import pdb 
from functools import partial 

def greedy(model, sample, vocab, reverse_state_vocab, max_len):
    
    obj_pattern = re.compile('<obj\d+>') 
    frame_pattern = re.compile('<frame\d+>')
    ds = torch.ones(1, 1).fill_(vocab['<state>']).long() 
    curr_obj = None
    curr_slot = None 
    start_frame = -1
    end_frame = -1 
    seq = [] 
    
    for l in range(max_len): 
        state = model.decode_state(sample, ds.cuda())
        if type(state) == tuple:
            assert state[1] == 'sequicity'
            state = state[0] 
        indices = torch.argsort(state).cpu().numpy()[::-1]
        found_token = False 
        for idx in indices: 
            token = reverse_state_vocab[idx] 
            if frame_pattern.match(token): # is a frame 
                if start_frame == -1: 
                    start_frame = token 
                elif end_frame == -1: 
                    end_frame = token 
                else:
                    continue # ignore this token 
            elif obj_pattern.match(token): # is an object 
                if token in seq: 
                    continue # duplicate object, skipping this token 
                else:
                    curr_obj = token 
            elif token in ['SIZE', 'COLOR', 'MATERIAL', 'SHAPE']: 
                if curr_obj is None: 
                    continue # no assign object, skipping this token 
                else: 
                    curr_slot = token 
            elif token not in vocab: # e.g. output response token 
                continue 
            else:
                if curr_obj is None or curr_slot is None: 
                    continue # no assign object/slot, skipping this token
            seq.append(token)
            in_token_idx = vocab[token]
            ds = torch.cat([ds, torch.ones(1,1).long().fill_(in_token_idx)], dim=1)
            found_token = True 
            found_eos = (token == '<end_state>')
            break 
        if not found_token: break # avoid infinite loops 
        if found_eos: break
    if found_eos:
        seq = seq[:-1] 
        ds = ds[:, :-1] 
    return seq, ds
        

def beam_search(model, sample, vocab, reverse_state_vocab, max_len, config): 
    
    min_len, beam, penalty, nbest = 1, 5, 1.0, 5
    end_symbol, pad_symbol = vocab['<end_state>'], vocab['<pad>']
    start_symbol = vocab['<state>']

    ds = torch.ones(1, 1).fill_(start_symbol).long()
    hyplist=[([], 0., ds)]                                                
    best_state=None     
    comp_hyplist=[]

    for l in range(max_len):
        new_hyplist = []
        argmin = 0
        for out, lp, st in hyplist:
            state = model.decode_state(sample, st.cuda()) 
            logp = F.log_softmax(state)
            lp_vec = logp.cpu().data.numpy() + lp 
            if l >= min_len:
                new_lp = lp_vec[end_symbol] + penalty * (len(out) + 1)
                comp_hyplist.append((out, new_lp, st)) 
                if best_state is None or best_state < new_lp:
                    best_state = new_lp
            count = 1   
            for o in np.argsort(lp_vec)[::-1]:
                new_lp = lp_vec[o]
                
                if True:
                    token = reverse_state_vocab[int(o)]
                    if token not in vocab: # e.g. output response token
                        continue 

                token_idx = vocab[token]

                if len(new_hyplist) == beam:
                    if new_hyplist[argmin][1] < new_lp:
                        new_st = torch.cat([st, torch.ones(1,1).long().fill_(token_idx)], dim=1)
                        new_hyplist[argmin] = (out + [token], new_lp, new_st)
                        argmin = min(enumerate(new_hyplist), key=lambda h:h[1][1])[0]
                    else:   
                        break
                else:       
                    new_st = torch.cat([st, torch.ones(1,1).long().fill_(token_idx)], dim=1)
                    new_hyplist.append((out + [token], new_lp, new_st))
                    if len(new_hyplist) == beam:
                        argmin = min(enumerate(new_hyplist), key=lambda h:h[1][1])[0]
                count += 1  
        hyplist = new_hyplist
    if len(comp_hyplist) > 0:
        maxhyps = sorted(comp_hyplist, key=lambda h: -h[1])[:nbest]
        return maxhyps, best_state
    else:
        return [([], 0, [])], None


def inference_dst_main(model_name: str, results_dir: str, train_config_path: str, inference_config_path: str, model_config_path: str, model_path, decode_style):
    with open(inference_config_path, "rb") as f:
        config: Dict[str, str] = json.load(f)

    # copy configs from training (e.g. dataset preprocessing params) 
    train_config = json.load(open(train_config_path, 'rb'))
    for k,v in train_config.items():
        if k not in config: 
            config[k] = v 
    
    state_maxlen = config['state_maxlen']

    with open(model_config_path, "rb") as f:
        model_config: Dict[str, int] = json.load(f)

    video_ft_dir = config["test_video_ft_dir"]
    video_label_dir = config["test_video_labels_dir"]
    resnext_dir = config['test_video_resnext_dir']
    dial_dir = config['test_dial_dir'] 

    batch_size = int(config["batch_size"])
    num_workers = int(config["num_workers"])
    device = torch.device(config["device"])

    # load vocab 
    vocab_file = Path(results_dir) / "vocab.pkl"
    vocab = pickle.load(open(vocab_file, 'rb'))
   
    # load model
    model: nn.Module = ModelsFactory.get_model(model_name, model_config, vocab['all'], vocab['state'], vocab['res'], config, model_path) 
    
    model.eval()
    model.to(device)
    
    config['mask_bb'] = 0
    config['mask_resnext'] = 0 
    config['track_bb'] = 0 

    dataset: data.Dataset = DatasetsFactory.get_inference_dst_dataset(model_name, 
        video_ft_dir, video_label_dir, dial_dir, resnext_dir, config, 
        vocab=vocab['all'], state_vocab=vocab['state'], res_vocab=vocab['res']) 
    data_loader = data.DataLoader(dataset, 
        collate_fn=partial(collate_fn, vocab=vocab['all'], config=config), 
        batch_size=batch_size, num_workers=num_workers)
    dataset_length = len(dataset) 
    
    with torch.no_grad():
        
        if True: 
            reverse_state_vocab = {v:k for k,v in vocab['state'].items()}
            reverse_res_vocab = {v:k for k,v in vocab['res'].items()}
            reverse_vocab = {v:k for k,v in vocab['all'].items()} 
            prior_state_token = vocab['all']['<prior_state>']
        
        all_preds = {} 
        for batch_idx, sample in tqdm(enumerate(data_loader), total=len(data_loader), position=0, leave=True, ncols=0):
            for k,v in sample.items():
                if type(v) == torch.Tensor:
                    sample[k] = v.to(device)
            vid, turn_idx = sample['vid'][0], sample['turn_idx'][0]
            if vid not in all_preds: all_preds[vid] = {}
            if turn_idx not in all_preds[vid]: all_preds[vid][turn_idx] = {} 
            
            if config['state_prediction'] and config['prior_state']:
                if turn_idx !=0: # update prior state as decoded state from prior turn 
                    sample['dial'] = torch.cat([prior_state.cuda(), sample['dial_only']], dim=-1)

            if config['state_prediction']:
                if decode_style == 'greedy':
                    # greedy decoding  
                    decoded_state, state = greedy(model, sample, vocab['all'], reverse_state_vocab, state_maxlen) 
                elif decode_style == 'beam_search': 
                    # beam search decoding 
                    output = beam_search(model, sample, vocab['all'], reverse_state_vocab, state_maxlen, config) 
                    decoded_state = output[0][0][0] 
                    state = output[0][0][2] 

                    if "<end_state>" in decoded_state:
                        end_state_idx = decoded_state.index('<end_state>') 
                        decoded_state = decoded_state[:end_state_idx]
                        state = state[:,:end_state_idx+1] #position zero is the <state> token 

                if config['prior_state']:
                    prior_state = torch.cat([torch.tensor([[prior_state_token]]), state[:,1:]], dim=-1)

            else:
                decoded_state = ['']

            decoded_res = ''

            all_preds[vid][turn_idx] = {
                    "gt_state": ' '.join(sample['state_raw'][0][1:-1]),
                    "pred_state": ' '.join(decoded_state),
                    "gt_res": sample['res_raw'][0],
                    "pred_res": decoded_res,
                    'gt_segment': sample['gt_temporal'][0], 
                    }

    result_file = Path(results_dir) / "all_preds_{}_maxlen{}.json".format(decode_style, state_maxlen)
    print('Saving results to {}'.format(result_file))
    json.dump(all_preds, open(result_file, 'w'), indent=4)

            

