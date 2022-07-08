import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
from functools import cmp_to_key

import nltk 
import itertools 
import copy
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence  
from tqdm import tqdm 
import math 

from object_indices import is_cone_object, OBJECTS_IDX_TO_NAME
from object_attributes import OBJECT_ATTRIBUTES 
from datasets.dst_utils import * 
from datasets.video_utils import * 

import pdb 

class DstAbstractDataset(Dataset):
    def __init__(self, video_ft_dir: str, video_label_dir: str, dial_dir: str, resnext_dir, 
        vocab, state_vocab, res_vocab, 
        config):
        self.video_ft_dir: Path = Path(video_ft_dir)
        self.video_label_dir: Path = Path(video_label_dir)
        self.dial_dir: Path = Path(dial_dir)
        self.resnext_dir: Path = Path(resnext_dir) 
        self.config = config 

        # add video names and labels data
        self.video_ft_data = {} 
        self.video_label_data = {} 

        # add dialogue turn names
        self.dial_data = [] 
        self.items = []

        # extra variables
        self.max_objects: int = config['max_objects'] 
        self.num_obj_classes = 193 
        self.frame_shapes = np.array([320, 240, 320, 240]) 
        self.frame_rate = config['frame_rate']
        self.max_frames = round(300/self.frame_rate)
        self.resnext_sampling_rate = -1 
        # start from frame 1
        self.frame_map = {i+1:j for i,j in enumerate(range(1,301, self.frame_rate))}
        self.frame_map[0] = 0 
        self.segment_map = None 

        # word vocabulary 
        self.vocab = vocab 
        self.state_vocab = state_vocab 
        self.res_vocab = res_vocab 

        # initialize dataset 
        self.num_dials = config['num_dials'] 

        self._init_dataset() 

    def _load_state_values(self, values):
        output = [] 
        if '<Z>' in values:
            output.extend(['SIZE', values['<Z>']])
        if '<C>' in values:
            output.extend(['COLOR', values['<C>']])
        if '<M>' in values:
            output.extend(['MATERIAL', values['<M>']])
        if '<S>' in values:
            output.extend(['SHAPE', values['<S>']])
        return output 

    def _load_dst_labels(self, vid, state, gt_segment, mode=None):
        state = {int(k):v for k,v in state.items()}
        sorted_keys = sorted(state.keys())
        output = ['<state>']
        state_objects = [] 

        if self.config['temporal_grounding'] and gt_segment is not None:  
            gt_start = '<frame{}>'.format(self.get_frame(gt_segment[0])+1)
            gt_end = '<frame{}>'.format(self.get_frame(gt_segment[1])+1)
            output += [gt_start] + [gt_end]

        if True: 
            for key in sorted_keys: 
                output.append('<obj{}>'.format(key))
                state_objects.append('<obj{}>'.format(key))
                output.extend(self._load_state_values(state[key]))
        output.append('<end_state>')
    
        encoded_state_in, encoded_state_out, encoded_objects = self._load_dst_sequence(vid, state, gt_segment, output, state_objects, mode)

        return encoded_state_in, encoded_state_out, output, encoded_objects

    def _load_dst_sequence(self, vid, state, gt_segment, gt_state_sequence, state_objects, mode): 
        if 'obj_label_filter' not in self.config or self.config['obj_label_filter']: 
            # only obj found in detector models are applied in training 
            # if missing, consider a limitation of the detection model 
            objects = self.video_ft_data[vid]['object_mapping']
            obj_state = ['<state>']
            state_objects = [] 
            if self.config['temporal_grounding'] and gt_segment is not None: 
                gt_start = '<frame{}>'.format(self.get_frame(gt_segment[0])+1)
                gt_end = '<frame{}>'.format(self.get_frame(gt_segment[1])+1)
                obj_state += [gt_start] + [gt_end]
            for obj_idx in range(self.video_ft_data[vid]['num_possible_objects']):
                obj_id = objects[obj_idx]
                if obj_id in state: 
                    obj_state.append('<obj{}>'.format(obj_id))
                    state_objects.append('<obj{}>'.format(obj_id))
                    obj_state.extend(self._load_state_values(state[obj_id]))
            obj_state.append('<end_state>')
        else:
            # directly use the object labels from the ground truth state 
            obj_state = gt_state_sequence 

        encoded_state_in = [self.word2id(t) for t in obj_state[:-1]]
        
        if mode == 'in_seq_only':
            return encoded_state_in, None, None
         
        encoded_state_out = [self.word2id(t, self.state_vocab) for t in obj_state[1:]]
        encoded_objects = [self.word2id(o) for o in state_objects]

        return encoded_state_in, encoded_state_out, encoded_objects   

    def _get_resnext_sampling_rate(self, clips): 
        clip1 = clips[0]['segment']
        clip2 = clips[1]['segment']
        return int(self.frame_rate / (clip2[0] - clip1[0]))

    def _get_video_ft(self): 
        for video_name in tqdm(self.video_names, total=len(self.video_names), desc="loading video ft"):
            ft_file = Path(self.video_ft_dir) / (video_name + '.pkl')
            vft = pickle.load(open(ft_file, 'rb')) 
            for k,v in vft.items():
                vft[k] = v[::self.frame_rate]
            vft = normalize_and_pad_bb(vft, self.max_objects, self.frame_shapes, self.vocab, self.config['temporal_grounding'])
            self.video_ft_data[video_name] = vft

            if self.config['add_resnext']: 
                ft_file = Path(self.resnext_dir) / (video_name + '.pkl') 
                resnext_ft = pickle.load(open(ft_file, 'rb')) 
                if self.resnext_sampling_rate == -1: 
                    self.resnext_sampling_rate = self._get_resnext_sampling_rate(resnext_ft['clips'])
                resnext_ft['clips'] = resnext_ft['clips'][::self.resnext_sampling_rate]
                resnext_ft, segment_map = get_resnext_ft(resnext_ft, vft['num_possible_objects']) 
                self.video_ft_data[video_name]['resnext'] = resnext_ft
                if self.segment_map is None:
                    self.segment_map = segment_map 

    def _get_bb_labels(self):
        for video_name in tqdm(self.video_names, total=len(self.video_names), desc="loading bb labels"): 
            label_file =Path(self.video_label_dir) / (video_name + '.pkl')
            label = pickle.load(open(label_file, 'rb')) 
            for k,v in label.items():
                label[k] = v[::self.frame_rate]
            vft = self.video_ft_data[video_name]
            self.video_ft_data[video_name] = normalize_bb_label(label, vft, self.frame_shapes) 
    
    def _build_res_vocab(self):
        answer_options =  ['<pad>', '0', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9', 'False', 'True', 'blue', 'brown', 'cone', 'cube', 'cyan', 'cylinder', 'flying', 'flying,rotating', 'flying,rotating,sliding', 'flying,sliding', 'gold', 'gray', 'green', 'large', 'medium', 'metal', 'no action', 'purple', 'red', 'rotating', 'rotating,sliding', 'rubber', 'sliding', 'small', 'sphere', 'spl', 'yellow']
        res_vocab = {}
        for option in answer_options:
            res_vocab[option] = len(res_vocab)        
        
        self.res_vocab = res_vocab 
        self.reverse_res_vocab = {v:k for k,v in self.res_vocab.items()}

    def _build_vocab(self):
        word_freq = {} 
        for dialog in tqdm(self.dial_data, total=len(self.dial_data), desc="build vocab"): 
            for turn in dialog:
                for word in nltk.word_tokenize(turn['question']):
                    if word not in word_freq: 
                        word_freq[word] = 0 
                    word_freq[word] += 1 
                for word in nltk.word_tokenize(str(turn['answer'])): 
                    if word not in word_freq:
                        word_freq[word] = 0 
                    word_freq[word] += 1
        
        special_tokens = ['<pad>', 
                '<video>', '<frame>',
                '<unk>', 
                '<usr>', '<sys>', 
                '<context>', '<state>', '<end_state>', '<prior_state>', 
                '<response>', '<end_response>',
                'SIZE', 'COLOR', 'MATERIAL', 'SHAPE', '<start>', '<end>']
        if 'state_classify' in self.config and self.config['state_classify']:
            state_special_tokens = ['<pad>']
        else:
            state_special_tokens = ['<pad>', '<state>', '<end_state>', 'SIZE', 'COLOR', 'MATERIAL', 'SHAPE']
        answer_options =  ['<pad>', '0', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9', 'False', 'True', 'blue', 'brown', 'cone', 'cube', 'cyan', 'cylinder', 'flying', 'flying,rotating', 'flying,rotating,sliding', 'flying,sliding', 'gold', 'gray', 'green', 'large', 'medium', 'metal', 'no action', 'purple', 'red', 'rotating', 'rotating,sliding', 'rubber', 'sliding', 'small', 'sphere', 'spl', 'yellow']

        vocab = {}
        for token in special_tokens:
            vocab[token] = len(vocab) 
        
        state_vocab = {}
        for token in state_special_tokens:
            state_vocab[token] = len(state_vocab)

        for i in range(self.num_obj_classes):
            obj_class = '<obj{}>'.format(i) 
            vocab[obj_class] = len(vocab)
            if 'state_classify' not in self.config or not self.config['state_classify']:
                state_vocab[obj_class] = len(state_vocab)
        
        for k,v in OBJECT_ATTRIBUTES.items():
            for a in v:
                if a not in vocab:
                    vocab[a] = len(vocab)
                if a not in state_vocab: 
                    state_vocab[a] = len(state_vocab)
        
        if self.config['temporal_grounding']:
            for i in range(self.max_frames+1): #include last frame #300 
                frame_id = '<frame{}>'.format(i+1) #start from frame 1  
                vocab[frame_id] = len(vocab)
                state_vocab[frame_id] = len(state_vocab) 
        
        for word, freq in word_freq.items():
            if word not in vocab:
                vocab[word] = len(vocab) 

        self.vocab = vocab 
        self.reverse_vocab = {v:k for k,v in self.vocab.items()} 

        if self.config['share_vocab']: 
            self.state_vocab = vocab
        else:
            self.state_vocab = state_vocab 
        self.reverse_state_vocab = {v:k for k,v in self.state_vocab.items()}

    def words2ids(self, string, sos_token): 
        words = nltk.word_tokenize(string)
        sentence = np.ndarray(len(words)+1, dtype=np.int32)
        sentence[0] = self.vocab[sos_token]
        for i,w in enumerate(words):
            if w in self.vocab: 
                sentence[i+1] = self.vocab[w]
            else:
                sentence[i+1] = self.vocab['<unk>']
        return sentence 

    def word2id(self, word, vocab=None):
        if vocab is None: return self.vocab[word]
        return vocab[word] 

    def get_frame(self, frame_num):
        # if frame id > max frames, return max frames 
        # consider this as limitation of the video encoder model 
        return round(frame_num/self.frame_rate)

    def get_frame_cutoff(self, cutoff): 
        for k in range(len(self.frame_map)): 
            v = self.frame_map[k] 
            if v <= cutoff: 
                frame_cutoff = k 
            else:
                break 
        if frame_cutoff <= 0: pdb.set_trace() 
        return frame_cutoff 

    def get_segment_cutoff(self, cutoff, frame_cutoff): 
        segment_cutoff = min(frame_cutoff, len(self.segment_map)-1)
        segment = self.segment_map[segment_cutoff]
        while(segment[-1] > cutoff):
            segment_cutoff -= 1  # to avoid accidentially include segment features outside of temporal cutoff point 
            segment = self.segment_map[segment_cutoff] 
        if segment_cutoff <= 0: pdb.set_trace()
        return segment_cutoff 

    def _init_dataset(self) -> None:
        if True: 
            # load dialogue pdb.set_trace()s
            video_names = set()
            dial_files = list(Path(self.dial_dir).glob("*.json"))[:self.num_dials]
            for dial_file in tqdm(dial_files, total=len(dial_files), desc="loading dial data"): 
                dial_data = json.load(open(dial_file, 'r'))
                self.dial_data.append(dial_data)
                video_names.add(dial_file.stem) 
            self.video_names = sorted(list(video_names)) 

            # load vocab
            if self.vocab is None: 
                self._build_vocab() 
            
            self._build_res_vocab() 

            # load video features 
            self._get_video_ft()

            if self.config['track_bb']: 
                self._get_bb_labels() 

            # iterate dialogue to make data items by dialogue turns
            max_dial_len = 0 
            for dialog in tqdm(self.dial_data, total=len(self.dial_data), desc="process dial turns"):
                
                questions = [self.words2ids(t['question'], '<usr>') for t in dialog[:-1]]
                answers = [self.words2ids(str(t['answer']), '<sys>') for t in dialog[:-1]]
                
                responses = [self.res_vocab[str(t['answer'])] for t in dialog[:-1]]
                
                delex_questions = [self.words2ids(t['delex_question'], '<usr>') for t in dialog[:-1]]
                delex_answers = [self.words2ids(str(t['delex_answer']), '<sys>') for t in dialog[:-1]]

                qa_pair = [np.concatenate((q,a)).astype(np.int32) for q,a in zip(questions, answers)]
                delex_qa_pair = [np.concatenate((q,a)).astype(np.int32) for q,a in zip(delex_questions, delex_answers)]
                vid = dialog[0]['image']
                prior_state_seq = [self.word2id('<prior_state>')]
                for n in range(len(questions)):
    
                    history = np.asarray([]) 
                    turns = [[self.word2id('<context>')]]
                    delex_turns = [[self.word2id('<context>')]]
                    turn_indices= [[0]] 
                   
                    start_idx = max(0, n - self.config['max_turns'])
                    for m in range(start_idx, n):
                        history = np.append(history, qa_pair[m]) 
                        turns.append(qa_pair[m])
                        turn_indices.append([m+1]*len(qa_pair[m]))
                        delex_turns.append(delex_qa_pair[m])
                    question = questions[n]
                    turns.append(questions[n])
                    turn_indices.append([n+1]*len(questions[n]))
                    delex_turns.append(questions[n])

                    response_in = [self.word2id('<response>')]
                    response_out = [responses[n]]
                    dial_len = sum([len(t) for t in turns])
                    max_dial_len = max(max_dial_len, dial_len) 
                   
                    state_in, state_out, state_raw, state_objs = self._load_dst_labels(vid, dialog[n]['final_state_by_labels'], 
                            dialog[n]['gt_turn_temporal'])

                    prior_state_turns = [prior_state_seq] + turns 
 
                    gt_temporal = dialog[n]['gt_turn_temporal']
                    if gt_temporal is None: 
                        gt_temporal = (-1,-1) 
                        gt_frames = (-1,-1)
                    else:
                        gt_start = self.get_frame(gt_temporal[0])+1
                        gt_end = self.get_frame(gt_temporal[1])+1
                        gt_frames = (gt_start, gt_end) 
                        gt_temporal = tuple(gt_temporal) 

                    cutoff_raw = dialog[n]['temporal_cutoff']
                    frame_cutoff = self.get_frame_cutoff(cutoff_raw)
                    assert self.frame_map[frame_cutoff] <= cutoff_raw

                    if self.config['add_resnext']:
                        segment_cutoff = self.get_segment_cutoff(cutoff_raw, frame_cutoff)
                        assert self.segment_map[segment_cutoff][-1] <= cutoff_raw 
                    else:
                        segment_cutoff = -1 
                    
                    item = {
                            "vid":vid, 
                            "turn_idx":n, 
                            "history": history, 
                            "question": question, 
                            "turns": np.concatenate(turns),
                            "delex_turns": np.concatenate(delex_turns),
                            "turns_len": len(np.concatenate(turns)), 
                            "prior_state_turns": np.concatenate(prior_state_turns), 
                            "frame_cutoff": frame_cutoff, "segment_cutoff": segment_cutoff, 
                            'cutoff_raw': cutoff_raw, 
                            "gt_temporal": gt_temporal, 'gt_frames': gt_frames,  
                            'state_raw': state_raw, "state_objs": state_objs, 
                            "state_in": state_in, "state_out": state_out,
                            "original_state": dialog[n]['final_state_by_labels'], 
                            "response_raw": str(dialog[n]['answer']), 
                            "response_in": response_in, "response_out": response_out}
                    self.items.append(item) 
                    
                    if self.config['prior_state']:
                        prior_state_seq = [self.word2id('<prior_state>')] + state_in[1:]
                    
    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.tensor, torch.tensor], Tuple[torch.tensor, torch.tensor], str]:
        raise NotImplementedError("get item method must be implemented for dataset")

class DstDataset(DstAbstractDataset):
    def __init__(self, video_ft_dir: str, video_label_dir: str, 
            dial_dir: str, resnext_dir, vocab, state_vocab, res_vocab, config):
        super().__init__(video_ft_dir, video_label_dir, dial_dir, resnext_dir, 
            vocab, state_vocab, res_vocab, 
            config)

    def mask_video_by_state(self, vft, frames, objs):
        mask = []
        if frames != (-1,-1):
            start = frames[0]
            end = frames[1]
            assert start!=-1
            assert end!=-1
        else:
            start = 1
            end = len(vft) 
        for f_idx, f in enumerate(vft): 
            if f_idx == 0: 
                m = np.ones(len(f), dtype=bool) # <video> token 
            else:
                m = np.isin(f, objs)
                m[0] = True # <frame> token 
            mask.append(m) 
        mask = torch.tensor(np.concatenate(mask))
        return mask 

    def mask_resnext(self, vft, vft_tensor):
        output = [vft[0]] # <video> token position
        output_mask = [np.array([0])]
        prior_segment_idx = -1
        mask_segment = np.ones(vft[1].shape) * 0.00000001

        for i in range(1, len(vft)): 
            ft = vft[i] 
            mask = np.array([0] * len(ft))
           
            # for 75%, keep the original features 
            if np.random.rand() < 0.75: 
                output.append(ft) 
                output_mask.append(mask) 
                continue 

            if i == prior_segment_idx+1: # avoid masking 2 continous segments 
                output.append(ft)
                output_mask.append(mask) 
                continue 

            # only select one random position for prediction
            mask_index = np.random.randint(len(mask))
            mask[mask_index] = 1 
            
            output_mask.append(mask) 
            output.append(mask_segment) 
            prior_segment_idx = i 

        output = torch.tensor(np.concatenate(output), dtype=torch.float32)
        output_mask = torch.tensor(np.concatenate(output_mask), dtype=torch.int32)
        assert len(output) == len(output_mask)
        output_tensor = torch.zeros(vft_tensor.shape)
        output_tensor[:len(output)] = output 
        output_mask_tensor = torch.zeros(len(vft_tensor), dtype=torch.int32)
        output_mask_tensor[:len(output_mask)] = output_mask
        return output_tensor, output_mask_tensor

    def mask_bb(self, vft):
        original_output = torch.tensor(np.concatenate(vft), dtype=torch.float32)
        output = [vft[0]] # <video> token position 
        output_mask = [np.array([0])]
        prior_mask_obj = -1 
        mask_bb = np.ones(4)*0.00000001 
        for i in range(1, len(vft)):
            
            ft = vft[i]
            num_obj = len(ft) 
            mask = np.array([0] * num_obj)
            
            # position zero is <frame> token and is ignore 
            obj_indices = np.arange(1, num_obj)
            np.random.shuffle(obj_indices)
            mask_obj_idx = -1 
            
            for obj_idx in obj_indices: 
                # if padding bb, skip this iteration 
                if ft[obj_idx].sum() == 0:
                    continue
                # if this obj is mask in prior frame, skip this iteration 
                if obj_idx == prior_mask_obj:
                    continue 
                ft[obj_idx] = mask_bb # to avoid nan loss
                mask_obj_idx = obj_idx 
                mask[obj_idx] = 1 
                break

            output.append(ft)
            output_mask.append(mask)
            prior_mask_obj = mask_obj_idx 
        
        output = torch.tensor(np.concatenate(output), dtype=torch.float32)
        output_mask = torch.tensor(np.concatenate(output_mask), dtype=torch.int32) 

        assert len(output) == len(output_mask) 
        assert len(output) == len(original_output)
        return original_output, output, output_mask

    def __getitem__(self, idx: int): 
        item = self.items[idx]
        vid, turn_idx = item['vid'], item['turn_idx']
        turns = item['turns']
        frame_cutoff = item['frame_cutoff']    
        segment_cutoff = item['segment_cutoff'] 
        cutoff_raw = item['cutoff_raw']

        state_in = item['state_in']
        state_out = item['state_out']

        # load video ft 
        video_ft = self.video_ft_data[vid]
        # cut video by cutoff point (inclusive) 
        if self.config['mask_bb']:
            obj_bb, masked_obj_bb, bb_mask = self.mask_bb(video_ft['norm_bb'][:frame_cutoff+1])
        elif self.config['track_bb']: 
            obj_bb = torch.tensor(np.concatenate(video_ft['label_bb'][:frame_cutoff+1]), dtype=torch.float32)
            masked_obj_bb = torch.tensor(np.concatenate(video_ft['norm_bb'][:frame_cutoff+1]), dtype=torch.float32)
            bb_mask = torch.tensor(np.concatenate(video_ft['label_mask'][:frame_cutoff+1]), dtype=torch.int32)
        else:
            obj_bb = torch.tensor(np.concatenate(video_ft['norm_bb'][:frame_cutoff+1]), dtype=torch.float32)
            masked_obj_bb, bb_mask = None, None 
        objs = torch.tensor(np.concatenate(video_ft['encoded_objects'][:frame_cutoff+1]), dtype=torch.int32)
        assert len(obj_bb) == len(objs) 
        if self.config['max_objects']>0 and obj_bb.sum() == 0: pdb.set_trace()
        # filter video ft by state for response prediction 
        video_mask = None 
        
        # add resnext 
        resnext, masked_resnext, resnext_mask = None, None, None 
        if self.config['add_resnext']:
            resnext_ft = torch.tensor(np.concatenate(video_ft['resnext'][:segment_cutoff+1]), dtype=torch.float32)
            resnext = torch.zeros(len(objs), resnext_ft.shape[-1])
            resnext[:len(resnext_ft)] = resnext_ft 
            if self.config['mask_resnext']: 
                masked_resnext, resnext_mask = self.mask_resnext(video_ft['resnext'][:segment_cutoff+1], resnext)

        # load dialogue inputs 
        dial_tensor: torch.tensor = torch.tensor(turns, dtype=torch.int32)
        delex_dial_tensor = torch.tensor(item['delex_turns'], dtype=torch.int32) 

        if self.config['prior_state']:
            prior_state_dial = torch.tensor(item['prior_state_turns'], dtype=torch.int32) 
        else:
            prior_state_dial = None 

        state_in = torch.tensor(state_in, dtype=torch.int32)
        state_out = torch.tensor(state_out, dtype=torch.int32)

        res_in = torch.tensor(item['response_in'], dtype=torch.int32)
        res_out = torch.tensor(item['response_out'], dtype=torch.int32) 

        data_item = {
            'vid': vid, 'turn_idx': turn_idx,
            'frame_cutoff': frame_cutoff, 
            'obj_bb': obj_bb, 'objs': objs, 'resnext': resnext, 'video_len': len(resnext) if resnext is not None else 0,   
            'masked_obj_bb': masked_obj_bb, 'bb_mask': bb_mask, 
            'masked_resnext': masked_resnext, 'resnext_mask': resnext_mask, 
            'video_mask': video_mask, 
            'dial': dial_tensor, 'prior_state_dial': prior_state_dial, 'dial_len': item['turns_len'], 
            'delex_dial': delex_dial_tensor, 
            'state_in': state_in, 'state_out': state_out,
            'gt_temporal': item['gt_temporal'], 'gt_frames': item['gt_frames'], 
            'state_raw': item['state_raw'], 
            'original_state': item['original_state'], 
            'res_in': res_in, 'res_out': res_out, 'res_raw': item['response_raw'],  
        }
        return data_item  

def collate_fn(data, vocab, config): 
    batch = {}
    for k in data[0].keys():
        batch[k] = [d[k] for d in data]

    pad_token = vocab['<pad>']
    
    batch['obj_bb'] = pad_sequence(batch['obj_bb'], batch_first=True, padding_value=0.0) 
    batch['objs'] = pad_sequence(batch['objs'], batch_first=True, padding_value=pad_token) 
   
    if config['mask_bb'] or config['track_bb']: 
        batch['masked_obj_bb'] = pad_sequence(batch['masked_obj_bb'], batch_first=True, padding_value=0.0)
        batch['bb_mask'] = pad_sequence(batch['bb_mask'], batch_first=True, padding_value=0) 

    if config['add_resnext']: 
        batch['resnext'] = pad_sequence(batch['resnext'], batch_first=True, padding_value=0.0) 
        if config['mask_resnext']:
            batch['masked_resnext'] = pad_sequence(batch['masked_resnext'], batch_first=True, padding_value=0.0)
            batch['resnext_mask'] = pad_sequence(batch['resnext_mask'], batch_first=True, padding_value=0)

    batch['dial_only'] = pad_sequence(batch['dial'], batch_first=True, padding_value=pad_token)
   
    if config['prior_state']:
        batch['dial'] = pad_sequence(batch['prior_state_dial'], batch_first=True, padding_value=pad_token) 
    else:
        batch['dial'] = pad_sequence(batch['dial'], batch_first=True, padding_value=pad_token)
        batch['delex_dial'] = pad_sequence(batch['delex_dial'], batch_first=True, padding_value=pad_token)

    batch['state_in'] = pad_sequence(batch['state_in'], batch_first=True, padding_value=pad_token)
    batch['state_out'] = pad_sequence(batch['state_out'], batch_first=True, padding_value=pad_token)

    if config['no_state']:
        batch['state_in'] = torch.zeros(batch['state_in'].shape, dtype=torch.int32)
        batch['state_out'] = torch.zeros(batch['state_out'].shape, dtype=torch.int32)

    batch['res_in'] = pad_sequence(batch['res_in'], batch_first=True, padding_value=0)
    batch['res_out'] = pad_sequence(batch['res_out'], batch_first=True, padding_value=0) 
    
    obj_mask = batch['objs']!=pad_token
    dial_mask = batch['dial']!=pad_token
    
    if 'state_classify' in config and config['state_classify']:
        state_mask = batch['state_out'] != pad_token
        batch['state_mask'] = state_mask 
    else:
        state_mask = batch['state_in']!=pad_token
    res_mask = batch['res_in']!=pad_token 
    
    padding_mask = torch.cat([obj_mask, dial_mask, state_mask, res_mask], dim=-1)
    expanded_padding_mask = padding_mask.unsqueeze(1).repeat(1, padding_mask.shape[1], 1)

    video_dial_len = obj_mask.shape[1] + dial_mask.shape[1] 
    seq_len = padding_mask.shape[1]

    attn_mask = np.triu(np.ones((1, seq_len, seq_len)), k=1)
    attn_mask = torch.from_numpy(attn_mask) == 0 
    attn_mask = padding_mask.unsqueeze(1) & attn_mask
    
    # keep padding mask of video and dial features 
    attn_mask[:,:,:video_dial_len] = expanded_padding_mask[:,:,:video_dial_len]
    
    batch['attn_mask'] = attn_mask
    batch['padding_mask'] = padding_mask
    batch['res_attn_mask'] = expanded_padding_mask 
    return batch
