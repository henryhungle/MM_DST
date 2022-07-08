import pickle
import json
from typing import List, Tuple, Dict
from pathlib import Path
from tqdm import tqdm
import copy 
import torch
import numpy as np

from datasets.dst_utils import * 
from object_indices import * 
import pdb 

ATTRIBUTES = ['<Z>', '<C>', '<<M>', '<S>']

def delexicalize_dial(dial, mapping):
    delex_dial = [] 
    
    for turn_idx, turn in enumerate(dial): 
        delex_question = copy.deepcopy(turn['question'])
        delex_answer = str(copy.deepcopy(turn['answer']))
        for k,v in mapping.items():
            delex_question = delex_question.replace(k, v)
            delex_answer = delex_answer.replace(k, v) 
        turn = {'question': delex_question, 'answer': delex_answer}
        delex_dial.append(turn)
    
    return delex_dial

def delexicalize(sent, mapping):
    sent = copy.deepcopy(sent)
    for k,v in mapping.items():
        sent = sent.replace(k,v)
    return sent

def get_label_mapping(objects): 
    mapping = {}
    for obj_idx, obj in enumerate(objects): 
        key = '{}_{}_{}_{}'.format(obj['size'], obj['color'], obj['shape'], obj['material'])
        label = OBJECTS_NAME_TO_IDX[key]
        mapping[str(obj_idx)] = label 
    return mapping 

def get_state_by_labels(state, mapping): 
    new_state = {}
    for k,v in state.items():
        new_k = mapping[k]
        new_state[new_k] = v 
    return new_state 

def preprocess_dial(process_args: List):

    dial_path, attribute_mapping, results_dir, video_dir = process_args

    # read dialogues
    dial = json.load(open(dial_path, 'r'))[0]

    # read video scenes
    video_name = Path(dial_path).stem
    video = json.load(open(video_dir + '/' + video_name + '.json', 'r'))

    # map objects to labels 
    label_mapping = get_label_mapping(video['objects']) 

    prior_template = None
    prior_state = {} 

    for turn_idx, turn in enumerate(dial):
        if turn_idx == len(dial)-1: break 
        delex_question = delexicalize(turn['question'], attribute_mapping)
        delex_answer = delexicalize(str(turn['answer']), attribute_mapping)
        turn['delex_question'] = delex_question
        turn['delex_answer'] = delex_answer
	
        turn['temporal_cutoff'] = get_end_time(turn['template']['cutoff']) 
        turn['gt_turn_temporal'] = get_temporal_boundary(turn['template']['used_periods'][-1])

        raw_state = dial[turn_idx+1]['template']['used_objects']
        state1 = clean_state(raw_state)
        
        # original DVD object annotations include object information appearing in the answer of the current turn 
        # For DST tasks, object information is only up to the question, and any information from the answer are excluded 
        state2 = remove_answer_obj_from_state(prior_state, state1, 
            turn['template'], prior_template, turn['final_all_program']) 
                    
        # original DVD object annotations also inclue objects that are sampled to support the next turn's question
        # e.g. quesitons like "among these objects", there is a red rubber cube,...."
        # For DST tasks, only consider objects up to the current turn and so such these objects are excluded 
        state3 = remove_future_obj_from_state(prior_state, state2, 
                dial[turn_idx+1], turn_idx+1) 

        final_state = state3
        final_state_by_labels = get_state_by_labels(final_state, label_mapping) 

        turn['final_state'] = final_state
        turn['final_state_by_labels'] = final_state_by_labels

        prior_template = turn['template']
        prior_state = final_state

    # save the preprocess dialogues as json files 
    dial_path = Path(dial_path)
    results_dir = Path(results_dir)
    video_name = dial_path.stem
    output_path = results_dir / (video_name + ".json")
    json.dump(dial, open(output_path, "w"))

def get_experiment_dialogues(config): 
    dials_dir = config['dials_dir']
    return [str(path) for path in Path(dials_dir).glob("*.json")]

def get_attribute_mapping(metadata, synonyms): 
    mapping = {}
    for k,v in metadata['types'].items():
        if k in ['Color', 'Size', 'Material', 'Shape']: 
            token = k.upper() 
            for i in v: 
                if i in synonyms: 
                    for s in synonyms[i]:
                        mapping[s] = token
                else:
                    mapping[i] = token 
    return mapping 


def preprocess_dial_main(results_dir: str, config_path: str) -> None:
    
    with open(config_path, "rb") as f:
        config = json.load(f)

    # extract paths to dialogue files for the experiment
    experiment_dials = get_experiment_dialogues(config)
    num_dials = len(experiment_dials)

    metadata = json.load(open(config['dials_metadata'], 'r'))
    synonyms = json.load(open(config['dials_synonyms'], 'r')) 
    video_dir = config['video_dir']
    attribute_mapping = get_attribute_mapping(metadata, synonyms) 


    process_args = [(path, attribute_mapping, results_dir, video_dir) for path in experiment_dials]
    for i in tqdm(range(num_dials)):
        process_args_i = process_args[i]
        preprocess_dial(process_args_i)
