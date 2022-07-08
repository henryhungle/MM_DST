import json
import argparse
from typing import Dict, Any
import pdb 
from pathlib import Path 

from processes.training_dst_main import training_dst_main
from processes.inference_dst_main import inference_dst_main
from preprocess.preprocess_dial_main import preprocess_dial_main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing, Training, and inference on DVD-DST for multimodal DST tasks')
    subparsers = parser.add_subparsers()

    # create parser for dialogue preprocessing command 
    preprocess_dial_parser = subparsers.add_parser('preprocess_dial') 
    preprocess_dial_parser.set_defaults(mode='preprocess_dial') 
    preprocess_dial_parser.add_argument('--results_dir', type=str, required=True, 
                                help='Path to save preprocessed dialogues') 
    preprocess_dial_parser.add_argument('--config', type=str, required=True, 
                                help='Path to the preprocessing configuration file') 

    # create parser for mm dst training command 
    training_dst_parser = subparsers.add_parser('training_dst')
    training_dst_parser.set_defaults(mode='training_dst') 
    training_dst_parser.add_argument('--model_type', type=str, default='vdtn', 
                                    help='Type of DST model for training') 
    training_dst_parser.add_argument('--model_config', type=str, required=True, 
                                    help='Path to the model configuration file') 
    training_dst_parser.add_argument('--training_config', type=str, required=True, 
                                    help='Path to the training configuration file') 

    # create parser for the inference command
    inference_parser = subparsers.add_parser('inference_dst')
    inference_parser.set_defaults(mode='inference_dst')
    inference_parser.add_argument("--model_type", type=str, default='vdtn',
                                  help='Type of DST model for inference')
    inference_parser.add_argument('--model_path', type=str, required=True,
                                  help='Path to the saved model checkpoints')
    inference_parser.add_argument("--inference_config", type=str, required=True,
                                  help="Path to the inference configuration file")
    inference_parser.add_argument('--inference_style', type=str, required=False, default='beam_search',
                                  help='The style of generation: beam_search or greedy')

    args = parser.parse_args()
    mode = args.mode

    if mode == 'preprocess_dial': 
        results_dir = args.results_dir
        config_path = args.config 
        preprocess_dial_main(results_dir, config_path) 

    elif mode == 'training_dst': 
        model_type = args.model_type 
        model_config_path = args.model_config 
        train_config_path = args.training_config 
        
        with open(model_config_path, 'rb') as f: 
            model_config: Dict[str, int] = json.load(f)

        with open(train_config_path, 'rb') as f: 
            train_config: Dict[str, Any] = json.load(f) 

        training_dst_main(model_type, train_config, model_config) 

    elif mode == "inference_dst":
        model_type = args.model_type
        inference_config_path = args.inference_config
        model_path = args.model_path
        #state_path = args.state_path
        inference_style = args.inference_style

        results_dir = Path(model_path).parent
        model_config_path = Path(model_path).parent / "model_config.json"
        train_config_path = Path(model_path).parent / "train_config.json"

        inference_dst_main(model_type, results_dir, train_config_path, inference_config_path, model_config_path, model_path, inference_style) #, state_path)
