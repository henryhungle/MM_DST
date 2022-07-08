from typing import Dict, Any

import torch
import torch.nn as nn
import pdb 

from baselines.vdtn import TransformerDST

class ModelsFactory(object):

    @staticmethod
    def get_model(model_name: str, model_config: Dict[str, int], vocab, state_vocab, res_vocab, train_config, model_weights_path: str = None):
        
        if model_name == 'vdtn': 
            model = TransformerDST(model_config, vocab, state_vocab, res_vocab, train_config) 
        else:
            raise AttributeError("Model name is incorrect")

        if model_weights_path is not None:
            model.load_state_dict(torch.load(model_weights_path, map_location="cuda:0"))
            print(f"Loaded model parameters from {model_weights_path}")

        return model
