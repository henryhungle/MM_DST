from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import * 

import pdb 

class AbstractCaterModel(nn.Module):
    def __init__(self, config: Dict[str, int]):
        super().__init__()
        self.config: Dict[str, int] = config
        self.max_objects_in_frame = 10
        self.bb_in_dim = 4
        self.bb_out_dim = 4

class TransformerEncoder(nn.Module):
    def __init__(self, layer, N):
        super(TransformerEncoder, self).__init__() 
        self.layers = clones(layer, N) 
        self.norm = LayerNorm(layer.size)

    def forward(self, x, x_mask):
        for layer in self.layers: 
            x = layer(x, x_mask)
        return self.norm(x) 

class TransformerLayer(nn.Module): 
    def __init__(self, size, dropout, attn, ff):
        super(TransformerLayer, self).__init__() 
        self.size = size 
        self.attn = attn 
        self.ff = ff
        self.attn_sublayer = SublayerConnection(size, dropout) 
        self.ff_sublayer = SublayerConnection(size, dropout) 

    def forward(self, x, x_mask, y=None):
        if y is None:
            x = self.attn_sublayer(x, lambda x: self.attn(x, x, x, x_mask))
        else:
            x = self.attn_sublayer(x, lambda x: self.attn(x, y, y, x_mask))
        return self.ff_sublayer(x, self.ff)

class TransformerDST(AbstractCaterModel): 
    def __init__(self, config: Dict[str, int], vocab, state_vocab, res_vocab, train_config): 
        super().__init__(config)
        self.bb_in_dim = 4 
        self.bb_out_dim = 4 
        self.resnext_in_dim = 2048 
        self.resnext_out_dim = 2048
        self.config = config 
        self.add_resnext = train_config['add_resnext']
        self.add_bb = train_config['add_bb']
        self.share_vocab = train_config['share_vocab'] 
        self.bb_prediction = max(train_config['mask_bb'], train_config['track_bb'])
        self.resnext_prediction = train_config['mask_resnext']
        self.state_prediction = train_config['state_prediction'] 

        self.vocab = vocab 
        self.state_vocab = state_vocab 
        self.res_vocab = res_vocab 

        attn = MultiHeadedAttention(config['num_attention_heads'], config["transformer_dim"])
        ff = PositionwiseFeedForward(config['transformer_dim'], config['transformer_ff_dim'], config['dropout'])
        transformer_layer = TransformerLayer(config['transformer_dim'], config['dropout'], attn, ff)
        self.transformer = TransformerEncoder(transformer_layer, config['num_attention_layers']) 
        
        self.word_embedding = Embeddings(config["word_embedding_dim"], len(vocab))
        self.position_encoding = PositionalEncoding(config["word_embedding_dim"], config["dropout"]) 

        self.bb_linear = nn.Linear(in_features=self.bb_in_dim, out_features=config["transformer_dim"])

        if self.add_resnext:
            self.resnext_linear = nn.Linear(in_features=self.resnext_in_dim, out_features=config['transformer_dim'])

        if self.state_prediction:
            self.state_prediction_layer = nn.Linear(in_features=config['transformer_dim'], out_features=len(self.state_vocab))

        if self.bb_prediction: 
            self.bb_prediction_layer = nn.Linear(in_features=config['transformer_dim'], out_features=self.bb_out_dim)

        if self.resnext_prediction:
            self.resnext_prediction_layer = nn.Linear(in_features=config['transformer_dim'], out_features=self.resnext_out_dim)

        self.norm = LayerNorm(config['transformer_dim'])

    def forward(self, x, mode): 
        video_ft, dial_ft = self.encode_video_dial(x)
        state_ft = self.position_encoding(self.word_embedding(x['state_in'].long()))
        
        if mode == 'state': 
            if self.add_bb or self.add_resnext:
                in_ft = torch.cat([video_ft, dial_ft, state_ft], dim=1)
                seq_len = in_ft.shape[1] 
                mask = x['attn_mask'][:, :seq_len, :seq_len]
            else:
                in_ft = torch.cat([dial_ft, state_ft], dim=1)
                seq_len = in_ft.shape[1]
                video_len = x['objs'].shape[1] 
                mask = x['attn_mask'][:, video_len : video_len+seq_len, video_len : video_len+seq_len]

        in_ft = self.norm(in_ft) 
        output = self.transformer(in_ft, mask)

        if mode == 'state': 
            state_len = x['state_in'].shape[1] 
            state_output = output[:, -state_len:]
            state_output = self.state_prediction_layer(state_output)
            return state_output 
         
    def decode_masked_video(self, x, mode): 
        if mode == 'bb':
            video_ft = self.encode_video(x, bb_name='masked_obj_bb')
        elif mode == 'resnext': 
            video_ft = self.encode_video(x, resnext_name='masked_resnext') 
        elif mode == 'bb_resnext':
            video_ft = self.encode_video(x, bb_name='masked_obj_bb', resnext_name='masked_resnext')

        in_ft = self.norm(video_ft)
        seq_len = in_ft.shape[1]
        mask = x['attn_mask'][:, :seq_len, :seq_len]
        output = self.transformer(in_ft, mask)
        
        if mode == 'bb': 
            output = self.bb_prediction_layer(output)
            return output 
        elif mode == 'resnext': 
            output = self.resnext_prediction_layer(output)
            return output 
        elif mode == 'bb_resnext': 
            bb_output = self.bb_prediction_layer(output)
            resnext_output = self.resnext_prediction_layer(output)
            return bb_output, resnext_output 

    def encode_video(self, x, bb_name='obj_bb', resnext_name='resnext'):
        bb_ft = F.relu(self.bb_linear(x[bb_name]))
        obj_ft = self.word_embedding(x['objs'].long())
        video_ft = obj_ft + bb_ft 
        
        if self.add_resnext:
            resnext_ft = F.relu(self.resnext_linear(x[resnext_name]))
            video_ft = video_ft + resnext_ft    

        video_ft = self.position_encoding(video_ft)
        return video_ft 

    def encode_video_dial(self, x, bb_name='obj_bb'): 
        if self.add_bb or self.add_resnext:
            video_ft = self.encode_video(x, bb_name)
        else:
            video_ft = None 
        dial_ft = self.position_encoding(self.word_embedding(x['dial'].long()))
        return video_ft, dial_ft 

    def decode_state(self, x, state_in): 
        video_ft, dial_ft = self.encode_video_dial(x)
        state_ft = self.position_encoding(self.word_embedding(state_in.long()))
        
        if self.add_bb or self.add_resnext:
            in_ft = torch.cat([video_ft, dial_ft, state_ft], dim=1)
        else:
            in_ft = torch.cat([dial_ft, state_ft], dim=1)
        in_ft = self.norm(in_ft)
        
        mask = torch.ones(1, in_ft.shape[1], in_ft.shape[1]).bool().cuda()

        if self.add_bb or self.add_resnext:
            video_dial_len = video_ft.shape[1] + dial_ft.shape[1]
        else:
            video_dial_len = dial_ft.shape[1] 
        mask[:,:video_dial_len,video_dial_len:] = False
        state_output = self.transformer(in_ft, mask)[:,-1,:]

        state_output = self.state_prediction_layer(state_output).view(-1)

        return state_output 
    

