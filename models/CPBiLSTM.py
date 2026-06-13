import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
# from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, drop_rate):
        super(MLPBlock, self).__init__()
        self.in_linear = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.out_linear = nn.Linear(hidden_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
    
    def forward(self, x):
        '''
        x: [B, *, in_dim]
        '''
        out = self.in_linear(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.out_linear(out)
        out = self.ln(self.dropout(out) + x)
        return out
    
class Model(nn.Module):
    """
    Use BiLSTM as the baseline model
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.charge_discharge_length = configs.charge_discharge_length
        self.early_cycle_threshold = configs.early_cycle_threshold
        self.drop_rate = configs.dropout
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.e_layers = configs.e_layers
        # Embedding
        self.intra_flatten = nn.Flatten(start_dim=2)
        self.intra_embed = nn.Linear(self.charge_discharge_length*3, self.d_model)
        self.intra_MLP = nn.ModuleList([MLPBlock(self.d_model, self.d_ff, self.d_model, self.drop_rate) for _ in range(configs.e_layers)])
        # self.norm = nn.LayerNorm(configs.d_ff*2)
        self.inter_cycle_BiLSTM = nn.LSTM(configs.d_model, configs.d_ff, configs.d_layers, batch_first=True, dropout=self.drop_rate, bidirectional=True)

        self.projection = nn.Linear(configs.d_ff*2, configs.output_num)

    def classification(self, x_enc, curve_attn_mask, return_embedding=False):
        # Embedding
        # Intra-cycle modelling
        x_enc = self.intra_flatten(x_enc) # [B, early_cycle, fixed_len * num_var]
        x_enc = self.intra_embed(x_enc)
        for i in range(self.e_layers):
            x_enc = self.intra_MLP[i](x_enc) # [B, early_cycle, d_model]

        # Inter-cycle modelling
        lengths = torch.sum(curve_attn_mask, dim=1).cpu() # [N]
        x_enc = pack_padded_sequence(x_enc, lengths=lengths, batch_first=True, enforce_sorted=False)
        x_enc, (_,_) = self.inter_cycle_BiLSTM(x_enc)
        x_enc, lens_unpacked = pad_packed_sequence(x_enc, True)
        idx = (torch.as_tensor(lengths, device=x_enc.device, dtype=torch.long) - 1).view(-1, 1).expand(
            len(lengths), x_enc.size(2))
        idx = idx.unsqueeze(1)
        x_enc = x_enc.gather(1, idx).squeeze(1) # [B, 2*d_ff]

        output = self.projection(x_enc)
        if return_embedding:
            return output, x_enc
        return output

    def forward(self,  cycle_curve_data, curve_attn_mask, return_embedding=False):
        '''
        params:
            cycle_curve_data: [B, L, num_variables, fixed_length_of_curve]
            curve_attn_mask: [B, L]
        '''
        # tmp_curve_attn_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(cycle_curve_data)
        # cycle_curve_data[tmp_curve_attn_mask==0] = 0 # set the unseen data as zeros
        if return_embedding:
            dec_out, embedding = self.classification(cycle_curve_data, curve_attn_mask, return_embedding=return_embedding)
            return dec_out, embedding
        else:
            dec_out = self.classification(cycle_curve_data, curve_attn_mask, return_embedding=return_embedding)
            return dec_out  # [B, N]
