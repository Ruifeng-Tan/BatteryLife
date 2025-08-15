import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.charge_discharge_length = configs.charge_discharge_length
        self.early_cycle_threshold = configs.early_cycle_threshold
        padding = self.stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, padding, configs.dropout)
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )

        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.early_cycle_threshold*configs.charge_discharge_length - self.patch_len) / self.stride + 2)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, 1)
    
    def classification(self, x_enc, curve_attn_mask, return_embedding):
        '''
        x_enc: [B, L*fixed_length_of_curve, num_vars]
        curve_attn_mask: [B, L]
        '''
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.transpose(1, 2) # [B, num_vars, fixed_len*L]
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        curve_attn_mask = torch.repeat_interleave(curve_attn_mask, dim=1, repeats=self.charge_discharge_length) # [bs, fxied_len*L]
        curve_attn_mask = torch.repeat_interleave(curve_attn_mask, dim=0, repeats=n_vars) # [bs*nvars, fxied_len*L]
        curve_attn_mask = self.padding_patch_layer(curve_attn_mask)
        curve_attn_mask = curve_attn_mask.unfold(dimension=-1, size=self.patch_len, step=self.stride) # [bs, patch_num, patch_len]
        curve_attn_mask = torch.sum(curve_attn_mask, dim=-1) # [B, patch_num]
        curve_attn_mask = torch.where(curve_attn_mask>=1, torch.ones_like(curve_attn_mask), torch.zeros_like(curve_attn_mask))
        curve_attn_mask = curve_attn_mask.unsqueeze(1) # [B, 1, patch_num]
        curve_attn_mask = torch.repeat_interleave(curve_attn_mask, curve_attn_mask.shape[-1], dim=1) # [B, patch_num, patch_num]
        curve_attn_mask = curve_attn_mask.unsqueeze(1) # [B, 1, patch_num, patch_num]
        curve_attn_mask = curve_attn_mask==0 # set True to mask

        enc_out, attns = self.encoder(enc_out, attn_mask=curve_attn_mask)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        output = self.flatten(enc_out)
        output = self.dropout(output)
        embedding = output.reshape(output.shape[0], -1)
        output = self.projection(embedding)  # (batch_size, num_classes)
        if return_embedding:
            return output, embedding
        return output

    def forward(self, cycle_curve_data, curve_attn_mask, x_mark_enc=None, return_embedding=False):
        '''
        cycle_curve_data: [B, early_cycle, num_var, fixed_len]
        curve_attn_mask: [B, early_cycle]
        '''
        # tmp_curve_attn_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(cycle_curve_data)
        # cycle_curve_data[tmp_curve_attn_mask==0] = 0 # set the unseen data as zeros
        B, num_vars = cycle_curve_data.shape[0], cycle_curve_data.shape[2]
        cycle_curve_data = cycle_curve_data.transpose(2, 3) # [B, L, fixed_len, num_vars]
        cycle_curve_data = cycle_curve_data.reshape(B, -1, num_vars)  # [B, L*fixed_len, num_vars]
        x_enc = cycle_curve_data
        if return_embedding:
            dec_out, embedding = self.classification(x_enc, curve_attn_mask, return_embedding) # [B, N]
            return dec_out, embedding
        else:
            dec_out = self.classification(x_enc, curve_attn_mask, return_embedding) # [B, N]
            return dec_out

