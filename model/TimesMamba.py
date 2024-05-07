import torch
import torch.nn as nn

from layers.Embed import SeriesEmbedding
from layers.RevIN import RevIN
from model.mambacore import MambaForSeriesForecasting


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.use_norm = config.use_norm
        self.use_mark = config.use_mark
        self.channel_independence = config.channel_independence
        self.d_model = config.d_model

        if self.use_norm:
            self.revin_layer = RevIN(config.enc_in, affine=config.revin_affine)

        # Embedding
        self.enc_embedding = SeriesEmbedding(
            config.seq_len,
            config.d_model,
            config.dropout,
        )
        print(self.enc_embedding)

        # Encoder-only architecture
        self.mamba = MambaForSeriesForecasting(
            dims=[config.d_model],
            depths=[config.e_layers],
            ssm_expand=config.ssm_expand,
            ssm_drop_rate=config.dropout,
            mlp_ratio=config.r_ff,
            mlp_drop_rate=config.dropout,
        )
        print(self.mamba)

        self.projector = nn.Linear(
            config.d_model,
            config.pred_len,
            bias=True,
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, _, N = x_enc.shape  # b ts nv

        if self.use_norm:
            x_enc = self.revin_layer(x_enc, "norm")

        if not self.use_mark:
            x_mark_enc = None

        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # b nv c

        if self.channel_independence:
            enc_out = enc_out.reshape((-1, 1, self.d_model))  # b*nv 1 c

        enc_out = torch.unsqueeze(enc_out, 1)  # b 1 nv c
        enc_out = self.mamba(enc_out)  # b 1 nv c
        enc_out = torch.squeeze(enc_out, 1)  # b nv c

        if self.channel_independence:
            enc_out = enc_out.reshape((B, -1, self.d_model))  # b nv c

        enc_out = self.projector(enc_out).transpose(1, 2)[:, :, :N]  # b ts nv

        if self.use_norm:
            enc_out = self.revin_layer(enc_out, "denorm")

        return enc_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        enc_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return enc_out[:, -self.pred_len :, :]
