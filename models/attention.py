"""code modified from https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html"""
import torch.nn as nn

__all__ = ['attention1', 'attention4']


class AttentionLayer(nn.Module):
    def __init__(self, d_model, nhead=16, hidden_dim=2048,
                 norm=False, dropout=0.0):
        super().__init__()
        if dropout is None:
            dropout = 0.0
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, d_model)

        if norm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        self.norm = norm

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        if self.norm:
            src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        if self.norm:
            src = self.norm2(src)
        return src


class Attention(nn.Module):
    def __init__(self, d_model, seq_len, output_dim, n_layers=1, **kwargs):
        super().__init__()
        self.seq_len = seq_len
        layers = [AttentionLayer(d_model, **kwargs) for _ in range(n_layers)]
        self.layers = nn.Sequential(*layers)

        self.activation = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(d_model * seq_len, output_dim)

    def forward(self, src):
        src = src.view(src.size(0), self.seq_len, -1)
        features = self.activation(self.layers(src)).view(src.size(0), -1)
        return self.classifier(features)


def attention1(input_dim, output_dim, args):
    return Attention(
        int(input_dim / args.max_length),
        args.max_length,
        output_dim,
        n_layers=1,
        hidden_dim=args.hidden_dim,
        norm=args.bn,
        dropout=args.dropout
    )


def attention4(input_dim, output_dim, args):
    return Attention(
        int(input_dim / args.max_length),
        args.max_length,
        output_dim,
        n_layers=4,
        hidden_dim=args.hidden_dim,
        norm=args.bn,
        dropout=args.dropout
    )
