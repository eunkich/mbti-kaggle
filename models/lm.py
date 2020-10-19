import torch.nn as nn
from transformers import AutoModel

__all__ = ['lm_classifier']


class LMClassifier(nn.Module):
    def __init__(self, model_name, num_class, bn=False, dropout=None):
        super().__init__()
        self.lm = AutoModel.from_pretrained(model_name)
        layers = []
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        if bn:
            layers.append(nn.LayerNorm(self.lm.config.hidden_size))
        layers.append(nn.Linear(self.lm.config.hidden_size, num_class))
        self.fc = nn.Sequential(*layers)

    def forward(self, tokens):
        features = self.lm(input_ids=tokens)[0].mean(1)  # mean of output
        logits = self.fc(features)
        return logits


def lm_classifier(input_dim, output_dim, args):
    return LMClassifier(
        args.lm,
        output_dim,
        args.bn,
        args.dropout
    )
