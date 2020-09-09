import torch.nn as nn

__all__ = ['mlp3', 'mlp5']


class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim, bn=False, dropout=None):
        super().__init__()
        layers = [nn.Linear(input_dim, output_dim)]
        if bn:
            layers.append(nn.BatchNorm1d(output_dim))
        if dropout is not None:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MLP(nn.Module):
    def __init__(self, *dims, bn=False, dropout=None):
        super().__init__()
        assert len(dims) > 2, "MLP module must have more than 3 layers"
        features = []
        input_dim = dims[0]
        for output_dim in dims[1:-1]:
            block = BasicBlock(input_dim, output_dim, bn=bn, dropout=dropout)
            features.append(block)
            input_dim = output_dim
        self.features = nn.Sequential(*features)
        self.classifier = nn.Linear(input_dim, dims[-1])

    def forward(self, x):
        return self.classifier(self.features(x))


def mlp3(input_dim, output_dim, args):
    dims = [input_dim, args.hidden_dim, output_dim]
    return MLP(*dims, bn=args.bn, dropout=args.dropout)


def mlp5(input_dim, output_dim, args):
    dims = [input_dim] + [args.hidden_dim] * 3 + [output_dim]
    return MLP(*dims, bn=args.bn, dropout=args.dropout)
