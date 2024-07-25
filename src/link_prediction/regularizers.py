import torch

from torch import nn


class Regularizer(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, factors):
        pass


class L2(Regularizer):
    def __init__(self, weight: float):
        super().__init__(weight)

    def forward(self, factors):
        l = [torch.mean(factor**2) for factor in factors]
        l = sum(l) * self.weight / len(factors)
        return l


class N2(Regularizer):
    def __init__(self, weight: float):
        super().__init__(weight)

    def forward(self, factors):
        norms = [torch.sum(torch.norm(f, 2, 1) ** 3) for f in factors]
        norms = [self.weight * norm for norm in norms]
        norm = sum(norms) / factors[0].shape[0]

        return norm


class N3(Regularizer):
    def __init__(self, weight: float):
        super().__init__(weight)

    def forward(self, factors):
        norms = [torch.sum(torch.abs(f) ** 3) for f in factors]
        norms = [self.weight * norm for norm in norms]
        norm = sum(norms) / factors[0].shape[0]

        return norm
