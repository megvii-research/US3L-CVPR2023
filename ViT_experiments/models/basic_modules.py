import torch
import torch.nn as nn
import torch.nn.functional as F


class Reshape(nn.Module):
    def __init__(self, shape) -> None:
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return torch.reshape(x, self.shape)


class Add(nn.Module):
    def __init__(self) -> None:
        super(Add, self).__init__()

    def forward(self, x1, x2):
        return x1 + x2


class Concat(nn.Module):
    def __init__(self, dim) -> None:
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, x):
        out = torch.cat(x, dim=self.dim)
        return out


class Affine(nn.Module):
    def __init__(self, bn):
        super(Affine, self).__init__()
        self.has_affine = bn.affine
        if self.has_affine:
            self.k = bn.weight.reshape(1, bn.weight.shape[0], 1, 1)
            self.b = bn.bias.reshape(1, bn.bias.shape[0], 1, 1)
        self.running_mean = bn.running_mean.reshape(1, bn.running_mean.shape[0], 1, 1)
        running_var = bn.running_var
        self.safe_std = torch.sqrt(running_var + bn.eps).reshape(
            1, running_var.shape[0], 1, 1
        )

    def forward(self, x_in: torch.Tensor):
        x_normalized = (x_in - self.running_mean) / self.safe_std
        if not self.has_affine:
            return x_normalized
        out = self.k * x_normalized + self.b
        return out

