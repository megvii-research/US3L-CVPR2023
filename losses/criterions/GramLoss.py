import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from losses import BaseLoss as Base

class Criterion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = "GramLoss"

    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        print(G.size())

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    def __call__(self, student_output, teacher_output, epoch=None):
        total_loss = 0

        for iq, q in enumerate(teacher_output):
            for v in range(len(student_output)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                G_s = self.gram_matrix(student_output[v])
                G_t = self.gram_matrix(q)

                loss = F.mse_loss(G_s, G_t.detach())
                #loss = torch.mean(torch.square(student_output[v]-q.detach()))
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss