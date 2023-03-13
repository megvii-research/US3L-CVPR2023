import torch
from torch import nn
import numpy as np

class Regularizer(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.name = "GroupReg"
        self.model = model
        self.config = config
        self.reg_lambda = config.MODEL.OFA.REGULARIZER_LAMBDA
        self.decay_alpha = config.MODEL.OFA.REGULARIZER_DECAY_ALPHA
        self.decay_type = config.MODEL.OFA.REGULARIZER_DECAY_TYPE
        self.decay_bins = config.MODEL.OFA.REGULARIZER_DECAY_BINS
        self.warmup_epochs = config.MODEL.OFA.REGULARIZER_WARMUP_EPOCHS
        self.total_epochs = config.TRAIN.EPOCHS

    def cal_lambda(self, i, epoch):
        # i: index 0,1,2,3,...,decay_bins

        if epoch<self.warmup_epochs: # normal regularizer during warmup
            return self.reg_lambda
        # dynamic
        decay_alpha = self.decay_alpha
        #ramp_epochs = 0
        ramp_epochs = self.total_epochs//2 # 200 for 400 epochs
        if epoch<ramp_epochs:
            decay_alpha *= (epoch-self.warmup_epochs) / (ramp_epochs-self.warmup_epochs)
        
        if self.decay_type == 'exp':
            return self.reg_lambda * decay_alpha**i 
        elif self.decay_type == 'linear':
            # dynamic
            return self.reg_lambda * (1-decay_alpha*i) # v1
            #return self.reg_lambda * (1-decay_alpha*(self.decay_bins-1-i)) # v2
        else:
            print('Not Implemented Error')

    def __call__(self, width_mult=1.0, epoch=-1): 
        loss = 0.
        for n, p in self.model.named_parameters():
            if 'encoder_k' in n or 'proj_conv' in n: # skip encoder_k
                continue 
            if width_mult==1.0 and 'predictor' in n:
                continue
            length = len(p.size())
            lambda_mask = torch.ones(p.size(), dtype=torch.float64, device=p.device)

            if length == 1: # bn weight/bias, group L2
                dim = p.size()[0]
                group_dim = dim//self.decay_bins
                # generate mask
                for i in range(self.decay_bins):
                    end_dim = dim if i==self.decay_bins-1 else (i+1)*group_dim
                    lambda_mask[i*group_dim:end_dim] = self.cal_lambda(i, epoch)
            elif length == 4: # conv weight/bias, group L2
                dim_out, dim_in = p.size()[0], p.size()[1]
                group_dim_out, group_dim_in = dim_out//self.decay_bins, dim_in//self.decay_bins
                # generate mask
                for i in range(self.decay_bins):
                    end_dim = dim_out if i==self.decay_bins-1 else (i+1)*group_dim_out
                    lambda_mask[i*group_dim_out:end_dim, :, :, :] =  self.cal_lambda(i, epoch)
            else: # fc weight, normal L2
                lambda_mask[:] = self.reg_lambda
            loss += (lambda_mask * torch.square(p)).sum()

        #self.update(loss.item(), batch_size=pred[0].shape[0] if len(pred)>1 else pred.shape[0])
        return loss
