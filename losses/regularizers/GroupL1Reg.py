import torch
from torch import nn



class Regularizer(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.name = "GroupL1Reg"
        self.model = model
        self.config = config
        self.l1_reg_lambda = config.MODEL.OFA.REGULARIZER_L1_LAMBDA
        self.decay_alpha = config.MODEL.OFA.REGULARIZER_DECAY_ALPHA
        self.decay_type = config.MODEL.OFA.REGULARIZER_DECAY_TYPE
        self.decay_bins = config.MODEL.OFA.REGULARIZER_DECAY_BINS
        self.warmup_epochs = config.MODEL.OFA.REGULARIZER_WARMUP_EPOCHS
        self.total_epochs = config.TRAIN.EPOCHS
    
    def cal_lambda(self, i, epoch):
        # i: index 0,1,2,3,...,decay_bins
        # i=0, first channel; i=decay_bins-1, last channel
        if epoch<self.warmup_epochs: # normal regularizer during warmup
            return 0 # we don't use L1 normalization at first

        # dynamic
        decay_alpha = self.decay_alpha
        ramp_epochs = 0
        #ramp_epochs = self.total_epochs//2 # 200 for 400 epochs
        if epoch<ramp_epochs:
            decay_alpha *= (epoch-self.warmup_epochs) / (ramp_epochs-self.warmup_epochs)
        return self.l1_reg_lambda * i * decay_alpha


    def __call__(self, width_mult=1.0, epoch=-1):      
        loss = 0.
        for n, p in self.model.named_parameters():
            if 'encoder_k' in n or 'proj_conv' in n: # skip encoder_k
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
                loss += (lambda_mask * torch.abs(p)).sum()

        #self.update(loss.item(), batch_size=pred[0].shape[0] if len(pred)>1 else pred.shape[0])
        return loss
