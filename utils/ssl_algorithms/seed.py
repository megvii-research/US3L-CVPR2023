import torch
import torch.nn as nn
from copy import deepcopy
import models

class SEED(nn.Module):
    """
    Build a SEED model for Self-supervised Distillation: a student encoder, a teacher encoder (stay frozen),
    and an instance queue.
    Adapted from MoCo, He, Kaiming, et al. "Momentum contrast for unsupervised visual representation learning."
    """

    def __init__(self, base_encoder, config):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SEED, self).__init__()

        self.config = config
        
        dim = config.SSL.SETTING.DIM
        hidden_dim = config.SSL.SETTING.HIDDEN_DIM

        self.K = config.SSL.SETTING.MOCO_K
        self.t = config.MODEL.OFA.DISTILL_STUDENT_TEMP 
        self.temp = config.MODEL.OFA.DISTILL_TEACHER_TEMP 
        self.dim = dim

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder_q = base_encoder(num_classes=hidden_dim, bn_track_stats=True)

        # build a 3-layer projector
        fc_dim = hidden_dim

        self.encoder_q.fc = nn.Sequential(self.encoder_q.fc,
                                        nn.BatchNorm1d(fc_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(fc_dim, fc_dim, bias=False),
                                        nn.BatchNorm1d(fc_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(fc_dim, dim)) # output layer

        teacher_encoder = models.__dict__[config.DISTILL.TEACHER_ARCH]
        
        dim_teacher = config.DISTILL.TEACHER_DIM
        hidden_dim_teacher = config.DISTILL.TEACHER_HIDDEN_DIM

        self.encoder_k  = teacher_encoder(num_classes=hidden_dim_teacher)
        self.encoder_k.fc = nn.Sequential(self.encoder_k.fc,
                                        nn.BatchNorm1d(hidden_dim_teacher),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(hidden_dim_teacher, hidden_dim_teacher, bias=False),
                                        nn.BatchNorm1d(hidden_dim_teacher),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(hidden_dim_teacher, dim_teacher)) # output layer


        # we need the final dimension to be consistent
        assert dim_teacher==dim

        self.load_teacher_weight(config.DISTILL.TEACHER_PRETRAINED)
        
        #student manually set width
        self.encoder_q.apply(lambda m: setattr(m, 'width_mult', config.MODEL.OFA.WIDTH_MULT))

        # teacher is full width
        self.encoder_k.apply(lambda m: setattr(m, 'width_mult', 1.0))

        # not update by gradient
        for param_k in self.encoder_k.parameters():
            param_k.requires_grad = False

        # create the queue
        self.register_buffer("queue", torch.randn(dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def load_teacher_weight(self, path):
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt['state_dict']
        print(state_dict.keys())
        print(self.encoder_k.state_dict().keys())

        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            elif k.startswith('encoder_q'):
                # remove prefix
                state_dict[k[len("encoder_q."):]] = state_dict[k]
            del state_dict[k]
    
        msg = self.encoder_k.load_state_dict(state_dict, strict=True)
        print('missing', set(msg.missing_keys))
        print("=> loaded pre-trained model '{}'".format(path))

    # queue updation
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, concat=True):

        # gather keys before updating queue in distributed mode
        if concat:
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity as in MoCo-v2

        # replace the keys at ptr (de-queue and en-queue)
        self.queue[:, ptr:ptr + batch_size] = keys.T

        # move pointer
        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    def forward(self, x1, x2):
        """
        Input:
            image: a batch of images
        Output:
            student logits, teacher logits
        """
        image = x1

        # compute query features
        s_emb = self.encoder_q(image)  # NxC
        s_emb = nn.functional.normalize(s_emb, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            t_emb = self.encoder_k(image)  # keys: NxC
            t_emb = nn.functional.normalize(t_emb, dim=1)

        # cross-Entropy Loss
        logit_stu = torch.einsum('nc,ck->nk', [s_emb, self.queue.clone().detach()])
        logit_tea = torch.einsum('nc,ck->nk', [t_emb, self.queue.clone().detach()])

        logit_s_p = torch.einsum('nc,nc->n', [s_emb, t_emb]).unsqueeze(-1)
        logit_t_p = torch.einsum('nc,nc->n', [t_emb, t_emb]).unsqueeze(-1)

        logit_stu = torch.cat([logit_s_p, logit_stu], dim=1)
        logit_tea = torch.cat([logit_t_p, logit_tea], dim=1)

        # compute soft labels
        logit_stu /= self.t
        logit_tea = nn.functional.softmax(logit_tea/self.temp, dim=1)

        # de-queue and en-queue
        self._dequeue_and_enqueue(t_emb, concat=True)

        return logit_stu, logit_tea

# multi-GPU data collector
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class SEED_MSE(nn.Module):
    """
    Build a self-supervised distillation model.
    """
    def __init__(self, base_encoder, config):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SEED_MSE, self).__init__()

        self.config = config
        
        dim = config.SSL.SETTING.DIM
        hidden_dim = config.SSL.SETTING.HIDDEN_DIM

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder_q = base_encoder(num_classes=hidden_dim, bn_track_stats=True)

        # build a 3-layer projector
        fc_dim = hidden_dim

        self.encoder_q.fc = nn.Sequential(self.encoder_q.fc,
                                        nn.BatchNorm1d(fc_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(fc_dim, fc_dim, bias=False),
                                        nn.BatchNorm1d(fc_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(fc_dim, dim)) # output layer

        teacher_encoder = models.__dict__[config.DISTILL.TEACHER_ARCH]
        
        dim_teacher = config.DISTILL.TEACHER_DIM
        hidden_dim_teacher = config.DISTILL.TEACHER_HIDDEN_DIM

        self.encoder_k  = teacher_encoder(num_classes=hidden_dim_teacher)
        self.encoder_k.fc = nn.Sequential(self.encoder_k.fc,
                                        nn.BatchNorm1d(hidden_dim_teacher),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(hidden_dim_teacher, hidden_dim_teacher, bias=False),
                                        nn.BatchNorm1d(hidden_dim_teacher),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(hidden_dim_teacher, dim_teacher)) # output layer


        # we need the final dimension to be consistent
        assert dim_teacher==dim

        self.load_teacher_weight(config.DISTILL.TEACHER_PRETRAINED)
        
        #student manually set width
        self.encoder_q.apply(lambda m: setattr(m, 'width_mult', config.MODEL.OFA.WIDTH_MULT))

        # teacher is full width
        self.encoder_k.apply(lambda m: setattr(m, 'width_mult', 1.0))

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, hidden_dim, bias=False),
                                        nn.BatchNorm1d(hidden_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(hidden_dim, dim)) # output layer
        #self.predictor = nn.Identity()

    def load_teacher_weight(self, path):
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt['state_dict']
        print(state_dict.keys())
        print(self.encoder_k.state_dict().keys())

        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            elif k.startswith('encoder_q'):
                # remove prefix
                state_dict[k[len("encoder_q."):]] = state_dict[k]
            del state_dict[k]
    
        msg = self.encoder_k.load_state_dict(state_dict, strict=True)
        print('missing', set(msg.missing_keys))
        print("=> loaded pre-trained model '{}'".format(path))

    # for us backbone, we don't save bn mean and var
    def calibrate_teacher(self, dataloader):
        self.encoder_k.train()
        for i, (images, _) in enumerate(dataloader):
            images[0] = images[0].cuda(non_blocking=True)
            images[1] = images[1].cuda(non_blocking=True)
            with torch.no_grad():
                self.encoder_k(images[0])
                self.encoder_k(images[1])
    
    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
        """

        online_proj_one = self.encoder_q(x1)
        online_proj_two = self.encoder_q(x2)

        online_pred_one = self.predictor(online_proj_one)
        online_pred_two = self.predictor(online_proj_two)

        self.encoder_k.eval()

        with torch.no_grad():
            target_proj_one = self.encoder_k(x1)
            target_proj_two = self.encoder_k(x2)
        
        return [online_pred_one, online_pred_two], [target_proj_one.detach(), target_proj_two.detach()]


class SEED_SimCLR(SEED_MSE):
    """
    Build a self-supervised distillation model.
    """
    def __init__(self, base_encoder, config):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SEED_SimCLR, self).__init__(base_encoder, config)
    
    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
        """

        f1 = self.encoder_q(x1)
        f2 = self.encoder_q(x2)

        #f1, f2 = nn.functional.normalize(f1, dim=1), nn.functional.normalize(f2, dim=1)

        self.encoder_k.eval()
        with torch.no_grad():
            f1_t = self.encoder_k(x1)
            f2_t = self.encoder_k(x2)
            #f1_t, f2_t = nn.functional.normalize(f1_t, dim=1), nn.functional.normalize(f2_t, dim=1)

        return [f1, f2], [f1_t.detach(), f2_t.detach()]