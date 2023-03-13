import torch
from yacs.config import CfgNode as CN

FLAGS = None

_C = CN()
_C.BASE_CONFIG = ""  # will use this file as base config and cover self's unique changes
_C.DEVICE = None
_C.RANK = 0
_C.WORLD_SIZE = 1
_C.LOCAL_RANK = 0
_C.SEED = None
_C.OUTPUT = ""

_C.EVALUATOR = ""
_C.EVALUATION_DOMAIN = None  # support 'float'/'bn_merged'/'quant'

_C.MODEL = CN()
_C.MODEL.ARCH = ""
_C.MODEL.CHECKPOINT = ""
_C.MODEL.PRETRAINED = ""
_C.MODEL.NUM_CLASSES = 0
_C.MODEL.INPUTSHAPE = [-1, -1]  # h, w

#for us nets
_C.MODEL.OFA = CN()
_C.MODEL.OFA.BN_CAL_BATCH_NUM = 5
_C.MODEL.OFA.NUM_SAMPLE_TRAINING = 1
_C.MODEL.OFA.CALIBRATE_BN = False
_C.MODEL.OFA.CUMULATIVE_BN_STATS = True
_C.MODEL.OFA.SANDWICH = True # whether to include min and max model
_C.MODEL.OFA.SAMPLE_SCHEDULER = ""  # dynamic scheduler
_C.MODEL.OFA.MOMENTUM_UPDATE = True # whether to update momentum encoder during small network forward pass
_C.MODEL.OFA.DISTILL = True # whether to use teacher soft target
_C.MODEL.OFA.DISTILL_STUDENT_TEMP = 1.0
_C.MODEL.OFA.DISTILL_TEACHER_TEMP = 1.0
_C.MODEL.OFA.DISTILL_CRITERION = "CosineSimilarity"
_C.MODEL.OFA.DISTILL_LAMBDA = 1.0
_C.MODEL.OFA.DISTILL_FEATURE = False
_C.MODEL.OFA.DISTILL_FEATURE_CRITERION = ""
_C.MODEL.OFA.DISTILL_FEATURE_LAMBDA = 1.0 # weighted loss
_C.MODEL.OFA.DISTILL_FEATURE_NAME = ["layer3.0.bn2"]
_C.MODEL.OFA.DISTILL_FEATURE_DIM = [256]

_C.MODEL.OFA.REGULARIZER = False # regularizer for our model
_C.MODEL.OFA.REGULARIZER_CRITERION = "L2Reg" # regularizer for our model
_C.MODEL.OFA.REGULARIZER_LAMBDA = 5e-4 # lambda for regularizer
_C.MODEL.OFA.REGULARIZER_L1_LAMBDA = 0. # lambda for l1 regularizer
_C.MODEL.OFA.REGULARIZER_DECAY_ALPHA = 0.2 # decay rate for group regularizer
_C.MODEL.OFA.REGULARIZER_DECAY_TYPE = "exp" # decay strategy for group regularizer
_C.MODEL.OFA.REGULARIZER_DECAY_BINS = 4 # decay bins for group regularizer
_C.MODEL.OFA.REGULARIZER_WARMUP_EPOCHS = 0 # decay bins for group regularizer

_C.MODEL.OFA.BN_TRACK_STATS = False # set false during training, set true during evaluation
_C.MODEL.OFA.WIDTH_MULT = 1.0
# Do not modify default width_mult_list if you want to load pre-trained model with calibrated BN. Instead add your test width (can be arbitrary in width_mult_range) to width_mult_list_test
_C.MODEL.OFA.WIDTH_MULT_LIST = [0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0]
# uncomment here if you want to test other widths.
# width_mult_list_test: [0.26, 0.31415926]
_C.MODEL.OFA.WIDTH_MULT_RANGE = [0.25, 1.0]
_C.MODEL.OFA.USE_SLIMMABLE = False # set to True to enable slimmable network

_C.TRAIN = CN()
_C.TRAIN.USE_DDP = False
_C.TRAIN.USE_AMP = False
_C.TRAIN.SYNC_BN = False
_C.TRAIN.LINEAR_EVAL = False
_C.TRAIN.WARMUP_FC = False # load from a linear evaluation model
_C.TRAIN.RESUME = ""
_C.TRAIN.EPOCHS = 0
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.PRETRAIN = ""
_C.TRAIN.DATASET = ""
_C.TRAIN.LABEL_SMOOTHING = 0.0
_C.TRAIN.BATCH_SIZE = 1  # per-gpu
_C.TRAIN.NUM_WORKERS = 0
_C.TRAIN.PRINT_FREQ = 1
_C.TRAIN.DROP_LAST = False # drop last in dataloader

_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.TYPE = None  # support cosine / step / multiStep
_C.TRAIN.LR_SCHEDULER.WARMUP_EPOCHS = 0
_C.TRAIN.LR_SCHEDULER.WARMUP_LR = 0.0
_C.TRAIN.LR_SCHEDULER.BASE_LR = 0.0
_C.TRAIN.LR_SCHEDULER.FC_LR = 0.0 # specific learning rate for final fc layer
_C.TRAIN.LR_SCHEDULER.MIN_LR = 0.0
_C.TRAIN.LR_SCHEDULER.SPECIFIC_LRS = []
_C.TRAIN.LR_SCHEDULER.DECAY_MILESTONES = []
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCH = 0
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = ""
_C.TRAIN.OPTIMIZER.EPS = None
_C.TRAIN.OPTIMIZER.BETAS = None  # (0.9, 0.999)
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.0
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 0.0

_C.TRAIN.LOSS = CN()
_C.TRAIN.LOSS.CRITERION = CN()
_C.TRAIN.LOSS.REGULARIZER = CN()
_C.TRAIN.LOSS.LAMBDA = 0.0  # a ratio controls the importance of regularizer
_C.TRAIN.LOSS.CRITERION.NAME = ""  # support CrossEntropy / LpLoss
_C.TRAIN.LOSS.CRITERION.LPLOSS = CN()
_C.TRAIN.LOSS.CRITERION.LPLOSS.P = 2.0
_C.TRAIN.LOSS.CRITERION.LPLOSS.REDUCTION = "none"
_C.TRAIN.LOSS.REGULARIZER.NAME = ""  # support PACT/SLIMMING

_C.TRAIN.RUNNER = CN()
_C.TRAIN.RUNNER.NAME = ""  # support default

_C.TRAIN.METER = CN()
_C.TRAIN.METER.NAME = ""  # support ACC / MAP / MIOU
_C.TRAIN.METER.ACC = CN()
_C.TRAIN.METER.ACC.TOPK = []
_C.TRAIN.METER.MAP = CN()
_C.TRAIN.METER.MIOU = CN()

_C.AUG = CN()
_C.AUG.TRAIN = CN()
_C.AUG.TRAIN.RANDOMRESIZEDCROP = CN()
_C.AUG.TRAIN.RANDOMRESIZEDCROP.ENABLE = False
# _C.AUG.TRAIN.RANDOMRESIZEDCROP.SIZE = MODEL.INPUT_SHAPE
_C.AUG.TRAIN.RANDOMRESIZEDCROP.SCALE = (0.08, 1.0)
_C.AUG.TRAIN.RANDOMRESIZEDCROP.INTERPOLATION = "bilinear"
_C.AUG.TRAIN.RESIZE = CN()
_C.AUG.TRAIN.RESIZE.ENABLE = False
_C.AUG.TRAIN.RESIZE.SIZE = (-1, -1)  # h, w
_C.AUG.TRAIN.RESIZE.KEEP_RATIO = True
_C.AUG.TRAIN.RESIZE.INTERPOLATION = "bilinear"
_C.AUG.TRAIN.HORIZONTAL_FLIP = CN()
_C.AUG.TRAIN.HORIZONTAL_FLIP.PROB = 0.0
_C.AUG.TRAIN.VERTICAL_FLIP = CN()
_C.AUG.TRAIN.VERTICAL_FLIP.PROB = 0.0
_C.AUG.TRAIN.RANDOMCROP = CN()
_C.AUG.TRAIN.RANDOMCROP.ENABLE = False
# _C.AUG.TRAIN.RANDOMCROP.SIZE = MODEL.INPUT_SHAPE
_C.AUG.TRAIN.RANDOMCROP.PADDING = 0
_C.AUG.TRAIN.CENTERCROP = CN()
_C.AUG.TRAIN.CENTERCROP.ENABLE = False
_C.AUG.TRAIN.COLOR_JITTER = CN()
_C.AUG.TRAIN.COLOR_JITTER.PROB = 0.0
_C.AUG.TRAIN.COLOR_JITTER.BRIGHTNESS = 0.4
_C.AUG.TRAIN.COLOR_JITTER.CONTRAST = 0.4
_C.AUG.TRAIN.COLOR_JITTER.SATURATION = 0.2
_C.AUG.TRAIN.COLOR_JITTER.HUE = 0.1
_C.AUG.TRAIN.AUTO_AUGMENT = CN()
_C.AUG.TRAIN.AUTO_AUGMENT.ENABLE = False
_C.AUG.TRAIN.AUTO_AUGMENT.POLICY = 0.0
_C.AUG.TRAIN.RANDOMERASE = CN()
_C.AUG.TRAIN.RANDOMERASE.PROB = 0.0
_C.AUG.TRAIN.RANDOMERASE.MODE = "const"
_C.AUG.TRAIN.RANDOMERASE.MAX_COUNT = None
_C.AUG.TRAIN.MIX = CN()  # mixup & cutmix
_C.AUG.TRAIN.MIX.PROB = 0.0
_C.AUG.TRAIN.MIX.MODE = "batch"
_C.AUG.TRAIN.MIX.SWITCH_MIXUP_CUTMIX_PROB = 0.0
_C.AUG.TRAIN.MIX.MIXUP_ALPHA = 0.0
_C.AUG.TRAIN.MIX.CUTMIX_ALPHA = 0.0
_C.AUG.TRAIN.MIX.CUTMIX_MIXMAX = None
_C.AUG.TRAIN.NORMLIZATION = CN()
_C.AUG.TRAIN.NORMLIZATION.MEAN = []
_C.AUG.TRAIN.NORMLIZATION.STD = []
_C.AUG.EVALUATION = CN()
_C.AUG.EVALUATION.RESIZE = CN()
_C.AUG.EVALUATION.RESIZE.ENABLE = False
_C.AUG.EVALUATION.RESIZE.SIZE = (-1, -1)  # h, w
_C.AUG.EVALUATION.RESIZE.KEEP_RATIO = True
_C.AUG.EVALUATION.RESIZE.INTERPOLATION = "bilinear"
_C.AUG.EVALUATION.CENTERCROP = CN()
_C.AUG.EVALUATION.CENTERCROP.ENABLE = False
_C.AUG.EVALUATION.NORMLIZATION = CN()
_C.AUG.EVALUATION.NORMLIZATION.MEAN = []
_C.AUG.EVALUATION.NORMLIZATION.STD = []

_C.QUANT = CN()
_C.QUANT.TYPE = ""  # support 'qat' / ptq
_C.QUANT.BIT_ASSIGNER = CN()
_C.QUANT.BIT_ASSIGNER.NAME = None  # support 'HAWQ'
_C.QUANT.BIT_ASSIGNER.W_BIT_CHOICES = [2, 4, 8]
_C.QUANT.BIT_ASSIGNER.A_BIT_CHOICES = [2, 4, 8, 16]

_C.QUANT.BIT_ASSIGNER.HAWQ = CN()
_C.QUANT.BIT_ASSIGNER.HAWQ.EIGEN_TYPE = "avg"  # support 'max' / 'avg'
_C.QUANT.BIT_ASSIGNER.HAWQ.SENSITIVITY_CALC_ITER_NUM = 50
_C.QUANT.BIT_ASSIGNER.HAWQ.LIMITATION = CN()
_C.QUANT.BIT_ASSIGNER.HAWQ.LIMITATION.BIT_ASCEND_SORT = False
_C.QUANT.BIT_ASSIGNER.HAWQ.LIMITATION.BIT_WIDTH_COEFF = 1e10
_C.QUANT.BIT_ASSIGNER.HAWQ.LIMITATION.BOPS_COEFF = 1e10
_C.QUANT.BIT_CONFIG = (
    []
)  # a mapping, key is layer_name, value is {"w":w_bit, "a":a_bit}
_C.QUANT.FOLD_BN = False
_C.QUANT.W = CN()
_C.QUANT.W.QUANTIZER = None  # support "LSQ" / "DOREFA"
_C.QUANT.W.BIT = 8
_C.QUANT.W.SYMMETRY = True
_C.QUANT.W.GRANULARITY = (
    "channelwise"  # support "layerwise"/"channelwise" currently, default is channelwise
)
_C.QUANT.W.OBSERVER_METHOD = CN()
_C.QUANT.W.OBSERVER_METHOD.NAME = (
    "MINMAX"  # support "MINMAX"/"MSE" currently, default is MINMAX
)
_C.QUANT.W.OBSERVER_METHOD.ALPHA = 0.0001  # support percentile
_C.QUANT.W.OBSERVER_METHOD.BINS = 2049  # support kl_histogram
_C.QUANT.A = CN()
_C.QUANT.A.BIT = 8
_C.QUANT.A.QUANTIZER = None  # support "LSQ" / "DOREFA"
_C.QUANT.A.SYMMETRY = False
_C.QUANT.A.GRANULARITY = (
    "layerwise"  # support "layerwise"/"channelwise" currently, default is layerwise
)
_C.QUANT.A.OBSERVER_METHOD = CN()
_C.QUANT.A.OBSERVER_METHOD.NAME = (
    "MINMAX"  # support "MINMAX"/"MSE" currently, default is MINMAX
)
_C.QUANT.A.OBSERVER_METHOD.ALPHA = 0.0001  # support percentile
_C.QUANT.A.OBSERVER_METHOD.BINS = 2049  # support kl_histogram
_C.QUANT.CALIBRATION = CN()
_C.QUANT.CALIBRATION.PATH = ""
_C.QUANT.CALIBRATION.TYPE = ""  # support tarfile / python_module
_C.QUANT.CALIBRATION.MODULE_PATH = ""  # the import path of calibration dataset
_C.QUANT.CALIBRATION.SIZE = 0
_C.QUANT.CALIBRATION.BATCHSIZE = 1
_C.QUANT.CALIBRATION.NUM_WORKERS = 0

_C.QUANT.FINETUNE = CN()
_C.QUANT.FINETUNE.ENABLE = False
_C.QUANT.FINETUNE.METHOD = ""
_C.QUANT.FINETUNE.BATCHSIZE = 32
_C.QUANT.FINETUNE.ITERS_W = 0
_C.QUANT.FINETUNE.ITERS_A = 0

_C.QUANT.FINETUNE.BRECQ = CN()
_C.QUANT.FINETUNE.BRECQ.KEEP_GPU = True

_C.PRUNE = CN()
_C.PRUNE.TYPE = ""  # support structed / non_struncted
_C.PRUNE.ENABLE_PRUNE = False
_C.PRUNE.GLOBAL_FACTOR = 1.0
_C.PRUNE.ENABLE_RETRAIN = False
_C.PRUNE.STRATEGY = ""  # l1norm, slimming
_C.PRUNE.GRANULARITY = ""  # support layerwise / channelwise
_C.PRUNE.RETRAIN_GRANULARITY = "model"  # model / stage / block / layer
_C.PRUNE.PRUNE_NUM = []  # a mapping: {layer_name: pruned_num}
_C.PRUNE.PRUNE_PROB = []  # a mapping: {layer_name: pruned_prob}
_C.PRUNE.SLIMMING_FACTOR = 0.0  # a float in [0, 1]


_C.DISTILL = CN()
_C.DISTILL.TEACHER_ARCH = ""
_C.DISTILL.TEACHER_PRETRAINED = ""
_C.DISTILL.TEACHER_DIM = 128 # teacher output dimension for the MLP head
_C.DISTILL.TEACHER_HIDDEN_DIM = 2048 # teacher hidden dimension of the MLP head


_C.SSL = CN()
_C.SSL.TYPE = None # support mocov2
_C.SSL.SETTING = CN()
_C.SSL.SETTING.DIM = 128 # output dimension for the MLP head
_C.SSL.SETTING.HIDDEN_DIM = 2048 # hidden dimension for the MLP head

_C.SSL.SETTING.T = 0.07 # temperature for InfoNCE loss
_C.SSL.SETTING.MOCO_K = 65536 # size of memory bank for MoCo
_C.SSL.SETTING.MOMENTUM = 0.999 # MoCo momentum of updating key encoder
_C.SSL.SETTING.MLP = True # whether to use MLP head, default True
_C.SSL.SETTING.PREDICTOR = True # whether to use predictor
_C.SSL.SETTING.NEW_DISTILL_HEAD = True # whether to use predictor


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()

    config.defrost()
    if hasattr(args, "checkpoint"):
        config.MODEL.CHECKPOINT = args.checkpoint
    if hasattr(args, "pretrained"):
        config.MODEL.PRETRAINED = args.pretrained
    if hasattr(args, "calibration"):
        config.QUANT.CALIBRATION.PATH = args.calibration
    if hasattr(args, "batch_size"):
        config.QUANT.CALIBRATION.BATCHSIZE = args.batch_size
    if hasattr(args, "num_workers"):
        config.QUANT.CALIBRATION.NUM_WORKERS = args.num_workers
        config.TRAIN.NUM_WORKERS = args.num_workers
    if hasattr(args, "eval_domain"):
        config.EVALUATION_DOMAIN = args.eval_domain
    if hasattr(args, "print_freq"):
        config.TRAIN.PRINT_FREQ = args.print_freq
    if hasattr(args, "output"):
        config.OUTPUT = args.output

    config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # get config depend chain files recursively
    config_depends = []

    tmp_config = _C.clone()
    tmp_config.defrost()
    tmp_config.merge_from_file(args.config)
    config_depends.append(args.config)
    while tmp_config.BASE_CONFIG:
        next_config = tmp_config.BASE_CONFIG
        config_depends.append(next_config)
        tmp_config.BASE_CONFIG = ""
        tmp_config.merge_from_file(next_config)
    # tmp_config's merge order is reversed so can't use it directly

    for conf_path in reversed(config_depends):
        config.merge_from_file(conf_path)

    config.freeze()

    global FLAGS
    FLAGS = config
    return config


def update_config(config, key, value):
    config.defrost()
    keys = key.split(".")

    def _set_config_attr(cfg, keys, value):
        if len(keys) > 1:
            cfg = getattr(cfg, keys[0].upper())
            _set_config_attr(cfg, keys[1:], value)
        else:
            setattr(cfg, keys[0].upper(), value)

    _set_config_attr(config, keys, value)
    config.freeze()
    return config
