# -*- coding: utf-8 -*-
# @Time    : 2021/10/23 11:07
# @Author  : LIU YI

from collections import namedtuple
from os import path as osp
import torch
import torch.nn as nn
from cell_operations import OPS as OPS_searched
from operations import OPS as OPS_infer


class NASNetInferCell(nn.Module):
    def __init__(
        self,
        genotype,
        C_prev_prev,
        C_prev,
        C,
        reduction,
        reduction_prev,
        affine,
        track_running_stats,
    ):
        super(NASNetInferCell, self).__init__()
        if 'connects' in genotype:
            OPS = OPS_infer
            if reduction_prev:
                self.preprocess0 = OPS["skip_connect"](
                    C_prev_prev, C, 2, affine
                )
            else:
                self.preprocess0 = OPS["nor_conv_1x1"](
                    C_prev_prev, C, 1, affine
                )
            self.preprocess1 = OPS["nor_conv_1x1"](
                C_prev, C, 1, affine
            )
        else:
            OPS = OPS_searched
            if reduction_prev:
                self.preprocess0 = OPS["skip_connect"](
                    C_prev_prev, C, 2, affine, track_running_stats
                )
            else:
                self.preprocess0 = OPS["nor_conv_1x1"](
                    C_prev_prev, C, 1, affine, track_running_stats
                )
            self.preprocess1 = OPS["nor_conv_1x1"](
                C_prev, C, 1, affine, track_running_stats
            )
        self.reduction = reduction

        if not reduction:
            nodes, concats = genotype["normal"], genotype["normal_concat"]
        else:
            if "reduce" in genotype:
                nodes, concats = genotype["reduce"], genotype["reduce_concat"]
            else:
                nodes =[(('max_pool_3x3', 0), ('max_pool_3x3', 1)), # step 1
                        (('skip_connect', 2), ('max_pool_3x3', 0)), # step 2
                        (('max_pool_3x3', 0), ('skip_connect', 2)), # step 3
                        (('skip_connect', 2), ('avg_pool_3x3', 0))  # step 4
                    ]

                concats = [2, 3, 4, 5]
        self._multiplier = len(concats)
        self._concats = concats
        self._steps = len(nodes)
        self._nodes = nodes
        self.edges = nn.ModuleDict()
        for i, node in enumerate(nodes):
            for in_node in node:
                name, j = in_node[0], in_node[1]
                stride = 2 if reduction and j < 2 else 1
                node_str = "{:}<-{:}".format(i + 2, j)
                if 'connects' in genotype:
                    self.edges[node_str] = OPS[name](
                        C, C, stride, affine
                    )
                else:
                    self.edges[node_str] = OPS[name](
                        C, C, stride, affine, track_running_stats
                    )

    # [TODO] to support drop_prob in this function..
    def forward(self, s0, s1, unused_drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i, node in enumerate(self._nodes):
            clist = []
            for in_node in node:
                name, j = in_node[0], in_node[1]
                node_str = "{:}<-{:}".format(i + 2, j)
                op = self.edges[node_str]
                clist.append(op(states[j]))
            states.append(sum(clist))
        return torch.cat([states[x] for x in self._concats], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):
    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(
                5, stride=3, padding=0, count_include_pad=False
            ),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class NASNetonCIFAR(nn.Module):
    def __init__(
        self,
        C,
        N,
        stem_multiplier,
        num_classes,
        genotype,
        auxiliary,
        affine=True,
        track_running_stats=True,
    ):
        super(NASNetonCIFAR, self).__init__()
        self._C = C
        self._layerN = N
        self.stem = nn.Sequential(
            nn.Conv2d(3, C * stem_multiplier, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C * stem_multiplier),
        )

        # config for each layer
        layer_channels = (
            [C] * N + [C * 2] + [C * 2] * (N - 1) + [C * 4] + [C * 4] * (N - 1)
        )
        layer_reductions = (
            [False] * N + [True] + [False] * (N - 1) + [True] + [False] * (N - 1)
        )

        C_prev_prev, C_prev, C_curr, reduction_prev = (
            C * stem_multiplier,
            C * stem_multiplier,
            C,
            False,
        )
        self.auxiliary_index = None
        self.auxiliary_head = None
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(
            zip(layer_channels, layer_reductions)
        ):
            cell = NASNetInferCell(
                genotype,
                C_prev_prev,
                C_prev,
                C_curr,
                reduction,
                reduction_prev,
                affine,
                track_running_stats,
            )
            self.cells.append(cell)
            C_prev_prev, C_prev, reduction_prev = (
                C_prev,
                cell._multiplier * C_curr,
                reduction,
            )
            if reduction and C_curr == C * 4 and auxiliary:
                self.auxiliary_head = AuxiliaryHeadCIFAR(C_prev, num_classes)
                self.auxiliary_index = index
        self._Layer = len(self.cells)
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.drop_path_prob = -1

    def update_drop_path(self, drop_path_prob):
        self.drop_path_prob = drop_path_prob

    def auxiliary_param(self):
        if self.auxiliary_head is None:
            return []
        else:
            return list(self.auxiliary_head.parameters())

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += "\n {:02d}/{:02d} :: {:}".format(
                i, len(self.cells), cell.extra_repr()
            )
        return string

    def extra_repr(self):
        return "{name}(C={_C}, N={_layerN}, L={_Layer})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    def forward(self, inputs):
        stem_feature, logits_aux = self.stem(inputs), None
        cell_results = [stem_feature, stem_feature]
        for i, cell in enumerate(self.cells):
            cell_feature = cell(cell_results[-2], cell_results[-1], self.drop_path_prob)
            cell_results.append(cell_feature)
            if (
                self.auxiliary_index is not None
                and i == self.auxiliary_index
                and self.training
            ):
                logits_aux = self.auxiliary_head(cell_results[-1])
        out = self.lastact(cell_results[-1])
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        if logits_aux is None:
            return out, logits
        else:
            return out, [logits, logits_aux]


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat connectN connects')
#Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES_small = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'conv_3x1_1x3',
]

PRIMITIVES_large = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'conv_3x1_1x3',
]

PRIMITIVES_huge = [
    'skip_connect',
    'nor_conv_1x1',
    'max_pool_3x3',
    'avg_pool_3x3',
    'nor_conv_3x3',
    'sep_conv_3x3',
    'dil_conv_3x3',
    'conv_3x1_1x3',
    'sep_conv_5x5',
    'dil_conv_5x5',
    'sep_conv_7x7',
    'conv_7x1_1x7',
    'att_squeeze',
]

PRIMITIVES = {'small': PRIMITIVES_small,
              'large': PRIMITIVES_large,
              'huge' : PRIMITIVES_huge}

NASNet = Genotype(
  normal = [
    (('sep_conv_5x5', 1), ('sep_conv_3x3', 0)),
    (('sep_conv_5x5', 0), ('sep_conv_3x3', 0)),
    (('avg_pool_3x3', 1), ('skip_connect', 0)),
    (('avg_pool_3x3', 0), ('avg_pool_3x3', 0)),
    (('sep_conv_3x3', 1), ('skip_connect', 1)),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    (('sep_conv_5x5', 1), ('sep_conv_7x7', 0)),
    (('max_pool_3x3', 1), ('sep_conv_7x7', 0)),
    (('avg_pool_3x3', 1), ('sep_conv_5x5', 0)),
    (('skip_connect', 3), ('avg_pool_3x3', 2)),
    (('sep_conv_3x3', 2), ('max_pool_3x3', 1)),
  ],
  reduce_concat = [4, 5, 6],
  connectN=None, connects=None,
)

PNASNet = Genotype(
  normal = [
    (('sep_conv_5x5', 0), ('max_pool_3x3', 0)),
    (('sep_conv_7x7', 1), ('max_pool_3x3', 1)),
    (('sep_conv_5x5', 1), ('sep_conv_3x3', 1)),
    (('sep_conv_3x3', 4), ('max_pool_3x3', 1)),
    (('sep_conv_3x3', 0), ('skip_connect', 1)),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    (('sep_conv_5x5', 0), ('max_pool_3x3', 0)),
    (('sep_conv_7x7', 1), ('max_pool_3x3', 1)),
    (('sep_conv_5x5', 1), ('sep_conv_3x3', 1)),
    (('sep_conv_3x3', 4), ('max_pool_3x3', 1)),
    (('sep_conv_3x3', 0), ('skip_connect', 1)),
  ],
  reduce_concat = [2, 3, 4, 5, 6],
  connectN=None, connects=None,
)


DARTS_V1 = Genotype(
  normal=[
    (('sep_conv_3x3', 1), ('sep_conv_3x3', 0)), # step 1
    (('skip_connect', 0), ('sep_conv_3x3', 1)), # step 2
    (('skip_connect', 0), ('sep_conv_3x3', 1)), # step 3
    (('sep_conv_3x3', 0), ('skip_connect', 2))  # step 4
  ],
  normal_concat=[2, 3, 4, 5],
  reduce=[
    (('max_pool_3x3', 0), ('max_pool_3x3', 1)), # step 1
    (('skip_connect', 2), ('max_pool_3x3', 0)), # step 2
    (('max_pool_3x3', 0), ('skip_connect', 2)), # step 3
    (('skip_connect', 2), ('avg_pool_3x3', 0))  # step 4
  ],
  reduce_concat=[2, 3, 4, 5],
  connectN=None, connects=None,
)

# DARTS: Differentiable Architecture Search, ICLR 2019
DARTS_V2 = Genotype(
  normal=[
    (('sep_conv_3x3', 0), ('sep_conv_3x3', 1)), # step 1
    (('sep_conv_3x3', 0), ('sep_conv_3x3', 1)), # step 2
    (('sep_conv_3x3', 1), ('skip_connect', 0)), # step 3
    (('skip_connect', 0), ('dil_conv_3x3', 2))  # step 4
  ],
  normal_concat=[2, 3, 4, 5],
  reduce=[
    (('max_pool_3x3', 0), ('max_pool_3x3', 1)), # step 1
    (('skip_connect', 2), ('max_pool_3x3', 1)), # step 2
    (('max_pool_3x3', 0), ('skip_connect', 2)), # step 3
    (('skip_connect', 2), ('max_pool_3x3', 1))  # step 4
  ],
  reduce_concat=[2, 3, 4, 5],
  connectN=None, connects=None,
)


# One-Shot Neural Architecture Search via Self-Evaluated Template Network, ICCV 2019
SETN = Genotype(
  normal=[
    (('skip_connect', 0), ('sep_conv_5x5', 1)),
    (('sep_conv_5x5', 0), ('sep_conv_3x3', 1)),
    (('sep_conv_5x5', 1), ('sep_conv_5x5', 3)),
    (('max_pool_3x3', 1), ('conv_3x1_1x3', 4))],
  normal_concat=[2, 3, 4, 5],
  reduce=[
    (('sep_conv_3x3', 0), ('sep_conv_5x5', 1)),
    (('avg_pool_3x3', 0), ('sep_conv_5x5', 1)),
    (('avg_pool_3x3', 0), ('sep_conv_5x5', 1)),
    (('avg_pool_3x3', 0), ('skip_connect', 1))],
  reduce_concat=[2, 3, 4, 5],
  connectN=None, connects=None
)


# Searching for A Robust Neural Architecture in Four GPU Hours, CVPR 2019
GDAS_V1 = Genotype(
  normal=[
    (('skip_connect', 0), ('skip_connect', 1)),
    (('skip_connect', 0), ('sep_conv_5x5', 2)),
    (('sep_conv_3x3', 3), ('skip_connect', 0)),
    (('sep_conv_5x5', 4), ('sep_conv_3x3', 3))],
  normal_concat=[2, 3, 4, 5],
  reduce=[
    (('sep_conv_5x5', 0), ('sep_conv_3x3', 1)),
    (('sep_conv_5x5', 2), ('sep_conv_5x5', 1)),
    (('dil_conv_5x5', 2), ('sep_conv_3x3', 1)),
    (('sep_conv_5x5', 0), ('sep_conv_5x5', 1))],
  reduce_concat=[2, 3, 4, 5],
  connectN=None, connects=None
)



Networks = {'DARTS_V1': DARTS_V1,
            'DARTS_V2': DARTS_V2,
            'DARTS'   : DARTS_V2,
            'NASNet'  : NASNet,
            'GDAS_V1' : GDAS_V1,
            'PNASNet' : PNASNet,
            'SETN'    : SETN,
           }

# This function will return a Genotype from a dict.
def build_genotype_from_dict(xdict):
  def remove_value(nodes):
    return [tuple([(x[0], x[1]) for x in node]) for node in nodes]
  genotype = Genotype(
      normal=remove_value(xdict['normal']),
      normal_concat=xdict['normal_concat'],
      reduce=remove_value(xdict['reduce']),
      reduce_concat=xdict['reduce_concat'],
      connectN=None, connects=None
      )
  return genotype

def obtain_model(config, extra_path=None):
    if config.dataset == "cifar":
        return get_cifar_models(config, extra_path)
    # elif config.dataset == "imagenet":
    #     return get_imagenet_models(config)
    else:
        raise ValueError("invalid dataset in the model config : {:}".format(config))


def get_cifar_models(config, extra_path=None):
    super_type = getattr(config, "super_type", "basic")
    if super_type == "basic":
        pass
        # from .CifarResNet import CifarResNet
        # from .CifarDenseNet import DenseNet
        # from .CifarWideResNet import CifarWideResNet
        #
        # if config.arch == "resnet":
        #     return CifarResNet(
        #         config.module, config.depth, config.class_num, config.zero_init_residual
        #     )
        # elif config.arch == "densenet":
        #     return DenseNet(
        #         config.growthRate,
        #         config.depth,
        #         config.reduction,
        #         config.class_num,
        #         config.bottleneck,
        #     )
        # elif config.arch == "wideresnet":
        #     return CifarWideResNet(
        #         config.depth, config.wide_factor, config.class_num, config.dropout
        #     )
        # else:
        #     raise ValueError("invalid module type : {:}".format(config.arch))
    elif super_type.startswith("infer"):
        # from .shape_infers import InferWidthCifarResNet
        # from .shape_infers import InferDepthCifarResNet
        # from .shape_infers import InferCifarResNet
        assert len(super_type.split("-")) == 2, "invalid super_type : {:}".format(
            super_type
        )
        infer_mode = super_type.split("-")[1]
        # if infer_mode == "width":
        #     return InferWidthCifarResNet(
        #         config.module,
        #         config.depth,
        #         config.xchannels,
        #         config.class_num,
        #         config.zero_init_residual,
        #     )
        # elif infer_mode == "depth":
        #     return InferDepthCifarResNet(
        #         config.module,
        #         config.depth,
        #         config.xblocks,
        #         config.class_num,
        #         config.zero_init_residual,
        #     )
        # elif infer_mode == "shape":
        #     return InferCifarResNet(
        #         config.module,
        #         config.depth,
        #         config.xblocks,
        #         config.xchannels,
        #         config.class_num,
        #         config.zero_init_residual,
        #     )
        if infer_mode == "nasnet.cifar":
            genotype = config.genotype
            if isinstance(extra_path, type(str)):  # reload genotype by extra_path
                if not osp.isfile(extra_path):
                    raise ValueError("invalid extra_path : {:}".format(extra_path))
                xdata = torch.load(extra_path)
                current_epoch = xdata["epoch"]
                genotype = xdata["genotypes"][current_epoch - 1]

            elif isinstance(extra_path, dict):
                genotype = extra_path
            else:  # reload genotype by extra_path
                genotype = extra_path._asdict()

            C = config.C if hasattr(config, "C") else config.ichannel
            N = config.N if hasattr(config, "N") else config.layers
            return NASNetonCIFAR(
                C, N, config.stem_multi, config.class_num, genotype, config.auxiliary
            )
        else:
            raise ValueError("invalid infer-mode : {:}".format(infer_mode))
    else:
        raise ValueError("invalid super-type : {:}".format(super_type))
