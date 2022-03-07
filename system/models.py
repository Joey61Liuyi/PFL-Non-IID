# -*- coding: utf-8 -*-
# @Time    : 2021/10/23 11:07
# @Author  : LIU YI

from collections import namedtuple
from os import path as osp
import torch
import torch.nn as nn
from cell_operations import OPS as OPS_searched
from operations import OPS as OPS_infer
import numpy as np



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

def count_parameters(model_or_parameters, unit="mb", deprecated=False):
    if isinstance(model_or_parameters, nn.Module):
        counts = sum(np.prod(v.size()) for v in model_or_parameters.parameters())
    elif isinstance(model_or_parameters, nn.Parameter):
        counts = model_or_parameters.numel()
    elif isinstance(model_or_parameters, (list, tuple)):
        counts = sum(
            count_parameters(x, None, deprecated) for x in model_or_parameters
        )
    else:
        counts = sum(np.prod(v.size()) for v in model_or_parameters)
    if not isinstance(unit, str) and unit is not None:
        raise ValueError("Unknow type of unit: {:}".format(unit))
    elif unit is None:
        counts = counts
    elif unit.lower() == "kb" or unit.lower() == "k":
        counts /= 1e3 if deprecated else 2 ** 10  # changed from 1e3 to 2^10
    elif unit.lower() == "mb" or unit.lower() == "m":
        counts /= 1e6 if deprecated else 2 ** 20  # changed from 1e6 to 2^20
    elif unit.lower() == "gb" or unit.lower() == "g":
        counts /= 1e9 if deprecated else 2 ** 30  # changed from 1e9 to 2^30
    else:
        raise ValueError("Unknow unit: {:}".format(unit))
    return counts


def count_parameters_in_MB(model):
    return count_parameters(model, "mb", deprecated=True)
def get_model_infos(model, shape):
    # model = copy.deepcopy( model )

    model = add_flops_counting_methods(model)
    # model = model.cuda()
    model.eval()

    # cache_inputs = torch.zeros(*shape).cuda()
    # cache_inputs = torch.zeros(*shape)
    cache_inputs = torch.rand(*shape)
    if next(model.parameters()).is_cuda:
        cache_inputs = cache_inputs.cuda()
    # print_log('In the calculating function : cache input size : {:}'.format(cache_inputs.size()), log)
    with torch.no_grad():
        _____ = model(cache_inputs)
    FLOPs = compute_average_flops_cost(model) / 1e6
    Param = count_parameters_in_MB(model)

    if hasattr(model, "auxiliary_param"):
        aux_params = count_parameters_in_MB(model.auxiliary_param())
        print("The auxiliary params of this model is : {:}".format(aux_params))
        print(
            "We remove the auxiliary params from the total params ({:}) when counting".format(
                Param
            )
        )
        Param = Param - aux_params

    # print_log('FLOPs : {:} MB'.format(FLOPs), log)
    torch.cuda.empty_cache()
    model.apply(remove_hook_function)
    return FLOPs, Param

# ---- Public functions
def add_flops_counting_methods(model):
    model.__batch_counter__ = 0
    add_batch_counter_hook_function(model)
    model.apply(add_flops_counter_variable_or_reset)
    model.apply(add_flops_counter_hook_function)
    return model


def compute_average_flops_cost(model):
    """
    A method that will be available after add_flops_counting_methods() is called on a desired net object.
    Returns current mean flops consumption per image.
    """
    batches_count = model.__batch_counter__
    flops_sum = 0
    # or isinstance(module, torch.nn.AvgPool2d) or isinstance(module, torch.nn.MaxPool2d) \
    for module in model.modules():
        if (
            isinstance(module, torch.nn.Conv2d)
            or isinstance(module, torch.nn.Linear)
            or isinstance(module, torch.nn.Conv1d)
            or hasattr(module, "calculate_flop_self")
        ):
            flops_sum += module.__flops__
    return flops_sum / batches_count


# ---- Internal functions
def pool_flops_counter_hook(pool_module, inputs, output):
    batch_size = inputs[0].size(0)
    kernel_size = pool_module.kernel_size
    out_C, output_height, output_width = output.shape[1:]
    assert out_C == inputs[0].size(1), "{:} vs. {:}".format(out_C, inputs[0].size())

    overall_flops = (
        batch_size * out_C * output_height * output_width * kernel_size * kernel_size
    )
    pool_module.__flops__ += overall_flops


def self_calculate_flops_counter_hook(self_module, inputs, output):
    overall_flops = self_module.calculate_flop_self(inputs[0].shape, output.shape)
    self_module.__flops__ += overall_flops


def fc_flops_counter_hook(fc_module, inputs, output):
    batch_size = inputs[0].size(0)
    xin, xout = fc_module.in_features, fc_module.out_features
    assert xin == inputs[0].size(1) and xout == output.size(1), "IO=({:}, {:})".format(
        xin, xout
    )
    overall_flops = batch_size * xin * xout
    if fc_module.bias is not None:
        overall_flops += batch_size * xout
    fc_module.__flops__ += overall_flops


def conv1d_flops_counter_hook(conv_module, inputs, outputs):
    batch_size = inputs[0].size(0)
    outL = outputs.shape[-1]
    [kernel] = conv_module.kernel_size
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups
    conv_per_position_flops = kernel * in_channels * out_channels / groups

    active_elements_count = batch_size * outL
    overall_flops = conv_per_position_flops * active_elements_count

    if conv_module.bias is not None:
        overall_flops += out_channels * active_elements_count
    conv_module.__flops__ += overall_flops


def conv2d_flops_counter_hook(conv_module, inputs, output):
    batch_size = inputs[0].size(0)
    output_height, output_width = output.shape[2:]

    kernel_height, kernel_width = conv_module.kernel_size
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups
    conv_per_position_flops = (
        kernel_height * kernel_width * in_channels * out_channels / groups
    )

    active_elements_count = batch_size * output_height * output_width
    overall_flops = conv_per_position_flops * active_elements_count

    if conv_module.bias is not None:
        overall_flops += out_channels * active_elements_count
    conv_module.__flops__ += overall_flops


def batch_counter_hook(module, inputs, output):
    # Can have multiple inputs, getting the first one
    inputs = inputs[0]
    batch_size = inputs.shape[0]
    module.__batch_counter__ += batch_size


def add_batch_counter_hook_function(module):
    if not hasattr(module, "__batch_counter_handle__"):
        handle = module.register_forward_hook(batch_counter_hook)
        module.__batch_counter_handle__ = handle


def add_flops_counter_variable_or_reset(module):
    if (
        isinstance(module, torch.nn.Conv2d)
        or isinstance(module, torch.nn.Linear)
        or isinstance(module, torch.nn.Conv1d)
        or isinstance(module, torch.nn.AvgPool2d)
        or isinstance(module, torch.nn.MaxPool2d)
        or hasattr(module, "calculate_flop_self")
    ):
        module.__flops__ = 0


def add_flops_counter_hook_function(module):
    if isinstance(module, torch.nn.Conv2d):
        if not hasattr(module, "__flops_handle__"):
            handle = module.register_forward_hook(conv2d_flops_counter_hook)
            module.__flops_handle__ = handle
    elif isinstance(module, torch.nn.Conv1d):
        if not hasattr(module, "__flops_handle__"):
            handle = module.register_forward_hook(conv1d_flops_counter_hook)
            module.__flops_handle__ = handle
    elif isinstance(module, torch.nn.Linear):
        if not hasattr(module, "__flops_handle__"):
            handle = module.register_forward_hook(fc_flops_counter_hook)
            module.__flops_handle__ = handle
    elif isinstance(module, torch.nn.AvgPool2d) or isinstance(
        module, torch.nn.MaxPool2d
    ):
        if not hasattr(module, "__flops_handle__"):
            handle = module.register_forward_hook(pool_flops_counter_hook)
            module.__flops_handle__ = handle
    elif hasattr(module, "calculate_flop_self"):  # self-defined module
        if not hasattr(module, "__flops_handle__"):
            handle = module.register_forward_hook(self_calculate_flops_counter_hook)
            module.__flops_handle__ = handle


def remove_hook_function(module):
    hookers = ["__batch_counter_handle__", "__flops_handle__"]
    for hooker in hookers:
        if hasattr(module, hooker):
            handle = getattr(module, hooker)
            handle.remove()
    keys = ["__flops__", "__batch_counter__", "__flops__"] + hookers
    for ckey in keys:
        if hasattr(module, ckey):
            delattr(module, ckey)


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
        cell_results.pop(0)
        return logits, cell_results
        # if logits_aux is None:
        #     return out, logits
        # else:
        #     return out, [logits, logits_aux]



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
