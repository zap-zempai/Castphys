# -*- coding: utf-8 -*-
"""
@author: KDH
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.nn.functional as F

import pretrainedmodels
import pretrainedmodels.utils as utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'alexnet'  # 'bninception'
# resnext = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet').to(device)
alexnet = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet').to(device)
print()


def _encoder2():
    encoder2 = Encoder2()
    return encoder2


def _regressor():
    regressor2 = Regressor_light_new(is_attention=0)
    return regressor2


def nn_output():
    encoder2 = _encoder2().to(device)
    regressor = _regressor().to(device)
    return encoder2, regressor


class Encoder2(nn.Module):

    def __init__(self):
        super(Encoder2, self).__init__()
        try:
            self.features = alexnet._features
        except:
            net = alexnet
            modules = list(net.children())[:-1]  # delete the last fc layer.
            self.features = nn.Sequential(*modules)

    def forward(self, x):
        x = self.features(x)
        return x


class Regressor(nn.Module):

    def __init__(self):
        super(Regressor, self).__init__()

        self.avgpool = alexnet.avgpool
        self.lin0 = alexnet.linear0
        self.lin1 = alexnet.linear1
        self.relu0 = alexnet.relu0
        self.relu1 = alexnet.relu1
        self.drop0 = alexnet.dropout0
        self.drop1 = alexnet.dropout0
        self.last_linear = alexnet.last_linear
        self.va_regressor = nn.Linear(1000, 2)

    def forward(self, x):
        x = torch.flatten(self.avgpool(x), 1)
        x = self.relu0(self.lin0(self.drop0(x)))
        x = self.relu1(self.lin1(self.drop1(x)))

        x = self.last_linear(x)
        x = self.va_regressor(x)
        return x


class Regressor_light(nn.Module):

    def __init__(self):
        super(Regressor_light, self).__init__()

        self.avgpool = alexnet.avgpool
        self.lin0 = nn.Linear(9216, 64)  # 64
        self.lin1 = nn.Linear(64, 8)
        self.relu0 = alexnet.relu0
        self.relu1 = alexnet.relu1
        self.drop0 = alexnet.dropout0
        self.drop1 = alexnet.dropout0
        #        self.last_linear = alexnet.last_linear
        self.va_regressor = nn.Linear(8, 2)

    def forward(self, x):
        x = torch.flatten(self.avgpool(x), 1)
        x = self.relu0(self.lin0(self.drop0(x)))
        x = self.relu1(self.lin1(self.drop1(x)))

        #        x = self.last_linear(x)
        x = self.va_regressor(x)
        return x


class Regressor_light_new(nn.Module):

    def __init__(self, is_attention):
        super(Regressor_light_new, self).__init__()
        self.avgpool = alexnet.avgpool
        self.lin0 = nn.Linear(9216, 32)
        self.lin1 = nn.Linear(32, 256)
        self.lin2 = nn.Linear(9216, 32)  # TODO(): add independent linear unit
        self.relu0 = alexnet.relu0
        self.relu1 = alexnet.relu1
        self.drop0 = alexnet.dropout0
        self.drop1 = alexnet.dropout0
        self.va_regressor = nn.Linear(256, 2)

        self.pw_conv = nn.Conv2d(512, 256, kernel_size=1)
        self.mpool = nn.MaxPool2d(2, stride=2, return_indices=False)
        self.apool = nn.AvgPool2d(2, stride=2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.sigmoid = nn.Sigmoid()

        self.is_attention = is_attention

    def forward(self, x):

        if self.is_attention:
            x1 = torch.flatten(self.avgpool(x), 1)  # shape: [BS, 9216]
            x1 = self.relu0(self.lin0(self.drop0(x1)))
            x1 = self.relu1(self.lin1(self.drop1(x1)))
            x_va = self.va_regressor(x1)

            mout = self.mpool(x)
            aout = self.apool(x)  # shape: [BS, 256, 3, 3]
            x2_res = self.sigmoid(self.upsample(self.pw_conv(torch.cat([mout, aout], dim=1))))
            #            x2 = x2_res * self.avgpool(x)
            x2 = x2_res + self.avgpool(x)  # GPU 2 (mul -> add)

            x2 = torch.flatten(x2, 1)  # shape: [BS, 9216]
            x_btl_1 = self.relu0(self.lin0(self.drop0(x2)))
        #            x_btl_1 = self.relu0(self.lin2(self.drop0(x2)))  # GPU 3 (independent linear unit)
        else:
            x = torch.flatten(self.avgpool(x), 1)  # shape: [BS, 9216]
            x_btl_1 = self.relu0(self.lin0(self.drop0(x)))
            x_btl_2 = self.relu1(self.lin1(self.drop1(x_btl_1)))
            x_va = self.va_regressor(x_btl_2)

        #        return x_va, x_btl_1
        return x_va
