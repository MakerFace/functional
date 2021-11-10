#!/usr/bin/env python3
# encoding=utf-8
import torch
from torch import functional as F
from torch import nn
from math import log10
from matplotlib.font_manager import FontManager

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import pyplot as plt


def smoothL1(x):
    res = []
    for i in x:
        if (i > -1.) and (i < 1.):
            res.append(0.5*i*i)
        else:
            res.append(abs(i)-0.5)
    return res


def softmax(x):
    from math import exp
    res = []
    for i in x:
        res.append(1/(1+exp(-i)))
    return res


class WeightedFocalLoss():
    def __init__(self, alpha=.25, gamma=2):
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs):
        res = []
        for i in inputs:
            res.append(-self.alpha * pow(1-i, self.gamma) * log10(i))
        return res

class CELoss():
    def forward(self, inputs):
        res = []
        for i in inputs:
            res.append(-0.25*log10(i))
        return res

# focal0 = CELoss()
focal0 = WeightedFocalLoss(gamma=0)
focal1 = WeightedFocalLoss(gamma=0.5)
focal2 = WeightedFocalLoss(gamma=1)
focal3 = WeightedFocalLoss(gamma=2)
focal4 = WeightedFocalLoss(gamma=5)

x = np.arange(0.01, 1, 0.01)
y0 = focal0.forward(x)
y1 = focal1.forward(x)
y2 = focal2.forward(x)
y3 = focal3.forward(x)
y4 = focal4.forward(x)

fig, ax = plt.subplots()
ax.plot(x, y0, label='gamma=0')
ax.plot(x, y1, label='gamma=0.5')
ax.plot(x, y2, label='gamma=1')
ax.plot(x, y3, label="gamma=2")
ax.plot(x, y4, label="gamma=5")
ax.legend()
plt.show()
