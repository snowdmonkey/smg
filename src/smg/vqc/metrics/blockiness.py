"""this module provide functionality of calculating blockiness related metrics"""
from typing import Union

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from smg.vqc.metrics.base import Metric


EPSILON = 1e-5


class BlockinessPerra(Metric):
    """Blockiness metric based on C. Perra, “A Low Computational Complexity Blockiness Estimation based on Spatial
    Analysis,” IEEE 22nd Telecommunications Forum (Telfor), 2014"""

    k = 2.3

    def __init__(self, height: int, width: int, use_cuda: bool = None):
        super(BlockinessPerra, self).__init__()

        if use_cuda is None:
            self._use_cuda = torch.cuda.is_available()
        else:
            self._use_cuda = use_cuda

        self._Mx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        self._My = self._Mx.t()
        self._Mx.unsqueeze_(0).unsqueeze_(0)
        self._My.unsqueeze_(0).unsqueeze_(0)
        self._height = height
        self._width = width

        self._omega1h = self._get_omega1h()
        self._omega1v = self._get_omega1v()
        self._omega2 = self._get_omega2()

        if self._use_cuda:
            self._Mx = self._Mx.cuda()
            self._My = self._My.cuda()
            self._omega1h = self._omega1h.cuda()
            self._omega1v = self._omega1v.cuda()
            self._omega2 = self._omega2.cuda()

    def _get_omega1h(self) -> Tensor:
        mask = torch.zeros(size=(8, 8), dtype=torch.float32)
        mask[[0, 7], :] = 1/16
        mask.unsqueeze_(0).unsqueeze_(0)
        return mask

    def _get_omega1v(self) -> Tensor:
        mask = torch.zeros(size=(8, 8), dtype=torch.float32)
        mask[:, [0, 7]] = 1/16
        mask.unsqueeze_(0).unsqueeze_(0)
        return mask

    def _get_omega2(self) -> Tensor:
        mask = torch.zeros(size=(8, 8), dtype=torch.float32)
        mask[1, range(1, 7)] = 1/20
        mask[6, range(1, 7)] = 1/20
        mask[range(1, 7), 1] = 1/20
        mask[range(1, 7), 6] = 1/20
        mask.unsqueeze_(0).unsqueeze_(0)
        return mask

    def process(self, frame: np.ndarray) -> float:
        frame = torch.tensor(frame, dtype=torch.float32)
        frame.unsqueeze_(0).unsqueeze_(0)

        if self._use_cuda:
            frame = frame.cuda()

        Dx = F.conv2d(frame, weight=self._Mx, stride=1, padding=1).abs()
        Dy = F.conv2d(frame, weight=self._My, stride=1, padding=1).abs()
        max_Dx = F.max_pool2d(Dx, kernel_size=8, stride=8)
        max_Dy = F.max_pool2d(Dy, kernel_size=8, stride=8)

        max_Dx[max_Dx < EPSILON] = EPSILON
        max_Dy[max_Dy < EPSILON] = EPSILON

        Bx = F.conv2d(Dx.abs(), weight=self._omega1v, stride=8) / max_Dx
        By = F.conv2d(Dy.abs(), weight=self._omega1h, stride=8) / max_Dy
        B = torch.max(Bx, By)
        D = (Dx ** 2 + Dy ** 2) ** 0.5

        max_D = F.max_pool2d(D, kernel_size=8, stride=8)
        max_D[max_D < EPSILON] = EPSILON
        I = F.conv2d(D, weight=self._omega2, stride=8) / max_D
        L = torch.abs(B ** self.k - I ** self.k) / torch.abs(B ** self.k + I ** self.k + EPSILON)
        return L.mean().item()
