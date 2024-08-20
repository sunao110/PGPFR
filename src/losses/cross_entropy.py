from typing import Optional, Sequence

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .helpers import one_hot
except:
    from helpers import one_hot


class Loss(nn.Module):
    def __init__(self,
                 n_classes: int,
                 ignore: Optional[int] = None,
                 weights: Optional[Sequence[float]] = None,
                 ) -> None:

        super().__init__()

        self.n_classes = n_classes
        if weights is None:
            weights = [1.] * self.n_classes

        assert isinstance(weights, (list, tuple)), \
            f"weights must be of type (list, tuple)."
        weights = np.array(weights)
        if ignore is not None:
            weights[ignore] = 0
        weights = torch.from_numpy(weights).to(torch.get_default_dtype())
        self.weights = weights.view(1, -1)

    def forward(self,
                logits: torch.Tensor,
                target: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                mask_sample_weight: Optional[torch.Tensor] = None
                ) -> torch.Tensor:

        device = logits.device
        log_pred = F.log_softmax(logits, dim=1)  # [bs, n_classes]
        # 改变one_hot 维度
        if target.ndim < 2:
            target = one_hot(target, self.n_classes).to(device)  # [bs, n_classes]

        view_shape = [1, -1] + [1] * (target.ndim - 2)
        weights = self.weights.view(view_shape).to(device)  # [1, n_classes]
        loss = - log_pred * target  # [bs, n_classes]
        loss = loss.sum(1)  # [bs]
        if mask is not None and mask_sample_weight is not None:
            # print("cross", mask.shape)
            mask_sample_weight = mask_sample_weight.to(loss.device)
            mask = mask.to(loss.device)
            loss = loss * mask * mask_sample_weight
            # mask 相当于已经做了有权均值处理
            # loss = loss.sum() / torch.count_nonzero(mask)
            loss = loss.sum()
            return loss

        mult_ = (log_pred.shape[1] / np.prod(log_pred.shape))
        loss = loss.sum() * mult_
        return loss
