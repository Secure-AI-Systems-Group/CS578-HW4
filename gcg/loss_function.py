# Some useful libraries, feel free to import any others you need.
import torch.nn.functional as F

from torch import Tensor

# Your job is to compute the loss used by GCG, which is used in line 538 of algorithm.py (and in other places...)

def compute_loss(shift_logits: Tensor, shift_labels: Tensor) -> Tensor:
    loss = ...

    return loss
