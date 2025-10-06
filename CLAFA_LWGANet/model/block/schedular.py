# """
# https://github.com/huggingface/transformers
# """

# import math
# from functools import partial
# from torch.optim import Optimizer
# from torch.optim.lr_scheduler import LambdaLR


# def _get_cosine_schedule_with_warmup_lr_lambda(
#     current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
# ):
#     if current_step < num_warmup_steps:
#         return float(current_step) / float(max(1, num_warmup_steps))
#     progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
#     return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


# def get_cosine_schedule_with_warmup(
#     optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1):
#     """
#     Create a schedule with a learning rate that decreases following the values of the cosine function between the
#     initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
#     initial lr set in the optimizer.
#     Args:
#         optimizer ([`~torch.optim.Optimizer`]):
#             The optimizer for which to schedule the learning rate.
#         num_warmup_steps (`int`):
#             The number of steps for the warmup phase.
#         num_training_steps (`int`):
#             The total number of training steps.
#         num_cycles (`float`, *optional*, defaults to 0.5):
#             The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
#             following a half-cosine).
#         last_epoch (`int`, *optional*, defaults to -1):
#             The index of the last epoch when resuming training.
#     Return:
#         `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
#     """

#     lr_lambda = partial(
#         _get_cosine_schedule_with_warmup_lr_lambda,
#         num_warmup_steps=num_warmup_steps,
#         num_training_steps=num_training_steps,
#         num_cycles=num_cycles,
#     )
#     return LambdaLR(optimizer, lr_lambda, last_epoch)

import math
from functools import partial
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float,
    min_lr_ratio: float = 0.0
):
    """
    Compute LR multiplier with linear warmup and cosine decay.

    Args:
        current_step (int): Current training step.
        num_warmup_steps (int): Steps used for linear warmup.
        num_training_steps (int): Total number of training steps.
        num_cycles (float): Cosine cycles (e.g. 0.5 = cosine decay).
        min_lr_ratio (float): Minimum lr ratio (min_lr = min_lr_ratio * base_lr)

    Returns:
        float: learning rate multiplier (0 ~ 1)
    """
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
    return max(min_lr_ratio, cosine_decay)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.0,
    last_epoch: int = -1
):
    """
    Enhanced cosine LR scheduler with warmup and optional min_lr_ratio floor.

    Args:
        optimizer (Optimizer): PyTorch optimizer.
        num_warmup_steps (int): Linear warmup steps.
        num_training_steps (int): Total training steps.
        num_cycles (float): Cosine wave cycles. 0.5 = decay to 0 once.
        min_lr_ratio (float): Minimum LR as a fraction of base LR.
        last_epoch (int): Last epoch for resuming training.

    Returns:
        LambdaLR: Learning rate scheduler.
    """
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
