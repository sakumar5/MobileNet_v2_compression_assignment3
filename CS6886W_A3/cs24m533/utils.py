"""----------------------------------------------------------------
Modules:
    os    : Handles file paths and directory operations.
    math  : Provides mathematical functions for scheduling.
    torch : PyTorch utilities for deep learning operations.
----------------------------------------------------------------"""
import os
import math
import torch



"""---------------------------------------------
* def name :
*       set_seed
*
* purpose:
*       Sets random seeds across Python, NumPy, and PyTorch
*       to ensure reproducible and deterministic results
*       during model training and evaluation.
*
* Input parameters:
*       seed : integer value used as the random seed
*              (default = 42)
*
* return:
*       None
---------------------------------------------"""
def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

"""---------------------------------------------
* def name :
*       accuracy
*
* purpose:
*       Computes Top-k classification accuracy for
*       model predictions.
*
* Input parameters:
*       output : model output logits or scores
*       target : ground-truth class labels
*       topk   : tuple indicating which Top-k
*                accuracies to compute (e.g., (1,), (1,5))
*
* return:
*       List of Top-k accuracy values (in percentage)
---------------------------------------------"""
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

"""---------------------------------------------
* def name :
*       save_checkpoint
*
* purpose:
*       Saves the current training state (model,
*       optimizer, epoch information) to disk.
*
* Input parameters:
*       state    : dictionary containing training state
*       filename : path where the checkpoint is saved
*
* return:
*       None
---------------------------------------------"""
def save_checkpoint(state, filename: str):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)

"""---------------------------------------------
* def name :
*       bits_to_megabytes
*
* purpose:
*       Converts a value from bits to megabytes (MB).
*
* Input parameters:
*       bits : size in bits
*
* return:
*       Size converted to megabytes (float)
---------------------------------------------"""
def bits_to_megabytes(bits: int) -> float:
    return bits / 8.0 / (1024.0 * 1024.0)


"""---------------------------------------------
* def name :
*       cosine_lr_scheduler
*
* purpose:
*       Applies cosine decay to the learning rate
*       during training for smoother convergence.
*
* Input parameters:
*       optimizer     : optimizer whose learning rate is updated
*       base_lr       : initial learning rate
*       epoch         : current training epoch
*       total_epochs  : total number of training epochs
*
* return:
*       Updated learning rate value
---------------------------------------------"""
def cosine_lr_scheduler(optimizer, base_lr, epoch, total_epochs):
    lr = 0.5 * base_lr * (1 + math.cos(math.pi * epoch / total_epochs))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr
