"""----------------------------------------------------------------
Modules:
    torch        : Core PyTorch library for tensor operations.
    nn           : Neural network layers used for weight inspection.
    quantize_tensor : Utility function for quantizing tensors.
----------------------------------------------------------------"""

import torch
import torch.nn as nn
from .quant_ops import quantize_tensor

"""---------------------------------------------
* def name :
*       quantize_model_weights_inplace
*
* purpose:
*       Applies post-training quantization to
*       model weights in-place for compression
*       analysis.
*
* Input parameters:
*       model       : neural network model to be quantized
*       num_bits    : bit-width used for weight quantization
*       symmetric   : flag to enable symmetric quantization
*       per_channel : flag to enable per-channel quantization
*
* return:
*       None
---------------------------------------------"""
def quantize_model_weights_inplace(model: nn.Module,
                                   num_bits: int = 4,
                                   symmetric: bool = True,
                                   per_channel: bool = True) -> None:
    """
    Post-training quantization of weights only.

    - Loops over Conv2d / Linear
    - Quantizes weights to `num_bits`, then dequantizes back to float
      (so you can still run normal PyTorch inference)
    - For compression *analysis*, you assume stored weights are `num_bits`.
    """
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            with torch.no_grad():
                w = module.weight
                # per-channel along out_channels for conv/linear
                ch_axis = 0
                q_w_int, scale, zero_point = quantize_tensor(
                    w,
                    num_bits=num_bits,
                    symmetric=symmetric,
                    per_channel=per_channel,
                    ch_axis=ch_axis,
                )
                # dequantize to float, but now values are quantized
                w_q = (q_w_int.to(w.dtype) - zero_point.to(w.dtype)) * scale
                module.weight.copy_(w_q)
