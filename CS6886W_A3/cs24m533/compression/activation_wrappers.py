"""----------------------------------------------------------------
Modules:
    torch              : Core PyTorch library for tensor operations.
    nn                 : Neural network modules and layers.
    typing             : Provides type hints for dictionaries.
    fake_quantize_tensor : Performs fake quantization of tensors.
    tensor_num_bits    : Computes number of bits required for tensors.
----------------------------------------------------------------"""
import torch
import torch.nn as nn
from typing import Dict
from .quant_ops import fake_quantize_tensor, tensor_num_bits

"""---------------------------------------------
* class name :
*       QuantizedActivationWrapper
*
* purpose:
*       Wraps a module to apply fake quantization
*       on activations and track activation
*       bit usage statistics.
*
* Input parameters:
*       module       : neural network module to wrap
*       num_bits     : bit-width used for activation quantization
*       symmetric    : flag to enable symmetric quantization
*       per_channel  : flag to enable per-channel quantization
*       name         : name identifier for the wrapped module
*
* return:
*       QuantizedActivationWrapper module
---------------------------------------------"""
class QuantizedActivationWrapper(nn.Module):
    def __init__(self,
                 module: nn.Module,
                 num_bits: int,
                 symmetric: bool = True,
                 per_channel: bool = False,
                 name: str = ""):
        super().__init__()
        self.module = module
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.name = name

        self.register_buffer("activation_bits_accum", torch.zeros(1))
        self.register_buffer("num_samples", torch.zeros(1))

    """---------------------------------------------
    * def name :
    *       forward
    *
    * purpose:
    *       Applies fake quantization to activations
    *       while preserving gradients using STE.
    *
    * Input parameters:
    *       x : input activation tensor
    *
    * return:
    *       Quantized activation tensor
    ---------------------------------------------"""
    def forward(self, x):
        # 1) Update stats WITHOUT breaking grad
        with torch.no_grad():
            # estimate bits used for this activation (toy example)
            # you may already have smarter code here:
            self.num_samples += 1
            self.activation_bits_accum += x.numel() * self.num_bits

        # 2) If we’re not actually quantizing (e.g., bits >= 32), just pass through
        if self.num_bits >= 32:
            return x

        # 3) Quantize activations – but keep gradient using STE
        #    DO NOT wrap this in torch.no_grad()
        from compression.quant_ops import quantize_tensor

        q_x_int, scale, zero_point = quantize_tensor(
            x,
            num_bits=self.num_bits,
            symmetric=self.symmetric,
            per_channel=self.per_channel,
        )

        # Convert back to float for the network
        q_x = (q_x_int.to(x.dtype) - zero_point.to(x.dtype)) * scale

        # Straight-Through Estimator (STE):
        # Forward: use quantized value q_x
        # Backward: gradient flows as if it were identity (through x)
        x_q = x + (q_x - x).detach()

        return x_q

    """---------------------------------------------
    * def name :
    *       get_avg_activation_bits
    *
    * purpose:
    *       Computes the average number of bits
    *       used per activation sample.
    *
    * Input parameters:
    *       None
    *
    * return:
    *       Average activation bits (float)
    ---------------------------------------------"""
    def get_avg_activation_bits(self) -> float:
        if self.num_samples.item() == 0:
            return 0.0
        return self.activation_bits_accum.item() / self.num_samples.item()
        
  
 
"""---------------------------------------------
* def name :
*       wrap_activations_in_model
*
* purpose:
*       Recursively wraps model layers with
*       activation quantization wrappers.
*
* Input parameters:
*       model       : neural network model
*       num_bits    : bit-width for activation quantization
*       symmetric   : flag to enable symmetric quantization
*       per_channel : flag to enable per-channel quantization
*
* return:
*       None
---------------------------------------------"""
def wrap_activations_in_model(model: nn.Module,
                              num_bits: int,
                              symmetric: bool = True,
                              per_channel: bool = False):
    for name, child in list(model.named_children()):
        if isinstance(child, (nn.Conv2d, nn.Linear, nn.ReLU, nn.ReLU6, nn.Sequential)):
            wrapped = QuantizedActivationWrapper(
                module=child,
                num_bits=num_bits,
                symmetric=symmetric,
                per_channel=per_channel,
                name=name,
            )
            setattr(model, name, wrapped)
        else:
            wrap_activations_in_model(child, num_bits, symmetric, per_channel)

"""---------------------------------------------
* def name :
*       collect_activation_stats
*
* purpose:
*       Collects average activation bit usage
*       from all wrapped activation layers.
*
* Input parameters:
*       model : neural network model
*
* return:
*       Dictionary mapping layer names to
*       average activation bit values
---------------------------------------------"""
def collect_activation_stats(model: nn.Module) -> Dict[str, float]:
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, QuantizedActivationWrapper):
            stats[name] = module.get_avg_activation_bits()
    return stats
