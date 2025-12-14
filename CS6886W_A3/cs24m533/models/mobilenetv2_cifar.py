"""----------------------------------------------------------------
Modules:
    torch  : Core PyTorch library for tensor operations.
    nn     : Neural network layers and model components.
    typing : Provides type hints for cleaner model definitions.
----------------------------------------------------------------"""
import torch
import torch.nn as nn
from typing import List

__all__ = ["mobilenet_v2_cifar"]

"""---------------------------------------------
* def name :
*       _make_divisible
*
* purpose:
*       Ensures channel dimensions are divisible
*       by a given divisor for hardware efficiency.
*
* Input parameters:
*       v          : original channel value
*       divisor    : value to ensure divisibility
*       min_value  : minimum allowed channel value
*
* return:
*       Adjusted channel value
---------------------------------------------"""
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

"""---------------------------------------------
* class name :
*       ConvBNReLU
*
* purpose:
*       Combines convolution, batch normalization,
*       and ReLU6 activation into a single block.
*
* Input parameters:
*       in_planes   : number of input channels
*       out_planes  : number of output channels
*       kernel_size : convolution kernel size
*       stride      : convolution stride
*       groups      : number of convolution groups
*
* return:
*       Sequential convolutional block
---------------------------------------------"""
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )

"""---------------------------------------------
* class name :
*       InvertedResidual
*
* purpose:
*       Implements the inverted residual block
*       with optional skip connection.
*
* Input parameters:
*       inp          : number of input channels
*       oup          : number of output channels
*       stride       : stride for depthwise convolution
*       expand_ratio : channel expansion factor
*
* return:
*       Inverted residual block module
---------------------------------------------"""
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.append(ConvBNReLU(hidden_dim, hidden_dim,
                                 stride=stride, groups=hidden_dim))
        layers.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(oup))
        self.conv = nn.Sequential(*layers)

    """---------------------------------------------
    * def name :
    *       forward
    *
    * purpose:
    *       Performs forward propagation through
    *       the inverted residual block.
    *
    * Input parameters:
    *       x : input feature tensor
    *
    * return:
    *       Output feature tensor
    ---------------------------------------------"""
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

"""---------------------------------------------
* class name :
*       MobileNetV2CIFAR
*
* purpose:
*       Defines the MobileNetV2 architecture
*       customized for CIFAR datasets.
*
* Input parameters:
*       num_classes  : number of output classes
*       width_mult   : width multiplier for channels
*       round_nearest: channel rounding divisor
*       dropout      : dropout probability
*
* return:
*       MobileNetV2CIFAR model instance
---------------------------------------------"""
class MobileNetV2CIFAR(nn.Module):
    def __init__(self,
                 num_classes: int = 10,
                 width_mult: float = 1.0,
                 round_nearest: int = 8,
                 dropout: float = 0.2):
        super().__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],  # stride=1 for CIFAR
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(
            last_channel * max(1.0, width_mult), round_nearest
        )

        features: List[nn.Module] = [ConvBNReLU(3, input_channel, stride=1)]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(input_channel, output_channel, stride, expand_ratio=t)
                )
                input_channel = output_channel

        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    """---------------------------------------------
    * def name :
    *       forward
    *
    * purpose:
    *       Executes the forward pass of the
    *       MobileNetV2 CIFAR model.
    *
    * Input parameters:
    *       x : input image tensor
    *
    * return:
    *       Classification logits
    ---------------------------------------------"""
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



""""---------------------------------------------
* def name :
*       mobilenet_v2_cifar
*
* purpose:
*       Factory function to create a
*       MobileNetV2CIFAR model instance.
*
* Input parameters:
*       num_classes : number of output classes
*       width_mult  : width multiplier for channels
*       dropout     : dropout probability
*
* return:
*       MobileNetV2CIFAR model
---------------------------------------------"""
def mobilenet_v2_cifar(num_classes=10, width_mult=1.0, dropout=0.2):
    return MobileNetV2CIFAR(
        num_classes=num_classes,
        width_mult=width_mult,
        dropout=dropout,
    )
