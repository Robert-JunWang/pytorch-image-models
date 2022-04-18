
from functools import partial
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .hub import load_state_dict_from_url
from torchvision.models.mobilenetv2 import _make_divisible

from timm.models.layers import DropBlock2d, DropPath

from .registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


__all__ = ['PeleeNetV3', 'peleenet17', 'peleenet18', 'peleenet182', 'peleenet31', 'peleenet36', 'peleenet3s', 'peleenet3s2', 'peleenet3m', 'peleenet3m2', 'peleenet3m3', 'peleenet3xs']

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (1, 1),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'features.stemblock.stem1', 'classifier': 'classifier',
        **kwargs
    }

# default_cfgs = {
#     'peleenet1x': _cfg(url='https://github.com/edge-cv/benchmark/releases/download/pretrained/peleenet1x.pth'),
#     'peleenet2x': _cfg(url='https://github.com/edge-cv/benchmark/releases/download/pretrained/peleenet2x.pth')
# }


model_urls = {
    'peleenet3s': 'pretrained/peleenet3s.pth',
    'peleenet31': 'pretrained/peleenet31a.pth'
}


@register_model
def peleenet3xs(pretrained: bool = False, progress: bool = True, **kwargs: Any):

    width_mult=0.75
    depth_mult=1.0

    block_setting = [
        BlockConfig(3, 32, 1, 128, activation='relu', use_se=False, width_mult=width_mult, depth_mult=depth_mult),
        BlockConfig(4, 32, 2, 256, activation='relu', use_se=False, width_mult=width_mult, depth_mult=depth_mult),
        BlockConfig(6, 64, 4, 512, activation='relu', use_se=False, width_mult=width_mult, depth_mult=depth_mult),
        BlockConfig(4, 64, 4, 896, activation='relu', use_se=False, width_mult=width_mult, depth_mult=depth_mult, stride=1),
    ]
    return _peleenet('peleenet2x', pretrained, progress,
                     block_setting=block_setting,
                     first_layer = (3, 24, 3, 2),
                     **kwargs)

@register_model
def peleenet3s(pretrained: bool = False, progress: bool = True, **kwargs: Any):

    block_setting = [
        BlockConfig(3, 32, 2, 128, activation='relu', use_se=False),
        BlockConfig(4, 32, 4, 256, activation='relu', use_se=False),
        BlockConfig(6, 64, 4, 512, activation='hs', use_se=False),
        BlockConfig(4, 64, 4, 896, activation='hs', use_se=False, stride=1),
    ]

    return _peleenet('peleenet3s', pretrained, progress,
                     block_setting=block_setting,
                     **kwargs)


@register_model
def peleenet17(pretrained: bool = False, progress: bool = True, **kwargs: Any):

    block_setting = [
        BlockConfig(3, 32, 2, 128, activation='relu', use_se=False),
        BlockConfig(4, 32, 4, 256, activation='relu', use_se=False),
        BlockConfig(6, 64, 4, 512, activation='relu', use_se=False),
        BlockConfig(4, 64, 4, 896, activation='relu', use_se=False, stride=1),
    ]

    return _peleenet('peleenet17', pretrained, progress,
                     block_setting=block_setting,
                     **kwargs)

@register_model
def peleenet18(pretrained: bool = False, progress: bool = True, **kwargs: Any):

    block_setting = [
        BlockConfig(3, 32, 2, 128, activation='relu', use_se=False),
        BlockConfig(3, 32, 4, 256, activation='relu', use_se=False),
        BlockConfig(9, 64, 4, 512, activation='relu', use_se=False),
        BlockConfig(3, 64, 4, 896, activation='relu', use_se=False, stride=1),
    ]

    return _peleenet('peleenet18', pretrained, progress,
                     block_setting=block_setting,
                     **kwargs)

@register_model
def peleenet182(pretrained: bool = False, progress: bool = True, **kwargs: Any):

    block_setting = [
        BlockConfig(3, 32, 2, 128, activation='relu', use_se=False),
        BlockConfig(3, 32, 4, 224, activation='relu', use_se=False),
        BlockConfig(9, 64, 4, 512, activation='relu', use_se=False),
        BlockConfig(3, 64, 4, 704, activation='relu', use_se=False, stride=1),
    ]

    return _peleenet('peleenet182', pretrained, progress,
                     block_setting=block_setting,
                     **kwargs)

@register_model
def peleenet27(pretrained: bool = False, progress: bool = True, **kwargs: Any):

    block_setting = [
        BlockConfig(3, 32, 4, 128, activation='relu', use_se=False),
        BlockConfig(3, 48, 4, 256, activation='relu', use_se=False),
        BlockConfig(18, 64, 4, 512, activation='relu', use_se=False),
        BlockConfig(3, 64, 4, 1024, activation='relu', use_se=False, stride=1),
    ]
    return _peleenet('peleenet36', pretrained, progress,
                     block_setting=block_setting,
                     **kwargs)

@register_model
def peleenet31(pretrained: bool = False, progress: bool = True, **kwargs: Any):

    block_setting = [
        BlockConfig(3, 32, 4, 128, activation='relu', use_se=False),
        BlockConfig(8, 48, 4, 256, activation='relu', use_se=False),
        BlockConfig(12, 64, 4, 512, activation='relu', use_se=False),
        BlockConfig(8, 64, 4, 1024, activation='relu', use_se=False, stride=1),
    ]
    return _peleenet('peleenet31', pretrained, progress,
                     block_setting=block_setting,
                     **kwargs)


@register_model
def peleenet36(pretrained: bool = False, progress: bool = True, **kwargs: Any):

    block_setting = [
        BlockConfig(3, 32, 4, 128, activation='relu', use_se=False),
        BlockConfig(3, 48, 4, 256, activation='relu', use_se=False),
        BlockConfig(27, 64, 4, 512, activation='relu', use_se=False),
        BlockConfig(3, 64, 4, 1024, activation='relu', use_se=False, stride=1),
    ]
    return _peleenet('peleenet36', pretrained, progress,
                     block_setting=block_setting,
                     **kwargs)

@register_model
def peleenet3s2(pretrained: bool = False, progress: bool = True, **kwargs: Any):

    block_setting = [
        BlockConfig(3, 32, 2, 128, activation='relu', use_se=False),
        BlockConfig(4, 32, 4, 256, activation='relu', use_se=True),
        BlockConfig(6, 64, 4, 512, activation='hs', use_se=False),
        BlockConfig(4, 64, 4, 896, activation='hs', use_se=True, stride=1),
    ]

    return _peleenet('peleenet1x', pretrained, progress,
                     block_setting=block_setting,
                     **kwargs)

@register_model
def peleenet3m(pretrained: bool = False, progress: bool = True, **kwargs: Any):

    block_setting = [
        BlockConfig(3, 32, 2, 128, activation='relu', use_se=False),
        BlockConfig(8, 48, 4, 256, activation='relu', use_se=False),
        BlockConfig(12, 64, 4, 512, activation='hs', use_se=False),
        BlockConfig(8, 64, 4, 1024, activation='hs', use_se=False, stride=1),
    ]
    return _peleenet('peleenet2x', pretrained, progress,
                     block_setting=block_setting,
                     **kwargs)

@register_model
def peleenet3m2(pretrained: bool = False, progress: bool = True, **kwargs: Any):

    block_setting = [
        BlockConfig(3, 32, 4, 128, activation='relu', use_se=False),
        BlockConfig(8, 48, 4, 256, activation='relu', use_se=False),
        BlockConfig(12, 64, 4, 512, activation='hs', use_se=False),
        BlockConfig(8, 64, 4, 1024, activation='hs', use_se=False, stride=1),
    ]
    return _peleenet('peleenet2x', pretrained, progress,
                     block_setting=block_setting,
                     **kwargs)

@register_model
def peleenet3m3(pretrained: bool = False, progress: bool = True, **kwargs: Any):

    block_setting = [
        BlockConfig(4, 40, 2, 128, activation='relu', use_se=False),
        BlockConfig(8, 48, 4, 256, activation='relu', use_se=False),
        BlockConfig(12, 64, 4, 512, activation='hs', use_se=False),
        BlockConfig(8, 64, 4, 1024, activation='hs', use_se=False, stride=1),
    ]
    return _peleenet('peleenet2x', pretrained, progress,
                     block_setting=block_setting,
                     **kwargs)


@register_model
def peleenet3mb(pretrained: bool = False, progress: bool = True, **kwargs: Any):

    # width_mult=1.2
    # depth_mult=1.4
    width_mult=1.0
    depth_mult=1.0

    block_setting = [
        BlockConfig(4, 40, 2, 128, activation='relu', use_se=False),
        BlockConfig(5, 56, 4, 304, activation='relu', use_se=False, width_mult=width_mult, depth_mult=depth_mult),
        BlockConfig(8, 80, 4, 616, activation='hs', use_se=False, width_mult=width_mult, depth_mult=depth_mult),
        BlockConfig(5, 80, 4, 1024, activation='hs', use_se=False, width_mult=width_mult, depth_mult=depth_mult, stride=1),
    ]
    return _peleenet('peleenet2x', pretrained, progress,
                     block_setting=block_setting,
                     **kwargs)



@register_model
def peleenet3m3(pretrained: bool = False, progress: bool = True, **kwargs: Any):

    block_setting = [
        BlockConfig(3, 32, 4, 128, activation='relu', use_se=False),
        BlockConfig(8, 48, 4, 256, activation='relu', use_se=False),
        BlockConfig(12, 64, 4, 512, activation='hs', use_se=False),
        BlockConfig(8, 64, 4, 1024, activation='hs', use_se=False, stride=1),
    ]
    return _peleenet('peleenet2x', pretrained, progress,
                     block_setting=block_setting,
                     **kwargs)


@register_model
def peleenet3l(pretrained: bool = False, progress: bool = True, **kwargs: Any):

    width_mult=1.2
    depth_mult=1.4

    block_setting = [
        BlockConfig(3, 32, 2, 128, activation='relu', use_se=False, width_mult=width_mult, depth_mult=depth_mult),
        BlockConfig(8, 48, 4, 256, activation='relu', use_se=False, width_mult=width_mult, depth_mult=depth_mult),
        BlockConfig(12, 64, 4, 512, activation='hs', use_se=False, width_mult=width_mult, depth_mult=depth_mult),
        BlockConfig(8, 64, 4, 1024, activation='hs', use_se=False, width_mult=width_mult, depth_mult=depth_mult, stride=1),
    ]
    return _peleenet('peleenet2x', pretrained, progress,
                     block_setting=block_setting,
                     **kwargs)

# class SqueezeExcitation(nn.Module):
#     # Implemented as described at Figure 4 of the MobileNetV3 paper
#     def __init__(self, input_channels: int, squeeze_factor: int = 4):
#         super().__init__()
#         squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
#         self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)

#     def _scale(self, input: Tensor, inplace: bool) -> Tensor:
#         scale = F.adaptive_avg_pool2d(input, 1)
#         scale = self.fc1(scale)
#         scale = self.relu(scale)
#         scale = self.fc2(scale)
#         return F.hardsigmoid(scale, inplace=inplace)

#     def forward(self, input: Tensor) -> Tensor:
#         scale = self._scale(input, True)
#         return scale * input



class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return torch.mul(x, y)

class BlockConfig:
    def __init__(self,
                 num_layers: int, growth_rate: int, bottleneck_width: int, out_channels: int,
                 activation: str = 'relu',
                 stride: int = 2,
                 use_se: bool = False,
                 width_mult: float = 1.0,
                 depth_mult: float = 1.0,
                 drop_block:Any = None):
        self.num_layers = math.ceil(num_layers*depth_mult)
        self.growth_rate = self.adjust_channels(growth_rate, width_mult)
        self.bottleneck_width = bottleneck_width
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.activation = activation
        self.drop_block = drop_block
        self.stride = stride

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)



class BasicConv2d(nn.Sequential):

    def __init__(self, in_channels, out_channels, activation='relu', drop_block=None, inplace=True, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels) 
        if activation == 'hs':
            self.activation = nn.Hardswish(inplace=inplace)
        elif activation == 'silu':
            self.activation = nn.SiLU(inplace=inplace)
        else:
            self.activation = nn.ReLU(inplace=inplace)
        if drop_block is not None:
            self.drop_block = drop_block


class _DenseLayer(nn.Module):
    def __init__(self, in_channels, config):
        super(_DenseLayer, self).__init__()


        growth_rate = config.growth_rate // 2
        self.out_channels = in_channels + 3 * growth_rate 
        inter_channels = growth_rate  * config.bottleneck_width  

        self.branch1a = BasicConv2d(in_channels, inter_channels, activation=config.activation, kernel_size=1)
        self.branch1b = BasicConv2d(inter_channels, growth_rate, activation=config.activation, kernel_size=3, padding=1, drop_block=config.drop_block)

        self.branch2a = BasicConv2d(in_channels, inter_channels, activation=config.activation, kernel_size=1)
        self.branch2b = BasicConv2d(inter_channels, growth_rate, activation=config.activation, kernel_size=3, padding=1)
        self.branch2c = BasicConv2d(growth_rate, growth_rate, activation=config.activation, kernel_size=3, padding=1, drop_block=config.drop_block)
        if config.use_se:
            # self.se = SqueezeExcitation(growth_rate*3, growth_rate)
            self.se = SqueezeExcitation(growth_rate*3)
        else:
            self.se = None

    def forward(self, x):
        branch1 = self.branch1a(x)
        branch1 = self.branch1b(branch1)

        branch2 = self.branch2a(x)
        branch2b = self.branch2b(branch2)
        branch2c = self.branch2c(branch2b)

        if self.se is not None:
            out = torch.cat([branch1, branch2b, branch2c], 1)
            out = self.se(out)
            return torch.cat([x, out], 1)
        else:
            return torch.cat([x, branch1, branch2b, branch2c], 1)



class _DenseBlock(nn.Sequential):
    def __init__(self, in_channels: int, config: BlockConfig):
        super(_DenseBlock, self).__init__()
        self.out_channels = in_channels
        for i in range(config.num_layers):
            layer = _DenseLayer(self.out_channels, config)
            self.add_module('denselayer%d' % (i + 1), layer)
            self.out_channels = layer.out_channels

class _StemBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_StemBlock, self).__init__()

        self.stem2a = BasicConv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        self.stem2b = BasicConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.stem3 = BasicConv2d(3*in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):

        branch1 = self.pool(x)
        branch2a = self.stem2a(x)
        branch2b = self.stem2b(branch2a)

        out = torch.cat([branch1, branch2a, branch2b], 1)
        out = self.stem3(out)

        return out


class PeleeNetV3(nn.Module):
    r"""PeleeNet model class
    Args:

        drop_rate (float) - dropout rate
        drop_path_rate (float) - drop path rate
        num_classes (int) - number of classification classes
    """
    def __init__(self, 
            block_setting: List[BlockConfig],
            first_layer = (3, 32, 3, 2),
            num_classes: int = 1000,
            drop_rate=0.2, **kwargs):

        super().__init__()


        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (isinstance(block_setting, Sequence) or
                  all([isinstance(s, BlockConfig) for s in block_setting])):
            raise TypeError("The block_setting should be List[BlockConfig]")

        self.num_classes = num_classes


        in_channels, out_channels, first_kernel_size, first_stride = first_layer
        padding = first_kernel_size // 2

        self.features = nn.Sequential(OrderedDict([
                ( 'conv0', BasicConv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=first_kernel_size, stride=first_stride, padding=padding)),
                ('stemblock', _StemBlock(in_channels=out_channels,out_channels=out_channels)), 
            ]))  


        # Each denseblock
        in_channels = out_channels
        for i, config in enumerate(block_setting):
            block = _DenseBlock(in_channels=in_channels, config=config)
            self.features.add_module('denseblock%d' % (i + 1), block)
            in_channels = block.out_channels

            self.features.add_module('transition%d' % (i + 1), BasicConv2d(in_channels, config.out_channels, activation=config.activation, kernel_size=1))
            in_channels = config.out_channels

            if config.stride != 1:
                self.features.add_module('pool%d' % (i + 1), nn.AvgPool2d(kernel_size=config.stride, stride=config.stride))
  
        # Linear layer
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate, inplace=True),
            nn.Linear(in_channels, num_classes),
        )

        self._initialize_weights()



    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                # m.eps = 1e-3
                # m.momentum = 0.03

                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

def _peleenet(arch: str, pretrained: bool = False, progress: bool = True, **kwargs: Any):
    print(kwargs)
    model = PeleeNetV3(**kwargs)

    if pretrained:
        model_url = model_urls[arch]

        if model_url.startswith('http'):
            state_dict = load_state_dict_from_url(model_url, progress=progress)
        else:
            state_dict = torch.load(model_url)

        print("Loading pretrained weights from %s" %(model_url))
        model.load_state_dict(state_dict)

    return model


if __name__ == '__main__':
    input_var = torch.Tensor(1,3,224,224)
    model = peleenet3xs()

    layer_types  = []
    output_shapes = []
    def print_size(self, input, output):
        layer_types.append(torch.typename(self).split('.')[-1])
        output_shapes.append(output.data.size())
        # print(torch.typename(self).split('.')[-1], ' output size:',output.data.size())

    names = list(model.features._modules.keys())
    for layer in model.features:
        layer.register_forward_hook(print_size)


    output = model.forward(input_var)
    for i, (name, type, size) in enumerate(zip(names, layer_types, output_shapes)):
        print(i, name, type, size)


