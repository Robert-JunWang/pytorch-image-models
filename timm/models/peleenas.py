
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.mobilenetv2 import _make_divisible

from .rep_vgg_block import RepVGGBlock


from .registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


__all__ = ['peleenas1x', 'peleenas1x2']

model_urls = {
    'peleenas1x': 'https://github.com/edge-cv/benchmark/releases/download/pretrained/peleenet1x.pth',
    'peleenas1x': 'https://github.com/edge-cv/benchmark/releases/download/pretrained/peleenet2x.pth'
}

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (1, 1),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'features.stemblock.stem1', 'classifier': 'classifier',
        **kwargs
    }

@register_model
def peleenas1x2(pretrained: bool = False, progress: bool = True, **kwargs: Any):

    block_setting = [
        BlockConfig(3, 32, 2, 128, activation='relu', use_se=False),
        BlockConfig(6, 32, 4, 256, activation='relu', use_se=False),
        BlockConfig(9, 64, 4, 512, activation='relu', use_se=False),
        BlockConfig(6, 64, 4, 896, activation='relu', use_se=False, stride=1),
    ]

    return _peleenet('peleenet3s', pretrained, progress,
                     block_setting=block_setting,
                     **kwargs)

@register_model
def peleenas1x(pretrained: bool = False, progress: bool = True, **kwargs: Any):

    block_setting = [
        BlockConfig(4, 32, 2, 128, activation='relu', use_se=False),
        BlockConfig(6, 32, 4, 256, activation='relu', use_se=False),
        BlockConfig(12, 64, 4, 512, activation='relu', use_se=False),
        BlockConfig(6, 64, 4, 896, activation='relu', use_se=False, stride=1),
    ]

    return _peleenet('peleenet3s', pretrained, progress,
                     block_setting=block_setting,
                     **kwargs)

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



# class _DenseLayer(nn.Module):
#     def __init__(self, in_channels, growth_rate, bottleneck_width, use_se, deploy=False):
#         super(_DenseLayer, self).__init__()


#         inter_channel = growth_rate  * bottleneck_width  

#         self.conv1 = BasicConv2d(in_channels, inter_channel, kernel_size=1)
#         self.rep1 = RepVGGBlock(inter_channel, growth_rate, kernel_size=3, padding=1, deploy=deploy)
#         self.rep2 = RepVGGBlock(growth_rate, growth_rate, kernel_size=3, padding=1, deploy=deploy)
#         self.out_channels = in_channels + growth_rate*2

#     def forward(self, x):
#         out = self.conv1(x)
#         out1 = self.rep1(out)
#         out2 = self.rep2(out1)

#         return torch.cat([x, out1, out2], 1)



# class _DenseBlock(nn.Sequential):
#     def __init__(self, num_layers, in_channels, bn_size, growth_rate, use_se):
#         super(_DenseBlock, self).__init__()
#         self.out_channels = in_channels
#         for i in range(num_layers):
#             layer = _DenseLayer(self.out_channels, growth_rate, bn_size, use_se)
#             self.out_channels = layer.out_channels
#             self.add_module('denselayer%d' % (i + 1), layer)

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




class _DenseBlock(nn.Module):
    def __init__(self, in_channels: int, config: BlockConfig, deploy=False):
        super(_DenseBlock, self).__init__()
        self.out_channels = in_channels
        growth_rate = config.growth_rate // 2
        inter_channels = growth_rate  * config.bottleneck_width  

        branch1 = []
        branch2 = []

        self.branch1a = BasicConv2d(in_channels, inter_channels, activation=config.activation, kernel_size=1)
        self.branch2a = BasicConv2d(in_channels, inter_channels, activation=config.activation, kernel_size=1)

        in_channels = inter_channels

        for i in range(config.num_layers):
            branch1.append(RepVGGBlock(in_channels, growth_rate, kernel_size=3, padding=1, deploy=deploy))
            branch2.append(RepVGGBlock(in_channels, growth_rate, kernel_size=3, padding=1, deploy=deploy))
            in_channels = growth_rate

            self.out_channels += 2*growth_rate

        self.branch1 = nn.ModuleList(branch1)
        self.branch2 = nn.ModuleList(branch2)

    def forward(self, x):
        out = [x]

        x1 = self.branch1a(x)
        x2 = self.branch2a(x)

        for (b1, b2) in zip(self.branch1, self.branch2):
            x1 = b1(x1)
            x2 = b2(x2)
            out.extend([x1,x2])

        return torch.cat(out, 1)

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
            drop_rate=0.2,
            deploy: bool = False):

        super().__init__()


        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (isinstance(block_setting, Sequence) or
                  all([isinstance(s, BlockConfig) for s in block_setting])):
            raise TypeError("The block_setting should be List[BlockConfig]")

        self.num_classes = num_classes
        self.deploy = deploy

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
            block = _DenseBlock(in_channels=in_channels, config=config, deploy=deploy)
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
    model = peleenas1s2()

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

