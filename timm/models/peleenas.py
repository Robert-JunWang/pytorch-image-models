
from functools import partial
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence
import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.mobilenetv2 import _make_divisible



from .registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


__all__ = ['PeleeNas', 'peleenas1x', 'peleenas2x']

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (1, 1),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'features.stemblock.stem1', 'classifier': 'classifier.3',
        **kwargs
    }

default_cfgs = {
    'peleenas1x': _cfg(url='https://github.com/edge-cv/benchmark/releases/download/pretrained/peleenet1x.pth'),
    'peleenas2x': _cfg(url='https://github.com/edge-cv/benchmark/releases/download/pretrained/peleenet2x.pth')
}

@register_model
def peleenas1x(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    model_args = dict(**kwargs)

    block_setting = [
        BlockConfig(3, 32, 2, 128, activation='relu', use_se=False),
        BlockConfig(4, 32, 4, 256, activation='relu', use_se=True),
        BlockConfig(6, 64, 4, 512, activation='hs', use_se=False),
        BlockConfig(4, 64, 4, 896, activation='hs', use_se=True, stride=1),
    ]
    return _peleenas('peleenas1x', pretrained, progress,
                     block_setting=block_setting,
                     last_channel=1024,
                     **kwargs)

@register_model
def peleenas2x(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    model_args = dict(**kwargs)

    block_setting = [
        BlockConfig(3, 32, 4, 128, activation='relu', use_se=False),
        BlockConfig(8, 48, 4, 256, activation='relu', use_se=True),
        BlockConfig(12, 64, 4, 512, activation='hs', use_se=False),
        BlockConfig(8, 64, 4, 1024, activation='hs', use_se=True, stride=1),
    ]
    return _peleenas('peleenas2x', pretrained, progress,
                     block_setting=block_setting,
                     last_channel=1024,
                     **kwargs)

class SqueezeExcitation(nn.Module):
    # Implemented as described at Figure 4 of the MobileNetV3 paper
    def __init__(self, input_channels: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)

    def _scale(self, input: Tensor, inplace: bool) -> Tensor:
        scale = F.adaptive_avg_pool2d(input, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        return F.hardsigmoid(scale, inplace=inplace)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input, True)
        return scale * input


class BlockConfig:
    def __init__(self,
                 num_layers: int, growth_rate: int, bottleneck_width: int, out_channels: int,
                 activation: str = 'relu',
                 stride: int = 2,
                 use_se: bool = False,
                 width_mult: float = 1.0,
                 drop_rate: float = 0.05):
        self.num_layers = num_layers
        self.growth_rate = self.adjust_channels(growth_rate, width_mult)
        self.bottleneck_width = bottleneck_width
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.activation = activation
        self.drop_rate = drop_rate
        self.stride = stride

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class BasicConv2d(nn.Sequential):

    def __init__(self, in_channels, out_channels, activation='relu', **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels) 
        if activation == 'hs':
            self.activation = nn.Hardswish()
        else:
            self.activation = nn.ReLU()


class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bottleneck_width, activation, use_se):
        super(_DenseLayer, self).__init__()


        growth_rate = growth_rate // 2
        self.out_channels = in_channels + 3 * growth_rate 
        inter_channels = growth_rate  * bottleneck_width  

        self.branch1a = BasicConv2d(in_channels, inter_channels, activation=activation, kernel_size=1)
        self.branch1b = BasicConv2d(inter_channels, growth_rate, activation=activation, kernel_size=3, padding=1)

        self.branch2a = BasicConv2d(in_channels, inter_channels, activation=activation, kernel_size=1)
        self.branch2b = BasicConv2d(inter_channels, growth_rate, activation=activation, kernel_size=3, padding=1)
        self.branch2c = BasicConv2d(growth_rate, growth_rate, activation=activation, kernel_size=3, padding=1)
        if use_se:
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
            layer = _DenseLayer(self.out_channels, config.growth_rate, config.bottleneck_width, config.activation, config.use_se)
            self.add_module('denselayer%d' % (i + 1), layer)
            self.out_channels = layer.out_channels

        self.add_module('transition', BasicConv2d(self.out_channels, config.out_channels, activation=config.activation, kernel_size=1))
        self.out_channels = config.out_channels

class _StemBlock(nn.Module):
    def __init__(self, num_input_channels, num_init_features, kernel_size=3, stride=2):
        super(_StemBlock, self).__init__()

        padding = kernel_size // 2

        self.stem1 = BasicConv2d(num_input_channels, num_init_features, kernel_size=kernel_size, stride=stride, padding=padding)
        self.stem2a = BasicConv2d(num_init_features, num_init_features, kernel_size=3, stride=2, padding=1)
        self.stem2b = BasicConv2d(num_init_features, num_init_features, kernel_size=3, stride=1, padding=1)
        self.stem3 = BasicConv2d(3*num_init_features, num_init_features, kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        out = self.stem1(x)

        branch1 = self.pool(out)
        branch2a = self.stem2a(out)
        branch2b = self.stem2b(branch2a)

        out = torch.cat([branch1, branch2a, branch2b], 1)
        out = self.stem3(out)

        return out




class PeleeNas(nn.Module):
    r"""PeleeNet model class
    Args:
        growth_rate (list of 4 ints) - how many filters to add each layer 
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bottleneck_width (list of 4 ints) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        compression_factor (list of 4 ints) - 
        drop_rate (float) - dropout rate
        num_classes (int) - number of classification classes
    """
    def __init__(self, 
            block_setting: List[BlockConfig],
            last_channel: int,
            stem_block=(3, 32, 3, 2),
            num_classes: int = 1000,
            drop_rate=0.05):

        super().__init__()

        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (isinstance(block_setting, Sequence) or
                  all([isinstance(s, BlockConfig) for s in block_setting])):
            raise TypeError("The block_setting should be List[BlockConfig]")

        self.num_classes = num_classes

        in_channels, out_channels, first_kernel_size, first_stride = stem_block
        self.features = nn.Sequential(OrderedDict([
                ('stemblock', _StemBlock(
                    in_channels, out_channels,
                    first_kernel_size, first_stride)), 
            ]))     


        # Each denseblock
        in_channels = out_channels
        for i, config in enumerate(block_setting):
            block = _DenseBlock(in_channels=in_channels, config=config)
            self.features.add_module('denseblock%d' % (i + 1), block)
            in_channels = block.out_channels

            if config.stride == 2:
                self.features.add_module('pool%d' % (i + 1), nn.AvgPool2d(kernel_size=2, stride=2))
  
        # Linear layer
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.classifier = nn.Sequential(
        #     nn.Linear(in_channels, last_channel),
        #     nn.Hardswish(inplace=True),
        #     nn.Dropout(p=drop_rate, inplace=True),
        #     nn.Linear(last_channel, num_classes),
        # )
        self.drop_rate = drop_rate
    
        # Linear layer
        if num_classes is not None:
            self.classifier = nn.Linear(in_channels, num_classes)
        else:
            self.classifier = None

        self._initialize_weights()

    # def forward(self, x):
    #     x = self.features(x)

    #     x = self.avgpool(x)
    #     x = torch.flatten(x, 1)
    #     x = self.classifier(x)
        
    #     return x
    def forward(self, x):
        features = self.features(x)

        out = features.mean([2, 3])
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        if self.classifier is not None:
            out = self.classifier(out)
        return out

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

def _peleenas(arch: str, pretrained: bool = False, progress: bool = True, **kwargs: Any):
    model = PeleeNas(**kwargs)

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
    input_var = torch.autograd.Variable(torch.Tensor(1,3,224,224))
    model = peleenas1x(num_classes=120)

    # print(model)

    def print_size(self, input, output):
        print(torch.typename(self).split('.')[-1], ' output size:',output.data.size())

    for layer in model.features:
        layer.register_forward_hook(print_size)

    o = model.forward(input_var)

