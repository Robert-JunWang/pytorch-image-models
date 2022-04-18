import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .hub import load_state_dict_from_url
from collections import OrderedDict
from typing import Type, Any, Callable, Union, List, Optional

from .registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


__all__ = ['PeleeNet', 'peleenet1x', 'peleenet2x', 'peleenet1x_se', 'peleenet2x_se']

# model_urls = {
#     'peleenet1x': 'https://github.com/edge-cv/benchmark/releases/download/pretrained/peleenet1x.pth',
#     'peleenet2x': 'https://github.com/edge-cv/benchmark/releases/download/pretrained/peleenet2x.pth'
# }
# # model_urls = {
# #     'peleenet1x': 'pretrained/peleenet1x.pth',
# #     'peleenet2x': 'pretrained/peleenet2x.pth'
# # }


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (1, 1),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'features.stemblock.stem1', 'classifier': 'classifier',
        **kwargs
    }


default_cfgs = {
    'peleenet1x': _cfg(url='https://github.com/edge-cv/benchmark/releases/download/pretrained/peleenet1x.pth'),
    'peleenet1x_se': _cfg(url='https://github.com/edge-cv/benchmark/releases/download/pretrained/peleenet1x.pth'),
    'peleenet2x': _cfg(url='https://github.com/edge-cv/benchmark/releases/download/pretrained/peleenet2x.pth'),
    'peleenet2x_se': _cfg(url='https://github.com/edge-cv/benchmark/releases/download/pretrained/peleenet2x.pth')
}


@register_model
def peleenet1x(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    model_args = dict(**kwargs)
    return _peleenet('peleenet1x', pretrained, progress,
        block_config=[3, 4, 6, 4], 
        bottleneck_width=[2,4,4,4], 
        **kwargs)


@register_model
def peleenet1x_se(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    return _peleenet('peleenet1x_se', pretrained, progress,
        block_config=[3, 4, 6, 4], 
        bottleneck_width=[2,4,4,4], 
        use_se=True,
        **kwargs)

@register_model
def peleenet2x(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    return _peleenet('peleenet2x', pretrained, progress,
        block_config = [3,8,12,8], 
        growth_rate=[32,48,64,64], 
        bottleneck_width=[4,4,4,4], 
        out_channels=(128,256,512,1024), **kwargs)

@register_model
def peleenet2x_se(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    return _peleenet('peleenet2x', pretrained, progress,
        block_config = [3,8,12,8], 
        growth_rate=[32,48,64,64], 
        bottleneck_width=[4,4,4,4], 
        out_channels=(128,256,512,1024), 
        use_se=True,
        **kwargs)

class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bottleneck_width, use_se):
        super(_DenseLayer, self).__init__()


        growth_rate = growth_rate // 2
        self.out_channels = in_channels + 3 * growth_rate 
        inter_channels = growth_rate  * bottleneck_width  

        self.branch1a = BasicConv2d(in_channels, inter_channels, kernel_size=1)
        self.branch1b = BasicConv2d(inter_channels, growth_rate, kernel_size=3, padding=1)

        self.branch2a = BasicConv2d(in_channels, inter_channels, kernel_size=1)
        self.branch2b = BasicConv2d(inter_channels, growth_rate, kernel_size=3, padding=1)
        self.branch2c = BasicConv2d(growth_rate, growth_rate, kernel_size=3, padding=1)
        if use_se:
            self.se = SELayer(growth_rate*3, growth_rate)
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
    def __init__(self, num_layers, in_channels, bn_size, growth_rate, use_se):
        super(_DenseBlock, self).__init__()
        self.out_channels = in_channels
        for i in range(num_layers):
            layer = _DenseLayer(self.out_channels, growth_rate, bn_size, use_se)
            self.add_module('denselayer%d' % (i + 1), layer)
            self.out_channels = layer.out_channels

class _StemBlock(nn.Module):
    def __init__(self, num_input_channels, num_init_features, kernel_size=5, stride=3):
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

class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
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

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, activation=True, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels) 
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation:
            return F.relu(x, inplace=True)
        else:
            return x

class PeleeNet(nn.Module):
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
    def __init__(self, growth_rate=[32, 32, 64, 64], 
                block_config=[3, 4, 6, 4], 
                stem_block=(32, 3, 2), 
                bottleneck_width=[1, 2, 4, 4], 
                drop_rate=0.05, 
                in_channels=3,
                num_classes=1000, 
                out_channels=(128,256,512,896),
                use_se=False):

        super(PeleeNet, self).__init__()

        self.num_classes = num_classes

        num_init_features, first_kernel_size, first_stride = stem_block
        self.features = nn.Sequential(OrderedDict([
                ('stemblock', _StemBlock(
                    in_channels, num_init_features,
                    first_kernel_size, first_stride)), 
            ]))     

        if type(growth_rate) is not list:
            growth_rate = [growth_rate] * 4

        if type(bottleneck_width) is not list:
            bottleneck_width = [bottleneck_width] * 4


        assert len(growth_rate) == len(bottleneck_width) and len(out_channels) == len(block_config), 'The length of the growth rate and the bottleneck width must be the same'

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, in_channels=num_features,
                                bn_size=bottleneck_width[i], growth_rate=growth_rate[i], use_se=use_se)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = out_channels[i]

            self.features.add_module('transition%d' % (i + 1), BasicConv2d(block.out_channels, num_features, kernel_size=1))
            if i != len(block_config) - 1:
                self.features.add_module('pool%d' % (i + 1), nn.AvgPool2d(kernel_size=2, stride=2))

        self.drop_rate = drop_rate
    
        # Linear layer
        if num_classes is not None:
            self.classifier = nn.Linear(num_features, num_classes)
        else:
            self.classifier = None

        self._initialize_weights()

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

def _peleenet(arch: str, pretrained: bool = False, progress: bool = True, **kwargs: Any):
    model = PeleeNet(**kwargs)

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
    model = peleenet1x(num_classes=120)

    print(model)

    def print_size(self, input, output):
        print(torch.typename(self).split('.')[-1], ' output size:',output.data.size())

    for layer in model.features:
        layer.register_forward_hook(print_size)

    output = model.forward(input_var)

