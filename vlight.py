import math
import torch
from torch.nn import functional as F
from torch import nn as nn

from csl_common.utils.nn import count_parameters, to_numpy


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def norm2d(type):
    if type == 'batch':
        return nn.BatchNorm2d
    elif type == 'instance':
        return nn.InstanceNorm2d
    elif type == 'none':
        return nn.Identity
    else:
        raise ValueError("Invalid normalization type: ", type)


class depthwise_conv(nn.Module):
    def __init__(self, nin, nout, stride=1):
        super(depthwise_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, stride=stride, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class DWBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, layer_normalization='batch', final_relu=True):
        super().__init__()
        self.conv1 = depthwise_conv(inplanes, planes, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = depthwise_conv(planes, planes)
        self.layer_norm = layer_normalization
        self.bn1 = norm2d(layer_normalization)(planes)
        self.bn2 = norm2d(layer_normalization)(planes)
        self.downsample = downsample
        self.stride = stride
        self.final_relu = final_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.final_relu:
            out = self.relu(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, layer_normalization='batch', final_relu=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.layer_norm = layer_normalization
        self.bn1 = norm2d(layer_normalization)(planes)
        self.bn2 = norm2d(layer_normalization)(planes)
        self.downsample = downsample
        self.stride = stride
        self.final_relu = final_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.final_relu:
            out = self.relu(out)
        return out


class Encoder(nn.Module):
    def __init__(self, channels, block=BasicBlock, num_blocks=2, input_channels=3, layer_normalization='batch',
                 depthwise=False):
        super().__init__()
        if depthwise:
            block = DWBasicBlock

        self.inplanes = channels[0]
        self.conv1 = nn.Conv2d(input_channels, channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.layer_norm = layer_normalization
        self.bn1 = norm2d(self.layer_norm)(channels[0])
        self.bn2 = norm2d(self.layer_norm)(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, channels[1], num_blocks, stride=2)
        self.layer2 = self._make_layer(block, channels[2], num_blocks, stride=2)
        if len(channels) > 3:
            self.layer3 = self._make_layer(block, channels[3], num_blocks, stride=1)
        # if len(channels) > 4:
        #     self.layer4 = self._make_layer(block, channels[3], num_blocks, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, final_relu=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm2d(self.layer_norm)(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, layer_normalization=self.layer_norm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, layer_normalization=self.layer_norm, final_relu=final_relu))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        if hasattr(self, 'layer3'):
           x = self.layer3(x)

        return x


class InvBasicBlockBilinear(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, layer_normalization='batch',
                 with_spectral_norm=False):
        super().__init__()
        self.layer_normalization = layer_normalization
        if upsample is not None:
            self.conv1 = torch.nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                conv1x1(inplanes, planes)
            )
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm2d(layer_normalization)(planes)
        self.bn2 = norm2d(layer_normalization)(planes)
        self.in1 = norm2d(layer_normalization)(planes)
        self.in2 = norm2d(layer_normalization)(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.upsample = upsample
        self.stride = stride
        if with_spectral_norm and upsample is None:
            self.conv1 = torch.nn.utils.spectral_norm(self.conv1)
            self.conv2 = torch.nn.utils.spectral_norm(self.conv2)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.layer_normalization == 'batch':
            out = self.bn1(out)
        elif self.layer_normalization == 'instance':
            out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.layer_normalization == 'batch':
            out = self.bn2(out)
        elif self.layer_normalization == 'instance':
            out = self.in2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        return self.relu(out)


class InvBasicBlockPxsh(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, layer_normalization='batch',
                 with_spectral_norm=False, depthwise=False):
        super().__init__()
        self.layer_normalization = layer_normalization
        if upsample is not None:
            # self.conv1 = deconv4x4(inplanes, planes, stride)
            self.conv1 = torch.nn.Sequential(
                torch.nn.PixelShuffle(2),
                conv1x1(inplanes//4, planes)
            )
        else:
            if depthwise:
                self.conv1 = depthwise_conv(inplanes, planes, stride=stride)
            else:
                self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm2d(layer_normalization)(planes)
        self.bn2 = norm2d(layer_normalization)(planes)
        self.in1 = norm2d(layer_normalization)(planes)
        self.in2 = norm2d(layer_normalization)(planes)
        self.relu = nn.ReLU(inplace=True)

        if depthwise:
            self.conv2 = depthwise_conv(planes, planes)
        else:
            self.conv2 = conv3x3(planes, planes)
        self.upsample = upsample
        self.stride = stride
        if with_spectral_norm and upsample is None:
            self.conv1 = torch.nn.utils.spectral_norm(self.conv1)
            self.conv2 = torch.nn.utils.spectral_norm(self.conv2)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.layer_normalization == 'batch':
            out = self.bn1(out)
        elif self.layer_normalization == 'instance':
            out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.layer_normalization == 'batch':
            out = self.bn2(out)
        elif self.layer_normalization == 'instance':
            out = self.in2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        return self.relu(out)


class Decoder(nn.Module):
    def __init__(self, channels, num_blocks=1, output_channels=3, pixel_shuffle=False, bilinear=False, depthwise=False,
                 extra_up=False, spectral_norm=False, layer_normalization='batch'):
        super().__init__()
        self.pixel_shuffle = pixel_shuffle
        self.bilinear = bilinear
        self.depthwise = depthwise
        self.extra_up = extra_up
        if pixel_shuffle:
            block = InvBasicBlockPxsh
        else:
            block = invresnet.InvBasicBlock
        self.layer_normalization = layer_normalization
        self.with_spectral_norm = spectral_norm
        if self.with_spectral_norm:
            self.sn = torch.nn.utils.spectral_norm
        else:
            self.sn = lambda x: x

        self.inplanes = channels[0]
        self.output_channels = output_channels

        self.norm = norm2d(layer_normalization)
        self.bn1 = self.norm(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, channels[1], num_blocks, stride=1)
        self.layer2 = self._make_layer(block, channels[2], num_blocks, stride=2)
        if len(channels) > 3:
            self.layer3 = self._make_layer(block, channels[3], num_blocks, stride=2)

        if pixel_shuffle:
            self.up_conv = nn.Sequential(
                nn.PixelShuffle(2),
                conv1x1(channels[-1]//4, channels[-1])
            )
            self.conv_out = nn.Sequential(
                nn.PixelShuffle(2),
                conv3x3(channels[-1]//4, output_channels)
            )
        elif self.bilinear:
            self.conv_out = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                conv3x3(64, output_channels)
            )
        else:
            self.up_conv = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False)
            self.conv_out = nn.ConvTranspose2d(channels[-1], output_channels, kernel_size=4, stride=2, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        upsample = None

        if self.pixel_shuffle:
            up = torch.nn.Sequential(
                torch.nn.PixelShuffle(2),
                conv1x1(self.inplanes // 4, planes)
            )
        elif self.bilinear:
            up = torch.nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                conv1x1(self.inplanes, planes)
            )
        else:
            up = nn.ConvTranspose2d(self.inplanes, planes * block.expansion,
                                    kernel_size=4, stride=stride, padding=1, bias=False)
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                self.sn(up),
                self.norm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample,
                            layer_normalization=self.layer_normalization,
                            with_spectral_norm=self.with_spectral_norm,
                            depthwise=self.depthwise
                            ))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, depthwise=self.depthwise))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        if hasattr(self, 'layer3'):
            x = self.layer3(x)

        if self.extra_up:
            x = self.up_conv(x)

        x = self.conv_out(x)

        if self.output_channels == 1:
            x = torch.sigmoid(x)
        return x


class VLight(nn.Module):
    def __init__(self, input_channels=3, output_channels=1, **kwargs):
        super().__init__()

        channels_down = [64, 256, 256, 256]
        channels_up = [256, 256, 256, 64]

        self.Q = Encoder(channels_down, num_blocks=1,input_channels=input_channels, depthwise=True)
        self.P = Decoder(channels_up, num_blocks=1, output_channels=output_channels, bilinear=False,
                         pixel_shuffle=True, depthwise=True, extra_up=True)

        print("Trainable params Q: {:,}".format(count_parameters(self.Q)))
        print("Trainable params P: {:,}".format(count_parameters(self.P)))

        self.total_iter = 0
        self.iter = 0
        self.z = None

    def forward(self, X):
        self.z = self.Q(X)
        outputs = self.P(self.z)
        return outputs


def load_net(model):
    from csl_common.utils import nn
    net = VLight(3,1)
    print("Loading model {}...".format(model))
    nn.read_model(model, 'saae', net)
    return net

