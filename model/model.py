import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from model.senet import se_resnext50_32x4d

def conv3x3(in_, out, bias=True):
    return nn.Conv2d(in_, out, 3, padding=1, bias=bias)

def conv7x7(in_, out, bias=True):
    return nn.Conv2d(in_, out, 7, padding=3, bias=bias)

def conv5x5(in_, out, bias=True):
    return nn.Conv2d(in_, out, 5, padding=2, bias=bias)

def conv1x1(in_, out, bias=True):
    return nn.Conv2d(in_, out, 1, padding=0, bias=bias)

class ConvRelu(nn.Module):
    def __init__(self, in_, out, kernel_size, norm_type = None):
        super(ConvRelu,self).__init__()

        is_bias = True
        self.norm_type = norm_type
        if norm_type == 'batch_norm':
            self.norm = nn.BatchNorm2d(out)
            is_bias = False

        elif norm_type == 'instance_norm':
            self.norm = nn.InstanceNorm2d(out)
            is_bias = True

        if kernel_size == 3:
            self.conv = conv3x3(in_, out, is_bias)
        elif kernel_size == 7:
            self.conv = conv7x7(in_, out, is_bias)
        elif kernel_size == 5:
            self.conv = conv5x5(in_, out, is_bias)
        elif kernel_size == 1:
            self.conv = conv1x1(in_, out, is_bias)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.norm_type is not None:
            x = self.conv(x)
            x = self.norm(x)
            x = self.activation(x)
        else:
            x = self.conv(x)
            x = self.activation(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, int(channel/reduction), bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(int(channel/reduction), channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SCSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SCSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel//reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel//reduction), channel),
                                                nn.Sigmoid())

        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = self.channel_excitation(chn_se).view(bahs, chs, 1, 1)
        chn_se = torch.mul(x, chn_se)

        spa_se = self.spatial_se(x)
        spa_se = torch.mul(x, spa_se)
        return torch.add(chn_se, 1, spa_se)

class Decoder(nn.Module):
    def __init__(self,in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, channels, kernel_size=3,padding=1),
                                   nn.BatchNorm2d(channels),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(channels, out_channels, kernel_size=3,padding=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True))
        self.SCSE = SCSEBlock(out_channels)

    def forward(self, x, e = None):
        x = F.upsample(x, scale_factor=2, mode='bilinear')
        if e is not None:
            x = torch.cat([x,e],1)
            x = F.dropout2d(x, p = 0.50)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.SCSE(x)
        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, outplanes):
        super(Bottleneck, self).__init__()
        planes = inplanes // 4

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)

        self.relu = nn.ReLU(inplace=True)

        self.is_skip = True
        if inplanes != outplanes:
            self.is_skip = False


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.is_skip:
            out += residual
        out = self.relu(out)

        return out

class Decoder_bottleneck(nn.Module):
    def __init__(self,in_channels, channels, out_channels):
        super(Decoder_bottleneck, self).__init__()

        self.block1 = Bottleneck(in_channels, channels)
        self.block2 = Bottleneck(channels, out_channels)
        self.SCSE = SCSEBlock(out_channels)

    def forward(self, x, e = None):
        x = F.upsample(x, scale_factor=2, mode='bilinear')
        if e is not None:
            x = torch.cat([x,e],1)

        x = self.block1(x)
        x = self.block2(x)
        x = self.SCSE(x)
        return x



class PhaseFIT(nn.Module):
    def __init__(self, num_classes=1,size=256):
        super(PhaseFIT, self).__init__()

        self.num_classes = num_classes
        self.encoder = se_resnext50_32x4d()
        self.size=size

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder.layer0.conv1,
                                   self.encoder.layer0.bn1,
                                   self.encoder.layer0.relu1)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center_global_pool = nn.AdaptiveAvgPool2d([1,1])
        self.center_conv1x1 = nn.Conv2d(512*4, 64, kernel_size=1)


        self.center = nn.Sequential(nn.Conv2d(512*4, 512, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2,stride=2))

        self.decoder5 = Decoder(256 + 512*4, 512, 64)
        self.decoder4 = Decoder(64 + 256*4, 256, 64)
        self.decoder3 = Decoder(64 + 128*4, 128, 64)
        self.decoder2 = Decoder(64 + 64*4, 64, 64)
        self.decoder1 = Decoder(64, 32, 64)


        self.logits_final = nn.Sequential(nn.Conv2d(320+64, 64, kernel_size=3, padding=1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(64, 1, kernel_size=1, padding=0))

    def forward(self, x):
        conv1 = self.conv1(x)     #1/2
        conv2 = self.conv2(conv1) #1/2

        conv3 = self.conv3(conv2) #1/4
        conv4 = self.conv4(conv3) #1/8
        conv5 = self.conv5(conv4) #1/16

        center_512 = self.center_global_pool(conv5)
        center_64 = self.center_conv1x1(center_512)


        f = self.center(conv5)
        d5 = self.decoder5(f, conv5)
        d4 = self.decoder4(d5, conv4)
        d3 = self.decoder3(d4, conv3)
        d2 = self.decoder2(d3, conv2)
        d1 = self.decoder1(d2)

        hypercol = torch.cat((
            d1,
            F.upsample(d2, scale_factor=2,mode='bilinear'),
            F.upsample(d3, scale_factor=4, mode='bilinear'),
            F.upsample(d4, scale_factor=8, mode='bilinear'),
            F.upsample(d5, scale_factor=16, mode='bilinear')),1)
        hypercol = F.dropout2d(hypercol, p = 0.50)

        hypercol_add_center = torch.cat((
            hypercol,
            F.upsample(center_64, scale_factor=self.size,mode='bilinear')),1)

        x_final = self.logits_final( hypercol_add_center)
        return x_final

