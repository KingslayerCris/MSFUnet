import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.vgg import VGG16
from timm.models.layers import trunc_normal_, DropPath
# -------------------------------------改进unet------------------------------------
# convnextV2
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
class Block(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
# 左侧上采样
class up_conv1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv1, self).__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1),
            nn.BatchNorm2d(ch_out),
            Block(ch_out)
        )
    def forward(self, x):
        x1 = self.up(x)
        x2 = self.conv1(x1)
        return x2

class up_conv2(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv2, self).__init__()
        self.up = nn.Upsample(scale_factor=4)
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )
    def forward(self, x):
        x1 = self.up(x)
        x2 = self.conv1(x1)
        return x2
class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class Cascade_Unet(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, backbone='vgg'):
        super(Cascade_Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)
            in_filters = [192, 384, 768, 1024]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]
        # 注意力

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        self.up1 = up_conv1(out_filters[3],out_filters[3])
        self.up1_1 = up_conv2(out_filters[3],out_filters[2])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        self.up2 = up_conv1(out_filters[3], out_filters[2])
        self.up2_2 = up_conv2(out_filters[3], out_filters[1])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        self.up3 = up_conv1(out_filters[2], out_filters[1])
        self.up3_3 = up_conv2(out_filters[2], out_filters[0])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])
        self.up4 = up_conv1(out_filters[1], out_filters[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)

        up4 = self.up_concat4(feat4, feat5)
        up4_1 = self.up1(feat5)
        up4_1_1 = self.up1_1(feat5)
        up4_2 = up4_1+up4

        up3 = self.up_concat3(feat3, up4_2)
        up3_1 = self.up2(up4_2)
        up3_1_1 = self.up2_2(up4_2)
        up3_2 = up3_1+up3+up4_1_1

        up2 = self.up_concat2(feat2, up3_2)
        up2_1 = self.up3(up3_2)
        up2_1_1 = self.up3_3(up3_2)
        up2_2 = up2_1+up2+up3_1_1

        up1 = self.up_concat1(feat1, up2_2)
        up1_1 = self.up4(up2_2)
        up1 = up1+up1_1+up2_1_1

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)

        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
