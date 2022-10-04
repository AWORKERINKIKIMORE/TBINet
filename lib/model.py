import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.mobilenetv3 import MobileNetV3Encoder_large, MobileNetV3Encoder_small, initialize_weights


def interpolate(x, size):
    return F.interpolate(x, size=size, mode='bilinear', align_corners=True)


def channel_shuffle(x, groups=2):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class Conv1x1(nn.Module):
    def __init__(self, in_channel, out_channel, relu=True):
        super(Conv1x1, self).__init__()
        conv = [
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel)
        ]
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class DWConv(nn.Module):
    def __init__(self, in_channel, k=3, d=1, relu=False):
        super(DWConv, self).__init__()
        conv = [
            nn.Conv2d(in_channel, in_channel, k, 1, (k//2)*d, d, in_channel, False),
            nn.BatchNorm2d(in_channel)
        ]
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, d=1, relu_dw=False, relu_p=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
            DWConv(in_channel, 3, d, relu_dw),
            Conv1x1(in_channel, out_channel, relu_p)
        )

    def forward(self, x):
        return self.conv(x)


class DSConv5x5(nn.Module):
    def __init__(self, in_channel, out_channel, d=1, relu_dw=False, relu_p=True):
        super(DSConv5x5, self).__init__()
        self.conv = nn.Sequential(
            DWConv(in_channel, 5, d, relu_dw),
            Conv1x1(in_channel, out_channel, relu_p)
        )

    def forward(self, x):
        return self.conv(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=2):
        super(ChannelAttention, self).__init__()
        squeeze_channels = max(in_planes // ratio, 16)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, squeeze_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze_channels, in_planes, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.fc(F.adaptive_avg_pool2d(x, 1)) * x
        return out


class MLP(nn.Module):
    def __init__(self, in_channel, out_channel, ratio=2):
        super(MLP, self).__init__()
        squeeze_channels = max(in_channel // ratio, 16)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channel, squeeze_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze_channels, out_channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.fc(F.adaptive_avg_pool2d(x, 1))
        return out


class SpatialAttention(nn.Module):
    def __init__(self, channel):
        super(SpatialAttention, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(channel, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class SalHead(nn.Module):
    def __init__(self, in_channel):
        super(SalHead, self).__init__()
        self.conv = nn.Conv2d(in_channel, 1, 1)

    def forward(self, x):
        x = self.conv(x)
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)


class BAG(nn.Module):
    def __init__(self, channel):
        super(BAG, self).__init__()
        self.r_sa = SpatialAttention(channel)
        self.d_sa = SpatialAttention(channel)

    def forward(self, rgb, dep):
        r = rgb * self.r_sa(dep) + rgb
        d = dep * self.d_sa(rgb) + dep

        return r, d


class BFS(nn.Module):
    def __init__(self, channel):
        super(BFS, self).__init__()
        self.conv1 = AFFM(channel, channel, channel)
        self.r_sa = SpatialAttention(channel)
        self.d_sa = SpatialAttention(channel)

    def forward(self, rgb, dep):
        rd = self.conv1(rgb, dep)

        r = rd * self.r_sa(rgb) + rgb
        d = rd * self.d_sa(dep) + dep

        return r, d


class AFFM(nn.Module):
    def __init__(self, in_channel1, in_channel2, out_channel):
        super(AFFM, self).__init__()
        self.redu_r = Conv1x1(in_channel1, out_channel)
        self.redu_d = Conv1x1(in_channel2, out_channel)
        self.conv_cat = DSConv3x3(out_channel * 2, out_channel)

    def forward(self, rgb, dep):
        rgb = self.redu_r(rgb)
        dep = self.redu_d(dep)
        mul = rgb * dep
        cat = self.conv_cat(torch.cat([rgb + mul, dep + mul], 1))
        return cat


class AFFM_CA(nn.Module):
    def __init__(self, in_channel1, in_channel2, out_channel):
        super(AFFM_CA, self).__init__()
        self.redu_r = nn.Sequential(
            Conv1x1(in_channel1, out_channel),
            ChannelAttention(out_channel)
        )
        self.redu_d = nn.Sequential(
            Conv1x1(in_channel2, out_channel),
            ChannelAttention(out_channel)
        )
        self.conv_cat = DSConv3x3(out_channel * 2, out_channel)

    def forward(self, rgb, dep):
        rgb = self.redu_r(rgb)
        dep = self.redu_d(dep)
        mul = rgb * dep
        cat = self.conv_cat(torch.cat([rgb + mul, dep + mul], 1))
        return cat


class GCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
        b_in_channel = in_channel // 2
        b_out_channel = out_channel // 2
        self.b1 = Conv1x1(b_in_channel, b_out_channel)
        self.b21 = DSConv3x3(b_in_channel, b_out_channel, d=3)
        self.b22 = DSConv3x3(b_in_channel, b_out_channel, d=5)
        self.b23 = DSConv3x3(b_in_channel, b_out_channel, d=7)
        self.b2 = DSConv3x3(b_out_channel*3, b_out_channel)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.b1(x1)
        x21 = self.b21(x2)
        x22 = self.b22(x2)
        x23 = self.b23(x2)
        x2 = self.b2(torch.cat([x21, x22, x23], 1))
        x = torch.cat([x1, x2], 1)
        x = channel_shuffle(x, 2)

        return x


class ChannelFuse(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ChannelFuse, self).__init__()
        self.ca = ChannelAttention(in_channel)
        self.conv = Conv1x1(in_channel, out_channel)

    def forward(self, x, x_):
        x = torch.cat([x, interpolate(x_, x.shape[2:])], dim=1)
        x = self.ca(x)
        x = self.conv(x)
        return x


class decoder(nn.Module):
    def __init__(self, c1=16, c2=24, c3=40, c4=80, c5=112, c6=160):
        super(decoder, self).__init__()
        self.channelfuse_r_h = AFFM_CA(c5, c6, 160)
        self.channelfuse_d_h = AFFM_CA(c5, c6, 160)
        self.channelfuse_r_m = AFFM_CA(c3, c4, 80)
        self.channelfuse_d_m = AFFM_CA(c3, c4, 80)
        self.channelfuse_r_l = AFFM_CA(c1, c2, 40)
        self.channelfuse_d_l = AFFM_CA(c1, c2, 40)

        self.fuse_ori_h = AFFM(160, 160, 160)
        self.gcm_h_r = GCM(160, 80)
        self.gcm_h_d = GCM(160, 80)
        self.gcm_h_f = GCM(160, 80)
        self.fuse_h = AFFM(80, 80, 80)
        self.out_h = Conv1x1(160, 80)

        self.fuse_ori_m = AFFM(80, 80, 80)
        self.gcm_m_r = GCM(160, 40)
        self.gcm_m_d = GCM(160, 40)
        self.gcm_m_f = GCM(160, 40)
        self.fuse_m = AFFM(40, 40, 40)
        self.out_m = Conv1x1(80, 40)

        self.fuse_ori_l = AFFM(40, 40, 40)
        self.gcm_l_r = GCM(80, 20)
        self.gcm_l_d = GCM(80, 20)
        self.gcm_l_f = GCM(80, 20)
        self.fuse_l = AFFM(20, 20, 20)
        self.out_l = Conv1x1(40, 20)

        self.s_r = SalHead(20)
        self.s_d = SalHead(20)
        self.s_f = SalHead(20)

    def forward(self, r1, r2, r3, r4, r5, r6, d1, d2, d3, d4, d5, d6):
        rh = self.channelfuse_r_h(r5, interpolate(r6, r5.shape[2:]))
        dh = self.channelfuse_d_h(d5, interpolate(d6, d5.shape[2:]))
        rm = self.channelfuse_r_m(r3, interpolate(r4, r3.shape[2:]))
        dm = self.channelfuse_d_m(d3, interpolate(d4, d3.shape[2:]))
        rl = self.channelfuse_r_l(r1, interpolate(r2, r1.shape[2:]))
        dl = self.channelfuse_d_l(d1, interpolate(d2, d1.shape[2:]))

        fh = self.fuse_ori_h(rh, dh)  # fuse stpe1
        rh = self.gcm_h_r(rh)
        dh = self.gcm_h_d(dh)
        fh = self.gcm_h_f(fh)
        fh_ = self.fuse_h(rh, dh)  #fuse stpe2
        fh = self.out_h(torch.cat([fh, fh_], 1))

        fh = interpolate(fh, rm.shape[2:])
        rh = interpolate(rh, rm.shape[2:])
        dh = interpolate(dh, rm.shape[2:])

        # m

        fm = self.fuse_ori_m(rm, dm)  # fuse stpe1

        rm = channel_shuffle(torch.cat([rm, rh], 1), 2)
        rm = self.gcm_m_r(rm)

        dm = channel_shuffle(torch.cat([dm, dh], 1), 2)
        dm = self.gcm_m_d(dm)

        fm = channel_shuffle(torch.cat([fm, fh], 1), 2)
        fm = self.gcm_m_f(fm)

        fm_ = self.fuse_m(rm, dm)  # fuse stpe2
        fm = self.out_m(torch.cat([fm, fm_], 1))

        fm = interpolate(fm, rl.shape[2:])
        rm = interpolate(rm, rl.shape[2:])
        dm = interpolate(dm, rl.shape[2:])

        # l

        fl = self.fuse_ori_l(rl, dl)  # fuse stpe1

        rl = channel_shuffle(torch.cat([rl, rm], 1), 2)
        rl = self.gcm_l_r(rl)

        dl = channel_shuffle(torch.cat([dl, dm], 1), 2)
        dl = self.gcm_l_d(dl)

        fl = channel_shuffle(torch.cat([fl, fm], 1), 2)
        fl = self.gcm_l_f(fl)

        fl_ = self.fuse_l(rl, dl)  # 后置融合
        fl = self.out_l(torch.cat([fl, fl_], 1))

        s_r = self.s_r(rl)
        s_d = self.s_d(dl)
        s_f = self.s_f(fl)

        return s_f, s_r, s_d


class Net(nn.Module):
    def __init__(self, c1=16, c2=24, c3=40, c4=80, c5=112, c6=160):
        super(Net, self).__init__()

        # Backbone model
        self.layer_rgb = MobileNetV3Encoder_large()
        self.layer_dep = MobileNetV3Encoder_small()
        initialize_weights(self.layer_rgb)
        initialize_weights(self.layer_dep)
        self.bd1 = BAG(c1)
        self.bd2 = BAG(c2)
        self.bd3 = BFS(c3)
        self.bd4 = BFS(c4)

        self.decoder = decoder()

    def forward(self, rgbs, depths):

        r_1 = self.layer_rgb.layer1(rgbs)
        d_1 = self.layer_dep.layer1(torch.cat((depths, depths, depths), 1))
        r_1, d_1 = self.bd1(r_1, d_1)

        r_2 = self.layer_rgb.layer2(r_1)
        d_2 = self.layer_dep.layer2(d_1)
        r_2, d_2 = self.bd2(r_2, d_2)

        r_3 = self.layer_rgb.layer3(r_2)
        d_3 = self.layer_dep.layer3(d_2)
        r_3, d_3 = self.bd3(r_3, d_3)

        r_4 = self.layer_rgb.layer4(r_3)
        d_4 = self.layer_dep.layer4(d_3)
        r_4, d_4 = self.bd4(r_4, d_4)

        pack = torch.cat([r_4, d_4], 0)

        pack = self.layer_rgb.layer5(pack)
        r_5, d_5 = pack.chunk(2, dim=0)

        pack = self.layer_rgb.layer6(pack)
        r_6, d_6 = pack.chunk(2, dim=0)

        s = self.decoder(r_1, r_2, r_3, r_4, r_5, r_6, d_1, d_2, d_3, d_4, d_5, d_6)
        return s
