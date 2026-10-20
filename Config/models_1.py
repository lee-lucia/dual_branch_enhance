import torch.nn as nn
import torch
import torch.nn.functional as F
from .restormer_arch import Restormer

# 浅层偏振特征提取
class ShallowPol(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.extractor1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        self.extractor2 = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.extractor1(x)
        x = self.extractor2(x)  # x: B×3×H×W
        return x
    
    
    
    
class ShallowPol_GroupShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super().__init__()
        mid_channels = 32  # 中间通道数，建议设置为 32 或 64
        self.groups = groups

        self.group_conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, groups=groups)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.group_conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, groups=groups)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def channel_shuffle(self, x):
        B, C, H, W = x.size()
        assert C % self.groups == 0
        x = x.view(B, self.groups, C // self.groups, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B, C, H, W)
        return x

    def forward(self, x):
        x = self.group_conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.channel_shuffle(x)

        x = self.group_conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.channel_shuffle(x)
        return x


# INC模块，颜色通道输入
class Inc(nn.Module):
    def __init__(self,in_channels,filters):
        super(Inc, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=(1, 1), stride=(1, 1),dilation=1,padding=(1-1) // 2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=(3, 3), stride=(1, 1),dilation=1,padding=(3-1) // 2),
            nn.LeakyReLU(),
            )   
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=(1, 1), stride=(1, 1),dilation=1,padding=(1-1) // 2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=(5, 5), stride=(1, 1),dilation=1,padding=(5-1) // 2),
            nn.LeakyReLU(),
            )  
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=(1, 1), stride=(1, 1),dilation=1),
            nn.LeakyReLU(),

        )   
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=(1, 1), stride=(1, 1),dilation=1),
            nn.LeakyReLU(),
        )    
    def forward(self,input):
        o1 = self.branch1(input)
        o2 = self.branch2(input)
        o3 = self.branch3(input)
        o4 = self.branch4(input)
        return torch.cat([o1,o2,o3,o4],dim=1)
class Flatten(nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), -1)


# CAM模块，颜色通道处理
class CAM(nn.Module):
    def __init__(self,in_channels,reduction_ratio):
        super(CAM, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            Flatten(),
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.Softsign(),
            nn.Linear(in_channels // reduction_ratio, in_channels ),
            nn.Softsign()
            )
    def forward(self,input):
        return input* self.module(input).unsqueeze(2).unsqueeze(3).expand_as(input)


# 颜色通道融合前处理
class IncCAM(nn.Module):
    def __init__(self, in_channels, filters, reduction_ratio):
        super(IncCAM, self).__init__()
        self.inc = Inc(in_channels, filters)
        self.cam = CAM(filters * 4, reduction_ratio)  # Inc 输出是 4x filters 通道

    def forward(self, x):
        x = self.inc(x)
        x = self.cam(x)
        return x
    
class GatedCrossFusion(nn.Module):
    def __init__(self, dim_q, dim_kv, out_dim):
        super().__init__()
        self.q_proj = nn.Conv2d(dim_q, out_dim, kernel_size=1)
        self.kv_proj = nn.Conv2d(dim_kv, out_dim, kernel_size=1)
        self.gate = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.fuse = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)

    def forward(self, q_feat, kv_feat):
        q_proj = self.q_proj(q_feat)
        kv_proj = self.kv_proj(kv_feat)
        concat = torch.cat([q_proj, kv_proj], dim=1)
        alpha = self.gate(concat)
        fused = alpha * q_proj + (1 - alpha) * kv_proj
        return self.fuse(fused)





class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out



# 用于偏振特征提取（输入为S1,S2或者AOLP和DOLP），灰度图
class PolarFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pol1 = ShallowPol_GroupShuffle(in_channels, out_channels)
        self.pol2 = ShallowPol_GroupShuffle(in_channels, out_channels)
        self.pol3 = ShallowPol_GroupShuffle(in_channels, out_channels)
        
        self.cbam = CBAM(gate_channels=192)
    def forward(self, x1, x2, x3):
        x1_pol = self.pol1(x1)
        x2_pol = self.pol2(x2)
        x3_pol = self.pol3(x3)
        x = torch.cat([x1_pol, x2_pol, x3_pol], dim=1)
        x = self.cbam(x)
        return x


# 输入为强度图像S0，彩图
class ColorFeatureExtractor(nn.Module):
    def __init__(self, filters, reduction_ratio):
        super().__init__()
        self.inccam_r = IncCAM(1, filters, reduction_ratio)
        self.inccam_g = IncCAM(1, filters, reduction_ratio)
        self.inccam_b = IncCAM(1, filters, reduction_ratio)

    def forward(self, x):
        r = torch.unsqueeze(x[:, 0, :, :], dim=1)
        g = torch.unsqueeze(x[:, 1, :, :], dim=1)
        b = torch.unsqueeze(x[:, 2, :, :], dim=1)

        r_feat = self.inccam_r(r)
        g_feat = self.inccam_g(g)
        b_feat = self.inccam_b(b)

        return torch.cat([r_feat, g_feat, b_feat], dim=1)  # B×(3*256)×H×W

class DICAM(nn.Module):
    # 可选depth 【2444】【4668】
    def __init__(self):
        super(DICAM, self).__init__()
        self.feat_polar = PolarFeatureExtractor(in_channels=1, out_channels=64)

        self.feat_color = ColorFeatureExtractor(filters=64, reduction_ratio=4)

        self.cross_attn = GatedCrossFusion(dim_q=192, dim_kv=768, out_dim=128)

        # 使用 1x1 Conv 将拼接后的通道缩到 embed_dim
        self.reduction = nn.Conv2d(128 + 768, 64, kernel_size=1)

        self.layer_2 = IncCAM(in_channels=64, filters=64, reduction_ratio=4)

        self.layer_tail = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=24,kernel_size=(3,3),stride=(1, 1),padding=(3-1) // 2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=24,out_channels=3,kernel_size=(1,1),stride=(1, 1),padding=(1-1) // 2),
            nn.Sigmoid()

        )

    def forward(self,x1, x2, x3, s0):
        f_polar = self.feat_polar(x1, x2, x3)  
        f_color = self.feat_color(s0)
        f_cross = self.cross_attn(f_polar, f_color)
        concat = torch.cat([f_cross, f_color], dim=1)
        reduced = self.reduction(concat)
        layer_3 = self.layer_2(reduced)
        out = self.layer_tail(layer_3)
        return out




