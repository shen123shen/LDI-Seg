import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.InceptionNext import inceptionnext_tiny
up_kwargs = {'mode': 'bilinear', 'align_corners': False}


from pytorch_wavelets import DWTForward, DWTInverse


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    return param_size


class LDI_Seg(nn.Module):
    def __init__(self, out_planes=1, encoder='inceptionnext_tiny'):
        super(LDI_Seg, self).__init__()
        self.encoder = encoder
        if self.encoder == 'inceptionnext_tiny':
            mutil_channel = [96, 192, 384, 768]
            self.backbone = inceptionnext_tiny()

        self.dropout = torch.nn.Dropout(0.3)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.mfc1 = MFC(mutil_channel[0], mutil_channel[1], mutil_channel[2], mutil_channel[3])


        self.decoder4 = BasicConv2d(mutil_channel[3], mutil_channel[2], 3, padding=1)
        self.decoder3 = BasicConv2d(mutil_channel[2], mutil_channel[1], 3, padding=1)
        self.decoder2 = BasicConv2d(mutil_channel[1], mutil_channel[0], 3, padding=1)
        self.decoder1 = nn.Sequential(nn.Conv2d(mutil_channel[0], 64, kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.ReLU(),
                                      nn.Conv2d(64, out_planes, kernel_size=1, stride=1))

        self.fu1 = DFE(96, 192,  96)
        self.fu2 = DFE(192, 384, 192)
        self.fu3 = DFE(384, 768,  384)
    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)

        x1, x2, x3, x4 = self.mfc1(x1, x2, x3, x4)


        x_f_3 = self.fu3(x3, x4)
        x_f_2 = self.fu2(x2, x_f_3)
        x_f_1 = self.fu1(x1, x_f_2)

        d1 = self.decoder1(x_f_1)
        d1 = self.dropout(d1)
        d1 = F.interpolate(d1, scale_factor=4, mode='bilinear')  # (1,1,224,224)
        return d1


class CAM_Module(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.edge_att = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        edge_att = torch.sigmoid(self.edge_att(x))

        q = x.view(B, C, -1)
        k = x.view(B, C, -1).permute(0,2,1)
        energy = torch.bmm(q, k)
        att = torch.softmax(energy, dim=-1)
        out = torch.bmm(att, q).view(B, C, H, W)

        out = self.gamma * out * edge_att + x
        return out
class GME(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 定义 Sobel 核（浮点型）
        sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                       [-2, 0, 2],
                                       [-1, 0, 1]], dtype=torch.float32)
        sobel_kernel_y = torch.tensor([[-1, -2, -1],
                                       [ 0,  0,  0],
                                       [ 1,  2,  1]], dtype=torch.float32)

        weight_x = sobel_kernel_x.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)
        weight_y = sobel_kernel_y.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)

        self.sobel_x = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                 padding=1, groups=in_channels, bias=False)
        self.sobel_y = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                 padding=1, groups=in_channels, bias=False)

        with torch.no_grad():
            self.sobel_x.weight.data = weight_x
            self.sobel_y.weight.data = weight_y

    def forward(self, x):
        grad_x = self.sobel_x(x)
        grad_y = self.sobel_y(x)
        magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        return magnitude


class WECA(nn.Module):
    def __init__(self, in_ch, out_ch, num_heads=8, window_size=8):
        super().__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')

        self.conv_HL = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_LH = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_HH = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(inplace=True)
        )

        self.high_gate = nn.Sequential(
            nn.Conv2d(in_ch * 3, in_ch, 1),
            nn.BatchNorm2d(in_ch),
            nn.Sigmoid()  # 0~1 门控权重
        )

        self.sobel_edge = GME(in_ch)
        self.edge_fuse = nn.Conv2d(in_ch * 2, in_ch, 1)

        self.fuse_high = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, 1),
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(inplace=True)
        )

        self.edge_cam = CAM_Module(out_ch)

    def forward(self, x, enc_feat=None):
        B, C, H, W = x.shape

        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, :]
        y_LH = yH[0][:, :, 1, :]
        y_HH = yH[0][:, :, 2, :]

        f_hl = self.conv_HL(y_HL)
        f_lh = self.conv_LH(y_LH)
        f_hh = self.conv_HH(y_HH)

        high_cat = torch.cat([f_hl, f_lh, f_hh], dim=1)
        high_gate = self.high_gate(high_cat)
        high_filtered = high_gate * (f_hl + f_lh + f_hh)

        if enc_feat is not None:
            eL, eH = self.wt(enc_feat)
            e_hl = eH[0][:, :, 0, :]
            e_lh = eH[0][:, :, 1, :]
            e_hh = eH[0][:, :, 2, :]
            e_high = self.conv_HL(e_hl) + self.conv_LH(e_lh) + self.conv_HH(e_hh)
            high_filtered = self.fuse_high(torch.cat([high_filtered, e_high], dim=1))

        yH_L = yL + high_filtered
        yH_L_up = F.interpolate(yH_L, size=(H, W), mode='bilinear', align_corners=True)

        edge = self.sobel_edge(x)
        y_with_edge = self.edge_fuse(torch.cat([yH_L_up, edge], dim=1))

        output = self.edge_cam(y_with_edge)

        return output

class DFE(nn.Module):
    def __init__(self, l_dim, g_dim, out_dim):
        super(DFE,self).__init__()
        self.extra_l = MFE(l_dim)
        self.bn = nn.BatchNorm2d(out_dim)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3x3 = BasicConv2d(g_dim, out_dim, 3, padding=1)
        self.selection = nn.Conv2d(out_dim, 1, 1)
        self.conv3x3_1 = BasicConv2d(out_dim, 2, 3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.proj = BasicConv2d(out_dim*2, out_dim, 1, padding=0)
        self.conv1x1 = BasicConv2d(out_dim*2, out_dim, 1, padding=0)
        self.fms = WECA(out_dim, out_dim)
    def forward(self,l,g):
        l = self.extra_l(l)
        g = self.conv3x3(self.upsample(g))
        fuse = self.proj(torch.cat([l, g], dim=1))
        fuse = self.fms(fuse,l)
        att = self.conv3x3_1(fuse)
        att = F.softmax(att, dim=1)

        att_1 = att[:, 0, :, :].unsqueeze(1)
        att_2 = att[:, 1, :, :].unsqueeze(1)
        output = att_1 * l + att_2 * g
        output = self.conv1x1(torch.cat([output, g], dim=1))
        return output

class MFE(nn.Module):
    """Multi-order Features with Dilated DWConv Kernel.

    Args:
        embed_dims (int): Number of input channels.
        dw_dilation (list): Dilations of three DWConv layers.
        channel_split (list): The raletive ratio of three splited channels.
    """

    def __init__(self,
                 embed_dims,
                 dw_dilation=[1, 2, 3,],
                 channel_split=[1, 3, 4,],
                ):
        super(MFE, self).__init__()

        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims
        assert len(dw_dilation) == len(channel_split) == 3
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3
        assert embed_dims % sum(channel_split) == 0

        # basic DW conv
        self.DW_conv0 = nn.Conv2d(
            in_channels=self.embed_dims,
            out_channels=self.embed_dims,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[0]) // 2,
            groups=self.embed_dims,
            stride=1, dilation=dw_dilation[0],
        )
        # DW conv 1
        self.DW_conv1 = nn.Conv2d(
            in_channels=self.embed_dims_1,
            out_channels=self.embed_dims_1,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[1]) // 2,
            groups=self.embed_dims_1,
            stride=1, dilation=dw_dilation[1],
        )
        # DW conv 2
        self.DW_conv2 = nn.Conv2d(
            in_channels=self.embed_dims_2,
            out_channels=self.embed_dims_2,
            kernel_size=7,
            padding=(1 + 6 * dw_dilation[2]) // 2,
            groups=self.embed_dims_2,
            stride=1, dilation=dw_dilation[2],
        )
        # a channel convolution
        self.PW_conv = nn.Conv2d(  # point-wise convolution
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1)

    def forward(self, x):
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(
            x_0[:, self.embed_dims_0: self.embed_dims_0+self.embed_dims_1, ...])
        x_2 = self.DW_conv2(
            x_0[:, self.embed_dims-self.embed_dims_2:, ...])
        x = torch.cat([
            x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)
        x = self.PW_conv(x)
        return x







class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # (1,768,14,14)
        x = self.conv(x) # (1,384,14,14)
        x = self.bn(x)
        return x




class ChannelSELayer(torch.nn.Module):
    """
    Implements Squeeze and Excitation
    """

    def __init__(self, num_channels):
        """
        Initialization

        Args:
            num_channels (int): No of input channels
        """

        super(ChannelSELayer, self).__init__()
        # 自适应平均池化操作,表示 1*1 大小的输出  -
        # 使用`AdaptiveAvgPool2d`进行全局平均池化，以获取每个通道的全局空间信息。
        self.gp_avg_pool = torch.nn.AdaptiveAvgPool2d(1)  # 全局平均池化层，用于将每个通道的空间信息压缩成一个单一的全局特征。

        self.reduction_ratio = 8  # default reduction ratio

        num_channels_reduced = num_channels // self.reduction_ratio
        # 两个全连接层，用于实现降维和升维操作。第一个全连接层将通道数减少到
        # num_channels // reduction_ratio，第二个全连接层将其恢复到原始通道数。
        self.fc1 = torch.nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = torch.nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.act = torch.nn.LeakyReLU()

        self.sigmoid = torch.nn.Sigmoid()
        self.bn = torch.nn.BatchNorm2d(num_channels)

    def forward(self, inp):  # inp输入张量 (2,128,56,56)

        batch_size, num_channels, H, W = inp.size()
        # 将池化后得到的特征图通过view函数。将其变为形状为(batch_size, num_channels)batch_size表示输入张量的批次大小，
        # num_channels表示卷积后的特征通道数，最后再经过激活函数
        # 通过两个全连接层（`Linear`）和`LeakyReLU`激活函数来学习通道之间的关系。
        out = self.act(self.fc1(self.gp_avg_pool(inp).view(batch_size, num_channels)))
        out = self.fc2(out)
        # 使用`Sigmoid`函数生成权重，并通过逐元素乘法将这些权重应用于输入特征图。
        out = self.sigmoid(out)
        # 将输入特征图 inp 与这些权重进行逐元素乘法，以调整每个通道的贡献。
        out = torch.mul(inp, out.view(batch_size, num_channels, 1, 1))
        # 最后，使用批量归一化（`BatchNorm2d`）和`LeakyReLU`激活函数输出最终的特征图。
        out = self.bn(out)
        out = self.act(out)

        return out


class MFC(nn.Module):
    def __init__(self, in_filters1, in_filters2, in_filters3, in_filters4, width=96, up_kwargs=None):
        super().__init__()
        self.c1 = in_filters1
        self.c2 = in_filters2
        self.c3 = in_filters3
        self.c4 = in_filters4
        self.width = width

        self.high_conv1_H = nn.Conv2d(self.c1, self.c1, kernel_size=1)
        self.high_conv1_V = nn.Conv2d(self.c1, self.c1, kernel_size=1)
        self.high_conv1_D = nn.Conv2d(self.c1, self.c1, kernel_size=1)

        self.high_conv2_H = nn.Conv2d(self.c2, self.c2, kernel_size=1)
        self.high_conv2_V = nn.Conv2d(self.c2, self.c2, kernel_size=1)
        self.high_conv2_D = nn.Conv2d(self.c2, self.c2, kernel_size=1)


        self.low_align1 = nn.Conv2d(self.c4, self.c1, kernel_size=1)
        self.low_align2 = nn.Conv2d(self.c4, self.c2, kernel_size=1)
        self.low_align3 = nn.Conv2d(self.c4, self.c3, kernel_size=1)


        self.attn_conv1 = nn.Sequential(
            nn.Conv2d(self.c3, self.c1, kernel_size=1),
            nn.Sigmoid()
        )
        self.attn_conv2 = nn.Sequential(
            nn.Conv2d(self.c3, self.c2, kernel_size=1),
            nn.Sigmoid()
        )


        self.res_weight1 = nn.Parameter(torch.tensor(0.3))
        self.res_weight2 = nn.Parameter(torch.tensor(0.2))

        self.dwt = DWTForward(J=1, wave='haar', mode='zero')
        self.idwt = DWTInverse(wave='haar', mode='zero')

        self.x4_enhance = ChannelSELayer(self.c4)
        self.sqe1 = ChannelSELayer(self.c1)
        self.sqe2 = ChannelSELayer(self.c2)
        self.sqe3 = ChannelSELayer(self.c3)
        self.sqe4 = ChannelSELayer(self.c4)

    def forward(self, x1, x2, x3, x4):
        B, C1, H, W = x1.shape

        # ==================== x1  ====================
        x1_L, x1_H_list = self.dwt(x1)
        x1_H = x1_H_list[0]  # (B, C1, 3, H/2, W/2)


        h1_H = self.high_conv1_H(x1_H[:, :, 0, :])  # (B, C1, H/2, W/2)
        h1_V = self.high_conv1_V(x1_H[:, :, 1, :])
        h1_D = self.high_conv1_D(x1_H[:, :, 2, :])


        x4_up = F.interpolate(x4, size=x1_L.shape[-2:], mode='bilinear', align_corners=True)
        x4_up = self.low_align1(x4_up)
        x1_L_fused = x1_L + x4_up

        x3_up = F.interpolate(x3, size=h1_H.shape[-2:], mode='bilinear', align_corners=True)
        attn1 = self.attn_conv1(x3_up)  # (B, C1, H/2, W/2)


        h1_H_fused = h1_H * attn1
        h1_V_fused = h1_V * attn1
        h1_D_fused = h1_D * attn1


        h1_fused = torch.stack([h1_H_fused, h1_V_fused, h1_D_fused], dim=2)


        x1_1 = self.sqe1(self.idwt((x1_L_fused, [h1_fused])))
        x1_1 = x1_1 + self.res_weight1 * x1

        # ==================== x2  ====================
        x2_L, x2_H_list = self.dwt(x2)
        x2_H = x2_H_list[0]  # (B, C2, 3, H/4, W/4)

        h2_H = self.high_conv2_H(x2_H[:, :, 0, :])
        h2_V = self.high_conv2_V(x2_H[:, :, 1, :])
        h2_D = self.high_conv2_D(x2_H[:, :, 2, :])

        x4_up2 = F.interpolate(x4, size=x2_L.shape[-2:], mode='bilinear', align_corners=True)
        x4_up2 = self.low_align2(x4_up2)
        x2_L_fused = x2_L + x4_up2

        x3_up2 = F.interpolate(x3, size=h2_H.shape[-2:], mode='bilinear', align_corners=True)
        attn2 = self.attn_conv2(x3_up2)

        h2_H_fused = h2_H * attn2
        h2_V_fused = h2_V * attn2
        h2_D_fused = h2_D * attn2

        h2_fused = torch.stack([h2_H_fused, h2_V_fused, h2_D_fused], dim=2)

        x2_1 = self.sqe2(self.idwt((x2_L_fused, [h2_fused])))
        x2_1 = x2_1 + self.res_weight2 * x2

        # ==================== x3 ====================
        x4_up3 = F.interpolate(x4, size=x3.shape[-2:], mode='bilinear', align_corners=True)
        x3_1 = self.sqe3(x3 + self.low_align3(x4_up3))

        # ==================== x4 ====================
        x4_1 = self.x4_enhance(x4)

        return x1_1, x2_1, x3_1, x4_1

