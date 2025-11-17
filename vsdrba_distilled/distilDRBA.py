import torch
import torch.nn as nn
import torch.nn.functional as F

from .warplayer import warp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.2, True)
    )


class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 16, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x, feat=False):
        x0 = self.cnn0(x)
        x = self.relu(x0)
        x1 = self.cnn1(x)
        x = self.relu(x1)
        x2 = self.cnn2(x)
        x = self.relu(x2)
        x3 = self.cnn3(x)
        if feat:
            return [x0, x1, x2, x3]
        return x3


class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super(ResConv, self).__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
        )
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * (13 + 1), 4, 2, 1),
            nn.PixelShuffle(2)
        )

    def forward(self, x, flow=None):
        if flow is not None:
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        flow = tmp[:, :4]
        mask = tmp[:, 4:5]
        feat = tmp[:, 5:13]
        tmap = torch.sigmoid(tmp[:, 13:])  # 必须在0到1内
        return flow, mask, feat, tmap


class IFNet(nn.Module):
    def __init__(self, scale=1):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(9 + 48 + 1, c=192)
        self.block1 = IFBlock(8 + 4 + 8 + 32, c=128)
        self.block2 = IFBlock(8 + 4 + 8 + 32, c=96)
        self.block3 = IFBlock(8 + 4 + 8 + 32, c=64)
        self.block4 = IFBlock(8 + 4 + 8 + 32, c=32)
        self.encode = Head()
        self.scale_list = [x / scale for x in [16, 8, 4, 2, 1]]

    def batch_interpolate(self, *tensor, scale_factor=1.0, mode="bilinear", align_corners=False):
        if scale_factor != 1:
            return [F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners) for x in tensor]
        return tensor

    def batch_warp(self, *tensor, flow):
        return [warp(x, flow, *self.get_grid(x)) for x in tensor]

    def forward(self, img0, img1, img2, timestep):
        encode_scale = min(1 / self.scale_list[-1], 1)  # 超过后续计算光流时的有效清晰度是无意义的
        img0_encode, img1_encode, img2_encode = self.batch_interpolate(img0[:, :3], img1[:, :3], img2[:, :3], scale_factor=encode_scale)
        f0 = self.encode(img0_encode)
        f1 = self.encode(img1_encode)
        f2 = self.encode(img2_encode)
        del img0_encode, img1_encode, img2_encode

        inp0 = img1.clone()
        h0 = f1.clone()

        cond_a = ((timestep >= 0.5) & (timestep < 1)).to(img0.dtype)
        cond_b = ((timestep > 1) & (timestep <= 1.5)).to(img0.dtype)
        inp1 = img0 * cond_a + img2 * cond_b
        h1 = f0 * cond_a + f2 * cond_b

        block = [self.block0, self.block1, self.block2, self.block3, self.block4]
        flow, mask = None, None
        # 用户正常使用时，无论scale=2.0，1.0，还是0.5，scale_list值一定是递减的
        for scale_idx in range(0, len(self.scale_list)):
            current_scale = 1 / self.scale_list[scale_idx]
            if flow is None:
                f0_s, f1_s, f2_s = self.batch_interpolate(f0, f1, f2, scale_factor=current_scale / encode_scale)
                img0_s, img1_s, img2_s = self.batch_interpolate(img0, img1, img2, scale_factor=current_scale)

                flow, mask, feat, tmap = block[scale_idx](torch.cat((img0_s[:, :3], img1_s[:, :3], img2_s[:, :3], f0_s,
                                                                     f1_s, f2_s, img0_s[:, :1].clone() * 0 + timestep
                                                                     ), 1))
            else:
                h0_s, h1_s = self.batch_interpolate(h0, h1, scale_factor=current_scale / encode_scale)
                inp0_s, inp1_s = self.batch_interpolate(inp0, inp1, scale_factor=current_scale)

                # 大于1.0尺寸上的warp操作耗时且无意义
                if current_scale <= 1:
                    wf0, warped_img0 = self.batch_warp(h0_s, inp0_s, flow=flow[:, :2])
                    wf1, warped_img1 = self.batch_warp(h1_s, inp1_s, flow=flow[:, 2:4])
                else:
                    wf0, warped_img0 = self.batch_warp(h0, inp0, flow=flow[:, :2])
                    wf1, warped_img1 = self.batch_warp(h1, inp1, flow=flow[:, 2:4])
                    wf0, wf1, warped_img0, warped_img1 = self.batch_interpolate(wf0, wf1, warped_img0, warped_img1,
                                                                                scale_factor=current_scale)
                    flow, mask, feat, tmap = self.batch_interpolate(flow, mask, feat, tmap, scale_factor=current_scale)
                    flow *= current_scale

                fd, mask, feat, tmap = block[scale_idx](
                    torch.cat((warped_img0[:, :3], warped_img1[:, :3], wf0, wf1, tmap, mask, feat), 1), flow)

                flow = flow + fd

            # 原流程(举例), 在1/4尺寸上infer -> 放大回1/1 做warp等操作 -> 下一次推理前缩放到1/2
            # 新流程, 在1/4尺寸上infer -> 放大到1/2 做warp操作 -> 下一次推理直接用
            # 分析, 省了几次F.interpolate操作和warp操作
            if scale_idx == len(self.scale_list) - 1:  # 最后一层缩放回原尺寸
                flow_scale = self.scale_list[scale_idx]
            elif self.scale_list[scale_idx + 1] >= 1:  # 下一次scale_list中的值如果小于1就代表下次流会放大到超出1/1
                flow_scale = self.scale_list[scale_idx] / self.scale_list[scale_idx + 1]
            else:
                flow_scale = 1

            if flow_scale != 1:
                flow, mask, feat, tmap = self.batch_interpolate(flow, mask, feat, tmap, scale_factor=flow_scale)
                flow *= flow_scale

        tenFlow_div, backwarp_tenGrid = self.get_grid(inp0)
        warped_img0 = warp(inp0, flow[:, :2], tenFlow_div, backwarp_tenGrid)
        warped_img1 = warp(inp1, flow[:, 2:4], tenFlow_div, backwarp_tenGrid)
        mask = torch.sigmoid(mask)
        merged = warped_img0 * mask + warped_img1 * (1 - mask)
        return merged

    def get_grid(self, x):
        _, _, ph, pw = x.shape
        tenFlow_div = torch.tensor([(pw - 1.0) / 2.0, (ph - 1.0) / 2.0], dtype=torch.float, device=x.device)

        tenHorizontal = torch.linspace(-1.0, 1.0, pw, dtype=torch.float, device=x.device)
        tenHorizontal = tenHorizontal.view(1, 1, 1, pw).expand(-1, -1, ph, -1)
        tenVertical = torch.linspace(-1.0, 1.0, ph, dtype=torch.float, device=x.device)
        tenVertical = tenVertical.view(1, 1, ph, 1).expand(-1, -1, -1, pw)
        backwarp_tenGrid = torch.cat([tenHorizontal, tenVertical], 1)

        return tenFlow_div, backwarp_tenGrid
