from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DPN", "dpn92"]


class CatBnAct(nn.Module):
    def __init__(self, in_chs: int, activation_fn: nn.Module | None = None) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(in_chs, eps=0.001)
        self.act = activation_fn or nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        return self.act(self.bn(x))


class BnActConv2d(nn.Module):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        kernel_size: int,
        stride: int,
        padding: int = 0,
        groups: int = 1,
        activation_fn: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(in_chs, eps=0.001)
        self.act = activation_fn or nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_chs,
            out_chs,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.act(self.bn(x)))


class InputBlock(nn.Module):
    def __init__(
        self,
        num_init_features: int,
        kernel_size: int = 7,
        padding: int = 3,
        num_channels: int = 3,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            num_channels,
            num_init_features,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(num_init_features, eps=0.001)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class DualPathBlock(nn.Module):
    def __init__(
        self,
        in_chs: int,
        num_1x1_a: int,
        num_3x3_b: int,
        num_1x1_c: int,
        inc: int,
        groups: int,
        block_type: str = "normal",
        b: bool = False,
    ) -> None:
        super().__init__()
        self.num_1x1_c = num_1x1_c
        self.inc = inc
        self.b = b
        if block_type == "proj":
            self.key_stride = 1
            self.has_proj = True
        elif block_type == "down":
            self.key_stride = 2
            self.has_proj = True
        else:
            self.key_stride = 1
            self.has_proj = False

        if self.has_proj:
            if self.key_stride == 2:
                self.c1x1_w_s2 = BnActConv2d(in_chs, num_1x1_c + 2 * inc, kernel_size=1, stride=2)
            else:
                self.c1x1_w_s1 = BnActConv2d(in_chs, num_1x1_c + 2 * inc, kernel_size=1, stride=1)
        self.c1x1_a = BnActConv2d(in_chs, num_1x1_a, kernel_size=1, stride=1)
        self.c3x3_b = BnActConv2d(
            num_1x1_a,
            num_3x3_b,
            kernel_size=3,
            stride=self.key_stride,
            padding=1,
            groups=groups,
        )
        if b:
            self.c1x1_c = CatBnAct(in_chs=num_3x3_b)
            self.c1x1_c1 = nn.Conv2d(num_3x3_b, num_1x1_c, kernel_size=1, bias=False)
            self.c1x1_c2 = nn.Conv2d(num_3x3_b, inc, kernel_size=1, bias=False)
        else:
            self.c1x1_c = BnActConv2d(num_3x3_b, num_1x1_c + inc, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x_in = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        if self.has_proj:
            if self.key_stride == 2:
                x_s = self.c1x1_w_s2(x_in)
            else:
                x_s = self.c1x1_w_s1(x_in)
            x_s1 = x_s[:, : self.num_1x1_c, :, :]
            x_s2 = x_s[:, self.num_1x1_c :, :, :]
        else:
            assert isinstance(x, tuple)
            x_s1 = x[0]
            x_s2 = x[1]
        x_in = self.c1x1_a(x_in)
        x_in = self.c3x3_b(x_in)
        if self.b:
            x_in = self.c1x1_c(x_in)
            out1 = self.c1x1_c1(x_in)
            out2 = self.c1x1_c2(x_in)
        else:
            x_in = self.c1x1_c(x_in)
            out1 = x_in[:, : self.num_1x1_c, :, :]
            out2 = x_in[:, self.num_1x1_c :, :, :]
        resid = x_s1 + out1
        dense = torch.cat([x_s2, out2], dim=1)
        return resid, dense


def adaptive_avgmax_pool2d(
    x: torch.Tensor,
    pool_type: str = "avg",
    padding: int = 0,
    count_include_pad: bool = False,
) -> torch.Tensor:
    if pool_type == "avgmax":
        x_avg = F.avg_pool2d(
            x,
            kernel_size=(x.size(2), x.size(3)),
            padding=padding,
            count_include_pad=count_include_pad,
        )
        x_max = F.max_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=padding)
        return 0.5 * (x_avg + x_max)
    if pool_type == "max":
        return F.max_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=padding)
    return F.avg_pool2d(
        x,
        kernel_size=(x.size(2), x.size(3)),
        padding=padding,
        count_include_pad=count_include_pad,
    )


class DPN(nn.Module):
    def __init__(
        self,
        *,
        num_init_features: int,
        k_r: int,
        groups: int,
        k_sec: tuple[int, int, int, int],
        inc_sec: tuple[int, int, int, int],
        num_classes: int = 1000,
        test_time_pool: bool = True,
        num_channels: int = 3,
    ) -> None:
        super().__init__()
        self.test_time_pool = test_time_pool
        self.blocks: OrderedDict[str, nn.Module] = OrderedDict()
        self.out_channels: list[int] = []

        self.blocks["conv1_1"] = InputBlock(num_init_features, kernel_size=7, padding=3, num_channels=num_channels)
        self.out_channels.append(num_init_features)

        bw = 64 * 4
        inc = inc_sec[0]
        r = (k_r * bw) // (64 * 4)
        self.blocks["conv2_1"] = DualPathBlock(num_init_features, r, r, bw, inc, groups, "proj", False)
        in_chs = bw + 3 * inc
        for idx in range(2, k_sec[0] + 1):
            self.blocks[f"conv2_{idx}"] = DualPathBlock(in_chs, r, r, bw, inc, groups, "normal", False)
            in_chs += inc
        self.out_channels.append(in_chs)

        bw = 128 * 4
        inc = inc_sec[1]
        r = (k_r * bw) // (64 * 4)
        self.blocks["conv3_1"] = DualPathBlock(in_chs, r, r, bw, inc, groups, "down", False)
        in_chs = bw + 3 * inc
        for idx in range(2, k_sec[1] + 1):
            self.blocks[f"conv3_{idx}"] = DualPathBlock(in_chs, r, r, bw, inc, groups, "normal", False)
            in_chs += inc
        self.out_channels.append(in_chs)

        bw = 256 * 4
        inc = inc_sec[2]
        r = (k_r * bw) // (64 * 4)
        self.blocks["conv4_1"] = DualPathBlock(in_chs, r, r, bw, inc, groups, "down", False)
        in_chs = bw + 3 * inc
        for idx in range(2, k_sec[2] + 1):
            self.blocks[f"conv4_{idx}"] = DualPathBlock(in_chs, r, r, bw, inc, groups, "normal", False)
            in_chs += inc
        self.out_channels.append(in_chs)

        bw = 512 * 4
        inc = inc_sec[3]
        r = (k_r * bw) // (64 * 4)
        self.blocks["conv5_1"] = DualPathBlock(in_chs, r, r, bw, inc, groups, "down", False)
        in_chs = bw + 3 * inc
        for idx in range(2, k_sec[3] + 1):
            self.blocks[f"conv5_{idx}"] = DualPathBlock(in_chs, r, r, bw, inc, groups, "normal", False)
            in_chs += inc
        self.blocks["conv5_bn_ac"] = CatBnAct(in_chs)
        self.out_channels.append(in_chs)

        self.features = nn.Sequential(self.blocks)
        self.classifier = nn.Conv2d(in_chs, num_classes, kernel_size=1, bias=True)

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        if not self.training and self.test_time_pool:
            x = F.avg_pool2d(features, kernel_size=7, stride=1)
            out = self.classifier(x)
            out = adaptive_avgmax_pool2d(out, pool_type="avgmax")
        else:
            x = adaptive_avgmax_pool2d(features, pool_type="avg")
            out = self.classifier(x)
        return out.view(out.size(0), -1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.features(inputs)
        x = self.logits(x)
        return x


def dpn92(num_classes: int = 1000, pretrained: str | None = None) -> DPN:
    del pretrained
    return DPN(
        num_init_features=64,
        k_r=96,
        groups=32,
        k_sec=(3, 4, 20, 3),
        inc_sec=(16, 32, 24, 128),
        num_classes=num_classes,
        test_time_pool=True,
    )
