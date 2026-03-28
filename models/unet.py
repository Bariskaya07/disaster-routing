from __future__ import annotations

from functools import partial

import torch
from torch import nn
from torch.nn import Dropout2d, UpsamplingBilinear2d

from .dpn import dpn92

encoder_params = {
    "dpn92": {
        "filters": [64, 336, 704, 1552, 2688],
        "decoder_filters": [64, 128, 256, 256],
        "last_upsample": 64,
        "init_op": dpn92,
        "url": None,
    }
}


class AbstractModel(nn.Module):
    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class ConvBottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, decoder_features: torch.Tensor, encoder_features: torch.Tensor) -> torch.Tensor:
        x = torch.cat([decoder_features, encoder_features], dim=1)
        return self.seq(x)


class UnetDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, middle_channels: int, out_channels: int) -> None:
        super().__init__()
        del middle_channels
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class EncoderDecoder(AbstractModel):
    def __init__(self, num_classes: int, num_channels: int = 3, encoder_name: str = "dpn92") -> None:
        super().__init__()
        self.first_layer_stride_two = True
        self.decoder_block = UnetDecoderBlock
        self.bottleneck_type = ConvBottleneck

        self.filters = encoder_params[encoder_name]["filters"]
        self.decoder_filters = encoder_params[encoder_name]["decoder_filters"]
        self.last_upsample_filters = encoder_params[encoder_name]["last_upsample"]
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.bottlenecks = nn.ModuleList(
            [
                self.bottleneck_type(self.filters[-idx - 2] + out_filters, out_filters)
                for idx, out_filters in enumerate(reversed(self.decoder_filters))
            ]
        )
        self.decoder_stages = nn.ModuleList(
            [self.get_decoder(idx) for idx in range(len(self.decoder_filters))]
        )
        self.last_upsample = UpsamplingBilinear2d(scale_factor=2)
        self.final = nn.Sequential(
            nn.Conv2d(self.last_upsample_filters, num_classes, 1, padding=0)
        )
        self.dropout = Dropout2d(p=0.25)

        encoder = encoder_params[encoder_name]["init_op"]()
        self.encoder_stages = nn.ModuleList(
            [self.get_encoder(encoder, idx) for idx in range(len(self.filters))]
        )
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_outputs: list[torch.Tensor] = []
        for stage in self.encoder_stages:
            x = stage(x)
            encoder_outputs.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())

        for idx, bottleneck in enumerate(self.bottlenecks):
            reverse_idx = -(idx + 1)
            x = self.decoder_stages[reverse_idx](x)
            x = bottleneck(x, encoder_outputs[reverse_idx - 1])

        x = self.last_upsample(x)
        x = self.dropout(x)
        return self.final(x)

    def get_decoder(self, layer: int) -> nn.Module:
        if layer + 1 == len(self.decoder_filters):
            in_channels = self.filters[layer + 1]
        else:
            in_channels = self.decoder_filters[layer + 1]
        return self.decoder_block(in_channels, self.decoder_filters[layer], self.decoder_filters[max(layer, 0)])

    def get_encoder(self, encoder: nn.Module, layer: int) -> nn.Module:
        raise NotImplementedError


class DPNUnet(EncoderDecoder):
    def __init__(self, seg_classes: int, backbone_arch: str = "dpn92", num_channels: int = 3) -> None:
        super().__init__(seg_classes, num_channels, backbone_arch)

    def get_encoder(self, encoder: nn.Module, layer: int) -> nn.Module:
        if layer == 0:
            return nn.Sequential(
                encoder.blocks["conv1_1"].conv,
                encoder.blocks["conv1_1"].bn,
                encoder.blocks["conv1_1"].act,
            )
        if layer == 1:
            return nn.Sequential(
                encoder.blocks["conv1_1"].pool,
                *[block for name, block in encoder.blocks.items() if name.startswith("conv2_")],
            )
        if layer == 2:
            return nn.Sequential(*[block for name, block in encoder.blocks.items() if name.startswith("conv3_")])
        if layer == 3:
            return nn.Sequential(*[block for name, block in encoder.blocks.items() if name.startswith("conv4_")])
        return nn.Sequential(*[block for name, block in encoder.blocks.items() if name.startswith("conv5_")])


dpn_unet = partial(DPNUnet)

__all__ = ["DPNUnet", "dpn_unet"]
