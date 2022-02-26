from typing import Union, Optional, Iterable

import torch
import torch.nn as nn
import numpy as np

from .layers import DivisiveNormalization2d, AdaptiveThreshold, Normalize, FractionalThreshold

vgg_depth_dict = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class HaH_VGG(nn.Module):
    def __init__(
        self, vgg_depth: str = "VGG16", conv_layer_name: str = "conv2d", conv_layer_type: type = nn.Conv2d, divisive_sigma: Optional[float] = None, threshold: Optional[float] = None
    ) -> None:
        super().__init__()
        self.normalize = Normalize((0.4914, 0.4822, 0.4465),
                                   (0.2471, 0.2435, 0.2616))
        self.vgg_depth = vgg_depth
        self.conv_layer_name = conv_layer_name
        self.conv_layer_type = conv_layer_type
        self.divisive_sigma = divisive_sigma
        self.threshold = threshold

        self.features = self.make_layers(vgg_depth_dict[vgg_depth])
        self.classifier = nn.Linear(512, 10)

        # self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, layer_list: Iterable[Union[str, int]]) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_channels = 3
        for layer_i, layer_info in enumerate(layer_list):
            if isinstance(layer_info, str):
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            else:
                layer_width = layer_info
                threshold_layer = []
                if type(self.divisive_sigma) == float and layer_i <= 7:
                    normalization_layer: nn.Module = DivisiveNormalization2d(
                        sigma=self.divisive_sigma)
                else:
                    normalization_layer = nn.BatchNorm2d(layer_width)

                if (self.threshold is not None and self.threshold > 0.0) and layer_i <= 7:
                    threshold_layer = [AdaptiveThreshold(
                        mean_scalar=self.threshold)]
                elif (self.threshold is not None and self.threshold < 0.0) and layer_i <= 7:
                    if self.threshold == -0.123:
                        if layer_i == 0:
                            threshold_layer = [FractionalThreshold(
                                remaining_ratio=0.1)]
                        elif layer_i == 1:
                            threshold_layer = [FractionalThreshold(
                                remaining_ratio=0.2)]
                        else:
                            threshold_layer = [FractionalThreshold(
                                remaining_ratio=0.3)]
                    else:
                        threshold_layer = [FractionalThreshold(
                                remaining_ratio=-self.threshold)]

                conv_layer = self.conv_layer_type(
                    in_channels, layer_width, kernel_size=3, padding=1, bias=False)

                layers += [conv_layer,  nn.ReLU(inplace=False),
                           normalization_layer]+threshold_layer

                in_channels = layer_width

        return nn.Sequential(*layers)

    @property
    def name(self) -> str:
        if self.conv_layer_name != "conv2d":
            s = f"{self.conv_layer_name}"
        else:
            s = ""
        if type(self.divisive_sigma) == float:
            s += f"_divisive_{self.divisive_sigma}"
        if self.threshold is not None and self.threshold > 0.0:
            s += f"_threshold_{self.threshold}"
        elif self.threshold is not None and self.threshold < 0.0:
            s += f"_fractional_{np.abs(self.threshold)}"
        s += f"_{self.vgg_depth}"
        return s
