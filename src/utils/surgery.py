from typing import Dict, Iterable, Callable
from collections import OrderedDict
import numpy as np

import torch
from torch import nn, Tensor
from torch.nn.functional import pad



from collections import OrderedDict

__all__ = ["extract_patches", "SpecificLayerTypeOutputExtractor_wrapper"]

def extract_patches(images, patch_shape, stride, padding=(0, 0), in_order="NHWC", out_order="NHWC"):
    assert images.ndim >= 2 and images.ndim <= 4
    if isinstance(images, np.ndarray):
        from sklearn.feature_extraction.image import _extract_patches

        if images.ndim == 2:  # single gray image
            images = np.expand_dims(images, 0)

        if images.ndim == 3:
            if images.shape[2] == 3:  # single color image
                images = np.expand_dims(images, 0)
            else:  # multiple gray images or single gray image with first index 1
                images = np.expand_dims(images, 3)

        elif in_order == "NCHW":
            images = images.transpose(0, 2, 3, 1)
        # numpy expects order NHWC
        patches = _extract_patches(
            images,
            patch_shape=(1, *patch_shape),
            extraction_step=(1, stride, stride, 1),
        ).reshape(-1, *patch_shape)
        # now patches' shape = NHWC

        if out_order == "NHWC":
            pass
        elif out_order == "NCHW":
            patches = patches.permute(0, 3, 1, 2)
        else:
            raise ValueError(
                'out_order not understood (expected "NHWC" or "NCHW")')

    elif isinstance(images, torch.Tensor):
        if images.ndim == 2:  # single gray image
            images = images.unsqueeze(0)

        if images.ndim == 3:
            if images.shape[2] == 3:  # single color image
                images = images.unsqueeze(0)
            else:  # multiple gray image
                images = images.unsqueeze(3)

        if in_order == "NHWC":
            images = images.permute(0, 3, 1, 2)
        # torch expects order NCHW

        images = pad(images, pad=padding)
        # if padding[0] != 0:
        #     breakpoint()

        patches = torch.nn.functional.unfold(
            images, kernel_size=patch_shape[:2], stride=stride
        )
        # at this point patches.shape = N, prod(patch_shape), n_patch_per_img

        # all these operations are done to circumvent pytorch's N,C,H,W ordering
        patches = patches.permute(0, 2, 1)
        n_patches = patches.shape[0] * patches.shape[1]
        patches = patches.reshape(n_patches, patch_shape[2], *patch_shape[:2])
        # now patches' shape = NCHW
        if out_order == "NHWC":
            patches = patches.permute(0, 2, 3, 1)
        elif out_order == "NCHW":
            pass
        else:
            raise ValueError(
                'out_order not understood (expected "NHWC" or "NCHW")')

    return patches


class SpecificLayerTypeOutputExtractor_wrapper(nn.Module):
    def __init__(self, model: nn.Module, layer_type):
        super().__init__()
        self._model = model
        self.layer_type = layer_type

        self.hook_handles = {}

        self.layer_outputs = OrderedDict()
        self.layer_inputs = OrderedDict()
        self.layers_of_interest = OrderedDict()
        for layer_id, layer in model.named_modules():
            if isinstance(layer, layer_type):
                self.layer_outputs[layer_id] = torch.empty(0)
                self.layer_inputs[layer_id] = torch.empty(0)
                self.layers_of_interest[layer_id] = layer

        for layer_id, layer in model.named_modules():
            if isinstance(layer, layer_type):
                self.hook_handles[layer_id] = layer.register_forward_hook(
                    self.generate_hook_fn(layer_id))

    def generate_hook_fn(self, layer_id: str) -> Callable:
        def fn(_, input, output):
            self.layer_outputs[layer_id] = output
            self.layer_inputs[layer_id] = input[0]
        return fn

    def close(self):
        [hook_handle.remove() for hook_handle in self.hook_handles.values()]

    def forward(self, x):
        return self._model(x)

    def __getattribute__(self, name: str):
        # the last three are used in nn.Module.__setattr__
        if name in ["_model", "layers_of_interest", "layer_outputs", "layer_inputs", "hook_handles", "generate_hook_fn", "close", "__dict__", "_parameters", "_buffers", "_non_persistent_buffers_set"]:
            return object.__getattribute__(self, name)
        else:
            return getattr(self._model, name)
