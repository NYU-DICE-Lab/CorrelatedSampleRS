"""
Patchwise smoothing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision  
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class PatchEnsembleModel(nn.Module):
    """
    Patchwise smoothing
    """
    def __init__(self, base_classifier, patch_size, patch_stride=1, reduction='mean'):
        super().__init__()
        self.base_classifier = base_classifier
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.reduction = reduction
    
    def get_patches(self, x):
        b, c, h, w = x.shape
        h2 = h/self.patch_stride
        w2 = w/self.patch_stride
        pad_row = (h2 -1) * self.patch_stride + self.patch_size - h
        pad_col = (w2 -1) * self.patch_stride + self.patch_size - w
        x = F.pad(x, (pad_row//2, pad_row - pad_row//2, pad_col//2, pad_col - pad_row//2))

        # get patches
        patches = x.unfold(2, self.patch_size, self.patch_stride).unfold(3, self.patch_size, self.patch_stride)
        patches = patches.reshape(b, c, self.num_patches, self.patch_size, self.patch_size)
        patches.permute(0,2,1,3,4).contiguous()
        return patches
    
    def forward(self, x):
        patches = self.get_patches(x)
        # get the output of each patch
        outputs = self.base_classifier(patches)
        if self.reduction == 'mean':
            outputs = outputs.mean(dim=1)
        return outputs

class PreprocessLayer(nn.Module):
    """
    Apply transformations for base classifier.
    Supports mean, std, deviation, normalization,
    """
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms
        self.preprocess_layer = self.transforms_imagenet_eval(**self.transforms)

    def transforms_imagenet_eval(
        self,
        img_size=224,
        crop_pct=None,
        interpolation='bilinear',
        use_prefetcher=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD):

        # Since we assume that we are using a patch model, we will disregard cropping and resizing. The arguments exist to
        # maintain compatibility with timm classifiers.
    
        # crop_pct = crop_pct or DEFAULT_CROP_PCT

        # if isinstance(img_size, (tuple, list)):
        #     assert len(img_size) == 2
        #     if img_size[-1] == img_size[-2]
        #     # fall-back to older behaviour so Resize scales to shortest edge if target is square
        #         scale_size = int(math.floor(img_size[0] / crop_pct))
        #     else:
        #         scale_size = tuple([int(x / crop_pct) for x in img_size])
        # else:
        #     scale_size = int(math.floor(img_size / crop_pct))

        # tfl = [
        #     transforms.Resize(scale_size, interpolation=str_to_interp_mode(interpolation)),
        #     transforms.CenterCrop(img_size),
        # ]
        if use_prefetcher:
            # prefetcher and collate will handle tensor conversion and norm
            tfl += [ToNumpy()]
        else:
            tfl += [
                transforms.ToTensor(),
                transforms.Normalize(
                         mean=torch.tensor(mean),
                         std=torch.tensor(std))
            ]

        return transforms.Sequential(tfl)

    def forward(self, x):
        return self.preprocess_layer(x)