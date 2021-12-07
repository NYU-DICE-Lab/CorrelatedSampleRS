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
import matplotlib
import math
matplotlib.use('Agg')
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT



class PatchModel(nn.Module):
    """
    Patchwise smoothing
    """
    def __init__(self, base_classifier, num_patches, patch_size, patch_stride=1, reduction='mean', num_classes=10):
        super().__init__()
        self.base_classifier = base_classifier
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.reduction = reduction
        self.num_patches = num_patches
        self.num_classes = num_classes
    
    def get_patches(self, x):
        # print('args2', self.patch_size, self.patch_stride)
        b, c, h, w = x.shape
        #print(b, c, h, w, self.patch_size, self.patch_stride)
        h2 = h//self.patch_stride
        w2 = w//self.patch_stride
        pad_row = (h2 -1) * self.patch_stride + self.patch_size - h
        pad_col = (w2 -1) * self.patch_stride + self.patch_size - w
        #print(pad_row, pad_col)
        #x = F.pad(x, (pad_row//2, pad_row - (pad_row//2), pad_col//2, pad_col - (pad_row//2)))
        #num_patches = (x.shape[2] // (self.patch_size - self.patch_stride)) * (x.shape[3] // (self.patch_size - self.patch_stride))
        #print(x.shape[2], x.shape[3])
        # get patches
        #print(x.shape, self.patch_stride, self.patch_size)
        patches = x.unfold(2, self.patch_size, self.patch_stride).unfold(3, self.patch_size, self.patch_stride)
        #print(patches.shape)
        _, _, px, py, _, _ = patches.shape
        gen_num_patches = px * py
        patches = patches.reshape(b, c, gen_num_patches, self.patch_size, self.patch_size)
        if gen_num_patches > self.num_patches:
            patches = patches[:, :, :self.num_patches, ...]
        patches = patches.permute(0,2,1,3,4).contiguous()
        # t = patches[4,35,...].detach().cpu().numpy().transpose(1,2,0)
        # t = (t - t.min())/t.ptp()
        # plt.imsave('./test11.png', t)
        # print(patches.shape)
        return patches
    
    def forward(self, x):
        # print(x.shape)
        # t = x[4,...].detach().cpu().numpy().transpose(1,2,0)
        # t = (t - t.min())/t.ptp()
        # plt.imsave('./test.png', t)
        patches = self.get_patches(x)
        #print(patches.shape)
        outputs = torch.zeros((patches.shape[0], patches.shape[1], self.num_classes), dtype=x.dtype, device=x.device)
        # get the output of each patch
        for i in range(patches.shape[0]):
            outputs[i] = self.base_classifier(patches[i, ...])
        # print('outputs', outputs.shape)
        if self.reduction == 'mean':
            outputs = outputs.mean(dim=1)
        elif self.reduction == 'max':
            outputs = outputs.max(dim=1)[0]
        elif self.reduction == 'min':
            outputs = outputs.min(dim=1)[0]
        # print('outputs2', outputs.shape)
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
        input_size=224,
        crop_pct=None,
        interpolation='bilinear',
        use_prefetcher=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD):

        # Since we assume that we are using a patch model, we will disregard cropping and resizing. The arguments exist to
        # maintain compatibility with timm classifiers.
        if isinstance(input_size, (tuple, list)):
            img_size = input_size[-2:]
        else:
            img_size = input_size
        crop_pct = crop_pct or DEFAULT_CROP_PCT

        if isinstance(img_size, (tuple, list)):
            assert len(img_size) == 2
            if img_size[-1] == img_size[-2]:
            # fall-back to older behaviour so Resize scales to shortest edge if target is square
                scale_size = int(math.floor(img_size[0] / crop_pct))
            else:
                scale_size = tuple([int(x / crop_pct) for x in img_size])
        else:
            scale_size = int(math.floor(img_size / crop_pct))

        tfl = [
             transforms.Resize(scale_size, interpolation=0),
             transforms.CenterCrop(img_size),
        ]
        if use_prefetcher:
            # prefetcher and collate will handle tensor conversion and norm
            tfl += [ToNumpy()]
        else:
            tfl += [
                transforms.Normalize(
                         mean=torch.tensor(mean),
                         std=torch.tensor(std))
            ]

        return nn.Sequential(*tfl)

    def forward(self, x):
        return self.preprocess_layer(x)