# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import torch
from torch import nn
import sys
import os


def _hub_dir():
    explicit = os.environ.get('TORCH_HUB_DIR')
    if explicit:
        return explicit
    torch_home = os.environ.get('TORCH_HOME')
    if torch_home:
        return os.path.join(os.path.expanduser(torch_home), 'hub')
    data_home = os.path.join('/data', os.environ.get('USER', ''), '.cache', 'torch', 'hub')
    if os.path.isdir(data_home):
        return data_home
    return torch.hub.get_dir()


class Dinov2Backbone(nn.Module):
    def __init__(self, name='dinov2_vitb14', pretrained=False, *args, **kwargs):
        super().__init__()
        self.name = name
        # Try direct import from cached hub to avoid GitHub network issues
        try:
            hub_dir = _hub_dir()
            dinov2_path = os.path.join(hub_dir, 'facebookresearch_dinov2_main')
            if os.path.exists(dinov2_path):
                sys.path.insert(0, dinov2_path)
            from dinov2.hub.backbones import dinov2_vitl14, dinov2_vitb14, dinov2_vits14, dinov2_vitg14
            backbone_map = {
                'dinov2_vitl14': dinov2_vitl14,
                'dinov2_vitb14': dinov2_vitb14,
                'dinov2_vits14': dinov2_vits14,
                'dinov2_vitg14': dinov2_vitg14,
            }
            if name in backbone_map:
                self.encoder = backbone_map[name](pretrained=pretrained)
            else:
                self.encoder = torch.hub.load('facebookresearch/dinov2', name, pretrained=pretrained)
        except Exception:
            # Fallback to torch.hub.load
            self.encoder = torch.hub.load('facebookresearch/dinov2', name, pretrained=pretrained)
        self.patch_size = self.encoder.patch_size
        self.embed_dim = self.encoder.embed_dim

    def forward(self, x):
        """
        Encode a RGB image using a ViT-backbone
        Args:
            - x: torch.Tensor of shape [bs,3,w,h]
        Return:
            - y: torch.Tensor of shape [bs,k,d] - image in patchified mode
        """
        assert len(x.shape) == 4
        y = self.encoder.get_intermediate_layers(x)[0] # ViT-L+896x896: [bs,4096,1024] - [bs,nb_patches,emb]
        return y
