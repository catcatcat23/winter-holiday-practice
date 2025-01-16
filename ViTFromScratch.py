#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/8/24 20:16
# @File    : ViTFromScratch.py
# @Email   : yancan@u.nus.edu / yancan@comp.nus.edu.sg
# @Software : PyCharm


import torch
import torch.nn as nn
from einops import rearrange
# pip install einops

class ViT(nn.Module):
    def __init__(self, *,
                 img_dim,
                 in_channels=3,
                 patch_dim=16,
                 num_classes=10,
                 dim=512,
                 blocks=6,
                 heads=4,
                 dim_linear_block=1024,
                 dropout=0, transformer=None, classification=True):
        """
        Args:
            img_dim: the spatial image size
            in_channels: number of img channels
            patch_dim: desired patch dim
            num_classes: classification task classes
            dim: the linear layer's dim to project the patches for MHSA
            blocks: number of transformer blocks
            heads: number of heads
            dim_linear_block: inner dim of the transformer linear block
            dropout: for pos emb and transformer
            transformer: in case you want to provide another transformer implementation
            classification: creates an extra CLS token
        """
        super().__init__()
        assert img_dim % patch_dim == 0, f'patch size {patch_dim} not divisible'
        self.p = patch_dim
        self.classification = classification
        tokens = (img_dim // patch_dim) ** 2
        self.token_dim = in_channels * (patch_dim ** 2)
        self.dim = dim

        self.project_patches = nn.Linear(self.token_dim, dim)

        self.emb_dropout = nn.Dropout(dropout)
        if self.classification:
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
            self.pos_emb1D = nn.Parameter(torch.randn(tokens + 1, dim)) # pos_emb1D是有梯度的，所以是Parameter w
            self.mlp_head = nn.Linear(dim, num_classes) # nn.Linear = w * x + b
        else:
            self.pos_emb1D = nn.Parameter(torch.randn(tokens, dim))

        # 问chatgpt，结合公式解释代码
        # TODO: Implement transformer encoder from scratch
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim_linear_block,
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=blocks
        )

    def expand_cls_to_batch(self, batch):
        """
        Args:
            batch: batch size
        Returns: cls token expanded to the batch size
        """
        return self.cls_token.expand([batch, -1, -1])

    def forward(self, img, mask=None):
        batch_size = img.shape[0]
        img_patches = rearrange(
            img, 'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
            patch_x=self.p, patch_y=self.p)
        # project patches with linear layer + add pos emb
        img_patches = self.project_patches(img_patches)

        if self.classification:
            img_patches = torch.cat(
                (self.expand_cls_to_batch(batch_size), img_patches), dim=1)

        patch_embeddings = self.emb_dropout(img_patches + self.pos_emb1D)
        # image patches embedding  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # positional embedding =     [0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # patch_embeddings =         [0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]

        # feed patch_embeddings into transformer encoder
        y = self.transformer(patch_embeddings, src_key_padding_mask=mask)

        if self.classification:
            # we index only the cls token for classification
            return self.mlp_head(y[:, 0, :])
        else:
            return y
