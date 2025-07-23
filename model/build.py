# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from .detection_transformer import build as build_ACT_model

def build_ACT_model_and_optimizer(config):
    model = build_ACT_model(config)
    model.cuda()
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": config['lr_backbone'],
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=config['lr'], weight_decay=config['weight_decay'])
    return model, optimizer