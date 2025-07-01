# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build
from .detr import build_vit

def build_model(args):
    return build(args)
def build_model_vit(args):
    return build_vit(args)