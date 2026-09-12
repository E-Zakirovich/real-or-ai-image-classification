"""
loader.py
~~~~~~~~~

With  this file I will return train, validation and  test dataset
to  artificial  model. Random  seed, random split  operations are
also written here. Btw, I will use augmentation.py and methods.py
to make worked datapipeline.
"""

# load packages
import torch
from torch.utils.data import random_split


class Load:
    def __init__(self):
        ...