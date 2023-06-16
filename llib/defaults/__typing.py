from typing import NewType
import torch 
import numpy 

Tensor = NewType('Tensor', torch.Tensor)

Array = NewType('Array', numpy.array)

class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)