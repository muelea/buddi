# borrowed from https://pytorch.org/vision/0.13/_modules/torchvision/ops/misc.html#MLP
# to use this script adhere their license

import torch 
from typing import List, Optional, Callable
from torch.nn import ReLU

class MLP(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = ReLU,
        bias: bool = True,
        dropout: float = 0.0,
        **kwargs
    ) -> None:
        ''' Simple MLP layer
        '''
        super(MLP, self).__init__()
        self.in_dim = in_channels
        self.out_dim = hidden_channels[-1]

        layers = []
        curr_in_channels = self.in_dim
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(curr_in_channels, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer())
            layers.append(torch.nn.Dropout(dropout))
            curr_in_channels = hidden_dim

        layers.append(torch.nn.Linear(curr_in_channels, self.out_dim, bias=bias))
        layers.append(torch.nn.Dropout(dropout))

        for idx, module in enumerate(layers):
            self.add_module(str(idx), module)

    def forward(self, input):
        for module in self:
            input = module(input)
        return input

def build_mlp(mlp_cfg):
    model = MLP(**mlp_cfg)
    return model