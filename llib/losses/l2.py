import torch 
import torch.nn as nn

class L2Loss(nn.Module):
    def __init__(
        self,
        squared: bool = False,
        translated: bool = False,
        weighted: bool = False,
        d1_aggregation: str = 'mean',
        **kwargs
    ):
        super().__init__()
       
        self.squared = squared 
        self.translated = translated
        self.weighted = weighted
        self.d1_aggregation = d1_aggregation

    def forward(self, x, y=None, weight=None, x_origin=None, y_origin=None, dim=None):
        """
        Compute L2 loss between x and y.
        x: (batch_size, j, n)
        y: (batch_size, j, n)
        weight: (batch_size, j)
        x_origin: (batch_size, n)
        y_origin: (batch_size, n)
        """
        
        if len(x.shape) == 2:
            dim = 1
        else:
            dim = 2

        # translate input
        if self.translated:
            assert x_origin is not None
            assert y_origin is not None
            x = x - x_origin.unsqueeze(1)
            y = y - y_origin.unsqueeze(1)

        # l2 loss
        d = x - y if y is not None else x
        if self.squared:
            loss = torch.pow(d, 2).sum(dim)
        else:
            loss = (d).norm(2, dim)
        
        # weight distance by weight
        if self.weighted:
            assert weight is not None
            loss = loss * weight

        # aggregate over j
        if self.d1_aggregation == 'mean':
            loss = loss.mean(dim - 1)
        elif self.d1_aggregation == 'sum':
            loss = loss.sum(dim - 1)
        elif self.d1_aggregation == 'none':
            pass
        else:
            raise NotImplementedError(f'Unknown aggregation type: {self.d1_aggregation}')

        return loss
