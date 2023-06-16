import torch
import torch.nn as nn 

class GMoF(nn.Module):
    def __init__(self, rho=1):
        super(GMoF, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, residual):
        squared_res = residual ** 2
        dist = self.rho ** 2 * torch.div(residual ** 2, residual ** 2 + self.rho ** 2)
        return self.rho ** 2 * dist