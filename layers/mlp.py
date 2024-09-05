import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class MLP(nn.Module):
    def __init__(self,  dim_in,dim_h, dec='concatenate'):
        super(MLP, self).__init__()
        self.dec = dec
        if dec=='concatenate':
            dim_in=3*dim_in+3
        self.mlp_out = nn.Sequential(
            nn.Linear(dim_in, 512, bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 8, bias=False),
        )

    def forward(self, z0,z1,z2,z3):
        z=torch.cat((z0,z1,z2,z3),dim=-1)
        h = self.mlp_out(z).squeeze()
        return h

    def reset_parameters(self):
        for lin in self.mlp_out:
            try:
                # lin.reset_parameters()
                nn.init.kaiming_normal_(lin.weight, mode='fan_in', nonlinearity='relu')
            except:
                continue
