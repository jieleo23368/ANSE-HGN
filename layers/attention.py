import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Attention(nn.Module):
    def __init__(self, rel_size, in_ft):
        super().__init__()
        self.w_list = nn.ModuleList([nn.Linear(in_ft, in_ft, bias=False) for _ in range(rel_size)])
        self.y_list = nn.ModuleList([nn.Linear(in_ft, 1) for _ in range(rel_size)])
        self.att_act1 = nn.Tanh()
        self.att_act2 = nn.Softmax(dim=-1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    
    def forward(self, h_list):
        h_combine_list = []
        for i, h in enumerate(h_list):
            h = self.w_list[i](h)
            h = self.y_list[i](h)
            h_combine_list.append(h)
        score = torch.cat(h_combine_list, -1)
        score = self.att_act1(score)
        score = self.att_act2(score)
        score = torch.unsqueeze(score, -1)
        h = torch.stack(h_list, dim=1)
        h = score * h
        h = torch.sum(h, dim=1)
        return h 





class SemanticAttention(nn.Module):

    def __init__(self, in_ft, out_ft):
        super().__init__()
        self.in_ft = in_ft
        self.out_ft = out_ft
        self.W = nn.Parameter(torch.zeros(size=(in_ft, out_ft)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.b = nn.Parameter(torch.zeros(size=(1, out_ft)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        self.q = nn.Parameter(torch.zeros(size=(1, out_ft)))
        nn.init.xavier_uniform_(self.q.data, gain=1.414)
        self.Tanh = nn.Tanh()
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x, nb_rel):
        h = torch.mm(x, self.W)
        h_prime = self.Tanh(h + self.b.repeat(h.size()[0],1))
        semantic_attentions = torch.mm(h_prime, torch.t(self.q)).view(nb_rel, -1)
        N = semantic_attentions.size()[1]
        semantic_attentions = semantic_attentions.mean(dim=1,keepdim=True)
        semantic_attentions = F.softmax(semantic_attentions, dim=0)
        semantic_attentions = semantic_attentions.view(nb_rel, 1,1)
        semantic_attentions = semantic_attentions.repeat(1, N, self.in_ft)
        input_embedding = x.view(nb_rel,N,self.in_ft)

        h_embedding = torch.mul(input_embedding, semantic_attentions)
        h_embedding = torch.sum(h_embedding, dim=0).squeeze()

        return h_embedding


