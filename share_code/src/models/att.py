import torch
import torch.nn as nn
from src.models.layers import *
import math
import numpy
from timm.models.layers import trunc_normal_


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head=8, in_feat=512, thres=1.0, n_feat=512, dropout_rate=0):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.scale = nn.Parameter(torch.ones(1), requires_grad=True)

        self.q = Layer_inAttention(100,n_feat,1,1,1.0)

        self.k = Layer_inAttention(n_feat,n_feat,1,1,1.0)

        self.v = Layer_inAttention(n_feat,n_feat,1,1,1.0)

        self.attn_lif = LIFSpike(thresh=thres)
        self.out = Layer_inAttention(n_feat,n_feat,1,1,1.0)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.scale.data.fill_(0.25)

    def forward(self, query, key, value):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        time = query.size(1)
        q = self.q(query).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)

        k = self.k(key).view(n_batch, -1, self.h, self.d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)

        v = self.v(value).view(n_batch, -1, self.h, self.d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
       
        mask = torch.zeros((n_batch, time, time), device = 'cuda')
        for i in range(n_batch):
            mask[i] = torch.triu(torch.ones(time, time),diagonal=1)
        mask = mask.unsqueeze(1).eq(1)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(mask, 0.0)
        x = attn @ v # (batch, head, time2, d_k)
        x = self.attn_lif(x)
        x = x.transpose(2, 3).reshape(n_batch, time, self.h*self.d_k).contiguous()
        x = self.out(x)
        return x
    