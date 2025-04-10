import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functional import get_layer_decomp
import numpy as np


class GumbelSoftmax(nn.Module):
    def __init__(self, vocab_size=50257,temperature=1):
        super(GumbelSoftmax, self).__init__()
        self.temperature = temperature

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).clamp(min=eps)
        return -torch.log(-torch.log(U + eps) + eps)

    def forward(self, logits, U_weight, hard=False):
        # logits: (1, vocab_size)
        # U_weight: (vocab_size, 1) or (1, vocab_size)
        # gumbel_noise = self.sample_gumbel(logits.shape).to(logits.device)
        if isinstance(U_weight, nn.Embedding):
            y = logits + U_weight.weight.T   # 关键改动：Us.weight 直接参与加法
        else:
            y = logits + U_weight 
        # y = torch.clamp(y, min=-0.5, max=0.5) 
        # y = y - y.max(dim=-1, keepdim=True)[0]
        y = F.softmax(y / self.temperature, dim=-1)
        if hard:
            index = y.max(dim=-1, keepdim=True)[1]
            y_hard = torch.zeros_like(y).scatter_(-1, index, 1.0)
            y = (y_hard - y).detach() + y  # straight-through
        return y
    
def get_matrices_expansions( true_grads, B=None, tol=5e-6):
        layer_ids = list(range(4, 137, 12))
        if B is None:
            max_rank = 0
            for i in layer_ids[:10]:
                grad = true_grads[i].detach().numpy()
                grad = grad.T
                B = np.linalg.matrix_rank(grad , tol=tol)
                if max_rank < B:
                    max_rank = B
            B = max_rank
        B = min(B, 768 - 20)
        
        R_Qs = []
        
        for i in range(2):
            grad_Q = true_grads[layer_ids[i]]
            grad_Q = grad_Q.T
            _, R_Q = get_layer_decomp(grad_Q, B=B, tol=tol)
            R_Q = R_Q.cpu()
            R_Qs.append(R_Q)
        return B, R_Qs

def get_embeddings(model,pos = None):
    gpt_embeddings_weight_position = model.transformer.wpe.weight.unsqueeze(0)
    emb = model.transformer.wte.weight.unsqueeze(0).cpu() + gpt_embeddings_weight_position[0][pos:pos+1,None,:]
    emb = model.transformer.h[0].ln_1(emb)
    return emb