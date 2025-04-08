import torch
import numpy as np
import torch.nn.functional as F
import math
# from constants import config

# def remove_padding(tokenizer, ids, left=False):
#     if left:
#         for i in range(ids.shape[0]):
#             if ids[i].item() != config['PAD_TOKEN']:
#                 ids = ids[i:]
#                 break
#     else:
#         for i in range(ids.shape[0] - 1, -1, -1):
#             if ids[i].item() != config['PAD_TOKEN']:
#                 ids = ids[:i+1]
#                 break
#     return tokenizer.decode(ids)

def grad_dist(grads1, grads2, args):
    ret = 0.0
    n_g = 0
    for g1, g2 in zip(grads1, grads2):
        if (g1 is not None) and (g2 is not None):
            if args.loss == 'cos':
                ret += 1.0 - (g1 * g2).sum() / (g1.view(-1).norm(p=2) * g2.view(-1).norm(p=2) + 1e-9)
            elif args.loss == 'dlg':
                ret += (g1 - g2).square().sum()
            elif args.loss == 'tag':
                ret += (g1 - g2).square().sum() + args.tag_factor * torch.abs(g1 - g2).sum()
            else:
                assert False
            n_g += 1
    if args.loss == 'cos':
        ret /= n_g
    return ret


def get_closest_tokens(inputs_embeds, unused_tokens, embeddings_weight, metric='cos'):
    embeddings_weight = embeddings_weight.repeat(inputs_embeds.shape[0], 1, 1)
    if metric == 'l2':
        d = torch.cdist(inputs_embeds, embeddings_weight, p=2)
    elif metric == 'cos':
        dp = torch.bmm(inputs_embeds, embeddings_weight.transpose(1, 2))
        norm1 = inputs_embeds.norm(p=2, dim=2).unsqueeze(2)
        norm2 = embeddings_weight.norm(p=2, dim=2).unsqueeze(1)
        d = -dp / (norm1 * norm2)
    else:
        assert False

    d[:, :, unused_tokens] = 1e9
    return d, d.min(dim=2)[1]


def get_layer_decomp(grad, B=None, tol=None, upcast=False):
    grad = grad.detach().cpu().numpy()
    if upcast:
        grad = grad.astype(np.float32)
    if B == None:
        if upcast:
            B = np.linalg.matrix_rank( grad.astype(np.float32) , tol=tol)
            grad = grad.float()
        else:
            B = np.linalg.matrix_rank( grad , tol=tol)
    U,S,Vh = torch.svd_lowrank(torch.tensor(grad),q=B,niter=10)
    if upcast:
        R = Vh.T.half()
    else:
        R = Vh.T
    return  B, torch.Tensor(R).detach()

def get_perplexity(gpt2, x_embeds, bert_embeddings_weight, gpt2_embeddings_weight, c=0.1):
    gpt2_embeddings_weight = gpt2_embeddings_weight.repeat(x_embeds.shape[0], 1, 1)

    # Get alphas on BERT embeddings --> transfer to GPT-2
    alpha, _ = get_closest_tokens(x_embeds, bert_embeddings_weight)
    # alpha = torch.cdist(x_embeds[:, :-1, :], bert_embeddings_weight, p=2)
    alpha = F.softmax(-alpha/c, dim=2)
    gpt2_embeds = alpha.bmm(gpt2_embeddings_weight)

    # Pass through GPT-2 and get average perplexity
    out_gpt2 = gpt2(inputs_embeds=gpt2_embeds)
    log_probs = out_gpt2.logits.log_softmax(dim=2)
    fuzzy_perplexity = -(log_probs[:, :-1, :] * alpha[:, 1:, :]).sum(dim=2).mean(dim=1).sum()
    return fuzzy_perplexity


def check_if_in_span(R_K_norm, v, norm='l2'):
    v /= v.pow(2).sum(-1,keepdim=True).sqrt()
    proj = torch.einsum('ik,ij,...j->...k', R_K_norm, R_K_norm, v ) 
    out_of_span = proj - v
    if norm == 'l2':
        size = out_of_span.pow(2).sum(-1).sqrt()
    elif norm == 'l1':
        size = out_of_span.abs().mean(-1)

    return size

def filter_in_span(R_K_norm, v, thresh, norm):
    size = check_if_in_span(R_K_norm, v, norm)
    bools = size < thresh
    return torch.where( bools )

def get_top_in_span(R_K_norm, v, thresh, norm,topk=5):
    size = check_if_in_span(R_K_norm, v, norm)
    size_probs = adaptive_sigmoid_probs(size)
    # _ , tokens = torch.sort(size_probs, descending=True)
    # bools = size < thresh
    # which = torch.where( bools )
    # _, idx = torch.sort( size[which] )
    # which_new = []
    # for w in which:
    #     which_new.append( w[idx] )
    # which_new = tuple( which_new )
    # return which_new[0],size_probs
    return size_probs
    # top_probs, top_indices = torch.topk(size_probs, topk)
    # return [
    #     (token.item(), math.log(prob.item())) 
    #     for prob, token in zip(top_probs.squeeze(0), top_indices.squeeze(0))
    # ]


def sigmoid_inverse_probs(distances, gamma=5.0):
    """
    基于Sigmoid逆变换的概率转换
    :param distances: [vocab_size] 距离张量（越小越可能）
    :param gamma: 陡峭度控制因子（建议5.0-20.0）
    :return: 概率分布向量
    """
    # 数据标准化
    mu = distances.mean()
    sigma = distances.std() + 1e-8
    normalized = (distances - mu) / sigma
    
    # Sigmoid变换
    scores = -gamma * normalized
    sigmoid_probs = torch.sigmoid(scores)
    
    # 归一化
    return sigmoid_probs / sigmoid_probs.sum()

def adaptive_sigmoid_probs(distances, min_gamma=5.0, max_gamma=50.0):
    """
    根据距离分布自动调整gamma值
    """
    # 计算距离分布的偏态系数
    skewness = torch.mean((distances - distances.mean())**3) / (distances.std()**3 + 1e-8)
    
    # 动态gamma：分布越集中，gamma越大
    gamma = min_gamma + (max_gamma - min_gamma) * torch.sigmoid(torch.tensor(-skewness))
    
    return sigmoid_inverse_probs(distances, gamma=gamma.item())

def normalize_each_row_in_a_torch_matrix(matrix):
    row_sums = torch.sqrt(torch.sum(matrix ** 2, dim=1, keepdim=True))
    normalized_matrix = matrix / row_sums
    return normalized_matrix

def reverse_text_embeddings(text_embeddings, model, to_numpy=True):
            true_embedding = model.transformer.wte.weight.data
            true_embedding = (true_embedding
                              - true_embedding.mean(dim=-1, keepdim=True))
            true_embedding = normalize_each_row_in_a_torch_matrix(
                matrix=true_embedding
            )

            reconstructed_embedding = text_embeddings
            # reconstructed_embedding = (reconstructed_embedding
            #                            - reconstructed_embedding.mean(dim=-1, keepdim=True))
            reconstructed_embedding = normalize_each_row_in_a_torch_matrix(
                matrix=reconstructed_embedding
            )

            similarity = torch.matmul(true_embedding,
                                      torch.transpose(reconstructed_embedding, 1, 2))
            corresponding_tokens = torch.argmax(similarity, dim=1)
            if to_numpy:
                corresponding_tokens = corresponding_tokens.detach().cpu().numpy()

            return corresponding_tokens