import numpy as np
from pprint import pprint

import math
from transformers import GPT2Tokenizer,GPT2LMHeadModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax
from torch.autograd import grad
from datasets import load_dataset
from utils import GumbelSoftmax, get_matrices_expansions, get_embeddings
from functional import get_top_in_span,adaptive_sigmoid_probs,reverse_text_embeddings
torch.manual_seed(50)
torch.autograd.set_detect_anomaly(True)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)

net = GPT2LMHeadModel.from_pretrained("gpt2")
net.eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # 设置填充token

def text_to_input(text, max_length=32):
    sample = tokenizer(
        text,
        return_tensors="pt",
        # padding="max_length",  # 确保所有序列等长
        max_length=32,         # 根据你的句子长度调整
        truncation=True
    )
    # print("first_token:",sample["input_ids"][0][0])
    sample["labels"] = sample["input_ids"].clone()
    return sample

class BeamCandidate:
    def __init__(self, tokens, score, log_probs):
        self.tokens = tokens    # Token序列
        self.score = score      # 累积对数概率
        self.log_probs = log_probs  # 各位置对数概率

def get_top_k_candidates(net, gumble, U, current_sequence, top_k=5):
    """生成当前position的top-k候选token及其对数概率"""
    # 将当前序列转换为模型输入
    input_ids = torch.tensor([current_sequence]).to(device)
    with torch.no_grad():
        outputs = net(input_ids)
        logits = outputs.logits[0, -1, :]  # 取最后一个位置的logits
    
    # 使用Gumbel采样获取概率分布
    # probs = gumble(logits.unsqueeze(0), U.weight.data, hard=False).squeeze()
    probs = gumble(logits.unsqueeze(0), U, hard=False).squeeze()
    
    # 选择top-k候选
    top_probs, top_indices = torch.topk(probs, top_k)
    return [
        (token.item(), math.log(prob.item())) 
        for prob, token in zip(top_probs, top_indices)
    ]
    # return [
    #     (token.item(), math.log(logits[token.item()].item())) 
    #     for token in top_indices
    # ]

def calculate_perplexity(log_probs):
    """计算序列的困惑度"""
    if not log_probs:
        return float('inf')
    n = len(log_probs)
    avg_log_prob = sum(log_probs) / n
    return math.exp(-avg_log_prob)

def beam_search(net, tokenizer, gumbles, Us, R_Q0, beam_width=3, max_len=20, top_k=5):
    # 初始化beam
    embeds_zero = get_embeddings(net, 0)
    init_candidates = get_top_in_span(R_Q0, embeds_zero, 0.005,'l2',topk=100)
    beams = [BeamCandidate([token], log_prob, [log_prob]) for token, log_prob in init_candidates]
    
    for pos in range(1, max_len):
        candidates = []
        # 扩展每个beam
        for beam in beams:
            if len(beam.tokens) >= max_len:
                candidates.append(beam)
                continue
            
            # 获取当前pos的候选token（修改后的候选生成）   
            token_candidates = get_top_k_candidates(
                net=net,
                gumble=gumbles[pos],
                # U = Us(torch.tensor([0])),
                U = Us[pos],
                current_sequence=beam.tokens,
                top_k=top_k
            )
            
            # 扩展候选
            for token, log_prob in token_candidates:
                new_tokens = beam.tokens + [token]
                new_log_probs = beam.log_probs + [log_prob]
                new_score = beam.score + log_prob
                candidates.append(BeamCandidate(new_tokens, new_score, new_log_probs))
        
        # 根据评分排序并剪枝
        candidates.sort(key=lambda x: x.score, reverse=True)
        beams = candidates[:beam_width]
    
    # 根据困惑度筛选最佳候选
    for beam in beams:
        beam.perplexity = calculate_perplexity(beam.log_probs)
    
    return min(beams, key=lambda x: x.perplexity)

text = "The muscles are your body's \"grand central station.\""  # 替换为你的句子
sample = text_to_input(text)
sample_len = sample["input_ids"].shape[1]


gumbels = [GumbelSoftmax() for _ in range(sample_len)]
# Us = [torch.randn(1, 50257).uniform_().requires_grad_(True) for _ in range(sample_len)]
Us = [nn.Parameter(torch.randn(1, 50257).uniform_(-0.5, 0.5)) for _ in range(sample_len)]
# Us = torch.nn.Embedding(num_embeddings=sample_len, embedding_dim=50257)
# Us = [torch.nn.Embedding(num_embeddings=50257, embedding_dim=1).to(device) for _ in range(sample_len)]



for i,m in net.named_parameters():
    if "weight" in i:
        m.data.uniform_(-0.5, 0.5)
    if "bias" in i:
        m.data.uniform_(-0.5, 0.5)

# for U in Us:   
#     U.weight.data.uniform_(-0.5, 0.5)
#     U.weight.requires_grad = True
#     U.weight.retain_grad()
    
def label_to_onehot(label,pad_to_len=sample_len):
    onehot = torch.zeros(pad_to_len, 50257)
    onehot_copy = onehot.clone().detach()
    onehot_copy.scatter_(1, label, 1)
    return onehot_copy

def onehot_criterion(x,pred):
    # x: (batch_size, 50257)
    # pred: (batch_size, 50257)
    loss = torch.nn.BCELoss(reduction='none')(x, pred)
    return loss.sum(dim=1).mean()



criterion = onehot_criterion
######### honest partipant #########
# compute original gradient 
out = net(**sample)
logits = torch.tensor([adaptive_sigmoid_probs(logit).item() for logit in out.logits.squeeze(0)])
orig_loss = criterion(logits, label_to_onehot(sample["labels"]))
dy_dx = torch.autograd.grad(orig_loss, net.parameters(), create_graph=True, allow_unused=True)


# share the gradients with other clients
original_dy_dx = list((_.detach().clone() for _ in dy_dx if _ is not None))

dummy_text = "Any Dummy Text To Be Replaced initially" 
dummy_sample = text_to_input(dummy_text)

dummy_ids = dummy_sample["input_ids"]

dummy_label = dummy_sample["labels"]

dummy_label = label_to_onehot(dummy_label)

def onehot_to_label(onehot):
    _, label = onehot.max(dim=1)
    return label

    
optimizer = torch.optim.LBFGS(Us,lr=1e-5)


history = []
for iters in range(300):
    def closure():
        global dummy_ids, dummy_label
        dummy_logits = torch.zeros_like(dummy_label).to(device)
        optimizer.zero_grad()  
        cand_dict = {i: [] for i in range(sample_len)}
        pred = net(input_ids=dummy_ids, labels=dummy_ids)
        _, R_Qs = get_matrices_expansions(original_dy_dx)
        for pos in range(sample_len):
            emb = get_embeddings(net, pos)
            probs = get_top_in_span(R_Qs[0], emb, 0.005,'l2')
            gumbel_probs = gumbels[pos](probs,Us[pos],hard=False)
            dummy_logits[pos]= gumbel_probs
        loss = criterion(dummy_label, dummy_logits) 
        dummy_dy_dx = torch.autograd.grad(loss, net.parameters(), create_graph=True,allow_unused=True)
        dummy_dy_dx = (_ for _ in dummy_dy_dx if _ is not None)

        grad_diff = 0
        grad_count = 0
        for gx, gy in zip(dummy_dy_dx, original_dy_dx): # TODO: fix the variablas here
            grad_diff += ((gx - gy) ** 2).sum()
            grad_count += gx.nelement()

        grad_diff.backward()
        dummy_ids = torch.argmax(dummy_logits, dim=-1)
        dummy_label = torch.tensor([gumbel(dummy_logits[i],Us[i],hard=True) for i,gumbel in enumerate(gumbels)])
        return grad_diff
        # perplex = calculate_perplexity(math.log(pred.logits))
        # loss = grad_diff + perplex
        # loss.backward()
        # return loss
    
    optimizer.step(closure)
    # current_loss = closure()
    # print(iters, "%.4f" % current_loss.item())
    # print("dummy_text:",dummy_text)
    # print("Gumbles change:",Us[0].weight.grad)
    dummy_text = tokenizer.decode(dummy_ids[0])

    if iters % 1 == 0: 
        current_loss = closure()
        print(iters, "%.4f" % current_loss.item())
        print("dummy_text:",dummy_text)
        print(f"[Iter {iters}] Grad Norm for U[{1}]:", None if Us[1].grad is None else Us[1].grad.norm().item())
