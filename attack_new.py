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

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)

net = GPT2LMHeadModel.from_pretrained("gpt2")
net.to(device)
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

def label_to_onehot(label,pad_to_len=20):
    onehot = torch.zeros(pad_to_len, 50257)
    onehot_copy = onehot.clone().detach().to(device)
    onehot_copy.scatter_(1, label, 1)
    return onehot_copy

def onehot_criterion(x,pred):
    # x: (batch_size, 50257)
    # pred: (batch_size, 50257)
    loss = torch.nn.BCELoss(reduction='none')(x, pred)
    return loss.sum(dim=1).mean()

text = "The muscles are your body's \"grand central station.\""  # 替换为你的句子
sample = text_to_input(text).to(device)
sample_len = sample["input_ids"].shape[1]


gumbels = [GumbelSoftmax().to(device) for _ in range(sample_len)]
# Us = [torch.randn(1, 50257).uniform_().requires_grad_(True) for _ in range(sample_len)]
Us = [nn.Parameter(torch.randn(1, 50257).to(device)) for _ in range(sample_len)]
# Us = torch.nn.Embedding(num_embeddings=sample_len, embedding_dim=50257)
# Us = [torch.nn.Embedding(num_embeddings=50257, embedding_dim=1).to(device) for _ in range(sample_len)]
print(Us[0].is_leaf)


for i,m in net.named_parameters():
    if "weight" in i:
        m.data.uniform_(-0.5, 0.5)
    if "bias" in i:
        m.data.uniform_(-0.5, 0.5)

for U in Us:
    U.data.uniform_(-0.5, 0.5) # Modified: Apply uniform to the leaf tensors

def onehot_criterion(x,pred):
    # x: (batch_size, 50257)
    # pred: (batch_size, 50257)
    loss = torch.nn.BCELoss(reduction='none')(x, pred)
    return 1/loss.sum(dim=1).mean()



criterion = onehot_criterion
######### honest partipant #########
# compute original gradient 
out = net(**sample)
logits =[adaptive_sigmoid_probs(logit) for logit in out.logits.squeeze(0)]
logits = torch.stack(logits, dim=0)
orig_loss = criterion(logits, label_to_onehot(sample["labels"],pad_to_len=sample_len).to(device))
dy_dx = torch.autograd.grad(orig_loss, net.parameters(), create_graph=True, allow_unused=True)


# share the gradients with other clients
original_dy_dx = list((_.detach().clone() for _ in dy_dx if _ is not None))
    
dummy_text = "Any Dummy Text To Be Replaced initially" 
dummy_sample = text_to_input(dummy_text).to(device)
dummy_ids = dummy_sample["input_ids"]
dummy_label = dummy_sample["labels"]
dummy_label = label_to_onehot(dummy_label,pad_to_len=sample_len)

 
# optimizer = torch.optim.LBFGS(Us,lr=1e-2,max_iter=4)
optimizer = torch.optim.AdamW(Us,lr=1e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

_, R_Qs = get_matrices_expansions(original_dy_dx) 
# pos_probs = []
# with torch.no_grad():
# for pos in range(sample_len):
#         emb = get_embeddings(net, pos)
#         probs = get_top_in_span(R_Qs[0], emb, 0.005,'l2')
#         pos_probs.append(probs)
R_Q = R_Qs[0].to(device)
del R_Qs

class BeamCandidate:
    def __init__(self, tokens, score, log_probs):
        self.tokens = tokens    # Token序列
        self.score = score      # 累积对数概率
        self.log_probs = log_probs  # 各位置对数概率

def get_topk(probs,topk):
    top_probs, top_indices = torch.topk(probs, topk)
    return top_probs, top_indices

def calculate_perplexity(dummy_ids):
      with torch.no_grad():  # GPT-2模型不参与梯度计算
          outputs = net(input_ids=dummy_ids)
          real_logits = outputs.logits  # [1, seq_len, vocab_size]

      # 将dummy_ids的labels作为真实标签，计算困惑度
      # 计算困惑度需要 CrossEntropyLoss
      ce_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

      # dummy_ids移除最后一个token，logits移除第一个位置以预测下一个token
      shift_logits = real_logits[:, :-1, :].contiguous().view(-1, real_logits.size(-1))
      shift_labels = dummy_ids[:, 1:].contiguous().view(-1)

      # 计算困惑度
      perplexity_loss = ce_loss(shift_logits, shift_labels)
      return perplexity_loss
# def calculate_perplexity(log_probs):
#     """计算序列的困惑度"""
#     if not log_probs:
#         return float('inf')
#     n = len(log_probs)
#     avg_log_prob = sum(log_probs) / n
#     return math.exp(-avg_log_prob)

history = []
max_iter = 300
for iters in range(300):
    torch.cuda.empty_cache()
    embs = [get_embeddings(net, pos) for pos in range(sample_len)]
    beam_width = 5 + int(((1-(iters+1)/max_iter))*15)
    def closure():
      global dummy_label, dummy_ids
      optimizer.zero_grad()  
      dummy_logits = torch.zeros_like(dummy_label)
      beams = []
      for pos in range(sample_len):
          probs = get_top_in_span(R_Q, embs[pos], 0.005,'l2')
          gumbel_probs = gumbels[pos](probs, Us[pos], hard=False)
          dummy_logits[pos] = gumbel_probs   
          candidates = []
          if beams == []:
              top_probs, top_indices = get_topk(gumbel_probs,topk=500)
              top_probs = top_probs.squeeze(0)
              top_indices = top_indices.squeeze(0)    
              beams = [BeamCandidate([token], log_prob, [log_prob]) for token, log_prob in zip(top_indices, torch.log(top_probs))]

          else:
            top_probs, top_indices = get_topk(gumbel_probs,topk=beam_width)
            top_probs = top_probs.squeeze(0)
            top_indices = top_indices.squeeze(0)
            for beam in beams:
              for token, log_prob in zip(top_indices, top_probs):
                  new_tokens = beam.tokens + [token]
                  new_log_probs = beam.log_probs + [log_prob]
                  new_score = beam.score + log_prob
                  candidates.append(BeamCandidate(new_tokens, new_score, new_log_probs))

            candidates.sort(key=lambda x: x.score, reverse=True)
            beams = candidates[:beam_width] #beam width=5
      

      # 原损失 (梯度差异)
      loss = criterion(dummy_logits, dummy_label) 
      # loss = criterion(dummy_label, dummy_logits)
      dummy_dy_dx = torch.autograd.grad(loss, net.parameters(), create_graph=True, allow_unused=True)
      dummy_dy_dx = tuple(_.to(device) for _ in dummy_dy_dx if _ is not None)

      # dummy_ids = torch.argmax(dummy_logits.detach(), dim=-1).unsqueeze(0)
      # for beam in beams:
      #     beam.perplexity = calculate_perplexity(beam.log_probs)
      for beam in beams:
        beam.perplexity = calculate_perplexity(torch.tensor([beam.tokens]).to(device))
      beams.sort(key=lambda x: x.perplexity)
      dummy_ids = torch.tensor([beams[0].tokens]).to(device)
      if len(history) >= 5:
          history.pop(0)
      history.append(dummy_ids.cpu())
      dummy_label = label_to_onehot(dummy_ids, pad_to_len=sample_len)

      grad_diff = 0
      for gx, gy in zip(dummy_dy_dx, original_dy_dx):
          grad_diff += ((gx - gy) ** 2).sum()

      # # --- 新增部分：GPT-2计算真实困惑度损失 ---
      # # dummy_ids作为输入，得到GPT-2的真实输出logits
      # with torch.no_grad():  # GPT-2模型不参与梯度计算
      #     outputs = net(input_ids=dummy_ids)
      #     real_logits = outputs.logits  # [1, seq_len, vocab_size]

      # # 将dummy_ids的labels作为真实标签，计算困惑度
      # # 计算困惑度需要 CrossEntropyLoss
      # ce_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

      # # dummy_ids移除最后一个token，logits移除第一个位置以预测下一个token
      # shift_logits = real_logits[:, :-1, :].contiguous().view(-1, real_logits.size(-1))
      # shift_labels = dummy_ids[:, 1:].contiguous().view(-1)

      # # 计算困惑度
      # perplexity_loss = ce_loss(shift_logits, shift_labels)
      perplexity_loss = beams[0].perplexity
      # 最终总损失 = 梯度差损失 + 困惑度损失
      total_loss = grad_diff +  perplexity_loss

      total_loss.backward()
      return total_loss

    # def closure():
    #     global dummy_label, dummy_ids
    #     optimizer.zero_grad()  
    #     dummy_logits = torch.zeros_like(dummy_label)
    #     # for pos in tqdm(range(sample_len),desc=f"[Iter {iters}] Position:"):
    #     for pos in range(sample_len):
    #         # emb = get_embeddings(net, pos)
    #         probs = get_top_in_span(R_Q, embs[pos], 0.005,'l2')
    #         # gumbel_probs = gumbels[pos](pos_probs[pos],Us[pos],hard=False)
    #         gumbel_probs = gumbels[pos](probs,Us[pos],hard=False)
    #         dummy_logits[pos]= gumbel_probs

        
    #     loss = criterion(dummy_logits,dummy_label) 
    #     dummy_dy_dx = torch.autograd.grad(loss, net.parameters(), create_graph=True,allow_unused=True)
    #     dummy_dy_dx = (_.to(device) for _ in dummy_dy_dx if _ is not None)
    #     # set new ids
    #     dummy_ids = torch.argmax(dummy_logits.detach(), dim=-1).unsqueeze(0)
    #     if len(history) >= 5:
    #       history.pop(0)
    #     history.append(dummy_ids.cpu())
    #     dummy_label = label_to_onehot(dummy_ids,pad_to_len=sample_len)

    #     grad_diff = 0
    #     grad_count = 0
    #     for gx, gy in zip(dummy_dy_dx, original_dy_dx): # TODO: fix the variablas here
    #         grad_diff += ((gx - gy) ** 2).sum()
    #         grad_count += gx.nelement()
        
    #     grad_diff.backward()
    #     return grad_diff

    # optimizer.step(closure)
    # if iters % 5 == 0:
    #     print(f"[Iter {iters}] Current loss:", "%.4f" % closure())
    #     dummy_text = tokenizer.decode(history[-1][0])
    #     print("dummy_text:",dummy_text)
    #     print(f"[Iter {iters}] Grad Norm for U[{0}]:", None if Us[0].grad is None else Us[0].grad.norm().item())
    current_loss = closure()
    optimizer.step()
    if iters % 10 == 0:
        print(f"[Iter {iters}] Current loss:", "%.4f" % current_loss.item())
        dummy_text = tokenizer.decode(dummy_ids[0])
        print("dummy_text:",dummy_text)
        print(f"[Iter {iters}] Grad Norm for U[{0}]:", None if Us[0].grad is None else Us[0].grad.norm().item())

