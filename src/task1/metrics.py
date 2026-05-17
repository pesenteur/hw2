from __future__ import annotations

import torch


@torch.no_grad()
def accuracy_topk(logits: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1, 5)) -> list[float]:
    maxk = min(max(topk), logits.size(1))
    _, pred = logits.topk(maxk, dim=1)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    scores = []
    for k in topk:
        k = min(k, logits.size(1))
        correct_k = correct[:k].reshape(-1).float().sum(0)
        scores.append((correct_k / target.numel()).item())
    return scores

