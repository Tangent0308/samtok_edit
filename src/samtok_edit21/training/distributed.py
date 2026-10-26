"""Data-parallel helpers of the trainer: one LoRA initialisation for all ranks, gradient averaging and a cheap
cross-rank consistency check.  The trainer averages gradients itself (no DDP wrapper) because the code injector and
the routing calibrations train next to the LoRA and some updates touch only part of them."""
from __future__ import annotations

import torch


def broadcast_parameters(parameters, dist):
    """Give every rank rank 0's values (LoRA A matrices are random and would otherwise differ per rank)."""
    flat = torch.cat([p.detach().flatten().float() for p in parameters])
    dist.broadcast(flat, src=0)
    offset = 0
    with torch.no_grad():
        for p in parameters:
            p.copy_(flat[offset:offset + p.numel()].view_as(p).to(p.dtype))
            offset += p.numel()


def rank_divergence(parameters, dist, world):
    """Largest relative gap between any rank's parameter checksum and rank 0's (0 when the ranks agree)."""
    checksum = float(torch.stack([p.detach().double().abs().sum() for p in parameters]).sum())
    gathered = [None] * world
    dist.all_gather_object(gathered, checksum)
    return max(abs(value - gathered[0]) for value in gathered) / max(abs(gathered[0]), 1e-12)


def average_gradients(parameters, dist, world):
    """Replace each rank's gradients by their mean over ranks."""
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in parameters]
    flat = torch.cat([g.flatten().float() for g in grads])
    dist.all_reduce(flat)
    flat /= world
    offset = 0
    for p, g in zip(parameters, grads):
        p.grad = flat[offset:offset + g.numel()].view_as(p).to(p.dtype).clone()
        offset += g.numel()
