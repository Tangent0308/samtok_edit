"""Read-only, differentiable region statistics for Qwen-Image attention."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from samtok_edit21.data.protocol import spans_in
from samtok_edit21.regions.supervision import attention_regions


def require_attention_backend():
    """Fail before model loading when exact LSE gradients are unavailable."""
    from packaging.version import Version
    from diffsynth.core.attention import FLEX_ATTN_AVAILABLE
    if Version(torch.__version__.split("+")[0]) < Version("2.8") or not FLEX_ATTN_AVAILABLE:
        raise RuntimeError("Attention supervision requires PyTorch >= 2.8 and FlexAttention with differentiable LSE")


def span_positions(tokenizer, input_ids, prompt):
    """Positions in the actual unpadded, system-trimmed TE input."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Mask positions require the current batch-one contract")
    spans = spans_in(prompt)
    start = tokenizer.convert_tokens_to_ids("<|mt_start|>")
    positions = (input_ids[0] == start).nonzero().flatten()
    if len(positions) != len(spans):
        raise ValueError("Actual TE input mask count differs from prompt")
    result = positions[:, None] + torch.arange(4, device=input_ids.device)
    for index, span in enumerate(spans):
        expected = tokenizer.encode(span, add_special_tokens=False)
        if len(expected) != 4 or int(result[index, -1]) >= input_ids.shape[1] or input_ids[0, result[index]].tolist() != expected:
            raise ValueError("Actual TE input has a truncated or reordered mask span")
    return result.long()


@dataclass(frozen=True)
class AttentionSupervision:
    positions: torch.Tensor
    source: torch.Tensor
    target: torch.Tensor
    layers: tuple[int, ...]

    def bind_layout(self, repeats, image_ids, target_mask):
        if len(self.layers) != len(set(self.layers)) or not self.layers or tuple(sorted(self.layers)) != tuple(self.layers):
            raise ValueError("Attention layers must be nonempty, sorted and unique")
        device = repeats.device
        positions = self.positions.to(device)
        if int(positions.max()) >= len(repeats):
            raise ValueError("Mask positions exceed joint layout")
        offsets = repeats.cumsum(0) - repeats
        joint = offsets[positions]
        if (image_ids[joint] != -1).any():
            raise ValueError("Mask positions must map to text")
        source = (image_ids == 0).nonzero().flatten()
        target = target_mask.nonzero().flatten()
        if source.numel() != self.source[0].numel() or target.numel() != self.target[0].numel() or image_ids.max() != 1:
            raise ValueError("Source/target attention geometry mismatch")
        return BoundRegionProbe(joint[:, 1:], source, target,
                                attention_regions(self.source.to(device)).flatten(1),
                                attention_regions(self.target.to(device)).flatten(1))


@dataclass(frozen=True)
class BoundRegionProbe:
    mask_keys: torch.Tensor
    source_indices: torch.Tensor
    target_indices: torch.Tensor
    source_regions: torch.Tensor
    target_regions: torch.Tensor

    def __call__(self, query, key, lse):
        """Return [K,4] log(N_T,D_T,N_S,D_S), never a detached map.

        Log-space accumulation is algebraically identical to summing N/D,
        including across layers, and avoids underflow for tiny attention mass.
        q/k are post-norm/post-RoPE [1,S,H,D]; lse is natural-log [1,H,S].
        """
        with torch.autocast(device_type=query.device.type, enabled=False):
            q, k = query.float().transpose(1, 2), key.float().transpose(1, 2)
            if q.shape[0] != 1:
                raise ValueError("Attention supervision expects batch one")
            count = self.mask_keys.shape[0]
            ids = self.mask_keys.flatten()
            qt = q[:, :, self.target_indices]
            km = k[:, :, ids]
            log_t = torch.matmul(qt, km.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            log_t = log_t - lse[:, :, self.target_indices, None].float()
            # [K,H,T,3], with the same full-key normalization for every row.
            log_t = log_t[0].reshape(q.shape[1], -1, count, 3).permute(2, 0, 1, 3)
            log_s = torch.matmul(q[:, :, ids], k[:, :, self.source_indices].transpose(-1, -2)) / math.sqrt(q.shape[-1])
            log_s = log_s - lse[:, :, ids, None].float()
            log_s = log_s[0].reshape(q.shape[1], count, 3, -1).permute(1, 0, 2, 3)
            nt = torch.logsumexp((log_t + self.target_regions.log()[:, None, :, None]).flatten(1), 1)
            dt = torch.logsumexp(log_t.flatten(1), 1)
            ns = torch.logsumexp((log_s + self.source_regions.log()[:, None, None, :]).flatten(1), 1)
            ds = torch.logsumexp(log_s.flatten(1), 1)
            return torch.stack((nt, dt, ns, ds), -1)


def attention_loss(stats, *, read_weight=0.5, heads, target_tokens, layers):
    if not math.isfinite(read_weight) or read_weight < 0:
        raise ValueError("Invalid attention read weight")
    if stats.ndim != 3 or stats.shape[0] != len(layers) or stats.shape[-1] != 4 or not torch.isfinite(stats).all():
        raise ValueError("Invalid differentiable attention statistics")
    total = torch.logsumexp(stats.float(), dim=0)
    rt, rs = (total[:, 0] - total[:, 1]).exp(), (total[:, 2] - total[:, 3]).exp()
    main, read = (1 - rt).square().mean(), (1 - rs).square().mean()
    metrics = {"attn_main": main, "attn_read": read, "attn_r_target": rt.mean(), "attn_r_source": rs.mean()}
    for layer, item in zip(layers, stats):
        metrics[f"attn_l{layer}_r_target"] = (item[:, 0] - item[:, 1]).exp().mean()
        metrics[f"attn_l{layer}_r_source"] = (item[:, 2] - item[:, 3]).exp().mean()
        metrics[f"attn_l{layer}_target_mass"] = (item[:, 1] - math.log(heads * target_tokens)).exp().mean()
        metrics[f"attn_l{layer}_source_mass"] = (item[:, 3] - math.log(heads * 3)).exp().mean()
        metrics[f"attn_l{layer}_log_target_mass"] = item[:, 1].mean() - math.log(heads * target_tokens)
    return main + read_weight * read, metrics
