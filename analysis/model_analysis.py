"""Analysis model: weight-compatible with the trained Transformer, but its MoE
returns per-token routing information and supports causal interventions
(expert ablation, shared-expert ablation).

Reuses RMSNorm / RoPE / MLA from the training model unchanged, so a checkpoint
saved by `training.model_train.Transformer` loads directly (strict=False to skip
non-persistent buffers).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.model_train import RMSNorm, RoPE, MLA


@dataclass
class RoutingCapture:
    """Per-layer, per-token routing info collected during a forward pass."""
    topk_indices: list = field(default_factory=list)   # each [N, K] long
    topk_weights: list = field(default_factory=list)   # each [N, K] float
    shared_norm: list = field(default_factory=list)    # each [N] float
    routed_norm: list = field(default_factory=list)    # each [N] float


class AnalysisMoE(nn.Module):
    """MoE forward without capacity dropping (inference-equivalent), instrumented
    for routing capture and interventions."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.use_shared_expert = getattr(config, "use_shared_expert", True)

        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.w13 = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.intermediate_size))
        self.w2 = nn.Parameter(torch.empty(self.num_experts, self.intermediate_size, self.hidden_size))
        if self.use_shared_expert:
            self.shared_w13 = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
            self.shared_w2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    @torch.inference_mode()
    def forward(self, x, ablate_experts: Optional[Iterable[int]] = None,
                ablate_shared: bool = False, ablate_routed: bool = False,
                capture: Optional[RoutingCapture] = None):
        bsz, seq_len, hidden = x.shape
        x_flat = x.reshape(-1, hidden)
        N = x_flat.shape[0]
        E, K = self.num_experts, self.num_experts_per_token

        # shared expert
        if self.use_shared_expert and not ablate_shared:
            g = self.shared_w13(x_flat)
            gs, us = g.chunk(2, dim=-1)
            shared_out = self.shared_w2(F.silu(gs) * us)
        else:
            shared_out = x_flat.new_zeros((N, hidden))

        # whole routed branch removed (fair like-for-like vs shared-branch removal)
        if ablate_routed and capture is None:
            return shared_out.view(bsz, seq_len, hidden)

        # router
        router_logits = self.router(x_flat).to(torch.float32)  # [N, E]
        if ablate_experts:
            idx = torch.tensor(list(ablate_experts), device=x.device, dtype=torch.long)
            router_logits[:, idx] = float("-inf")

        topk_logits, topk_indices = torch.topk(router_logits, K, dim=-1)  # [N, K]
        topk_weights = F.softmax(topk_logits, dim=-1)

        expert_ids = topk_indices.reshape(-1)
        token_ids = torch.arange(N, device=x.device).unsqueeze(1).expand(N, K).reshape(-1)
        gates = topk_weights.reshape(-1).to(x_flat.dtype)

        expert_ids, sort_perm = torch.sort(expert_ids)
        token_ids = token_ids[sort_perm]
        gates = gates[sort_perm]

        counts = torch.bincount(expert_ids, minlength=E)
        max_count = int(counts.max().item()) if N > 0 else 0
        valid_mask = torch.arange(max_count, device=x.device).unsqueeze(0) < counts.unsqueeze(1)

        packed = x_flat.new_zeros((E, max_count, hidden))
        packed[valid_mask] = x_flat[token_ids]
        proj = torch.matmul(packed, self.w13)
        ga, up = proj.chunk(2, dim=-1)
        packed_out = torch.matmul(F.silu(ga) * up, self.w2)
        valid_out = packed_out[valid_mask]

        routed_out = x_flat.new_zeros((N, hidden))
        routed_out.index_add_(0, token_ids, valid_out * gates.unsqueeze(-1))

        if capture is not None:
            capture.topk_indices.append(topk_indices.detach().to("cpu"))
            capture.topk_weights.append(topk_weights.detach().to("cpu"))
            capture.shared_norm.append(shared_out.detach().float().norm(dim=-1).cpu())
            capture.routed_norm.append(routed_out.detach().float().norm(dim=-1).cpu())

        output = (shared_out + routed_out).view(bsz, seq_len, hidden)
        return output


class AnalysisBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config)
        self.ffn_norm = RMSNorm(config)
        self.attn = MLA(config)
        self.moe = AnalysisMoE(config)
        self.ls1 = nn.Parameter(torch.ones(1) * 0.1)
        self.ls2 = nn.Parameter(torch.ones(1) * 0.1)

    @torch.inference_mode()
    def forward(self, x, position_ids=None, ablate_experts=None, ablate_shared=False,
                ablate_routed=False, capture=None):
        h = x + self.ls1 * self.attn(self.attn_norm(x), position_ids)
        moe_out = self.moe(self.ffn_norm(h), ablate_experts=ablate_experts,
                           ablate_shared=ablate_shared, ablate_routed=ablate_routed, capture=capture)
        return h + self.ls2 * moe_out


class AnalysisTransformer(nn.Module):
    """Weight-compatible with training.model_train.Transformer."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([AnalysisBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    @torch.inference_mode()
    def forward(self, input_ids, position_ids=None,
                ablate_layer_experts=None,   # dict[layer_idx] -> iterable[int]
                ablate_shared=False,
                ablate_routed=False,
                capture_routing=False):
        x = self.embed_tokens(input_ids)
        captures = [] if capture_routing else None
        for li, layer in enumerate(self.layers):
            cap = RoutingCapture() if capture_routing else None
            ael = ablate_layer_experts.get(li) if ablate_layer_experts else None
            x = layer(x, position_ids, ablate_experts=ael, ablate_shared=ablate_shared,
                      ablate_routed=ablate_routed, capture=cap)
            if capture_routing:
                captures.append(cap)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, captures


def load_analysis_model(config, checkpoint_path, device="cuda"):
    """Load a safetensors checkpoint (saved by the trainer) into the analysis model."""
    from safetensors.torch import load_file
    model = AnalysisTransformer(config).to(device).eval()
    state = load_file(checkpoint_path)
    cleaned = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    # Only non-persistent RoPE buffers should be "missing"; report anything else.
    real_missing = [m for m in missing if "freqs_cis" not in m and "inv_freq" not in m]
    if real_missing:
        print(f"[load] WARNING missing keys: {real_missing}")
    if unexpected:
        print(f"[load] WARNING unexpected keys: {unexpected}")
    return model
