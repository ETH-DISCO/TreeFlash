from __future__ import annotations

import heapq
import time
from types import SimpleNamespace
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import DynamicCache
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)
from typing_extensions import Tuple, Unpack


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [int(round(start + (i * span) / (num_draft_layers - 1))) for i in range(num_draft_layers)]


def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]],
) -> torch.Tensor:
    offset = 1
    selected_states = [hidden_states[layer_id + offset] for layer_id in layer_ids]
    return torch.cat(selected_states, dim=-1)


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size) / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def _cuda_time() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3DFlashAttention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3DFlashBackbone(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [Qwen3DFlashDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.target_layer_ids = self.config.dflash_config.get(
            "target_layer_ids",
            build_target_layer_ids(config.num_target_layers, config.num_hidden_layers),
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = config.block_size
        self.mask_token_id = self.config.dflash_config.get("mask_token_id", None)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


def _treeflash_config(config: Qwen3Config) -> dict[str, Any]:
    value = getattr(config, "treeflash_config", None)
    return value if isinstance(value, dict) else {}


def _candidate_tokens(config: Qwen3Config) -> int:
    treeflash_config = _treeflash_config(config)
    return int(treeflash_config.get("candidate_tokens", getattr(config, "candidate_tokens", 16)))


def _ar_approximation_name(config: Qwen3Config) -> str | None:
    treeflash_config = _treeflash_config(config)
    return treeflash_config.get("ar_approximation", getattr(config, "ar_approximation", None))




class SwiGLUApproximation(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.ug = nn.Linear(hidden_size * 2, 2 * intermediate_size)
        self.d = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.d.weight.data.zero_()
        self.norm = Qwen3RMSNorm(hidden_size, eps=1e-6)

    def forward(self, prev_token_embds: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        prev_token_embds = self.norm(prev_token_embds)
        if prev_token_embds.shape != hidden_states.shape:
            raise ValueError("Expected previous-token embeddings and hidden states to have identical shape.")
        input_hs = torch.cat([prev_token_embds, hidden_states], dim=-1)
        u, g = self.ug(input_hs).chunk(2, dim=-1)
        return self.d(F.silu(g) * u)


def _build_candidate_tree_from_transition_tables(
    *,
    transition_tokens: torch.Tensor,
    transition_log_probs: torch.Tensor,
    transition_next_idx: torch.Tensor,
    block_size: int,
    tree_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tree_size = max(int(tree_size), 1)
    topk_candidates = int(transition_tokens.shape[-1])
    transition_tokens_np = transition_tokens.detach().to(device="cpu", dtype=torch.long).numpy()
    transition_log_probs_np = transition_log_probs.detach().to(device="cpu", dtype=torch.float32).numpy()
    transition_next_idx_np = transition_next_idx.detach().to(device="cpu", dtype=torch.long).numpy()

    candidate_tokens_np = np.zeros(tree_size, dtype=np.int64)
    candidate_depths_np = np.zeros(tree_size, dtype=np.int64)
    candidate_mask_np = np.zeros((tree_size, tree_size), dtype=np.bool_)
    candidate_mask_np[0, 0] = True

    heap: list[tuple[float, tuple[int, ...], int, int, int, float, int, int]] = []
    if tree_size > 1 and block_size > 1 and topk_candidates > 0:
        first_logw = float(transition_log_probs_np[0, 0, 0])
        if np.isfinite(first_logw):
            heap = [(-first_logw, (0,), 0, 1, 0, first_logw, 0, 0)]

    for node_index in range(1, tree_size):
        if not heap:
            break
        (
            _neg_logw,
            ranks,
            parent_index,
            depth,
            rank,
            logw,
            source_depth_idx,
            source_regular_rank,
        ) = heapq.heappop(heap)

        token_id = int(transition_tokens_np[source_depth_idx, source_regular_rank, rank])
        next_regular_rank = int(transition_next_idx_np[source_depth_idx, source_regular_rank, rank])
        candidate_tokens_np[node_index] = token_id
        candidate_depths_np[node_index] = depth
        candidate_mask_np[node_index] = candidate_mask_np[parent_index]
        candidate_mask_np[node_index, node_index] = True

        if rank + 1 < topk_candidates:
            local_log_prob = float(transition_log_probs_np[source_depth_idx, source_regular_rank, rank])
            sibling_rank = rank + 1
            sibling_logw = (
                logw
                - local_log_prob
                + float(transition_log_probs_np[source_depth_idx, source_regular_rank, sibling_rank])
            )
            if np.isfinite(sibling_logw):
                heapq.heappush(
                    heap,
                    (
                        -sibling_logw,
                        ranks[:-1] + (sibling_rank,),
                        parent_index,
                        depth,
                        sibling_rank,
                        sibling_logw,
                        source_depth_idx,
                        source_regular_rank,
                    ),
                )

        child_depth = depth + 1
        if child_depth < block_size and next_regular_rank != -1:
            child_source_depth_idx = child_depth - 1
            child_source_regular_rank = next_regular_rank
            child_logw = logw + float(transition_log_probs_np[child_source_depth_idx, child_source_regular_rank, 0])
            if np.isfinite(child_logw):
                heapq.heappush(
                    heap,
                    (
                        -child_logw,
                        ranks + (0,),
                        node_index,
                        child_depth,
                        0,
                        child_logw,
                        child_source_depth_idx,
                        child_source_regular_rank,
                    ),
                )

    candidate_tokens = torch.from_numpy(candidate_tokens_np).unsqueeze(0).to(device)
    candidate_depths = torch.from_numpy(candidate_depths_np).unsqueeze(0).to(device)
    candidate_mask = torch.from_numpy(candidate_mask_np).unsqueeze(0).to(device)
    return candidate_tokens, candidate_mask, candidate_depths


def _compute_tree_acceptance(
    draft_ids: torch.Tensor,
    target_ids: torch.Tensor,
    tree_mask: torch.Tensor,
) -> tuple[int, list[int]]:
    if draft_ids.shape[0] != 1:
        raise ValueError("TreeFlash speculative generation currently supports batch size 1.")
    ancestor_mask = tree_mask & ~torch.eye(
        tree_mask.shape[1],
        device=tree_mask.device,
        dtype=torch.bool,
    )[None, :, :]
    parent_indices = (
        ancestor_mask
        * torch.arange(tree_mask.shape[1], device=tree_mask.device)[None, None, :]
    ).max(dim=-1).values
    depths = tree_mask.sum(dim=-1)
    parent_target = target_ids[:, parent_indices[0, 1:]]
    equality = torch.cat(
        [
            torch.ones((draft_ids.shape[0], 1), device=draft_ids.device, dtype=torch.bool),
            draft_ids == parent_target,
        ],
        dim=1,
    )
    acceptance_mask = (equality[:, None, :] | ~tree_mask).all(dim=2)
    acceptance_lengths = acceptance_mask.to(torch.float32) * depths
    best_node = acceptance_lengths.argmax(dim=1)
    acceptance_length = int(acceptance_lengths[0, best_node[0]].item())
    accepted_nodes = tree_mask[0, best_node[0]].nonzero(as_tuple=True)[0].tolist()
    return acceptance_length, accepted_nodes


def _compact_appended_window(cache_tensor: torch.Tensor, past_length: int, keep_current_indices: torch.Tensor) -> None:
    current_length = cache_tensor.shape[-2] - past_length
    if current_length <= 0:
        return
    keep_count = int(keep_current_indices.numel())
    if keep_count == 0 or keep_count == current_length:
        return
    kept_tail = cache_tensor.narrow(-2, past_length, current_length).index_select(-2, keep_current_indices)
    cache_tensor.narrow(-2, past_length, keep_count).copy_(kept_tail)


def _compact_dynamic_cache_tail(
    past_key_values: DynamicCache,
    past_length: int,
    keep_current_indices: list[int],
) -> None:
    if len(keep_current_indices) == 0:
        past_key_values.crop(past_length)
        return

    keep_tensor_by_device: dict[torch.device, torch.Tensor] = {}

    def get_keep_tensor(device: torch.device) -> torch.Tensor:
        if device not in keep_tensor_by_device:
            keep_tensor_by_device[device] = torch.tensor(keep_current_indices, dtype=torch.long, device=device)
        return keep_tensor_by_device[device]

    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        for key_cache, value_cache in zip(past_key_values.key_cache, past_key_values.value_cache, strict=False):
            if key_cache is None or key_cache.numel() == 0:
                continue
            keep_tensor = get_keep_tensor(key_cache.device)
            _compact_appended_window(key_cache, past_length, keep_tensor)
            _compact_appended_window(value_cache, past_length, keep_tensor)
        past_key_values.crop(past_length + len(keep_current_indices))
        return

    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            if not hasattr(layer, "keys") or layer.keys is None or layer.keys.numel() == 0:
                continue
            keep_tensor = get_keep_tensor(layer.keys.device)
            _compact_appended_window(layer.keys, past_length, keep_tensor)
            _compact_appended_window(layer.values, past_length, keep_tensor)
        past_key_values.crop(past_length + len(keep_current_indices))
        return

    raise RuntimeError("Unsupported DynamicCache layout for TreeFlash cache compaction.")


def _compile_tree_attention_mask(
    *,
    candidate_mask: torch.Tensor,
    past_length: int,
    dtype: torch.dtype,
    attention_mask_buffer: torch.Tensor,
    previous_tree_start: int,
    previous_tree_length: int,
) -> tuple[torch.Tensor, int, int]:
    tree_length = int(candidate_mask.shape[1])
    if previous_tree_length > 0:
        attention_mask_buffer[
            0,
            0,
            :previous_tree_length,
            previous_tree_start : previous_tree_start + previous_tree_length,
        ] = 0

    tree_block = attention_mask_buffer[0, 0, :tree_length, past_length : past_length + tree_length]
    tree_block.fill_(torch.finfo(dtype).min)
    tree_block.masked_fill_(candidate_mask[0], 0)
    attention_mask = attention_mask_buffer[:, :, :tree_length, : past_length + tree_length]
    return attention_mask, past_length, tree_length


class TreeFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__(config)
        self.model = Qwen3DFlashBackbone(config)
        self.candidate_tokens = _candidate_tokens(config)
        self.ar_method = _ar_approximation_name(config)
        self.ar_approximation: SwiGLUApproximation | None = None
        if self.ar_method == "swiglu":
            self.ar_approximation = SwiGLUApproximation(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
            )
        elif self.ar_method is not None:
            raise ValueError(
                "Expected `treeflash_config.ar_approximation` to be `swiglu` or null, "
                f"got {self.ar_method!r}."
            )
        self.post_init()

    @property
    def device(self) -> torch.device:
        return self.model.device

    @property
    def target_layer_ids(self) -> list[int]:
        return self.model.target_layer_ids

    @property
    def mask_token_id(self) -> int:
        return self.model.mask_token_id

    @property
    def block_size(self) -> int:
        return int(self.model.block_size)

    def forward(
        self,
        position_ids: torch.LongTensor,
        prev_token_embds: torch.Tensor | None = None,
        noise_embedding: torch.Tensor | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden_states = noise_embedding
        target_hidden = self.model.hidden_norm(self.model.fc(target_hidden))
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        for layer in self.model.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        normed_hs = self.model.norm(hidden_states)
        if self.ar_approximation is not None:
            if prev_token_embds is None:
                raise ValueError("`prev_token_embds` is required when the TreeFlash AR approximation is enabled.")
            hidden_states = hidden_states + self.ar_approximation(
                prev_token_embds=prev_token_embds,
                hidden_states=normed_hs,
            )
            return self.model.norm(hidden_states)
        return normed_hs

    @torch.no_grad()
    def get_candidate_trees(
        self,
        position_ids: torch.LongTensor,
        noise_embedding: torch.Tensor,
        embed_tokens,
        lm_head,
        attention_mask: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        m_initial_token_embds: int | None = None,
        topk_candidates: int = 16,
        temperature: float = 1.0,
        is_chain: bool = False,
        tree_size: int | None = None,
        **kwargs: Any,
    ):
        removed_feature_args = {
            "oracle_prev_token_embds",
            "return_proposal_probs",
            "return_timings",
            "root_history_token_ids",
        }
        unsupported_args = sorted(arg_name for arg_name in removed_feature_args if arg_name in kwargs)
        if unsupported_args:
            unsupported_args_str = ", ".join(unsupported_args)
            raise TypeError(f"Unsupported TreeFlash candidate-tree arguments: {unsupported_args_str}.")

        hidden_states = noise_embedding
        target_hidden = self.model.hidden_norm(self.model.fc(target_hidden))
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        for layer in self.model.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        normed_hs = self.model.norm(hidden_states)
        logits = lm_head(normed_hs)
        block_size = normed_hs.shape[1]
        topk_candidates = min(int(topk_candidates), int(logits.shape[-1]))
        if m_initial_token_embds is None:
            m_initial_token_embds = 16 
        m_initial_token_embds = min(int(m_initial_token_embds), int(logits.shape[-1]))
        temperature = max(float(temperature), 1e-5)

        if self.ar_approximation is None:
            log_probs = F.log_softmax(logits.float() / temperature, dim=-1)
            log_probs_topk = log_probs.topk(topk_candidates, dim=-1)
            transition_tokens = log_probs_topk.indices[0, 1:, None, :]
            transition_log_probs = log_probs_topk.values[0, 1:, None, :]
            transition_next_idx = torch.zeros_like(transition_tokens)
        else:
            regular_top_preds = logits[:, :-1, :].topk(m_initial_token_embds, dim=-1).indices
            prev_token_embds = embed_tokens(regular_top_preds.view(1, -1)).view(
                1,
                block_size - 1,
                m_initial_token_embds,
                -1,
            )
            prev_token_embds[:, 0] = noise_embedding[:, 0]
            hidden_states_expanded = hidden_states[:, 1:, None, :].expand(
                -1,
                -1,
                prev_token_embds.shape[2],
                -1,
            )
            normed_hs_expanded = normed_hs[:, 1:, None, :].expand(
                -1,
                -1,
                prev_token_embds.shape[2],
                -1,
            )
            corrected_hidden = hidden_states_expanded + self.ar_approximation(
                prev_token_embds=prev_token_embds,
                hidden_states=normed_hs_expanded,
            )
            corrected_normed_hs = self.model.norm(corrected_hidden)
            ar_logits = lm_head(corrected_normed_hs)
            ar_log_probs = F.log_softmax(ar_logits.float() / temperature, dim=-1)
            ar_log_probs_topk = ar_log_probs.topk(topk_candidates, dim=-1)
            transition_tokens = ar_log_probs_topk.indices[0]
            transition_log_probs = ar_log_probs_topk.values[0]
            matches = (
                transition_tokens[:-1, :, :, None]
                == regular_top_preds[0, 1:, None, None, :]
            ).cpu()
            transition_next_idx = torch.cat(
                (
                    torch.where(
                        matches.any(dim=-1),
                        matches.float().argmax(dim=-1),
                        torch.full_like(matches.any(dim=-1), fill_value=-1, dtype=torch.long),
                    ),
                    torch.full(
                        (1, transition_tokens.shape[1], transition_tokens.shape[2]),
                        fill_value=-1,
                        dtype=torch.long,
                    ),
                ),
                dim=0,
            )

        device = noise_embedding.device
        if is_chain:
            candidate_tokens = torch.cat(
                (
                    torch.zeros((1, 1), dtype=torch.long, device=transition_tokens.device),
                    transition_tokens[None, :, 0, 0],
                ),
                dim=-1,
            ).to(device)
            candidate_depth = torch.arange(block_size, device=device)[None, :]
            candidate_mask = candidate_depth[:, :, None] >= candidate_depth[:, None, :]
            return candidate_tokens, candidate_mask, candidate_depth

        tree_size = tree_size or self.candidate_tokens
        return _build_candidate_tree_from_transition_tables(
            transition_tokens=transition_tokens,
            transition_log_probs=transition_log_probs,
            transition_next_idx=transition_next_idx,
            block_size=block_size,
            tree_size=tree_size,
            device=device,
        )

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int] | None = None,
        temperature: float = 0.0,
        drafter_temperature: float = 1.0,
        top_m: int | None = None,
        top_k: int | None = None,
        tree_size: int | None = None,
        is_chain: bool = False,
        is_vanilla: bool = False,
        return_stats: bool = False,
        token_callback: Callable[[torch.Tensor], None] | None = None,
        **kwargs: Any,
    ):
        self.eval()
        target.eval()
        dflash_block_size = 1 if is_vanilla else self.block_size
        if dflash_block_size <= 1 and not is_vanilla:
            raise ValueError("Expected block size > 1 for TreeFlash decoding.")

        removed_feature_args = {
            "benchmark_parts",
            "collect_trace",
            "oracle_ids",
            "oracle_token_ids",
            "use_specinfer_acceptance",
        }
        unsupported_args = sorted(arg_name for arg_name in removed_feature_args if arg_name in kwargs)
        if unsupported_args:
            unsupported_args_str = ", ".join(unsupported_args)
            raise TypeError(f"Unsupported TreeFlash generation arguments: {unsupported_args_str}.")

        old_attn = getattr(target.config, "_attn_implementation", None)
        set_attn_implementation = getattr(target, "set_attn_implementation", None)
        if callable(set_attn_implementation) and old_attn is not None:
            set_attn_implementation("sdpa")

        try:
            device = self.device
            input_ids = input_ids.to(device)
            num_input_tokens = input_ids.shape[1]
            max_length = num_input_tokens + int(max_new_tokens)
            candidate_buffer_size = int(tree_size) if tree_size is not None else self.candidate_tokens

            output_ids = torch.full(
                (1, max_length + max(dflash_block_size, candidate_buffer_size)),
                self.mask_token_id,
                dtype=torch.long,
                device=device,
            )
            position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)
            past_key_values_target = DynamicCache()
            past_key_values_draft = DynamicCache()
            attention_dtype = getattr(target, "dtype", None)
            if attention_dtype is None or not attention_dtype.is_floating_point:
                attention_dtype = target.model.embed_tokens.weight.dtype
            attention_mask_buffer = torch.zeros(
                (1, 1, candidate_buffer_size, max_length + candidate_buffer_size),
                dtype=attention_dtype,
                device=device,
            )

            prefill_start = _cuda_time()
            output = target(
                input_ids,
                position_ids=position_ids[:, :num_input_tokens],
                past_key_values=past_key_values_target,
                use_cache=True,
                logits_to_keep=1,
                output_hidden_states=True,
            )
            output_ids[:, :num_input_tokens] = input_ids
            output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature)
            first_token_ids = output_ids[:, num_input_tokens : num_input_tokens + 1]
            if token_callback is not None:
                token_callback(first_token_ids)
            target_hidden = extract_context_feature(output.hidden_states, self.target_layer_ids)
            time_to_first_token = _cuda_time() - prefill_start

            decode_start = _cuda_time()
            start = num_input_tokens
            stopped_after_prefill = False
            if stop_token_ids is not None:
                stop_token_ids_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
                stopped_after_prefill = bool(torch.isin(first_token_ids[0], stop_token_ids_tensor).any().item())
            acceptance_lengths: list[int] = []
            draft_prefill = True
            previous_tree_start = 0
            previous_tree_length = 0

            while start < max_length and not stopped_after_prefill:
                step_start = start
                drafter_block_output_ids = output_ids[:, start : start + dflash_block_size].clone()
                if is_vanilla:
                    candidate_tokens = output_ids[:, start : start + 1].clone()
                    candidate_position_ids = position_ids[:, start : start + 1].clone()
                    attention_mask = None
                else:
                    noise_embedding = target.model.embed_tokens(drafter_block_output_ids)
                    draft_position_ids = position_ids[
                        :,
                        past_key_values_draft.get_seq_length() : step_start + dflash_block_size,
                    ]
                    candidate_tree_kwargs: dict[str, Any] = {
                        "temperature": drafter_temperature,
                        "is_chain": is_chain,
                    }
                    if top_m is not None:
                        candidate_tree_kwargs["m_initial_token_embds"] = int(top_m)
                    if top_k is not None:
                        candidate_tree_kwargs["topk_candidates"] = int(top_k)
                    if tree_size is not None:
                        candidate_tree_kwargs["tree_size"] = int(tree_size)
                    candidate_tree_kwargs.update(kwargs)
                    candidate_tokens, candidate_mask, candidate_depths = self.get_candidate_trees(
                        target_hidden=target_hidden,
                        lm_head=target.lm_head,
                        embed_tokens=target.model.embed_tokens,
                        noise_embedding=noise_embedding,
                        position_ids=draft_position_ids,
                        past_key_values=past_key_values_draft,
                        use_cache=True,
                        is_causal=False,
                        **candidate_tree_kwargs,
                    )
                    past_key_values_draft.crop(step_start)
                    candidate_position_ids = candidate_depths + step_start
                    candidate_tokens[:, 0] = drafter_block_output_ids[:, 0]
                    if draft_prefill:
                        draft_prefill = False
                        decode_start = _cuda_time()
                    attention_mask, previous_tree_start, previous_tree_length = _compile_tree_attention_mask(
                        candidate_mask=candidate_mask,
                        past_length=step_start,
                        dtype=attention_dtype,
                        attention_mask_buffer=attention_mask_buffer,
                        previous_tree_start=previous_tree_start,
                        previous_tree_length=previous_tree_length,
                    )

                output = target(
                    candidate_tokens,
                    position_ids=candidate_position_ids,
                    past_key_values=past_key_values_target,
                    use_cache=True,
                    output_hidden_states=True,
                    attention_mask=attention_mask,
                )

                posterior = sample(output.logits, temperature)
                if is_vanilla:
                    if step_start + 1 >= max_length:
                        break
                    output_ids[:, step_start + 1] = posterior[:, 0]
                    emitted_ids = output_ids[:, step_start + 1 : step_start + 2]
                    start += 1
                else:
                    acceptance_length, accepted_nodes = _compute_tree_acceptance(
                        candidate_tokens[:, 1:],
                        posterior[:, :-1],
                        candidate_mask,
                    )
                    residual_token = posterior[:, accepted_nodes[-1]]
                    output_ids[:, step_start : step_start + acceptance_length] = candidate_tokens[:, accepted_nodes]
                    output_ids[:, step_start + acceptance_length] = residual_token
                    emitted_ids = output_ids[:, step_start + 1 : step_start + acceptance_length + 1]
                    acceptance_lengths.append(acceptance_length)
                    _compact_dynamic_cache_tail(past_key_values_target, step_start, accepted_nodes)
                    start = step_start + acceptance_length
                    target_hidden = extract_context_feature(output.hidden_states, self.target_layer_ids)[
                        :,
                        accepted_nodes,
                        :,
                    ]

                if stop_token_ids is not None:
                    stop_token_ids_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
                    stop_token_indices = torch.isin(emitted_ids[0], stop_token_ids_tensor).nonzero(as_tuple=True)[0]
                    if stop_token_indices.numel() > 0:
                        first_stop_index = int(stop_token_indices[0].item())
                        if token_callback is not None:
                            token_callback(emitted_ids[:, : first_stop_index + 1])
                        break
                if token_callback is not None and emitted_ids.numel() > 0:
                    token_callback(emitted_ids)

                if stop_token_ids is not None and any(
                    stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
                ):
                    break

            output_ids = output_ids[:, :max_length]
            output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
            if stop_token_ids is not None:
                stop_token_ids_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
                stop_token_indices = torch.isin(
                    output_ids[0][num_input_tokens:],
                    stop_token_ids_tensor,
                ).nonzero(as_tuple=True)[0]
                if stop_token_indices.numel() > 0:
                    output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

            if not return_stats:
                return output_ids

            num_output_tokens = output_ids.shape[1] - num_input_tokens
            total_decode_time = _cuda_time() - decode_start
            time_per_output_token = total_decode_time / max(num_output_tokens, 1)
            return SimpleNamespace(
                output_ids=output_ids,
                normal_output_ids=output_ids,
                num_input_tokens=num_input_tokens,
                num_output_tokens=num_output_tokens,
                time_to_first_token=time_to_first_token,
                time_per_output_token=time_per_output_token,
                acceptance_lengths=acceptance_lengths,
            )
        finally:
            if callable(set_attn_implementation) and old_attn is not None:
                set_attn_implementation(old_attn)


TreeFlash = TreeFlashDraftModel
