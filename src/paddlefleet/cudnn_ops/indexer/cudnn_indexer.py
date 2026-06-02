# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""cuDNN DSA indexer forward via DLPack zero-copy bridge.

Calls cudnn.DSA.indexer_forward_wrapper (CuTe-DSL score kernel, SM100) and
cudnn.DSA.indexer_top_k_wrapper (TRT-LLM radix top-K, SM100) on Paddle tensors.

Public API:
    cudnn_indexer_topk_fwd  — combined score + top-K (main entry point)
    cudnn_indexer_forward   — score only
    cudnn_indexer_topk      — top-K on precomputed scores
"""

import paddle


# =========================================================================
# Lazy imports
# =========================================================================

_DSA = None
_torch = None


def _ensure_cudnn_dsa():
    global _DSA, _torch
    if _DSA is not None:
        return
    import torch

    _torch = torch
    from cudnn import DSA

    _DSA = DSA


# =========================================================================
# DLPack bridge
# =========================================================================


def _paddle_to_torch(x):
    """Zero-copy Paddle GPU tensor -> torch tensor via DLPack."""
    _ensure_cudnn_dsa()
    return _torch.utils.dlpack.from_dlpack(x)


def _torch_to_paddle(t):
    """Zero-copy torch GPU tensor -> Paddle tensor via DLPack."""
    return paddle.utils.dlpack.from_dlpack(t)


# =========================================================================
# Input validation (mirrors csa_indexer.py style)
# =========================================================================


def _validate_indexer_inputs(index_q, index_k_comp, weights):
    if not isinstance(index_q, paddle.Tensor):
        raise TypeError(
            f"index_q must be a paddle.Tensor, got {type(index_q)!r}"
        )
    if not isinstance(index_k_comp, paddle.Tensor):
        raise TypeError(
            f"index_k_comp must be a paddle.Tensor, got {type(index_k_comp)!r}"
        )
    if not isinstance(weights, paddle.Tensor):
        raise TypeError(
            f"weights must be a paddle.Tensor, got {type(weights)!r}"
        )
    if len(index_q.shape) != 4:
        raise ValueError(
            f"index_q must have shape [B, S, H_i, D_i], got {index_q.shape}"
        )
    if len(index_k_comp.shape) != 3:
        raise ValueError(
            f"index_k_comp must have shape [B, S_comp, D_i], got {index_k_comp.shape}"
        )
    if len(weights.shape) != 3:
        raise ValueError(
            f"weights must have shape [B, S, H_i], got {weights.shape}"
        )

    batch, seq_len, heads, dim = index_q.shape
    batch_k, _, dim_k = index_k_comp.shape
    batch_w, seq_len_w, heads_w = weights.shape
    if batch != batch_k or batch != batch_w:
        raise ValueError(
            f"batch mismatch: index_q={index_q.shape}, "
            f"index_k_comp={index_k_comp.shape}, weights={weights.shape}"
        )
    if seq_len != seq_len_w or heads != heads_w or dim != dim_k:
        raise ValueError(
            f"shape mismatch: index_q={index_q.shape}, "
            f"index_k_comp={index_k_comp.shape}, weights={weights.shape}"
        )
    if heads not in (32, 64):
        raise ValueError(
            f"cuDNN IndexerForward requires H_i (qhead_per_kv_head) in {{32, 64}}, got {heads}"
        )
    if dim != 128:
        raise ValueError(
            f"cuDNN IndexerForward requires D_i=128, got {dim}"
        )


# =========================================================================
# Core API
# =========================================================================


def cudnn_indexer_forward(index_q, index_k_comp, weights, ratio=4):
    """Compute indexer scores using cuDNN CuTe-DSL kernel (SM100).

    Args:
        index_q: [B, S_q, H_i, D_i] bf16. Indexer queries (RoPE'd + rotated).
        index_k_comp: [B, S_k, D_i] bf16. Compressed indexer keys.
        weights: [B, S_q, H_i] bf16. Per-head weights (already scaled).
        ratio: compression ratio for the causal mask.

    Returns:
        scores: [B, S_q, S_k] fp32 Paddle tensor. Masked positions are -inf.
    """
    _ensure_cudnn_dsa()
    q_t = _paddle_to_torch(index_q.contiguous())
    k_4d = index_k_comp.unsqueeze(2)  # [B, S_k, 1, D]
    k_t = _paddle_to_torch(k_4d.contiguous())
    w_t = _paddle_to_torch(weights.contiguous())
    result = _DSA.indexer_forward_wrapper(q_t, k_t, w_t, ratio=int(ratio))
    scores_t = result["scores"]
    scores = _torch_to_paddle(scores_t)

    return scores


def cudnn_indexer_topk(scores, sq, ratio, topk):
    """Select top-K indices using cuDNN TRT-LLM radix kernel (SM100).

    Args:
        scores: [B, S_q, S_k] fp32 Paddle tensor (from cudnn_indexer_forward).
        sq: query sequence length.
        ratio: compression ratio.
        topk: number of top-k entries to select per query position.

    Returns:
        topk_indices: [B, S_q, topk] int32, invalid slots are -1.
        topk_length: [B, S_q] int32, per-row valid count.
    """
    _ensure_cudnn_dsa()
    b = int(scores.shape[0])
    sk = int(scores.shape[2])
    n_rows = b * int(sq)
    topk_k = min(int(topk), sk)

    scores_flat = scores.reshape([n_rows, sk])
    scores_flat_t = _paddle_to_torch(scores_flat.contiguous())

    q_idx = _torch.arange(int(sq), device=scores_flat_t.device)
    valid_per_q = ((q_idx + 1) // int(ratio)).clamp(max=sk).to(_torch.int32)
    seq_lens = valid_per_q.repeat(b)

    tk_result = _DSA.indexer_top_k_wrapper(
        scores_flat_t, seq_lens, top_k=topk_k, next_n=1, return_val=False
    )
    topk_indices_t = tk_result["indices"]
    topk_indices = _torch_to_paddle(topk_indices_t).reshape(
        [b, int(sq), topk_k]
    )

    if topk_k < int(topk):
        pad_shape = [b, int(sq), int(topk) - topk_k]
        padding = paddle.full(pad_shape, -1, dtype="int32")
        topk_indices = paddle.concat([topk_indices, padding], axis=-1)

    topk_length = (topk_indices >= 0).sum(axis=-1).cast("int32")
    return topk_indices.cast("int32"), topk_length


def cudnn_indexer_topk_fwd(
    index_q,
    index_k_comp,
    weights,
    ratio=4,
    topk_effective=64,
    indexer_softmax_scale=1.0,
):
    """Paddle entry for cuDNN DSA compressed indexer forward.

    Mirrors the semantics of Megatron's ``indexer_topk`` and TileLang's
    ``csa_indexer_topk_fwd`` but uses cuDNN CuTe-DSL + TRT-LLM radix kernels
    through DLPack zero-copy bridge.

    Args:
        index_q: [B, S, H_i, D_i] bf16. Indexer queries.
        index_k_comp: [B, S_comp, D_i] bf16. Compressed indexer keys.
        weights: [B, S, H_i] bf16. Per-head weights (scaled by n_heads**-0.5).
        ratio: compression ratio (e.g. 4).
        topk_effective: number of top-k entries to select per query position.
        indexer_softmax_scale: additional scale on weights (default 1.0).

    Returns:
        topk_indices: [B, S, topk_effective] int32, invalid slots are -1.
        topk_length: [B, S] int32, per-row valid count.
    """
    _validate_indexer_inputs(index_q, index_k_comp, weights)
    if int(topk_effective) <= 0:
        raise ValueError(
            f"topk_effective must be positive, got {topk_effective}"
        )

    sq = int(index_q.shape[1])

    if float(indexer_softmax_scale) != 1.0:
        w = (weights.cast("float32") * float(indexer_softmax_scale)).cast(
            weights.dtype
        )
    else:
        w = weights

    scores = cudnn_indexer_forward(index_q, index_k_comp, w, ratio=int(ratio))
    topk_indices, topk_length = cudnn_indexer_topk(
        scores, sq, int(ratio), int(topk_effective)
    )
    return topk_indices, topk_length
