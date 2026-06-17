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

"""Paddle wrapper around the cuDNN-frontend DSA indexer forward.

Calls ``paddlefleet_ops.cudnn.deepseek_sparse_attention.indexer_forward.api
.indexer_forward_wrapper`` and ``paddlefleet_ops.cudnn.deepseek_sparse_attention
.indexer_top_k.api.indexer_top_k_wrapper`` directly on Paddle tensors.

Returns selected compressed KV indices and per-row valid counts.
"""

from __future__ import annotations

import paddle
from paddlefleet_ops import CUDNN_FRONTEND_HINT, is_cudnn_frontend_available


def _require_cudnn_frontend():
    if not is_cudnn_frontend_available():
        raise ImportError(CUDNN_FRONTEND_HINT)


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
        raise ValueError(f"cuDNN IndexerForward requires D_i=128, got {dim}")


def cudnn_indexer_forward(
    index_q, index_k_comp, weights, ratio=4, sm_scale=None
):
    """Compute indexer scores using cuDNN CuTe-DSL kernel (SM100).

    Args:
        index_q:       [B, S_q, H_i, D_i] bf16, indexer queries.
        index_k_comp:  [B, S_k, D_i] bf16, compressed indexer keys.
        weights:       [B, S_q, H_i] bf16, per-head weights.
        ratio:         compression ratio for the causal mask.
        sm_scale:      scale factor applied to QK scores (default: dim**-0.5).

    Returns:
        scores: [B, S_q, S_k] fp32 Paddle tensor. Masked positions are -inf.
    """
    if sm_scale is None:
        sm_scale = float(index_q.shape[-1]) ** -0.5
    _require_cudnn_frontend()
    from paddlefleet_ops.cudnn.deepseek_sparse_attention.indexer_forward.api import (
        indexer_forward_wrapper,
    )

    result = indexer_forward_wrapper(
        index_q.contiguous(),
        index_k_comp.unsqueeze(2).contiguous(),
        weights.contiguous(),
        ratio=int(ratio),
        sm_scale=float(sm_scale),
    )
    return result["scores"]


def cudnn_indexer_topk(scores, sq, ratio, topk, valid_range=None):
    """Select top-K indices using cuDNN TRT-LLM radix kernel (SM100).

    Args:
        scores:  [B, S_q, S_k] fp32 Paddle tensor.
        sq:      query sequence length.
        ratio:   compression ratio.
        topk:    number of entries to select per query position.
        valid_range: optional [B, S_q, 2] int32 per-query left-closed
            compressed-KV range ``[valid_start, valid_end)`` for document-mask
            (packed multi-document) training. ``None`` => causal-only mode
            (legacy single-document behavior, byte-for-byte unchanged).

    Returns:
        topk_indices: [B, S_q, topk] int32 **global** compressed-buffer ids,
            invalid slots are -1.
        topk_length:  [B, S_q] int32, per-row valid count.
    """
    batch = int(scores.shape[0])
    sk = int(scores.shape[2])
    sq = int(sq)
    topk = int(topk)
    topk_k = min(topk, sk)

    _require_cudnn_frontend()
    from paddlefleet_ops.cudnn.deepseek_sparse_attention.indexer_top_k.api import (
        indexer_top_k_wrapper,
    )

    if valid_range is None:
        # Causal-only (single-document): the radix kernel's per-row prefix
        # length is exactly the ratio-causal limit. No id remap needed —
        # local == global because there is a single compressed buffer.
        q_idx = paddle.arange(sq, dtype="int32")
        seq_lens = paddle.clip((q_idx + 1) // int(ratio), max=sk).tile([batch])
        scores_for_topk = scores
        valid_range_for_remap = None
    else:
        # Document-mask: the valid window [valid_start, valid_end) is an
        # arbitrary sub-interval, but the radix kernel only honors prefixes
        # [0, seq_lens). Left-align each query's window to [0, count), run
        # top-k in that local space, then map the selected local ids back to
        # global compressed-buffer ids by adding valid_start.
        from .docmask_utils import (
            shift_scores_to_local_window,
            topk_local_to_global,
        )

        if valid_range.shape[0] != batch or valid_range.shape[1] != sq:
            raise ValueError(
                f"valid_range must have shape [{batch}, {sq}, 2], got "
                f"{list(valid_range.shape)}"
            )
        scores_for_topk, counts = shift_scores_to_local_window(
            scores, valid_range
        )
        seq_lens = counts.reshape([batch * sq]).cast("int32")
        valid_range_for_remap = valid_range

    result = indexer_top_k_wrapper(
        scores_for_topk.reshape([batch * sq, sk]).contiguous(),
        seq_lens,
        top_k=topk_k,
        next_n=1,
        return_val=False,
    )
    topk_indices = result["indices"].reshape([batch, sq, topk_k]).cast("int32")

    if valid_range_for_remap is not None:
        # local (per-document, [0, count)) -> global; -1 slots preserved.
        topk_indices = topk_local_to_global(topk_indices, valid_range_for_remap)

    if topk_k < topk:
        padding = paddle.full([batch, sq, topk - topk_k], -1, dtype="int32")
        topk_indices = paddle.concat([topk_indices, padding], axis=-1)

    topk_length = (topk_indices >= 0).sum(axis=-1).cast("int32")
    return topk_indices, topk_length


def cudnn_indexer_topk_fwd(
    index_q,
    index_k_comp,
    weights,
    ratio=4,
    topk_effective=64,
    indexer_softmax_scale=1.0,
    valid_range=None,
):
    """Run cuDNN-frontend DSA indexer forward on Paddle tensors.

    Args:
        index_q:                [B, S, H_i, D_i] bf16, indexer queries.
        index_k_comp:           [B, S_comp, D_i] bf16, compressed indexer keys.
        weights:                [B, S, H_i] bf16, per-head weights.
        ratio:                  compression ratio (e.g. 4).
        topk_effective:         number of entries to select per query position.
        indexer_softmax_scale:  additional scale on weights.
        valid_range:            optional [B, S, 2] int32 per-query left-closed
            compressed-KV range for document-mask (packed multi-document)
            training. ``None`` => causal-only single-document mode (unchanged).

    Returns:
        topk_indices: [B, S, topk_effective] int32 global compressed-buffer ids,
            invalid slots are -1.
        topk_length:  [B, S] int32, per-row valid count.
    """
    _validate_indexer_inputs(index_q, index_k_comp, weights)
    if int(topk_effective) <= 0:
        raise ValueError(
            f"topk_effective must be positive, got {topk_effective}"
        )

    # sm_scale combines base dim**-0.5 with any additional indexer_softmax_scale
    _sm = float(index_q.shape[-1]) ** -0.5
    if float(indexer_softmax_scale) != 1.0:
        _sm = _sm * float(indexer_softmax_scale)

    scores = cudnn_indexer_forward(
        index_q, index_k_comp, weights, ratio=ratio, sm_scale=_sm
    )
    return cudnn_indexer_topk(
        scores,
        int(index_q.shape[1]),
        ratio,
        topk_effective,
        valid_range=valid_range,
    )
