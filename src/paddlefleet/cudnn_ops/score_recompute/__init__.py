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

"""Sparse attention score recompute (target) for indexer KL loss.

Computes the L1-normalised head-summed softmax target distribution:
    P[b,q,h,i] = exp(Q_h . K_{topk[i]}^T * scale - LSE[b,q,h])
    target[b,q,i] = sum_h(P[b,q,h,i]) / sum_i(sum_h(P[b,q,h,i]))
"""

import paddle
import paddle.nn.functional as F


def cudnn_attn_target_recompute(
    q_attn,
    k_attn,
    lse,
    topk_indices,
    softmax_scale,
    qhead_per_kv_head=None,
):
    """Compute L1-normalised attention target (pure Paddle).

    Args:
        q_attn: [B, S_q, H_q, D] bf16 Paddle tensor. MLA queries.
        k_attn: [B, S_k, D] bf16 Paddle tensor. Compressed KV (MQA).
        lse: [B, S_q, H_q] fp32 Paddle tensor. Log-sum-exp from attn forward.
        topk_indices: [B, S_q, topk] int32 Paddle tensor. Per-batch local KV ids.
        softmax_scale: float. Attention softmax scale.
        qhead_per_kv_head: int or None. (unused, kept for API compat)

    Returns:
        target: [B, S_q, topk] fp32 Paddle tensor. L1-normalised target.
    """
    B, S, H, D = q_attn.shape
    TOPK = topk_indices.shape[-1]

    query = q_attn.cast("float32")
    key_comp = k_attn.cast("float32")

    # Gather keys at topk positions (mask invalid with 0-index)
    valid = topk_indices >= 0
    safe_idx = paddle.where(valid, topk_indices, paddle.zeros_like(topk_indices))
    safe_flat = safe_idx.reshape([B, S * TOPK]).cast("int64")
    gathered_k = paddle.take_along_axis(
        key_comp, safe_flat.unsqueeze(-1).expand([B, S * TOPK, D]), axis=1
    ).reshape([B, S, TOPK, D])

    # logits = Q . K^T * scale
    logits = paddle.einsum("bshd,bstd->bsht", query, gathered_k) * softmax_scale

    # Mask invalid positions
    valid_4d = valid.unsqueeze(2).expand([B, S, H, TOPK])
    neg_inf = paddle.full([1], float("-inf"), dtype="float32")
    logits = paddle.where(valid_4d, logits, neg_inf)

    # per_head_score = exp(logits - lse)
    per_head_score = paddle.exp(logits - lse.unsqueeze(-1))
    per_head_score = paddle.where(valid_4d, per_head_score, paddle.zeros_like(per_head_score))

    # L1-normalize across topk dim
    head_sum = per_head_score.sum(axis=2)  # [B, S, TOPK]
    row_sum = head_sum.sum(axis=-1, keepdim=True).clip(min=1e-10)
    target = head_sum / row_sum
    target = paddle.where(valid, target, paddle.zeros_like(target))
    return target
