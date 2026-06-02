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

"""cuDNN sparse attention score recompute (target) via DLPack bridge.

Wraps cudnn.DSA.sparse_attn_score_recompute_wrapper to compute the L1-normalised
head-summed softmax target distribution used by the indexer KL loss.
"""

import paddle

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


def _paddle_to_torch(x):
    _ensure_cudnn_dsa()
    return _torch.utils.dlpack.from_dlpack(x)


def _torch_to_paddle(t):
    return paddle.utils.dlpack.from_dlpack(t)


def cudnn_attn_target_recompute(
    q_attn,
    k_attn,
    lse,
    topk_indices,
    softmax_scale,
    qhead_per_kv_head=None,
):
    """Compute L1-normalised attention target via cuDNN SparseAttnScoreRecompute.

    Args:
        q_attn: [B, S_q, H_q, D] bf16 Paddle tensor. MLA queries.
        k_attn: [B, S_k, D] bf16 Paddle tensor. Compressed KV (MQA).
        lse: [B, S_q, H_q] fp32 Paddle tensor. Log-sum-exp from attn forward.
        topk_indices: [B, S_q, topk] int32 Paddle tensor. Per-batch local KV ids.
        softmax_scale: float. Attention softmax scale.
        qhead_per_kv_head: int or None. Number of query heads per KV head.

    Returns:
        target: [B, S_q, topk] fp32 Paddle tensor. L1-normalised target.
    """
    _ensure_cudnn_dsa()

    q_t = _paddle_to_torch(q_attn.contiguous())
    k_t = _paddle_to_torch(k_attn.contiguous())
    lse_t = _paddle_to_torch(lse.contiguous())
    idx_t = _paddle_to_torch(topk_indices.contiguous())

    if qhead_per_kv_head is None:
        qhead_per_kv_head = int(q_attn.shape[2])

    result = _DSA.sparse_attn_score_recompute_wrapper(
        q_t,
        k_t,
        lse_t,
        idx_t,
        float(softmax_scale),
        qhead_per_kv_head=qhead_per_kv_head,
    )
    target = _torch_to_paddle(result["target"])
    return target
