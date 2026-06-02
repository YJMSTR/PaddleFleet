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

"""Stub for cuDNN sparse attention backward.

This module reserves the interface for the upcoming cuDNN sparse MQA backward
kernel. The signature mirrors Megatron's sparse_backward convention.
"""


def cudnn_sparse_attn_bwd(
    grad_output,
    query,
    kv_full,
    attn_sink,
    topk_idxs,
    output,
    lse,
    softmax_scale,
):
    """Compute sparse attention backward via cuDNN (not yet implemented).

    Args:
        grad_output: [B, S, H, D] gradient of attention output.
        query: [B, S, H, D] queries.
        kv_full: [B, S_kv, D] key-value (single-head MQA).
        attn_sink: [H] learnable attention sink.
        topk_idxs: [B, S, topk] int32 selected indices.
        output: [B, S, H, D] forward output (for reuse).
        lse: [B, S, H] log-sum-exp from forward.
        softmax_scale: float scaling factor.

    Returns:
        dq: [B, S, H, D] query gradient.
        dkv: [B, S_kv, D] kv gradient.
        d_attn_sink: [H] attn_sink gradient.
    """
    raise NotImplementedError(
        "cuDNN sparse attention backward is not yet implemented. "
        "Set csa_sparse_bwd_backend='tilelang' to use the TileLang path."
    )
