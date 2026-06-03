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

"""cuDNN DSA sparse attention backward via DLPack bridge.

Uses cudnn.DSA.sparse_attention_backward_wrapper (SM90/SM100).
TopK padding: SM90=128, SM100=64 (aligned with Megatron).
"""

from __future__ import annotations

import paddle
from paddle import Tensor

_DSA = None
_torch = None
_dlpack = None


def _ensure_imports():
    global _DSA, _torch, _dlpack
    if _DSA is not None:
        return
    import torch
    from torch.utils import dlpack
    from cudnn import DSA

    _torch = torch
    _dlpack = dlpack
    _DSA = DSA


def _get_topk_alignment() -> int:
    sm = paddle.device.get_device_properties("gpu:0").major
    return 64 if sm >= 10 else 128


def cudnn_sparse_attn_bwd(
    grad_output: Tensor,
    query: Tensor,
    kv_full: Tensor,
    attn_sink: Tensor,
    topk_idxs: Tensor,
    output: Tensor,
    lse: Tensor,
    softmax_scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sparse attention backward via cuDNN DSA.

    Args:
        grad_output: [B, S, H, D] bf16
        query: [B, S, H, D] bf16
        kv_full: [B, S_kv, D] bf16
        attn_sink: [H] fp32
        topk_idxs: [B, S, topk] int32
        output: [B, S, H, D] bf16 (forward output)
        lse: [B, S, H] fp32
        softmax_scale: float

    Returns:
        dq, dkv, d_attn_sink
    """
    _ensure_imports()

    B, S, H, D = query.shape
    S_kv = kv_full.shape[1]

    # Flatten to [B*S, ...] for cuDNN
    q_flat = query.reshape([B * S, H, D])
    kv_flat = kv_full.reshape([B * S_kv, D])
    o_flat = output.reshape([B * S, H, D])
    do_flat = grad_output.reshape([B * S, H, D])
    lse_flat = lse.reshape([B * S, H])

    # Convert batch-local topk indices to global indices into [B*S_kv, D]
    topk_flat = topk_idxs.reshape([B * S, -1])
    valid = topk_flat >= 0
    batch_ids = (
        paddle.arange(B, dtype="int32")
        .unsqueeze(1)
        .expand([B, S])
        .reshape([B * S])
    )
    batch_offsets = (batch_ids * S_kv).unsqueeze(1)
    topk_flat = paddle.where(valid, topk_flat + batch_offsets, topk_flat)

    # Pad topk
    topk = topk_flat.shape[-1]
    topk_align = _get_topk_alignment()
    padded_topk = (topk + topk_align - 1) // topk_align * topk_align
    if padded_topk != topk:
        pad = paddle.full([B * S, padded_topk - topk], -1, dtype=topk_flat.dtype)
        topk_flat = paddle.concat([topk_flat, pad], axis=-1)

    # DLPack zero-copy
    def _to_torch(x):
        return _dlpack.from_dlpack(paddle.utils.dlpack.to_dlpack(x.detach()))

    q_t = _to_torch(q_flat.contiguous())
    kv_t = _to_torch(kv_flat.contiguous())
    o_t = _to_torch(o_flat.contiguous())
    do_t = _to_torch(do_flat.contiguous())
    lse_t = _to_torch(lse_flat.contiguous())
    sink_t = _to_torch(attn_sink.contiguous())
    topk_t = _to_torch(topk_flat.contiguous())

    # Call cuDNN on Paddle's stream
    cur_stream = paddle.device.cuda.current_stream().cuda_stream
    torch_stream = _torch.cuda.ExternalStream(cur_stream)
    with _torch.cuda.stream(torch_stream):
        result = _DSA.sparse_attention_backward_wrapper(
            q=q_t,
            kv=kv_t,
            out=o_t,
            dout=do_t,
            lse=lse_t,
            attn_sink=sink_t,
            topk_idxs=topk_t,
            softmax_scale=softmax_scale,
            topk_length=None,
        )

    # Back to Paddle — clone to move grads from torch allocator to paddle allocator.
    # These grads persist as param.grad until optimizer step completes; keeping them
    # in torch's cache would hide ~1.3GB from paddle's allocator.
    def _to_paddle(t):
        view = paddle.utils.dlpack.from_dlpack(_dlpack.to_dlpack(t))
        return view.clone()

    dq = _to_paddle(result["dq"]).reshape([B, S, H, D])
    dkv = _to_paddle(result["dkv"]).reshape([B, S_kv, D])
    d_sink = _to_paddle(result["d_sink"])

    # Release torch tensor references so empty_cache can actually free them
    del result
    _torch.cuda.empty_cache()

    return dq, dkv, d_sink


def is_cudnn_dsa_available() -> bool:
    """Check if cuDNN DSA is available (SM >= 9)."""
    try:
        sm = paddle.device.get_device_properties("gpu:0").major
        return sm >= 9
    except Exception:
        return False


def set_cudnn_dsa_enabled(enabled: bool) -> None:
    """No-op kept for API compatibility. Backend is now config-driven."""
    pass
