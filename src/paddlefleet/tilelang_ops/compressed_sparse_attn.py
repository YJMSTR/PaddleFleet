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

from contextlib import contextmanager

import paddle

from .attn import sparse_mqa_bwd
from .attn.sparse_mqa import (
    _prepare_inputs,
    sparse_attn,
)


@contextmanager
def _nvtx_range(name: str):
    paddle.core.nvprof_nvtx_push(name)
    try:
        yield
    finally:
        paddle.core.nvprof_nvtx_pop()


class CSASparseAttention(paddle.autograd.PyLayer):
    _last_lse_indexer = None

    @staticmethod
    def forward(
        ctx, query, kv_full, attn_sink, topk_idxs, softmax_scale,
        sparse_fwd_backend="tilelang",
        sparse_bwd_backend="tilelang",
        indexer_topk: int = 0,
    ):
        b, sq, np_heads, hn = query.shape
        ctx.query_shape = (b, sq, np_heads, hn)
        ctx.softmax_scale = float(softmax_scale)
        ctx.attn_sink_dtype = attn_sink.dtype
        ctx.sparse_bwd_backend = str(sparse_bwd_backend)
        with _nvtx_range(f"sparse_attn_prepare_inputs[{sparse_fwd_backend}]"):
            query, kv_full, attn_sink, topk_idxs = _prepare_inputs(
                query,
                kv_full,
                attn_sink,
                topk_idxs,
            )
        with _nvtx_range(f"sparse_attn_fwd[{sparse_fwd_backend}]"):
            output, lse, lse_indexer, lse_kv_ln = sparse_attn(
                query,
                kv_full,
                attn_sink,
                topk_idxs,
                sm_scale=ctx.softmax_scale,
                use_flashmla=(sparse_fwd_backend == "flashmla"),
                indexer_topk=int(indexer_topk),
            )
        ctx.save_for_backward(query, kv_full, attn_sink, topk_idxs, output, lse)
        ctx.lse_kv_ln = lse_kv_ln
        ctx.lse_indexer = lse_indexer
        CSASparseAttention._last_lse_indexer = lse_indexer
        return output.reshape([b, sq, np_heads * hn])

    @staticmethod
    def backward(ctx, grad_output):
        query, kv_full, attn_sink, topk_idxs, output, lse = ctx.saved_tensor()
        b, sq, np_heads, hn = ctx.query_shape
        grad_output = grad_output.reshape([b, sq, np_heads, hn])

        if ctx.sparse_bwd_backend == "cudnn":
            from paddlefleet.cudnn_ops import cudnn_sparse_attn_bwd

            # cuDNN expects KV-only FP32 LSE excluding sink. FlashMLA provides
            # it in natural-log form.
            lse_for_cudnn = ctx.lse_kv_ln
            if lse_for_cudnn is None:
                # TileLang forward: convert full log2 LSE back to KV-only
                import math
                lse_full_ln = lse.cast("float32") * math.log(2.0)
                lse_for_cudnn = paddle.log(
                    paddle.exp(lse_full_ln) - paddle.exp(attn_sink.cast("float32"))
                )
            elif lse_for_cudnn.dtype != paddle.float32:
                lse_for_cudnn = lse_for_cudnn.cast("float32")

            with _nvtx_range("sparse_attn_bwd[cudnn]"):
                dq, dkv, d_attn_sink = cudnn_sparse_attn_bwd(
                    grad_output,
                    query,
                    kv_full,
                    attn_sink,
                    topk_idxs,
                    output,
                    lse_for_cudnn,
                    ctx.softmax_scale,
                )
        else:
            with _nvtx_range("sparse_attn_bwd[tilelang]"):
                dq, dkv, d_attn_sink = sparse_mqa_bwd.sparse_mqa_bwd_interface(
                    query,
                    kv_full,
                    attn_sink,
                    output,
                    grad_output,
                    topk_idxs,
                    lse,
                    ctx.softmax_scale,
                )
        dq = dq.reshape(query.shape)
        dkv = dkv.reshape(kv_full.shape)
        d_attn_sink = d_attn_sink.reshape(attn_sink.shape).cast(
            ctx.attn_sink_dtype
        )
        return (
            dq,
            dkv,
            d_attn_sink,
            None,
        )


def csa_sparse_attn(
    query,
    kv_full,
    attn_sink,
    topk_idxs,
    softmax_scale,
    sparse_fwd_backend="tilelang",
    sparse_bwd_backend="tilelang",
    indexer_topk: int = 0,
):
    output = CSASparseAttention.apply(
        query,
        kv_full,
        attn_sink,
        topk_idxs,
        softmax_scale,
        sparse_fwd_backend,
        sparse_bwd_backend,
        indexer_topk,
    )
    lse_indexer = CSASparseAttention._last_lse_indexer
    return output, lse_indexer
