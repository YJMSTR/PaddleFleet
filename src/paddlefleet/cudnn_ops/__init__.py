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

"""cuDNN frontend ops bridged into PaddleFleet via dlpack."""

__all__ = [
    "cudnn_indexer_topk_fwd",
    "csa_indexer_bwd",
    "cudnn_sparse_attn_bwd",
    "cudnn_attn_target_recompute",
    "sparse_attention_backward",
    "is_cudnn_dsa_available",
    "set_cudnn_dsa_enabled",
]


def __getattr__(name):
    if name == "cudnn_indexer_topk_fwd":
        from .indexer.cudnn_indexer import cudnn_indexer_topk_fwd

        globals()[name] = cudnn_indexer_topk_fwd
        return cudnn_indexer_topk_fwd
    if name == "csa_indexer_bwd":
        from .indexer.csa_indexer_bwd_cudnn import csa_indexer_bwd

        globals()[name] = csa_indexer_bwd
        return csa_indexer_bwd
    if name == "cudnn_sparse_attn_bwd":
        from .sparse_attn import cudnn_sparse_attn_bwd

        globals()[name] = cudnn_sparse_attn_bwd
        return cudnn_sparse_attn_bwd
    if name == "cudnn_attn_target_recompute":
        from .score_recompute import cudnn_attn_target_recompute

        globals()[name] = cudnn_attn_target_recompute
        return cudnn_attn_target_recompute
    if name == "sparse_attention_backward":
        from .sparse_attn import cudnn_sparse_attn_bwd

        globals()[name] = cudnn_sparse_attn_bwd
        return cudnn_sparse_attn_bwd
    if name == "is_cudnn_dsa_available":
        from .sparse_attn import is_cudnn_dsa_available

        globals()[name] = is_cudnn_dsa_available
        return is_cudnn_dsa_available
    if name == "set_cudnn_dsa_enabled":
        from .sparse_attn import set_cudnn_dsa_enabled

        globals()[name] = set_cudnn_dsa_enabled
        return set_cudnn_dsa_enabled
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
