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

import os
import unittest

import numpy as np
import paddle

os.environ["USE_FLASH_MLA"] = "1"

try:
    import paddlefleet_ops

    from paddlefleet.tilelang_ops.attn import sparse_mqa

    _HAS_FLASH_MLA = (
        paddlefleet_ops.is_flash_mla_available()
        and sparse_mqa._flash_mla_sparse_fwd is not None
    )
except (ImportError, RuntimeError):
    _HAS_FLASH_MLA = False


@unittest.skipUnless(
    paddle.is_compiled_with_cuda() and _HAS_FLASH_MLA,
    "FlashMLA sparse attention requires CUDA and flash_mla",
)
class TestSparseMQAFlashMLAForward(unittest.TestCase):
    def setUp(self):
        paddle.seed(2026)
        self.batch_size = 2
        self.seq_len = 1024
        self.num_heads = 64
        self.head_dim = 512
        self.topk = 128
        self.softmax_scale = self.head_dim**-0.5

    def _make_inputs(self):
        q = paddle.randn(
            [
                self.batch_size,
                self.seq_len,
                self.num_heads,
                self.head_dim,
            ],
            dtype=paddle.bfloat16,
        )
        kv = paddle.randn(
            [self.batch_size, self.seq_len, self.head_dim],
            dtype=paddle.bfloat16,
        )
        attn_sink = paddle.randn([self.num_heads], dtype=paddle.float32)
        topk_idxs = (
            paddle.arange(self.topk, dtype="int32")
            .reshape([1, 1, self.topk])
            .expand([self.batch_size, self.seq_len, self.topk])
        )
        return q, kv, attn_sink, topk_idxs

    def test_flash_mla_forward_matches_tilelang(self):
        q, kv, attn_sink, topk_idxs = self._make_inputs()

        old_use_flash_mla = sparse_mqa._USE_FLASH_MLA

        try:
            sparse_mqa._USE_FLASH_MLA = False
            tile_out, tile_lse = sparse_mqa.sparse_attn(
                q, kv, attn_sink, topk_idxs, sm_scale=self.softmax_scale
            )

            sparse_mqa._USE_FLASH_MLA = True
            flash_out, flash_lse = sparse_mqa.sparse_attn(
                q, kv, attn_sink, topk_idxs, sm_scale=self.softmax_scale
            )
        finally:
            sparse_mqa._USE_FLASH_MLA = old_use_flash_mla

        np.testing.assert_allclose(
            flash_out.float(),
            tile_out.float(),
            rtol=5e-2,
            atol=1e-2,
        )
        np.testing.assert_allclose(
            flash_lse,
            tile_lse,
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
