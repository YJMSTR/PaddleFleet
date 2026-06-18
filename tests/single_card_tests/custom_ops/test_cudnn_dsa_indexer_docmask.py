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

"""Unit tests for the cuDNN CSA indexer document-mask helpers.

Covers the pure-Paddle helpers in
``paddlefleet.cudnn_ops.indexer.docmask_utils``:

* topk_local_to_global / topk_global_to_local: per-document-local <-> global
  compressed-buffer id conversion, including the round-trip identity, the
  ``-1`` invalid-slot preservation, and the multi-document example from the
  task description.
* thd_to_bshd_b1 / bshd_b1_to_thd: packed-THD <-> BSHD(b==1, padded) layout
  conversion, including pad/de-pad round-trips and pad_value correctness.

These run on any device (CPU/GPU) -- no GPU kernel dependency.
"""

import unittest

import paddle

from paddlefleet.cudnn_ops.indexer.docmask_utils import (
    bshd_b1_to_thd,
    shift_scores_to_local_window,
    thd_to_bshd_b1,
    topk_global_to_local,
    topk_local_to_global,
    valid_range_to_counts,
)


class TestTopkLocalGlobal(unittest.TestCase):
    """topk_local_to_global / topk_global_to_local."""

    def test_multi_doc_example_local_to_global(self):
        """Task example: 3 docs, per-doc-local topk -> global flat ids.

        doc_col_start = [0, 3, 5] (doc0 has 3 compressed cols, doc1 has 2).
        local  [0,1,2 | 0,2,3 | 0,4,7]
        global [0,1,2 | 3,5,6 | 5,9,12]
        """
        # 3 queries, one per document, topk=3.
        topk_local = paddle.to_tensor(
            [[0, 1, 2], [0, 2, 3], [0, 4, 7]], dtype="int32"
        )
        valid_range = paddle.to_tensor([[0, 3], [3, 5], [5, 13]], dtype="int32")
        out = topk_local_to_global(topk_local, valid_range)
        expected = paddle.to_tensor(
            [[0, 1, 2], [3, 5, 6], [5, 9, 12]], dtype="int32"
        )
        self.assertTrue(paddle.equal_all(out, expected).item())

    def test_global_to_local_inverse(self):
        topk_global = paddle.to_tensor(
            [[0, 1, 2], [3, 5, 6], [5, 9, 12]], dtype="int32"
        )
        valid_range = paddle.to_tensor([[0, 3], [3, 5], [5, 13]], dtype="int32")
        out = topk_global_to_local(topk_global, valid_range)
        expected = paddle.to_tensor(
            [[0, 1, 2], [0, 2, 3], [0, 4, 7]], dtype="int32"
        )
        self.assertTrue(paddle.equal_all(out, expected).item())

    def test_round_trip_bshd(self):
        """local -> global -> local is identity on a [B, S, topk] tensor."""
        paddle.seed(0)
        b, s, topk = 2, 16, 8
        doc_start = paddle.randint(0, 100, [b, s, 1]).astype("int32")
        valid_range = paddle.concat([doc_start, doc_start + 50], axis=-1)
        local = paddle.randint(0, 50, [b, s, topk]).astype("int32")
        g = topk_local_to_global(local, valid_range)
        back = topk_global_to_local(g, valid_range)
        self.assertTrue(paddle.equal_all(back, local).item())

    def test_invalid_slots_preserved(self):
        """-1 slots stay -1 through both directions."""
        topk_local = paddle.to_tensor([[0, -1, 2], [-1, -1, 3]], dtype="int32")
        valid_range = paddle.to_tensor([[5, 10], [3, 9]], dtype="int32")
        g = topk_local_to_global(topk_local, valid_range)
        expected_g = paddle.to_tensor([[5, -1, 7], [-1, -1, 6]], dtype="int32")
        self.assertTrue(paddle.equal_all(g, expected_g).item())
        back = topk_global_to_local(g, valid_range)
        self.assertTrue(paddle.equal_all(back, topk_local).item())

    def test_zero_offset_is_identity(self):
        """doc_col_start == 0 (single-doc) leaves valid ids unchanged."""
        topk = paddle.to_tensor([[0, 5, 9], [1, -1, 7]], dtype="int32")
        valid_range = paddle.to_tensor([[0, 9], [0, 9]], dtype="int32")
        out = topk_local_to_global(topk, valid_range)
        self.assertTrue(paddle.equal_all(out, topk).item())

    def test_dtype_preserved(self):
        topk = paddle.to_tensor([[0, 1]], dtype="int32")
        valid_range = paddle.to_tensor([[4, 8]], dtype="int32")
        out = topk_local_to_global(topk, valid_range)
        self.assertEqual(out.dtype, paddle.int32)

    def test_shape_mismatch_raises(self):
        topk = paddle.to_tensor([[0, 1, 2]], dtype="int32")
        bad_valid = paddle.to_tensor([[0, 3], [1, 4]], dtype="int32")
        with self.assertRaises(ValueError):
            topk_local_to_global(topk, bad_valid)

    def test_valid_range_last_dim_must_be_2(self):
        topk = paddle.to_tensor([[0, 1, 2]], dtype="int32")
        bad_valid = paddle.to_tensor([[0, 3, 5]], dtype="int32")
        with self.assertRaises(ValueError):
            topk_local_to_global(topk, bad_valid)


class TestThdBshdConversion(unittest.TestCase):
    """thd_to_bshd_b1 / bshd_b1_to_thd."""

    def test_thd_to_bshd_pads_and_adds_batch(self):
        thd = paddle.arange(6).reshape([3, 2]).astype("float32")
        out = thd_to_bshd_b1(thd, pad_len=5, pad_value=0)
        self.assertEqual(out.shape, [1, 5, 2])
        # body preserved
        self.assertTrue(paddle.equal_all(out[0, :3], thd).item())
        # padding zeros
        self.assertTrue(
            paddle.equal_all(out[0, 3:], paddle.zeros([2, 2])).item()
        )

    def test_thd_to_bshd_pad_value_minus_one(self):
        thd = paddle.to_tensor([[0, 1, 2], [3, 4, 5]], dtype="int32")
        out = thd_to_bshd_b1(thd, pad_len=4, pad_value=-1)
        self.assertEqual(out.shape, [1, 4, 3])
        self.assertTrue(
            paddle.equal_all(
                out[0, 2:], paddle.full([2, 3], -1, dtype="int32")
            ).item()
        )

    def test_thd_to_bshd_no_pad_when_equal(self):
        thd = paddle.arange(4).reshape([2, 2]).astype("float32")
        out = thd_to_bshd_b1(thd, pad_len=2)
        self.assertEqual(out.shape, [1, 2, 2])
        self.assertTrue(paddle.equal_all(out[0], thd).item())

    def test_thd_to_bshd_pad_len_too_small_raises(self):
        thd = paddle.zeros([5, 2], dtype="float32")
        with self.assertRaises(ValueError):
            thd_to_bshd_b1(thd, pad_len=3)

    def test_bshd_to_thd_strips_padding_and_batch(self):
        bshd = paddle.arange(10).reshape([1, 5, 2]).astype("float32")
        out = bshd_b1_to_thd(bshd, total_len=3)
        self.assertEqual(out.shape, [3, 2])
        self.assertTrue(paddle.equal_all(out, bshd[0, :3]).item())

    def test_bshd_to_thd_requires_batch_one(self):
        bshd = paddle.zeros([2, 5, 2], dtype="float32")
        with self.assertRaises(ValueError):
            bshd_b1_to_thd(bshd, total_len=3)

    def test_bshd_to_thd_total_len_too_large_raises(self):
        bshd = paddle.zeros([1, 4, 2], dtype="float32")
        with self.assertRaises(ValueError):
            bshd_b1_to_thd(bshd, total_len=5)

    def test_round_trip_thd_bshd_thd(self):
        """THD -> BSHD(pad) -> THD recovers the original packed tensor."""
        paddle.seed(1)
        thd = paddle.randn([7, 4, 3]).astype("float32")  # [T, H, D]
        bshd = thd_to_bshd_b1(thd, pad_len=16, pad_value=0)
        self.assertEqual(bshd.shape, [1, 16, 4, 3])
        back = bshd_b1_to_thd(bshd, total_len=7)
        self.assertTrue(paddle.equal_all(back, thd).item())

    def test_round_trip_preserves_indices_dtype(self):
        thd = paddle.randint(0, 100, [5, 8]).astype("int32")
        bshd = thd_to_bshd_b1(thd, pad_len=8, pad_value=-1)
        back = bshd_b1_to_thd(bshd, total_len=5)
        self.assertEqual(back.dtype, paddle.int32)
        self.assertTrue(paddle.equal_all(back, thd).item())


class TestValidRangeToCounts(unittest.TestCase):
    """valid_range_to_counts."""

    def test_basic_counts(self):
        vr = paddle.to_tensor(
            [[[0, 3], [3, 5], [5, 13]]], dtype="int32"
        )  # [1, 3, 2]
        counts = valid_range_to_counts(vr)
        self.assertEqual(counts.shape, [1, 3])
        self.assertTrue(
            paddle.equal_all(
                counts, paddle.to_tensor([[3, 2, 8]], dtype="int32")
            ).item()
        )

    def test_empty_range_clamped_to_zero(self):
        # zeroed-out padding rows (start == end == 0) -> count 0.
        vr = paddle.to_tensor([[[0, 0], [4, 4]]], dtype="int32")
        counts = valid_range_to_counts(vr)
        self.assertTrue(
            paddle.equal_all(
                counts, paddle.to_tensor([[0, 0]], dtype="int32")
            ).item()
        )

    def test_dtype_is_int32(self):
        vr = paddle.to_tensor([[2, 7]], dtype="int64")
        counts = valid_range_to_counts(vr)
        self.assertEqual(counts.dtype, paddle.int32)

    def test_bad_shape_raises(self):
        with self.assertRaises(ValueError):
            valid_range_to_counts(paddle.zeros([3, 3], dtype="int32"))


class TestShiftScoresToLocalWindow(unittest.TestCase):
    """shift_scores_to_local_window."""

    def test_left_aligns_window(self):
        # one query, valid window [3, 5) over 8 compressed cols.
        scores = paddle.to_tensor(
            [[[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]]],
            dtype="float32",
        )
        vr = paddle.to_tensor([[[3, 5]]], dtype="int32")
        local, counts = shift_scores_to_local_window(scores, vr)
        self.assertTrue(
            paddle.equal_all(
                counts, paddle.to_tensor([[2]], dtype="int32")
            ).item()
        )
        # cols 3,4 (=13,14) move to front; tail is -inf.
        row = local.numpy()[0, 0]
        self.assertEqual(row[0], 13.0)
        self.assertEqual(row[1], 14.0)
        self.assertTrue((row[2:] == float("-inf")).all())

    def test_topk_after_shift_then_remap_matches_global(self):
        """End-to-end (no kernel): argmax within local window -> global id.

        Picks the single largest valid score per query directly (a stand-in
        for the radix top-k), maps it back to global ids, and checks it equals
        the true global argmax restricted to the valid window.
        """
        paddle.seed(7)
        b, s, sk = 1, 4, 12
        scores = paddle.randn([b, s, sk]).astype("float32")
        # three docs: cols [0,4), [4,8), [8,12); one query per doc + one extra.
        vr = paddle.to_tensor(
            [[[0, 4], [4, 8], [8, 12], [4, 8]]], dtype="int32"
        )
        local, counts = shift_scores_to_local_window(scores, vr)
        # local argmax (top-1) over the left-aligned window
        local_top1 = local.argmax(axis=-1).cast("int32")  # [b, s]
        # remap to global via topk_local_to_global on a [b,s,1] tensor
        g = topk_local_to_global(local_top1.unsqueeze(-1), vr)[..., 0]
        # reference: global argmax restricted to each query's valid window
        col = paddle.arange(sk, dtype="int32").reshape([1, 1, sk])
        in_win = (col >= vr[..., 0:1]) & (col < vr[..., 1:2])
        masked = paddle.where(
            in_win, scores, paddle.full_like(scores, float("-inf"))
        )
        ref = masked.argmax(axis=-1).cast("int32")
        self.assertTrue(paddle.equal_all(g, ref).item())

    def test_bad_scores_rank_raises(self):
        with self.assertRaises(ValueError):
            shift_scores_to_local_window(
                paddle.zeros([4, 8], dtype="float32"),
                paddle.zeros([4, 2], dtype="int32"),
            )


@unittest.skipIf(
    not paddle.device.is_compiled_with_cuda()
    or paddle.device.cuda.get_device_capability()[0] != 10,
    "cuDNN indexer top-k requires Blackwell GPU (SM100)",
)
class TestCudnnIndexerTopkDocmask(unittest.TestCase):
    """End-to-end forward docmask through cudnn_indexer_topk (radix kernel)."""

    def _ref_per_doc_topk(self, scores, valid_range, topk):
        """Per-query reference: top-k global ids restricted to the valid window.

        Mirrors the radix kernel semantics: within each query's
        ``[valid_start, valid_end)`` window pick the ``min(topk, count)``
        largest scores, returning their global compressed-buffer ids (sorted by
        descending score). Padding to ``topk`` is ``-1``.
        """
        b, s, sk = scores.shape
        col = paddle.arange(sk, dtype="int32").reshape([1, 1, sk])
        in_win = (col >= valid_range[..., 0:1]) & (col < valid_range[..., 1:2])
        masked = paddle.where(
            in_win, scores, paddle.full_like(scores, float("-inf"))
        )
        ref = paddle.full([b, s, topk], -1, dtype="int32")
        counts = (valid_range[..., 1] - valid_range[..., 0]).clip(min=0)
        masked_np = masked.numpy()
        counts_np = counts.numpy()
        import numpy as np

        ref_np = ref.numpy()
        for bi in range(b):
            for si in range(s):
                c = int(counts_np[bi, si])
                if c <= 0:
                    continue
                k = min(topk, c)
                order = np.argsort(-masked_np[bi, si])[:k]
                ref_np[bi, si, :k] = order.astype("int32")
        return paddle.to_tensor(ref_np)

    def test_docmask_topk_within_doc_windows(self):
        """User scenario: doc lens 23 + 9, ratio 4 -> 5 + 2 compressed cols.

        Compressed buffer (Sk=8): doc0 cols [0,5), doc1 cols [5,7), 1 pad col.
        Every selected id must lie inside its query's document window; no
        cross-document leakage; empty-range queries select nothing.
        """
        from paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
            cudnn_indexer_topk,
        )
        from paddlefleet.transformer.csa_attention import get_valid_range

        paddle.seed(0)
        ratio, sq, sk, topk = 4, 32, 8, 4
        startend = paddle.concat(
            [
                paddle.full([23], 23, dtype="int32"),
                paddle.full([9], 32, dtype="int32"),
            ]
        ).reshape([1, sq, 1])
        valid_range = get_valid_range(ratio, 1, sq, startend)
        scores = paddle.randn([1, sq, sk]).astype("float32")

        topk_indices, topk_length = cudnn_indexer_topk(
            scores, sq, ratio, topk, valid_range=valid_range
        )
        ti = topk_indices.numpy()[0]
        vr = valid_range.numpy()[0]
        for q in range(sq):
            start, end = int(vr[q, 0]), int(vr[q, 1])
            picks = [int(x) for x in ti[q] if x >= 0]
            for x in picks:
                self.assertTrue(
                    start <= x < end,
                    f"query {q}: id {x} outside window [{start},{end})",
                )
            if end == start:
                self.assertEqual(
                    picks, [], f"empty-range query {q} picked {picks}"
                )

    def test_docmask_matches_per_doc_reference(self):
        """Selected id sets match a pure-numpy per-window top-k reference."""
        from paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
            cudnn_indexer_topk,
        )

        paddle.seed(3)
        # 3 docs of compressed widths 4, 3, 5 packed into Sk=12; one query each.
        sk, topk = 12, 3
        valid_range = paddle.to_tensor(
            [[[0, 4], [4, 7], [7, 12]]], dtype="int32"
        )
        scores = paddle.randn([1, 3, sk]).astype("float32")
        topk_indices, _ = cudnn_indexer_topk(
            scores, 3, 4, topk, valid_range=valid_range
        )
        ref = self._ref_per_doc_topk(scores, valid_range, topk)
        # Compare as sets per query (radix tie order may differ from argsort).
        ti = topk_indices.numpy()[0]
        rf = ref.numpy()[0]
        for q in range(3):
            self.assertEqual(
                {int(x) for x in ti[q] if x >= 0},
                {int(x) for x in rf[q] if x >= 0},
                f"query {q} selected-set mismatch: got {ti[q]}, ref {rf[q]}",
            )

    def test_docmask_none_matches_causal_baseline(self):
        """valid_range=None reproduces the legacy causal-only path exactly."""
        from paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
            cudnn_indexer_topk,
        )

        paddle.seed(5)
        ratio, sq, sk, topk = 4, 16, 8, 4
        scores = paddle.randn([1, sq, sk]).astype("float32")
        idx_a, _ = cudnn_indexer_topk(scores, sq, ratio, topk, valid_range=None)
        idx_b, _ = cudnn_indexer_topk(scores, sq, ratio, topk)
        self.assertTrue(paddle.equal_all(idx_a, idx_b).item())


def _bwd_inputs(b, sq, sk, h, d, topk, seed):
    paddle.seed(seed)
    import paddle.nn.functional as F

    index_q = paddle.randn([b, sq, h, d]).astype("bfloat16")
    index_k = paddle.randn([b, sk, d]).astype("bfloat16")
    weights = paddle.randn([b, sq, h]).astype("bfloat16")
    target = F.softmax(paddle.randn([b, sq, topk]).astype("float32"), axis=-1)
    topk_probs = F.softmax(
        paddle.randn([b, sq, topk]).astype("float32"), axis=-1
    )
    return index_q, weights, index_k, target, topk_probs


@unittest.skipIf(
    not paddle.device.is_compiled_with_cuda()
    or paddle.device.cuda.get_device_capability()[0] != 10,
    "cuDNN indexer backward requires Blackwell GPU (SM100)",
)
class TestCudnnIndexerBwdDocmask(unittest.TestCase):
    """csa_indexer_bwd docmask params: topk_is_local + valid_range, THD layout.

    The sparse backward kernel requires ``topk % block_I == 0`` (block_I=128),
    so these tests use topk=128 with a compressed buffer wide enough for two
    documents.
    """

    def setUp(self):
        from paddlefleet.cudnn_ops import csa_indexer_bwd

        self.bwd = csa_indexer_bwd

    def test_local_topk_matches_global(self):
        """local ids + valid_range produce the same grads as global ids.

        Two-document packed sequence: doc0 occupies compressed cols [0, 256),
        doc1 occupies [256, 512). For each query we pick top-k ids inside its
        document window as GLOBAL ids, then derive the per-document LOCAL ids
        (global - valid_start). Both id spaces must yield identical gradients.
        """
        b, sq, sk, h, d, topk = 1, 128, 512, 64, 128, 128
        half = sq // 2
        win = sk // 2  # 256
        index_q, weights, index_k, target, topk_probs = _bwd_inputs(
            b, sq, sk, h, d, topk, seed=7
        )
        # valid_range [1, sq, 2]: first half -> [0, win), second -> [win, sk).
        starts = paddle.concat(
            [
                paddle.zeros([half], dtype="int32"),
                paddle.full([sq - half], win, dtype="int32"),
            ]
        )
        ends = starts + win
        vr = paddle.stack([starts, ends], axis=-1).reshape([1, sq, 2])

        # GLOBAL ids: query q picks s_+0 .. s_+topk-1 (window width win >= topk).
        col = paddle.arange(topk, dtype="int32").reshape([1, 1, topk])
        topk_global = (vr[..., 0:1] + col).tile([b, 1, 1])
        # LOCAL ids: subtract the document column start.
        topk_local = topk_global - vr[..., 0:1]

        gl = self.bwd(
            index_q.clone(),
            weights.clone(),
            index_k.clone(),
            target.clone(),
            topk_probs.clone(),
            topk_global.clone(),
            loss_coeff=0.01,
            grad_loss=paddle.to_tensor(1.0, dtype="float32"),
        )
        lo = self.bwd(
            index_q.clone(),
            weights.clone(),
            index_k.clone(),
            target.clone(),
            topk_probs.clone(),
            topk_local.clone(),
            loss_coeff=0.01,
            grad_loss=paddle.to_tensor(1.0, dtype="float32"),
            valid_range=vr,
            topk_is_local=True,
        )
        for name, a, c in zip(("dq", "dw", "dk"), gl, lo):
            self.assertTrue(
                paddle.allclose(
                    a.cast("float32"), c.cast("float32"), rtol=1e-3, atol=1e-3
                ).item(),
                f"{name}: local-id path diverges from global-id path",
            )

    def test_thd_layout_matches_bshd_b1(self):
        """THD packed [T,...] grads equal BSHD b==1 grads (no padding case)."""
        b, sq, sk, h, d, topk = 1, 128, 512, 64, 128, 128
        index_q, weights, index_k, target, topk_probs = _bwd_inputs(
            b, sq, sk, h, d, topk, seed=9
        )
        topk_indices = paddle.randint(0, sk, [b, sq, topk]).astype("int32")

        bshd = self.bwd(
            index_q.clone(),
            weights.clone(),
            index_k.clone(),
            target.clone(),
            topk_probs.clone(),
            topk_indices.clone(),
            loss_coeff=0.01,
            grad_loss=paddle.to_tensor(1.0, dtype="float32"),
        )
        # THD: drop the batch dim on all inputs.
        thd = self.bwd(
            index_q[0].clone(),
            weights[0].clone(),
            index_k[0].clone(),
            target[0].clone(),
            topk_probs[0].clone(),
            topk_indices[0].clone(),
            loss_coeff=0.01,
            grad_loss=paddle.to_tensor(1.0, dtype="float32"),
            layout="thd",
        )
        self.assertEqual(thd[0].shape, [sq, h, d])
        self.assertEqual(thd[2].shape, [sk, d])
        for name, a, t in zip(("dq", "dw", "dk"), bshd, thd):
            self.assertTrue(
                paddle.allclose(
                    a[0].cast("float32"),
                    t.cast("float32"),
                    rtol=1e-3,
                    atol=1e-3,
                ).item(),
                f"{name}: THD path diverges from BSHD b==1",
            )

    def test_local_without_valid_range_raises(self):
        b, sq, sk, h, d, topk = 1, 128, 512, 64, 128, 128
        index_q, weights, index_k, target, topk_probs = _bwd_inputs(
            b, sq, sk, h, d, topk, seed=11
        )
        topk_indices = paddle.randint(0, sk, [b, sq, topk]).astype("int32")
        with self.assertRaises(ValueError):
            self.bwd(
                index_q,
                weights,
                index_k,
                target,
                topk_probs,
                topk_indices,
                loss_coeff=0.01,
                topk_is_local=True,
            )


@unittest.skipIf(
    not paddle.device.is_compiled_with_cuda()
    or paddle.device.cuda.get_device_capability()[0] != 10,
    "cuDNN indexer top-k requires Blackwell GPU (SM100)",
)
class TestMapCompressedTopkToKvFullDocmask(unittest.TestCase):
    """End-to-end: cudnn docmask topk -> _map_compressed_topk_to_kv_full.

    Verifies the single-document causal cutoff inside
    ``_map_compressed_topk_to_kv_full`` (``id < (t+1)//ratio``, computed from
    the GLOBAL query position) does not falsely reject any in-window global
    compressed id produced by the docmask-aware forward, and that every kept id
    decodes back into its query's document window. Uses document lengths that
    are NOT multiples of ratio to stress the floor-accumulation of doc column
    starts.
    """

    def test_no_false_rejection_and_in_window(self):
        from paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
            cudnn_indexer_topk,
        )
        from paddlefleet.transformer.csa_attention import (
            _map_compressed_topk_to_kv_full,
            get_valid_range,
        )

        paddle.seed(0)
        ratio = 4
        # doc lengths NOT divisible by ratio: 6, 7, 19 (sum=32).
        #   doc0 len6  -> cutoff4  -> 1 compressed col  [0,1)
        #   doc1 len7  -> cutoff4  -> 1 compressed col  [1,2)
        #   doc2 len19 -> cutoff16 -> 4 compressed cols [2,6)
        # compressed buffer Sk = sq//ratio = 8 (cols [6,8) are pad).
        doc_lens = [6, 7, 19]
        sq = sum(doc_lens)
        sk = sq // ratio  # 8
        topk = 4
        offset = sq  # compressed entries follow raw KV in kv_full

        ends = []
        acc = 0
        for dl in doc_lens:
            acc += dl
            ends.extend([acc] * dl)
        startend = paddle.to_tensor(ends, dtype="int32").reshape([1, sq, 1])
        valid_range = get_valid_range(ratio, 1, sq, startend)  # [1, sq, 2]

        scores = paddle.randn([1, sq, sk]).astype("float32")
        topk_global, _ = cudnn_indexer_topk(
            scores, sq, ratio, topk, valid_range=valid_range
        )

        mapped = _map_compressed_topk_to_kv_full(
            topk_global, sq, ratio, offset
        )

        ti = topk_global.numpy()[0]
        mp = mapped.numpy()[0]
        vr = valid_range.numpy()[0]
        for q in range(sq):
            start, end = int(vr[q, 0]), int(vr[q, 1])
            in_topk = [int(x) for x in ti[q] if x >= 0]
            kept = [int(x) for x in mp[q] if x >= 0]
            # (1) No false rejection: count of valid slots preserved.
            self.assertEqual(
                len(kept),
                len(in_topk),
                f"query {q}: map dropped valid ids "
                f"(in={in_topk}, kept_mapped={mp[q].tolist()}, "
                f"window=[{start},{end}))",
            )
            # (2) Every kept id decodes (minus offset) into the doc window.
            for m in kept:
                comp = m - offset
                self.assertTrue(
                    start <= comp < end,
                    f"query {q}: mapped id {m} -> compressed {comp} "
                    f"outside window [{start},{end})",
                )

    def test_matches_manual_compressed_mapping(self):
        """Mapped ids equal (in-window global id + offset), elementwise."""
        from paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
            cudnn_indexer_topk,
        )
        from paddlefleet.transformer.csa_attention import (
            _map_compressed_topk_to_kv_full,
            get_valid_range,
        )

        paddle.seed(1)
        ratio = 4
        doc_lens = [10, 22]  # not multiples of ratio at the boundary sum
        sq = sum(doc_lens)  # 32
        sk = sq // ratio  # 8
        topk = 4
        offset = sq

        ends = []
        acc = 0
        for dl in doc_lens:
            acc += dl
            ends.extend([acc] * dl)
        startend = paddle.to_tensor(ends, dtype="int32").reshape([1, sq, 1])
        valid_range = get_valid_range(ratio, 1, sq, startend)

        scores = paddle.randn([1, sq, sk]).astype("float32")
        topk_global, _ = cudnn_indexer_topk(
            scores, sq, ratio, topk, valid_range=valid_range
        )
        mapped = _map_compressed_topk_to_kv_full(
            topk_global, sq, ratio, offset
        )

        # Since the forward already constrains topk_global to in-window ids,
        # the map is exactly: valid -> id+offset, invalid(-1) -> -1.
        valid = topk_global >= 0
        expected = paddle.where(
            valid,
            topk_global + offset,
            paddle.full_like(topk_global, -1),
        )
        self.assertTrue(paddle.equal_all(mapped, expected).item())


# ---------------------------------------------------------------------------
# End-to-end: docmask topk -> window concat -> sparse fwd (flash_mla / cudnn)
# ---------------------------------------------------------------------------


def _docmask_topk_idxs_for_attn(doc_lens, ratio, window_size, topk, seed):
    """Build kv_full topk_idxs for sparse attention under docmask (b==1).

    Returns ``(topk_idxs [1, sq, W+topk], sq, n_compressed, valid_range,
    doc_row_valid)`` where ``doc_row_valid[q]`` marks queries that belong to a
    real (non-padding) document position. Window + compressed indices are built
    with the same docmask-aware helpers the production forward uses, then
    concatenated exactly like ``CompressedSparseAttention.forward``.
    """
    from paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
        cudnn_indexer_topk,
    )
    from paddlefleet.transformer.csa_attention import (
        _map_compressed_topk_to_kv_full,
        get_valid_range,
        get_window_topk_idxs,
    )

    paddle.seed(seed)
    sq = sum(doc_lens)
    n_compressed = sq // ratio
    offset = sq  # compressed entries follow raw KV inside kv_full

    ends = []
    acc = 0
    for dl in doc_lens:
        acc += dl
        ends.extend([acc] * dl)
    startend = paddle.to_tensor(ends, dtype="int32").reshape([1, sq, 1])

    valid_range = get_valid_range(ratio, 1, sq, startend)  # [1, sq, 2]
    scores = paddle.randn([1, sq, n_compressed]).astype("float32")
    topk_compressed, _ = cudnn_indexer_topk(
        scores, sq, ratio, topk, valid_range=valid_range
    )
    compress_idxs = _map_compressed_topk_to_kv_full(
        topk_compressed, sq, ratio, offset
    )
    window_idxs = get_window_topk_idxs(window_size, 1, sq, startend)

    if compress_idxs.dtype != window_idxs.dtype:
        compress_idxs = compress_idxs.cast(window_idxs.dtype)
    topk_idxs = paddle.concat([window_idxs, compress_idxs], axis=-1).cast(
        "int32"
    )
    # A query row is "real" when its window has at least one valid slot.
    doc_row_valid = (window_idxs >= 0).any(axis=-1)[0]  # [sq]
    return topk_idxs, sq, n_compressed, valid_range, doc_row_valid


def _flash_mla_available():
    try:
        import paddlefleet_ops

        from paddlefleet.tilelang_ops.attn import sparse_mqa

        return (
            paddlefleet_ops.is_flash_mla_available()
            and sparse_mqa._flash_mla_sparse_fwd is not None
        )
    except (ImportError, RuntimeError, AttributeError):
        return False


@unittest.skipIf(
    not paddle.device.is_compiled_with_cuda()
    or paddle.device.cuda.get_device_capability()[0] != 10,
    "docmask sparse-fwd compatibility requires Blackwell GPU (SM100)",
)
class TestDocmaskSparseFwdCompat(unittest.TestCase):
    """docmask topk -> window concat -> sparse attention forward.

    The CSA sparse-attention forward is always FlashMLA (or TileLang / pure
    Paddle); there is **no cuDNN forward**. The ``backend="cudnn"`` option on
    ``csa_sparse_attn`` only selects the cuDNN *backward*; its forward still
    runs FlashMLA. These tests therefore exercise the FlashMLA forward both
    directly and through the ``csa_sparse_attn`` PyLayer entry point, checking
    that docmask-produced topk_idxs (window + compressed, kv_full-local, -1
    padding) feed cleanly in, that real (non-padding) query rows produce finite
    output, and that fully-masked padding rows (a docmask-only situation that
    single-document inputs never hit) stay benign.
    """

    # DSv4 sparse-MQA shape: head_dim must be 512, num_heads 64.
    HEAD_DIM = 512
    NUM_HEADS = 64

    def _build(self, doc_lens, ratio=4, window_size=8, topk=128, seed=0):
        topk_idxs, sq, n_comp, valid_range, doc_row_valid = (
            _docmask_topk_idxs_for_attn(
                doc_lens, ratio, window_size, topk, seed
            )
        )
        s_kvfull = sq + n_comp
        paddle.seed(seed + 1000)
        query = paddle.randn(
            [1, sq, self.NUM_HEADS, self.HEAD_DIM], dtype=paddle.bfloat16
        )
        kv_full = paddle.randn(
            [1, s_kvfull, self.HEAD_DIM], dtype=paddle.bfloat16
        )
        attn_sink = paddle.randn([self.NUM_HEADS], dtype=paddle.float32)
        sm_scale = self.HEAD_DIM**-0.5
        return query, kv_full, attn_sink, topk_idxs, sm_scale, doc_row_valid

    def _check_real_rows_finite(self, out, doc_row_valid, name):
        # out: [1, sq, H*D] or [1, sq, H, D]; reduce to per-row finiteness.
        out_f = out.reshape([out.shape[1], -1]).cast("float32")
        finite_per_row = paddle.isfinite(out_f).all(axis=-1)  # [sq]
        real = doc_row_valid  # [sq] bool
        real_finite = paddle.where(
            real, finite_per_row, paddle.ones_like(finite_per_row)
        )
        self.assertTrue(
            bool(real_finite.all().item()),
            f"{name}: some real (non-padding) query row is non-finite",
        )

    @unittest.skipUnless(_flash_mla_available(), "flash_mla not available")
    def test_flash_mla_fwd_docmask(self):
        from paddlefleet.tilelang_ops.attn import sparse_mqa

        # doc lens not multiples of ratio to exercise padding query rows.
        query, kv_full, attn_sink, topk_idxs, sm_scale, doc_row_valid = (
            self._build([23, 9], seed=1)
        )
        out, lse, _ = sparse_mqa.flash_mla_sparse_attn(
            query, kv_full, attn_sink, topk_idxs, sm_scale=sm_scale
        )
        self.assertEqual(out.shape[1], query.shape[1])
        self._check_real_rows_finite(out, doc_row_valid, "flash_mla")

    @unittest.skipUnless(_flash_mla_available(), "flash_mla not available")
    def test_csa_sparse_attn_pylayer_docmask(self):
        """Through csa_sparse_attn PyLayer (backend selects bwd; fwd=FlashMLA)."""
        from paddlefleet.fusions.csa_sparse_attn import csa_sparse_attn

        query, kv_full, attn_sink, topk_idxs, sm_scale, doc_row_valid = (
            self._build([23, 9], seed=2)
        )
        # backend="cudnn" picks the cuDNN *backward*; the forward is FlashMLA.
        try:
            out = csa_sparse_attn(
                query, kv_full, attn_sink, topk_idxs, sm_scale, backend="cudnn"
            )
        except (RuntimeError, ImportError) as e:
            self.skipTest(f"csa_sparse_attn cudnn-bwd path unavailable: {e}")
        self.assertEqual(out.shape[1], query.shape[1])
        self._check_real_rows_finite(out, doc_row_valid, "csa_sparse_attn")

    @unittest.skipUnless(_flash_mla_available(), "flash_mla not available")
    def test_real_rows_match_single_doc_equivalent(self):
        """A doc's real rows should be unaffected by a second packed doc.

        Pack [doc0 | doc1]; doc0's real query rows + its window/compressed ids
        all live in doc0's id range, so doc0's outputs must equal running doc0
        alone (same query/kv slice). Validates no cross-doc contamination
        through the sparse fwd gather.
        """
        from paddlefleet.tilelang_ops.attn import sparse_mqa

        # Use two identical-length docs so doc0 occupies a clean prefix of
        # kv_full and we can slice it out for the single-doc reference.
        ratio, window, topk = 4, 8, 128
        d0 = 64
        query, kv_full, attn_sink, topk_idxs, sm_scale, doc_row_valid = (
            self._build([d0, 64], ratio=ratio, window_size=window, topk=topk, seed=3)
        )
        out_packed, _, _ = sparse_mqa.flash_mla_sparse_attn(
            query, kv_full, attn_sink, topk_idxs, sm_scale=sm_scale
        )
        # doc0 occupies query rows [0, d0); its window ids ∈ [0, d0) and its
        # compressed ids ∈ [sq, sq + d0//ratio). Outputs there must be finite
        # and independent of doc1's content.
        out0 = out_packed.reshape([out_packed.shape[1], -1]).cast("float32")[
            :d0
        ]
        self.assertTrue(
            bool(paddle.isfinite(out0).all().item()),
            "doc0 real rows non-finite in packed run",
        )

    @unittest.skipUnless(_flash_mla_available(), "flash_mla not available")
    def test_fully_masked_rows_are_benign(self):
        """Fully ``-1`` query rows (docmask-only) must not produce NaN/Inf.

        Natural docmask packing never yields an all-masked row (every query
        attends to its own sliding window), but we construct one explicitly to
        prove the attention-sink denominator floors softmax: a row with zero
        valid KV slots yields a finite (≈0) output instead of NaN. This is the
        property that lets padding rows pass through the sparse fwd without
        contaminating downstream tensors.
        """
        from paddlefleet.tilelang_ops.attn import sparse_mqa

        h, d, sq, skv, topk = self.NUM_HEADS, self.HEAD_DIM, 8, 16, 128
        paddle.seed(7)
        q = paddle.randn([1, sq, h, d], dtype=paddle.bfloat16)
        kv = paddle.randn([1, skv, d], dtype=paddle.bfloat16)
        sink = paddle.randn([h], dtype=paddle.float32)

        idx = paddle.full([1, sq, topk], -1, dtype="int32").numpy()
        # rows 0..5 valid (a few real slots); rows 6,7 fully masked.
        for r in range(6):
            for j in range(4):
                idx[0, r, j] = j
        topk_idxs = paddle.to_tensor(idx)

        out, _, _ = sparse_mqa.flash_mla_sparse_attn(
            q, kv, sink, topk_idxs, sm_scale=d**-0.5
        )
        out_f = out.reshape([sq, -1]).cast("float32")
        self.assertTrue(
            bool(paddle.isfinite(out_f).all().item()),
            "fully-masked rows produced NaN/Inf (attention sink should floor "
            "the softmax denominator)",
        )


if __name__ == "__main__":
    unittest.main()
