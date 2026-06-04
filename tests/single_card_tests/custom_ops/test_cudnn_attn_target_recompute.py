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

"""Numerical parity tests for cudnn_attn_target_recompute and CuDNNIndexerLossFunc.

Test 1 — Target parity:
    Computes a Paddle reference target using the same formula as cuDNN:
      per_head_score[h,i] = exp(Q[h] . K[topk[i]]^T * scale - LSE[h])
      target[i] = sum_h(per_head_score[h,i]) / sum_i(sum_h(per_head_score[h,i]))
    LSE is computed from the SAME query/key/topk so cuDNN and reference are identical.

Test 2 — Gradient parity:
    Calls CuDNNIndexerLossFunc.apply(...) with precomputed grads, sets the loss
    scale via DSAIndexerLossAutoScaler, runs backward(), and checks that .grad on
    each indexer tensor matches direct csa_indexer_bwd(..., grad_loss=scale).
"""

import unittest

import paddle
import paddle.nn.functional as F

paddle.enable_compat(scope={"tilelang"}, silent=True)


def _cuda_or_skip(tc):
    if not paddle.device.is_compiled_with_cuda():
        tc.skipTest("CUDA build required")
    if paddle.device.cuda.device_count() == 0:
        tc.skipTest("No CUDA device")


def _try_import_cudnn_target():
    try:
        from paddlefleet.cudnn_ops import cudnn_attn_target_recompute
        return cudnn_attn_target_recompute
    except (ImportError, RuntimeError):
        return None


def _try_import_cudnn_bwd():
    try:
        from paddlefleet.cudnn_ops import csa_indexer_bwd
        return csa_indexer_bwd
    except (ImportError, RuntimeError):
        return None


def _paddle_reference_target(query_bf16, key_comp_bf16, topk_indices, sm_scale):
    """Pure-Paddle reference: same math as cuDNN SparseAttnScoreRecompute.

    Handles all-invalid rows safely (no NaN propagation).
    """
    B, S, H, D = query_bf16.shape
    TOPK = topk_indices.shape[-1]
    query = query_bf16.cast("float32")
    key_comp = key_comp_bf16.cast("float32")

    safe_idx = paddle.where(topk_indices >= 0, topk_indices, paddle.zeros_like(topk_indices))
    safe_flat = safe_idx.reshape([B, S * TOPK]).cast("int64")
    gathered_k = paddle.take_along_axis(
        key_comp, safe_flat.unsqueeze(-1).expand([B, S * TOPK, D]), axis=1
    ).reshape([B, S, TOPK, D])

    logits = paddle.einsum("bshd,bstd->bsht", query, gathered_k) * sm_scale

    valid = (topk_indices >= 0).unsqueeze(2).expand([B, S, H, TOPK])
    neg_inf = paddle.full([1], float("-inf"), dtype="float32")
    logits = paddle.where(valid, logits, neg_inf)

    # Safe LSE: for all-invalid rows, logsumexp(-inf) = -inf -> use 0.0 instead
    # to avoid NaN in exp(-inf - -inf).
    has_any_valid = (topk_indices >= 0).any(axis=-1)  # [B, S]
    lse = paddle.logsumexp(logits, axis=-1)  # [B, S, H]
    # Replace -inf LSE (all-invalid rows) with 0.0 to prevent NaN
    row_invalid = ~has_any_valid.unsqueeze(-1).expand([B, S, H])
    lse = paddle.where(row_invalid, paddle.zeros_like(lse), lse)

    per_head_score = paddle.exp(logits - lse.unsqueeze(-1))
    per_head_score = paddle.where(valid, per_head_score, paddle.zeros_like(per_head_score))

    head_sum = per_head_score.sum(axis=2)
    row_sum = head_sum.sum(axis=-1, keepdim=True).clip(min=1e-10)
    target = head_sum / row_sum

    valid_2d = topk_indices >= 0
    target = paddle.where(valid_2d, target, paddle.zeros_like(target))
    return target, lse


def _assert_close(actual, expected, rtol, atol, name):
    a = actual.cast("float32")
    e = expected.cast("float32")
    if not paddle.allclose(a, e, rtol=rtol, atol=atol).item():
        diff = (a - e).abs()
        denom = e.abs().clip(min=1e-12)
        raise AssertionError(
            f"{name} mismatch: max abs={diff.max().item():.4e} "
            f"max rel={(diff / denom).max().item():.4e}"
        )


def _make_causal_topk(B, S, S_COMP, TOPK, ratio=4, shuffle=False):
    """Build causal topk indices for all batches."""
    topk_indices = paddle.full([B, S, TOPK], -1, dtype="int32")
    for bi in range(B):
        for i in range(S):
            n_valid = min((i + 1) // ratio, S_COMP)
            n_valid = min(n_valid, TOPK)
            if n_valid > 0:
                idxs = list(range(n_valid))
                if shuffle:
                    idxs = idxs[::-1]
                for k in range(n_valid):
                    topk_indices[bi, i, k] = idxs[k]
    return topk_indices


# =========================================================================
# Test 1: Target numerical parity
# =========================================================================


class TestCuDNNAttnTargetParity(unittest.TestCase):
    """Numerical parity: cudnn_attn_target_recompute vs Paddle reference."""

    B = 1
    S = 32
    H = 64
    D = 512
    S_COMP = 8
    TOPK = 8
    SM_SCALE = 1.0 / (512 ** 0.5)

    def setUp(self):
        _cuda_or_skip(self)
        self.cudnn_target_fn = _try_import_cudnn_target()
        if self.cudnn_target_fn is None:
            self.skipTest("cuDNN DSA target recompute not available")
        paddle.set_device("gpu:0")
        paddle.seed(2026)

    def _run_parity(self, topk_indices, msg=""):
        B, S, H, D = self.B, self.S, self.H, self.D
        query = paddle.randn([B, S, H, D]).cast("bfloat16")
        key_comp = paddle.randn([B, self.S_COMP, D]).cast("bfloat16")

        ref_target, lse = _paddle_reference_target(
            query, key_comp, topk_indices, self.SM_SCALE
        )
        cudnn_target = self.cudnn_target_fn(
            query, key_comp, lse, topk_indices,
            self.SM_SCALE, qhead_per_kv_head=H,
        )
        _assert_close(cudnn_target, ref_target, rtol=5e-3, atol=5e-3,
                      name=f"target {msg}")

    def test_normal_parity(self):
        topk_indices = _make_causal_topk(self.B, self.S, self.S_COMP, self.TOPK)
        self._run_parity(topk_indices, "normal")

    def test_invalid_negative_one(self):
        topk_indices = _make_causal_topk(self.B, self.S, self.S_COMP, self.TOPK)
        topk_indices[:, :, -1] = -1
        self._run_parity(topk_indices, "invalid_-1")

    def test_early_causal_empty_rows(self):
        topk_indices = _make_causal_topk(self.B, self.S, self.S_COMP, self.TOPK)
        self._run_parity(topk_indices, "early_empty")

    def test_topk_reorder(self):
        topk_indices = _make_causal_topk(self.B, self.S, self.S_COMP, self.TOPK,
                                         shuffle=True)
        self._run_parity(topk_indices, "reorder")

    def test_bf16_input_fp32_output(self):
        topk_indices = _make_causal_topk(self.B, self.S, self.S_COMP, self.TOPK)
        query = paddle.randn([self.B, self.S, self.H, self.D]).cast("bfloat16")
        key_comp = paddle.randn([self.B, self.S_COMP, self.D]).cast("bfloat16")
        _, lse = _paddle_reference_target(query, key_comp, topk_indices, self.SM_SCALE)
        target = self.cudnn_target_fn(
            query, key_comp, lse, topk_indices, self.SM_SCALE, qhead_per_kv_head=self.H
        )
        self.assertEqual(target.dtype, paddle.float32)


# =========================================================================
# Test 2: CuDNNIndexerLossFunc backward — real autograd integration
# =========================================================================


class TestCuDNNIndexerLossFuncBackward(unittest.TestCase):
    """End-to-end gradient test for CuDNNIndexerLossFunc PyLayer.

    Calls CuDNNIndexerLossFunc.apply(...), sets the loss scale, runs backward(),
    and checks .grad on indexer tensors against direct csa_indexer_bwd.
    """

    B = 1
    S = 1024
    S_COMP = 256
    H = 64
    D = 128
    TOPK = 128
    LOSS_COEFF = 0.01

    def setUp(self):
        _cuda_or_skip(self)
        self.cudnn_bwd_fn = _try_import_cudnn_bwd()
        if self.cudnn_bwd_fn is None:
            self.skipTest("cuDNN csa_indexer_bwd not available")
        paddle.set_device("gpu:0")
        paddle.seed(2026)

    def test_pylayer_backward_grads_match_direct(self):
        """CuDNNIndexerLossFunc.backward routes correct scaled grads to leaves."""
        from paddlefleet.transformer.csa_attention import CuDNNIndexerLossFunc
        from paddlefleet.transformer.dsa_attention import DSAIndexerLossAutoScaler

        B, S, S_COMP, H, D, TOPK = (
            self.B, self.S, self.S_COMP, self.H, self.D, self.TOPK
        )
        loss_coeff = self.LOSS_COEFF
        scale_val = 0.73
        scale_tensor = paddle.to_tensor(scale_val, dtype="float32")

        index_q = paddle.randn([B, S, H, D]).cast("bfloat16")
        index_k = paddle.randn([B, S_COMP, D]).cast("bfloat16")
        weights = paddle.randn([B, S, H]).cast("bfloat16")
        target = F.softmax(paddle.randn([B, S, TOPK]).cast("float32"), axis=-1)
        topk_probs = F.softmax(paddle.randn([B, S, TOPK]).cast("float32"), axis=-1)
        topk_indices = paddle.randint(0, S_COMP, [B, S, TOPK]).cast("int32")
        mask = paddle.rand([B, S, TOPK]) < 0.05
        topk_indices = paddle.where(
            mask, paddle.full_like(topk_indices, -1), topk_indices
        )

        # Precompute grads with grad_loss=None (=1.0)
        try:
            pre_q, pre_w, pre_k = self.cudnn_bwd_fn(
                index_q.clone(), weights.clone(), index_k.clone(),
                target.clone(), topk_probs.clone(), topk_indices.clone(),
                loss_coeff=loss_coeff, grad_loss=None,
            )
        except Exception as exc:
            self.skipTest(f"cuDNN kernel error: {exc}")

        # Leaf tensors that receive gradients. Paddle PyLayer does not allow
        # inplace strategy on leaf vars, so pass non-leaf identity views.
        q_leaf = index_q.detach().clone()
        q_leaf.stop_gradient = False
        q_arg = q_leaf + 0
        w_leaf = weights.detach().clone()
        w_leaf.stop_gradient = False
        w_arg = w_leaf + 0
        k_leaf = index_k.detach().clone()
        k_leaf.stop_gradient = False
        k_arg = k_leaf + 0

        output_leaf = paddle.randn([B, S, H * D]).cast("bfloat16")
        output_leaf.stop_gradient = False
        output = output_leaf + 0
        indexer_loss_leaf = paddle.to_tensor(0.123, dtype="float32")
        indexer_loss_leaf.stop_gradient = False
        indexer_loss = indexer_loss_leaf + 0

        DSAIndexerLossAutoScaler.set_loss_scale(scale_tensor)
        result = CuDNNIndexerLossFunc.apply(
            output, indexer_loss,
            q_arg, w_arg, k_arg,
            pre_q, pre_w, pre_k,
        )
        result.backward(paddle.ones_like(result))

        # Verify indexer leaf grads = precomputed * scale
        _assert_close(q_leaf.grad, pre_q * scale_val, rtol=1e-5, atol=1e-5,
                      name="PyLayer grad_q")
        _assert_close(w_leaf.grad, pre_w * scale_val, rtol=1e-5, atol=1e-5,
                      name="PyLayer grad_w")
        _assert_close(k_leaf.grad, pre_k * scale_val, rtol=1e-5, atol=1e-5,
                      name="PyLayer grad_k")

        # Output grad passes through unchanged
        _assert_close(output_leaf.grad, paddle.ones_like(output_leaf), rtol=0.0, atol=0.0,
                      name="PyLayer grad_output")

        # indexer_loss receives scaled grad
        self.assertAlmostEqual(indexer_loss_leaf.grad.item(), scale_val, places=5)
        DSAIndexerLossAutoScaler.set_loss_scale(None)

    def test_pylayer_grads_vs_direct_bwd_call(self):
        """Cross-check: PyLayer grads == direct csa_indexer_bwd(grad_loss=scale)."""
        from paddlefleet.transformer.csa_attention import CuDNNIndexerLossFunc
        from paddlefleet.transformer.dsa_attention import DSAIndexerLossAutoScaler

        B, S, S_COMP, H, D, TOPK = (
            self.B, self.S, self.S_COMP, self.H, self.D, self.TOPK
        )
        loss_coeff = self.LOSS_COEFF
        scale_val = 1.5
        scale_tensor = paddle.to_tensor(scale_val, dtype="float32")

        index_q = paddle.randn([B, S, H, D]).cast("bfloat16")
        index_k = paddle.randn([B, S_COMP, D]).cast("bfloat16")
        weights = paddle.randn([B, S, H]).cast("bfloat16")
        target = F.softmax(paddle.randn([B, S, TOPK]).cast("float32"), axis=-1)
        topk_probs = F.softmax(paddle.randn([B, S, TOPK]).cast("float32"), axis=-1)
        topk_indices = paddle.randint(0, S_COMP, [B, S, TOPK]).cast("int32")

        # Precompute with grad_loss=None
        try:
            pre_q, pre_w, pre_k = self.cudnn_bwd_fn(
                index_q.clone(), weights.clone(), index_k.clone(),
                target.clone(), topk_probs.clone(), topk_indices.clone(),
                loss_coeff=loss_coeff, grad_loss=None,
            )
        except Exception as exc:
            self.skipTest(f"cuDNN kernel error: {exc}")

        # Direct call with grad_loss=scale (ground truth)
        direct_q, direct_w, direct_k = self.cudnn_bwd_fn(
            index_q.clone(), weights.clone(), index_k.clone(),
            target.clone(), topk_probs.clone(), topk_indices.clone(),
            loss_coeff=loss_coeff, grad_loss=scale_tensor,
        )

        # PyLayer path
        q_leaf = index_q.detach().clone()
        q_leaf.stop_gradient = False
        q_arg = q_leaf + 0
        w_leaf = weights.detach().clone()
        w_leaf.stop_gradient = False
        w_arg = w_leaf + 0
        k_leaf = index_k.detach().clone()
        k_leaf.stop_gradient = False
        k_arg = k_leaf + 0
        output_leaf = paddle.randn([B, S, H * D]).cast("bfloat16")
        output_leaf.stop_gradient = False
        output = output_leaf + 0
        indexer_loss_leaf = paddle.to_tensor(0.0, dtype="float32")
        indexer_loss_leaf.stop_gradient = False
        indexer_loss = indexer_loss_leaf + 0

        DSAIndexerLossAutoScaler.set_loss_scale(scale_tensor)
        result = CuDNNIndexerLossFunc.apply(
            output, indexer_loss,
            q_arg, w_arg, k_arg,
            pre_q, pre_w, pre_k,
        )
        result.backward(paddle.ones_like(result))

        _assert_close(q_leaf.grad, direct_q, rtol=1e-4, atol=1e-4,
                      name="grad_q vs direct")
        _assert_close(w_leaf.grad, direct_w, rtol=1e-4, atol=1e-4,
                      name="grad_w vs direct")
        _assert_close(k_leaf.grad, direct_k, rtol=1e-4, atol=1e-4,
                      name="grad_k vs direct")
        DSAIndexerLossAutoScaler.set_loss_scale(None)


if __name__ == "__main__":
    unittest.main()
