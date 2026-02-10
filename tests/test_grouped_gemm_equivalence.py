"""
Tests for Grouped GEMM Backend Equivalence (Triton vs PyTorch Loop)
===================================================================

The autograd_grouped_gemm module provides a GroupedGemmFunc that dispatches
to one of two backends selected via the MOEDL_GEMM_IMPL environment variable:

  - "triton" (default): Uses a Triton grouped GEMM kernel via grouped_gemm_triton.py.
  - "loop":             Uses a pure-PyTorch loop of torch.mm calls.

Both backends must produce equivalent results for the MoE model to be
backend-agnostic. This test module verifies:

  1. **Forward equivalence** – Given identical inputs (X, W), both backends
     return the same Y = X @ W.T within BF16 tolerance.

  2. **Backward equivalence** – Gradients dL/dX and dL/dW from both backends
     match within BF16 tolerance.

  3. **Multi-step training equivalence** – When used inside a full MoeBlk (or
     MoedlForCausalLM), the parameter gradients and gradient norms remain
     consistent across backends over several optimiser steps. This catches
     issues like accumulated numerical divergence, autocast/AMP interactions,
     and autograd graph differences.

Usage:
  # Run with pytest (recommended)
  CUDA_VISIBLE_DEVICES=0 pytest tests/test_grouped_gemm_equivalence.py -v

  # Or directly
  CUDA_VISIBLE_DEVICES=0 python tests/test_grouped_gemm_equivalence.py
"""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from moelab.moedl.autograd_grouped_gemm import GroupedGemmFunc, loop_gemm_fn, EXPERT_GEMM_FN
from moelab.moedl.naive_autograd_grouped_gemm import NaiveGroupedGemmFunc
from moelab.moedl.grouped_gemm_triton import group_gemm_fn as triton_gemm_fn
from moelab.moedl import MoedlConfig, MoedlForCausalLM, MoeBlk


# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda")

# BF16 has ~7.8e-3 relative precision; we use tolerances accordingly.
BF16_ATOL = 2e-2
BF16_RTOL = 2e-2

# For FP32 reference comparisons
FP32_ATOL = 1e-5
FP32_RTOL = 1e-5


def _clone_tensors(tensors, requires_grad=True, dtype=None):
    """Deep-clone a list of tensors, optionally casting dtype."""
    out = []
    for t in tensors:
        c = t.detach().clone()
        if dtype is not None:
            c = c.to(dtype)
        c.requires_grad_(requires_grad)
        out.append(c)
    return out


def _make_random_inputs(n_group, token_counts, D, K, dtype=torch.bfloat16):
    """
    Create random X_list and W_list for grouped GEMM.
    X_list[i]: (token_counts[i], D) — tokens dispatched to expert i.
    W_list[i]: (K, D) — weight matrix (K output features, D input features).
    Y = X @ W.T  →  (token_counts[i], K)
    """
    X_list = [torch.randn(t, D, device=DEVICE, dtype=dtype) for t in token_counts]
    W_list = [torch.randn(K, D, device=DEVICE, dtype=dtype) for t in token_counts]
    return X_list, W_list


# ===================================================================
# Level 1: Raw kernel equivalence (triton vs loop)
# ===================================================================

class TestRawKernelEquivalence:
    """Compare the raw gemm functions (triton vs loop) without autograd."""

    @pytest.mark.parametrize("transA,transB", [
        (False, True),   # NN layout used in forward: Y = X @ W.T
        (False, False),  # used in backward: grad_X = grad_Y @ W
        (True, False),   # used in backward: grad_W = grad_Y.T @ X
    ])
    @pytest.mark.parametrize("n_group", [1, 4, 8])
    def test_kernel_output_matches(self, n_group, transA, transB):
        """Raw triton kernel output must match loop-of-torch.mm output."""
        torch.manual_seed(42)
        # For (transA, transB) we need shapes that make the inner dim match.
        # A: (M, K) if not transA else (K, M);  B: (K, N) if not transB else (N, K)
        M_list = [torch.randint(1, 64, (1,)).item() for _ in range(n_group)]
        N, K = 32, 48

        A_list, B_list = [], []
        for m in M_list:
            if transA:
                A_list.append(torch.randn(K, m, device=DEVICE, dtype=torch.bfloat16))
            else:
                A_list.append(torch.randn(m, K, device=DEVICE, dtype=torch.bfloat16))
            if transB:
                B_list.append(torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16))
            else:
                B_list.append(torch.randn(K, N, device=DEVICE, dtype=torch.bfloat16))

        C_triton = triton_gemm_fn(A_list, B_list, transA=transA, transB=transB)
        C_loop = loop_gemm_fn(A_list, B_list, transA=transA, transB=transB)

        for i in range(n_group):
            torch.testing.assert_close(
                C_triton[i], C_loop[i],
                atol=BF16_ATOL, rtol=BF16_RTOL,
                msg=f"Kernel mismatch at group {i}, transA={transA}, transB={transB}",
            )


# ===================================================================
# Level 2: Autograd Function equivalence (GroupedGemmFunc backends)
# ===================================================================

class TestAutogradForwardBackward:
    """
    Compare GroupedGemmFunc with triton backend vs loop backend (and also
    against NaiveGroupedGemmFunc as an independent reference).
    """

    @pytest.fixture(params=[
        # (n_group, token_counts, D, K)
        (4, [8, 12, 6, 10], 64, 128),
        (8, [4, 4, 4, 4, 4, 4, 4, 4], 32, 64),
        (2, [1, 32], 128, 64),
        (4, [16, 0, 8, 4], 64, 128),   # one expert gets 0 tokens
    ])
    def gemm_inputs(self, request):
        """Parameterized fixture providing (n_group, X_list, W_list)."""
        n_group, token_counts, D, K = request.param
        torch.manual_seed(123)
        X_list, W_list = _make_random_inputs(n_group, token_counts, D, K, dtype=torch.bfloat16)
        return n_group, X_list, W_list, token_counts, D, K

    # ------ helpers ------

    @staticmethod
    def _run_forward_backward(gemm_fn_key, n_group, X_list_orig, W_list_orig):
        """
        Run forward + backward through GroupedGemmFunc using a specific backend.
        Returns (Y_list, grad_X_list, grad_W_list).
        """
        # Monkey-patch the module-level gemm_fn used by GroupedGemmFunc
        import moelab.moedl.autograd_grouped_gemm as _mod
        saved = _mod.gemm_fn
        _mod.gemm_fn = EXPERT_GEMM_FN[gemm_fn_key]

        X_list = _clone_tensors(X_list_orig)
        W_list = _clone_tensors(W_list_orig)

        Y_tuple = GroupedGemmFunc.apply(n_group, *(X_list + W_list))

        # scalar loss = sum of all elements
        loss = sum(y.sum() for y in Y_tuple)
        loss.backward()

        grad_X = [x.grad.clone() for x in X_list]
        grad_W = [w.grad.clone() for w in W_list]

        _mod.gemm_fn = saved  # restore
        return list(Y_tuple), grad_X, grad_W

    @staticmethod
    def _run_naive_forward_backward(n_group, X_list_orig, W_list_orig):
        """Run forward + backward through NaiveGroupedGemmFunc."""
        X_list = _clone_tensors(X_list_orig)
        W_list = _clone_tensors(W_list_orig)

        Y_tuple = NaiveGroupedGemmFunc.apply(n_group, *(X_list + W_list))
        loss = sum(y.sum() for y in Y_tuple)
        loss.backward()

        grad_X = [x.grad.clone() for x in X_list]
        grad_W = [w.grad.clone() for w in W_list]
        return list(Y_tuple), grad_X, grad_W

    # ------ tests ------

    def test_forward_triton_vs_loop(self, gemm_inputs):
        """Forward outputs must match between triton and loop backends."""
        n_group, X_list, W_list, token_counts, D, K = gemm_inputs

        Y_triton, _, _ = self._run_forward_backward("triton", n_group, X_list, W_list)
        Y_loop, _, _ = self._run_forward_backward("loop", n_group, X_list, W_list)

        for i in range(n_group):
            if token_counts[i] == 0:
                assert Y_triton[i].numel() == 0 and Y_loop[i].numel() == 0
                continue
            torch.testing.assert_close(
                Y_triton[i], Y_loop[i],
                atol=BF16_ATOL, rtol=BF16_RTOL,
                msg=f"Forward mismatch at expert {i}",
            )

    def test_backward_grad_X_triton_vs_loop(self, gemm_inputs):
        """grad_X must match between triton and loop backends."""
        n_group, X_list, W_list, token_counts, D, K = gemm_inputs

        _, gX_triton, _ = self._run_forward_backward("triton", n_group, X_list, W_list)
        _, gX_loop, _ = self._run_forward_backward("loop", n_group, X_list, W_list)

        for i in range(n_group):
            if token_counts[i] == 0:
                continue
            torch.testing.assert_close(
                gX_triton[i], gX_loop[i],
                atol=BF16_ATOL, rtol=BF16_RTOL,
                msg=f"grad_X mismatch at expert {i}",
            )

    def test_backward_grad_W_triton_vs_loop(self, gemm_inputs):
        """grad_W must match between triton and loop backends."""
        n_group, X_list, W_list, token_counts, D, K = gemm_inputs

        _, _, gW_triton = self._run_forward_backward("triton", n_group, X_list, W_list)
        _, _, gW_loop = self._run_forward_backward("loop", n_group, X_list, W_list)

        for i in range(n_group):
            torch.testing.assert_close(
                gW_triton[i], gW_loop[i],
                atol=BF16_ATOL, rtol=BF16_RTOL,
                msg=f"grad_W mismatch at expert {i}",
            )

    def test_forward_vs_naive_reference(self, gemm_inputs):
        """Both backends must also match the NaiveGroupedGemmFunc reference."""
        n_group, X_list, W_list, token_counts, D, K = gemm_inputs

        Y_triton, _, _ = self._run_forward_backward("triton", n_group, X_list, W_list)
        Y_naive, _, _ = self._run_naive_forward_backward(n_group, X_list, W_list)

        for i in range(n_group):
            if token_counts[i] == 0:
                continue
            torch.testing.assert_close(
                Y_triton[i], Y_naive[i],
                atol=BF16_ATOL, rtol=BF16_RTOL,
                msg=f"Forward triton vs naive mismatch at expert {i}",
            )

    def test_backward_vs_naive_reference(self, gemm_inputs):
        """Gradients must match the NaiveGroupedGemmFunc reference."""
        n_group, X_list, W_list, token_counts, D, K = gemm_inputs

        _, gX_triton, gW_triton = self._run_forward_backward("triton", n_group, X_list, W_list)
        _, gX_naive, gW_naive = self._run_naive_forward_backward(n_group, X_list, W_list)

        for i in range(n_group):
            if token_counts[i] == 0:
                continue
            torch.testing.assert_close(
                gX_triton[i], gX_naive[i],
                atol=BF16_ATOL, rtol=BF16_RTOL,
                msg=f"grad_X triton vs naive mismatch at expert {i}",
            )
            torch.testing.assert_close(
                gW_triton[i], gW_naive[i],
                atol=BF16_ATOL, rtol=BF16_RTOL,
                msg=f"grad_W triton vs naive mismatch at expert {i}",
            )


# ===================================================================
# Level 3: MoeBlk-level equivalence over multiple training steps
# ===================================================================

def _make_moeblk_config(**overrides):
    """Small MoeBlk config for testing."""
    defaults = dict(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        num_experts=4,
        num_active_experts=2,
        num_shared_experts=0,
        lb_coeff=0.0,
        lb_gamma=0.0,
        capacity_factor=0.0,
    )
    defaults.update(overrides)
    return MoedlConfig(**defaults)


def _run_moeblk_steps(backend_key, config, n_steps, seed, lr=1e-3):
    """
    Run `n_steps` training steps of MoeBlk with a specific GEMM backend.

    Returns:
      step_outputs: list of output tensors (detached) per step
      step_grad_norms: list of total gradient L2 norms per step
      param_state: dict of {name: param.data.clone()} after all steps
    """
    import moelab.moedl.autograd_grouped_gemm as _mod
    saved = _mod.gemm_fn
    _mod.gemm_fn = EXPERT_GEMM_FN[backend_key]

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    moeblk = MoeBlk(config).to(DEVICE).to(torch.bfloat16)
    # Deep-copy for deterministic init: caller must use same seed.
    optimizer = torch.optim.AdamW(moeblk.parameters(), lr=lr)

    step_outputs = []
    step_grad_norms = []
    step_router_logits = []

    B, L, D = 2, 8, config.hidden_size
    # Use a fixed input across steps for reproducibility
    torch.manual_seed(seed + 999)
    x_base = torch.randn(B, L, D, device=DEVICE, dtype=torch.bfloat16)

    for step in range(n_steps):
        optimizer.zero_grad()
        # Same input each step so differences come from weight updates only
        x = x_base.clone().detach()
        out, router_logits = moeblk(x)

        loss = out.sum()
        loss.backward()

        # Collect gradient norm before optimizer step
        total_norm = 0.0
        for p in moeblk.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.float().norm().item() ** 2
        total_norm = total_norm ** 0.5

        step_outputs.append(out.detach().clone())
        step_grad_norms.append(total_norm)
        step_router_logits.append(router_logits.detach().clone())

        optimizer.step()

    param_state = {n: p.data.clone() for n, p in moeblk.named_parameters()}

    _mod.gemm_fn = saved
    return step_outputs, step_grad_norms, step_router_logits, param_state


class TestMoeBlkMultiStepEquivalence:
    """
    End-to-end test: run MoeBlk for several training steps with each
    backend and verify that outputs, gradients, and parameters stay aligned.
    """

    N_STEPS = 5
    SEED = 2026

    @pytest.fixture(autouse=True)
    def _setup_config(self):
        self.config = _make_moeblk_config()

    def _run_both_backends(self):
        """Helper: run training loop with both backends, return results."""
        # We must create the model from the SAME seed so weights are identical.
        res_triton = _run_moeblk_steps("triton", self.config, self.N_STEPS, self.SEED)
        res_loop = _run_moeblk_steps("loop", self.config, self.N_STEPS, self.SEED)
        return res_triton, res_loop

    def test_output_equivalence_per_step(self):
        """MoeBlk output must match at every training step."""
        (outs_t, _, _, _), (outs_l, _, _, _) = self._run_both_backends()
        for step in range(self.N_STEPS):
            torch.testing.assert_close(
                outs_t[step], outs_l[step],
                atol=BF16_ATOL, rtol=BF16_RTOL,
                msg=f"Output mismatch at step {step}",
            )

    def test_router_logits_equivalence_per_step(self):
        """Router logits (expert selection) must match at every step."""
        (_, _, rl_t, _), (_, _, rl_l, _) = self._run_both_backends()
        for step in range(self.N_STEPS):
            torch.testing.assert_close(
                rl_t[step], rl_l[step],
                atol=BF16_ATOL, rtol=BF16_RTOL,
                msg=f"Router logits mismatch at step {step}",
            )

    def test_grad_norm_equivalence_per_step(self):
        """Total gradient L2 norm must match at every training step."""
        (_, gnorms_t, _, _), (_, gnorms_l, _, _) = self._run_both_backends()
        for step in range(self.N_STEPS):
            rel_err = abs(gnorms_t[step] - gnorms_l[step]) / (gnorms_t[step] + 1e-12)
            assert rel_err < 0.05, (
                f"Grad norm diverged at step {step}: "
                f"triton={gnorms_t[step]:.6f}  loop={gnorms_l[step]:.6f}  "
                f"rel_err={rel_err:.4f}"
            )

    def test_parameter_equivalence_after_training(self):
        """Final parameter values must match after N training steps."""
        (_, _, _, params_t), (_, _, _, params_l) = self._run_both_backends()
        for name in params_t:
            torch.testing.assert_close(
                params_t[name], params_l[name],
                atol=BF16_ATOL, rtol=BF16_RTOL,
                msg=f"Parameter '{name}' diverged after {self.N_STEPS} steps",
            )

    def test_grad_norm_nonzero(self):
        """Sanity check: gradient norms should be non-trivially large."""
        (_, gnorms, _, _), _ = self._run_both_backends()
        for step, gn in enumerate(gnorms):
            assert gn > 1e-6, f"Gradient norm suspiciously small at step {step}: {gn}"


# ===================================================================
# Level 4: Full MoedlForCausalLM multi-step equivalence
# ===================================================================

def _run_causal_lm_steps(backend_key, config, n_steps, seed, lr=1e-3):
    """
    Run `n_steps` of causal-LM training with a specific GEMM backend.
    """
    import moelab.moedl.autograd_grouped_gemm as _mod
    saved = _mod.gemm_fn
    _mod.gemm_fn = EXPERT_GEMM_FN[backend_key]

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model = MoedlForCausalLM(config).to(DEVICE).to(torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    B, L = 2, 16
    torch.manual_seed(seed + 777)
    input_ids = torch.randint(0, config.vocab_size, (B, L), device=DEVICE)
    labels = torch.randint(0, config.vocab_size, (B, L), device=DEVICE)

    step_losses = []
    step_grad_norms = []

    for step in range(n_steps):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.float().norm().item() ** 2
        total_norm = total_norm ** 0.5

        step_losses.append(loss.item())
        step_grad_norms.append(total_norm)
        optimizer.step()

    param_state = {n: p.data.clone() for n, p in model.named_parameters()}
    _mod.gemm_fn = saved
    return step_losses, step_grad_norms, param_state


class TestCausalLMMultiStepEquivalence:
    """
    Full model test: MoedlForCausalLM trained for several steps.
    Both backends must produce the same loss trajectory and gradients.
    """

    N_STEPS = 3
    SEED = 1234

    @pytest.fixture(autouse=True)
    def _setup_config(self):
        self.config = _make_moeblk_config(
            num_hidden_layers=2,
            num_experts=4,
            num_active_experts=2,
        )

    def _run_both(self):
        res_t = _run_causal_lm_steps("triton", self.config, self.N_STEPS, self.SEED)
        res_l = _run_causal_lm_steps("loop", self.config, self.N_STEPS, self.SEED)
        return res_t, res_l

    def test_loss_trajectory(self):
        """Loss values must match at every step."""
        (losses_t, _, _), (losses_l, _, _) = self._run_both()
        for step in range(self.N_STEPS):
            rel_err = abs(losses_t[step] - losses_l[step]) / (abs(losses_t[step]) + 1e-12)
            assert rel_err < 0.02, (
                f"Loss diverged at step {step}: "
                f"triton={losses_t[step]:.6f}  loop={losses_l[step]:.6f}  "
                f"rel_err={rel_err:.4f}"
            )

    def test_grad_norms(self):
        """Gradient norms must match at every step."""
        (_, gnorms_t, _), (_, gnorms_l, _) = self._run_both()
        for step in range(self.N_STEPS):
            rel_err = abs(gnorms_t[step] - gnorms_l[step]) / (gnorms_t[step] + 1e-12)
            assert rel_err < 0.05, (
                f"Grad norm diverged at step {step}: "
                f"triton={gnorms_t[step]:.6f}  loop={gnorms_l[step]:.6f}  "
                f"rel_err={rel_err:.4f}"
            )

    def test_final_parameters(self):
        """Parameters must match after training."""
        (_, _, params_t), (_, _, params_l) = self._run_both()
        mismatched = []
        for name in params_t:
            try:
                torch.testing.assert_close(
                    params_t[name], params_l[name],
                    atol=BF16_ATOL, rtol=BF16_RTOL,
                )
            except AssertionError:
                mismatched.append(name)
        assert len(mismatched) == 0, (
            f"{len(mismatched)} parameters diverged: {mismatched[:10]}"
        )


# ===================================================================
# Level 5: Individual gradient values inspection
# ===================================================================

class TestGradientValues:
    """
    Detailed per-parameter gradient inspection between backends.
    This catches subtle differences that aggregate norms might mask.
    """

    def test_per_expert_weight_gradients(self):
        """Each expert's gate/up/down weight gradients must match."""
        config = _make_moeblk_config(num_experts=4, num_active_experts=2)
        B, L, D = 2, 8, config.hidden_size

        for backend in ["triton", "loop"]:
            import moelab.moedl.autograd_grouped_gemm as _mod
            _mod.gemm_fn = EXPERT_GEMM_FN[backend]

        # Run both and compare per-expert gradients
        results = {}
        for backend in ["triton", "loop"]:
            import moelab.moedl.autograd_grouped_gemm as _mod
            _mod.gemm_fn = EXPERT_GEMM_FN[backend]

            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            moeblk = MoeBlk(config).to(DEVICE).to(torch.bfloat16)

            torch.manual_seed(999)
            x = torch.randn(B, L, D, device=DEVICE, dtype=torch.bfloat16)
            out, _ = moeblk(x)
            out.sum().backward()

            grads = {}
            for name, p in moeblk.named_parameters():
                if p.grad is not None:
                    grads[name] = p.grad.clone()
            results[backend] = grads

        # Compare
        for name in results["triton"]:
            if name not in results["loop"]:
                continue
            g_t = results["triton"][name]
            g_l = results["loop"][name]
            torch.testing.assert_close(
                g_t, g_l,
                atol=BF16_ATOL, rtol=BF16_RTOL,
                msg=f"Gradient mismatch for {name}",
            )

    def test_gradient_max_abs_error(self):
        """Max absolute error between backends should be bounded."""
        config = _make_moeblk_config(num_experts=8, num_active_experts=2)
        B, L, D = 4, 16, config.hidden_size

        results = {}
        for backend in ["triton", "loop"]:
            import moelab.moedl.autograd_grouped_gemm as _mod
            _mod.gemm_fn = EXPERT_GEMM_FN[backend]

            torch.manual_seed(7)
            torch.cuda.manual_seed(7)
            moeblk = MoeBlk(config).to(DEVICE).to(torch.bfloat16)

            torch.manual_seed(77)
            x = torch.randn(B, L, D, device=DEVICE, dtype=torch.bfloat16)
            out, _ = moeblk(x)
            out.sum().backward()

            grads = {}
            for name, p in moeblk.named_parameters():
                if p.grad is not None:
                    grads[name] = p.grad.clone()
            results[backend] = grads

        max_errors = {}
        for name in results["triton"]:
            if name not in results["loop"]:
                continue
            diff = (results["triton"][name] - results["loop"][name]).abs()
            max_errors[name] = diff.max().item()

        for name, max_err in max_errors.items():
            assert max_err < 0.1, (
                f"Max abs grad error too large for '{name}': {max_err:.6f}"
            )


# ===================================================================
# Level 6: AMP / autocast interaction
# ===================================================================

class TestAMPInteraction:
    """Ensure both backends behave identically under torch.autocast."""

    def test_autocast_forward_backward(self):
        """Forward/backward under autocast must produce same results."""
        n_group = 4
        torch.manual_seed(0)
        X_list_fp32, W_list_fp32 = _make_random_inputs(
            n_group, [8, 12, 6, 10], D=64, K=128, dtype=torch.float32,
        )

        results = {}
        for backend in ["triton", "loop"]:
            import moelab.moedl.autograd_grouped_gemm as _mod
            _mod.gemm_fn = EXPERT_GEMM_FN[backend]

            X = _clone_tensors(X_list_fp32, dtype=torch.float32)
            W = _clone_tensors(W_list_fp32, dtype=torch.float32)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                Y_tuple = GroupedGemmFunc.apply(n_group, *(X + W))
                loss = sum(y.sum() for y in Y_tuple)

            loss.backward()
            results[backend] = {
                "Y": [y.detach().clone() for y in Y_tuple],
                "gX": [x.grad.clone() for x in X],
                "gW": [w.grad.clone() for w in W],
            }

        for i in range(n_group):
            torch.testing.assert_close(
                results["triton"]["Y"][i], results["loop"]["Y"][i],
                atol=BF16_ATOL, rtol=BF16_RTOL,
                msg=f"AMP forward mismatch at group {i}",
            )
            torch.testing.assert_close(
                results["triton"]["gX"][i], results["loop"]["gX"][i],
                atol=BF16_ATOL, rtol=BF16_RTOL,
                msg=f"AMP grad_X mismatch at group {i}",
            )
            torch.testing.assert_close(
                results["triton"]["gW"][i], results["loop"]["gW"][i],
                atol=BF16_ATOL, rtol=BF16_RTOL,
                msg=f"AMP grad_W mismatch at group {i}",
            )


# ===================================================================
# Level 7: torch.compile compatibility
# ===================================================================

# ===================================================================
# Level 7: torch.compile compatibility
# ===================================================================

# torch.compile tests use subprocess because the MOEDL_GEMM_IMPL env var
# is read at import time. Monkey-patching gemm_fn at runtime doesn't work
# under torch.compile since dynamo captures the function reference at trace
# time. Subprocess ensures the env var is set before the module is imported.

_COMPILE_TEST_SCRIPT = '''\
import os, sys, json
os.environ["MOEDL_GEMM_IMPL"] = sys.argv[1]

import torch
from moelab.moedl import MoedlConfig, MoedlForCausalLM

torch.manual_seed(42)
torch.cuda.manual_seed(42)

config = MoedlConfig(
    vocab_size=256, hidden_size=64, intermediate_size=128,
    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    max_position_embeddings=64,
    num_experts=16, num_active_experts=2,
    num_shared_experts=0, lb_coeff=0.0, lb_gamma=0.0, capacity_factor=0.0,
)
model = MoedlForCausalLM(config).to("cuda").to(torch.bfloat16)
compiled_model = torch.compile(model, backend="inductor")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

B, L = 2, 16
torch.manual_seed(777)
input_ids = torch.randint(0, config.vocab_size, (B, L), device="cuda")
labels = torch.randint(0, config.vocab_size, (B, L), device="cuda")

results = {"losses": [], "grad_norms": [], "has_nan_grad": []}
for step in range(3):
    optimizer.zero_grad()
    out = compiled_model(input_ids=input_ids, labels=labels)
    out.loss.backward()
    has_nan = any(p.grad.isnan().any().item() for p in model.parameters() if p.grad is not None)
    gn = sum(p.grad.data.float().norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5
    results["losses"].append(out.loss.item())
    results["grad_norms"].append(gn)
    results["has_nan_grad"].append(has_nan)
    optimizer.step()

print(json.dumps(results))
'''

def _run_compile_subprocess(backend_key):
    """Run torch.compile training in subprocess with the given GEMM backend."""
    import subprocess, json
    result = subprocess.run(
        [sys.executable, "-c", _COMPILE_TEST_SCRIPT, backend_key],
        capture_output=True, text=True, timeout=120,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    if result.returncode != 0:
        pytest.fail(f"Subprocess failed ({backend_key}):\n{result.stderr[-2000:]}")
    # Parse the last line of stdout as JSON (ignore compile warnings on stderr)
    for line in reversed(result.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    pytest.fail(f"No JSON output from subprocess ({backend_key}):\n{result.stdout[-1000:]}")


class TestTorchCompileCompatibility:
    """
    Verify that both GEMM backends produce correct results under
    torch.compile (inductor). This catches the case where dynamo
    traces into GroupedGemmFunc and miscompiles the triton kernel's
    raw pointer arithmetic — producing NaN on the very first step.

    The fix is torch._dynamo.allow_in_graph(GroupedGemmFunc), which
    makes dynamo treat the custom autograd Function as an opaque node.

    Tests run in subprocesses so MOEDL_GEMM_IMPL is set before import.
    """

    def test_triton_compile_no_nan_loss(self):
        """Triton backend under torch.compile must not produce NaN loss."""
        res = _run_compile_subprocess("triton")
        for step, loss in enumerate(res["losses"]):
            assert not (loss != loss), (
                f"NaN loss at step {step} with triton + torch.compile"
            )

    def test_triton_compile_no_nan_grad(self):
        """Triton backend under torch.compile must not produce NaN gradients."""
        res = _run_compile_subprocess("triton")
        for step, has_nan in enumerate(res["has_nan_grad"]):
            assert not has_nan, (
                f"NaN gradient at step {step} with triton + torch.compile"
            )

    def test_loop_compile_no_nan(self):
        """Loop backend under torch.compile must also work (sanity check)."""
        res = _run_compile_subprocess("loop")
        for step in range(3):
            assert not (res["losses"][step] != res["losses"][step]), f"NaN loss at step {step}"
            assert not res["has_nan_grad"][step], f"NaN grad at step {step}"

    def test_compile_loss_decreases(self):
        """Loss must decrease over steps (training is actually working)."""
        res = _run_compile_subprocess("triton")
        losses = res["losses"]
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: step 0 = {losses[0]:.4f}, "
            f"step {len(losses)-1} = {losses[-1]:.4f}"
        )

    def test_compile_triton_vs_loop_equivalence(self):
        """Both backends must produce similar loss under torch.compile."""
        res_t = _run_compile_subprocess("triton")
        res_l = _run_compile_subprocess("loop")
        for step in range(3):
            rel_err = abs(res_t["losses"][step] - res_l["losses"][step]) / (abs(res_t["losses"][step]) + 1e-12)
            assert rel_err < 0.02, (
                f"Compiled loss diverged at step {step}: "
                f"triton={res_t['losses'][step]:.6f}  loop={res_l['losses'][step]:.6f}  "
                f"rel_err={rel_err:.4f}"
            )


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
