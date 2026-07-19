"""
Tests for LatentMoE: latent_factor.

Covers:
  1. Config validation
  2. MoeBlk construction (latent_down/up, expert_dim)
  3. End-to-end forward pass (output shape & finiteness)
  4. Trainer get_expert_stats (K/E resolution)
"""
import pytest
import torch
import torch.nn as nn
from transformers import TrainingArguments
from datasets import Dataset

from moelab.moedl import MoedlConfig, MoedlForCausalLM
from moelab.moedl.modeling_moedl import MoeBlk
from moelab.moedl.trainer import MoedlTrainer


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _moe_config(**kwargs):
    """Minimal MoE config; override any field via kwargs."""
    defaults = dict(
        vocab_size=500,
        hidden_size=64,          # divisible by 2, 4
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_experts=8,
        num_active_experts=2,
        lb_coeff=0.0,
    )
    defaults.update(kwargs)
    return MoedlConfig(**defaults)


@pytest.fixture
def training_args(tmp_path):
    return TrainingArguments(
        output_dir=str(tmp_path),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_steps=1,
        save_steps=100,
        report_to=[],
        use_cpu=True,
    )


@pytest.fixture
def tiny_dataset():
    return Dataset.from_dict({
        "input_ids": [[1, 2, 3, 4, 5]] * 10,
        "labels":    [[1, 2, 3, 4, 5]] * 10,
    })


# ---------------------------------------------------------------------------
# 1. Config validation
# ---------------------------------------------------------------------------

class TestLatentMoeConfig:

    def test_default_latent_factor_is_one(self):
        cfg = _moe_config()
        assert cfg.latent_factor == 1

    def test_latent_factor_stored(self):
        cfg = _moe_config(latent_factor=2)
        assert cfg.latent_factor == 2

    @pytest.mark.parametrize("bad", [0, -1, 1.5, "2"])
    def test_invalid_latent_factor_raises(self, bad):
        with pytest.raises((ValueError, TypeError)):
            _moe_config(latent_factor=bad)

    def test_latent_factor_ignored_for_dense_model(self):
        """Dense models (num_experts=1) do not store latent_factor."""
        cfg = MoedlConfig(
            vocab_size=500, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4,
            num_experts=1, num_active_experts=1,
            latent_factor=3,
        )
        # Dense path never stores latent_factor; attribute simply won't exist or is irrelevant.
        # The model should be constructable without error.
        model = MoedlForCausalLM(cfg)
        assert model is not None


# ---------------------------------------------------------------------------
# 2. MoeBlk construction
# ---------------------------------------------------------------------------

class TestLatentMoeBlkConstruction:

    def test_identity_projections_when_latent_factor_is_one(self):
        cfg = _moe_config(latent_factor=1)
        blk = MoeBlk(cfg)
        assert isinstance(blk.latent_down, nn.Identity)
        assert isinstance(blk.latent_up,   nn.Identity)

    def test_linear_projections_created_for_latent_factor_gt_one(self):
        cfg = _moe_config(hidden_size=64, latent_factor=2)
        blk = MoeBlk(cfg)
        assert isinstance(blk.latent_down, nn.Linear)
        assert isinstance(blk.latent_up,   nn.Linear)

    def test_expert_dim_equals_hidden_over_factor(self):
        cfg = _moe_config(hidden_size=64, latent_factor=4)
        blk = MoeBlk(cfg)
        assert blk.expert_dim == 16
        # latent_down: hidden_size → expert_dim
        assert blk.latent_down.in_features  == 64
        assert blk.latent_down.out_features == 16
        # latent_up: expert_dim → hidden_size
        assert blk.latent_up.in_features  == 16
        assert blk.latent_up.out_features == 64

    def test_router_stays_in_hidden_space(self):
        """Router always maps from hidden_size, not latent_dim."""
        cfg = _moe_config(hidden_size=64, latent_factor=2)
        blk = MoeBlk(cfg)
        assert blk.router.in_features == 64

    def test_latent_factor_does_not_change_expert_counts(self):
        """latent_factor only reduces the expert hidden dim, not the number of experts."""
        cfg = _moe_config(num_experts=8, num_active_experts=2, latent_factor=2)
        blk = MoeBlk(cfg)
        assert blk.num_experts == 8
        assert blk.num_active_experts == 2

    def test_indivisible_hidden_size_raises(self):
        """hidden_size must be divisible by latent_factor."""
        with pytest.raises(AssertionError):
            MoeBlk(_moe_config(hidden_size=65, latent_factor=2))  # 65 % 2 != 0


# ---------------------------------------------------------------------------
# 3. Forward pass
# ---------------------------------------------------------------------------

class TestLatentMoeForward:

    def _run_forward(self, cfg):
        model = MoedlForCausalLM(cfg)
        model.eval()
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        with torch.no_grad():
            out = model(input_ids)
        return out, cfg

    def test_output_shape_preserved_with_latent(self):
        cfg = _moe_config(hidden_size=64, latent_factor=2)
        out, cfg = self._run_forward(cfg)
        # logits: (batch, seq_len, vocab_size)
        assert out.logits.shape == (2, 8, cfg.vocab_size)

    def test_output_finite_with_latent(self):
        cfg = _moe_config(hidden_size=64, latent_factor=2)
        out, _ = self._run_forward(cfg)
        assert torch.isfinite(out.logits).all()

    def test_higher_latent_factor_output_shape_preserved(self):
        cfg = _moe_config(hidden_size=64, latent_factor=4)
        out, cfg = self._run_forward(cfg)
        assert out.logits.shape == (2, 8, cfg.vocab_size)
        assert torch.isfinite(out.logits).all()

    def test_latent_bottleneck_changes_output(self):
        """Same architecture but different latent_factor should produce different outputs."""
        torch.manual_seed(0)
        cfg_base   = _moe_config(hidden_size=64, latent_factor=1)
        cfg_latent = _moe_config(hidden_size=64, latent_factor=2)

        model_base   = MoedlForCausalLM(cfg_base).eval()
        model_latent = MoedlForCausalLM(cfg_latent).eval()

        input_ids = torch.randint(0, 500, (2, 8))
        with torch.no_grad():
            logits_base   = model_base(input_ids).logits
            logits_latent = model_latent(input_ids).logits

        assert not torch.allclose(logits_base, logits_latent)


# ---------------------------------------------------------------------------
# 4. Trainer – get_expert_stats K/E resolution
# ---------------------------------------------------------------------------

class TestLatentMoeTrainerExpertStats:
    """Verify get_expert_stats uses the correct effective E and K."""

    def _make_trainer(self, cfg, training_args, tiny_dataset):
        model = MoedlForCausalLM(cfg)
        return MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
            heatmap_on=False,
        )

    def _synthetic_router_logits(self, num_layers, num_tokens, num_experts):
        """Random router logits matching what the model would produce."""
        return [torch.randn(num_tokens, num_experts) for _ in range(num_layers)]

    def test_base_expert_stats_shape(self, training_args, tiny_dataset):
        """latent_factor=1 → count shape (L, K_base, E_base)."""
        cfg = _moe_config(num_experts=8, num_active_experts=2)
        trainer = self._make_trainer(cfg, training_args, tiny_dataset)

        logits = self._synthetic_router_logits(num_layers=2, num_tokens=10, num_experts=8)
        count, frac = trainer.get_expert_stats(logits)

        assert count.shape == (2, 2, 8)  # (L=2, K=2, E=8)
        assert frac.shape  == (2, 2, 8)

    def test_latent_expert_stats_shape(self, training_args, tiny_dataset):
        """latent_factor > 1 does not change K or E in stats — only expert_dim changes."""
        cfg = _moe_config(num_experts=8, num_active_experts=2, latent_factor=2)
        trainer = self._make_trainer(cfg, training_args, tiny_dataset)

        logits = self._synthetic_router_logits(num_layers=2, num_tokens=10, num_experts=8)
        count, frac = trainer.get_expert_stats(logits)

        assert count.shape == (2, 2, 8)  # (L=2, K=2, E=8) — unchanged by latent_factor
        assert frac.shape  == (2, 2, 8)

    def test_expert_stats_counts_sum_to_T(self, training_args, tiny_dataset):
        """Each k-slot's count across all experts should equal T (token count)."""
        cfg = _moe_config(num_experts=8, num_active_experts=2, latent_factor=2)
        trainer = self._make_trainer(cfg, training_args, tiny_dataset)

        T = 15
        logits = self._synthetic_router_logits(num_layers=2, num_tokens=T, num_experts=8)
        count, _ = trainer.get_expert_stats(logits)

        assert (count.sum(dim=-1) == T).all()
