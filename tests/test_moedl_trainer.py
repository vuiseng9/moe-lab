"""
Tests for MoedlTrainer.

Tests cover:
1. Dense model training (num_experts=1)
2. MoE model training with lb_coeff=0
3. MoE model training with lb_coeff>0
4. Expert statistics computation
5. Wandb logging integration
"""
import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from transformers import TrainingArguments
from datasets import Dataset

from moelab.moedl import MoedlConfig, MoedlForCausalLM
from moelab.moedl.trainer import MoedlTrainer


@pytest.fixture
def dense_model_config():
    """Config for dense Moedl model."""
    return MoedlConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_experts=1,
        num_active_experts=1,
        lb_coeff=0.0,
    )


@pytest.fixture
def moe_model_config():
    """Config for MoE Moedl model."""
    return MoedlConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_experts=8,
        num_active_experts=2,
        lb_coeff=0.0,
    )


@pytest.fixture
def tiny_dataset():
    """Create a tiny dataset for training tests."""
    data = {
        "input_ids": [[1, 2, 3, 4, 5] for _ in range(10)],
        "labels": [[1, 2, 3, 4, 5] for _ in range(10)],
    }
    return Dataset.from_dict(data)


@pytest.fixture
def training_args(tmp_path):
    """Basic training arguments."""
    return TrainingArguments(
        output_dir=str(tmp_path),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_steps=1,
        save_steps=100,
        report_to=[],  # No reporting by default
        use_cpu=True,  # Force CPU to avoid device mismatch in tests
    )


class TestMoedlTrainerProperties:
    """Test MoedlTrainer property methods."""
    
    def test_cfg_property_caching(self, dense_model_config, training_args, tiny_dataset):
        """Test that cfg property is cached correctly."""
        model = MoedlForCausalLM(dense_model_config)
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
        )
        
        # First access should set the cache
        cfg = trainer.cfg
        assert cfg is not None
        assert cfg.vocab_size == 1000
        
        # Second access should return cached value
        assert trainer.cfg is cfg
    
    def test_is_moe_property_dense_model(self, dense_model_config, training_args, tiny_dataset):
        """Test is_moe property returns False for dense model."""
        model = MoedlForCausalLM(dense_model_config)
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
        )
        
        assert trainer.is_moe is False
        assert trainer.E == 1
        assert trainer.K == 1
    
    def test_is_moe_property_moe_model(self, moe_model_config, training_args, tiny_dataset):
        """Test is_moe property returns True for MoE model."""
        model = MoedlForCausalLM(moe_model_config)
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
        )
        
        assert trainer.is_moe is True
        assert trainer.E == 8
        assert trainer.K == 2
    
    def test_is_moe_property_caching(self, moe_model_config, training_args, tiny_dataset):
        """Test that is_moe property is cached."""
        model = MoedlForCausalLM(moe_model_config)
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
        )
        
        # First access
        is_moe_first = trainer.is_moe
        # Second access should return cached value
        is_moe_second = trainer.is_moe
        assert is_moe_first is is_moe_second


class TestMoedlTrainerDenseModel:
    """Test MoedlTrainer with dense models."""
    
    def test_dense_model_training_step(self, dense_model_config, training_args, tiny_dataset):
        """Test that dense model can perform a training step."""
        model = MoedlForCausalLM(dense_model_config)
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
        )
        
        # Get a batch
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 5)),
            "labels": torch.randint(0, 1000, (2, 5)),
        }
        
        # Compute loss
        loss, outputs = trainer.compute_loss(
            model, batch, num_items_in_batch=2, return_outputs=True
        )
        
        assert loss is not None
        assert torch.isfinite(loss)
        assert outputs.router_logits is None  # Dense model has no router
        assert outputs.aux_loss is None  # No aux loss for dense model
    
    def test_dense_model_no_wandb_logging(self, dense_model_config, training_args, tiny_dataset):
        """Test that dense model doesn't attempt MoE-specific logging."""
        model = MoedlForCausalLM(dense_model_config)
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
        )
        
        # Mock wandb
        mock_wandb = Mock()
        trainer._wb_handler = mock_wandb
        
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 5)),
            "labels": torch.randint(0, 1000, (2, 5)),
        }
        
        # Compute loss - should not call wandb for dense model
        loss = trainer.compute_loss(model, batch, num_items_in_batch=2)
        
        # wandb.log should not be called (no MoE stats to log)
        mock_wandb.log.assert_not_called()


class TestMoedlTrainerMoeModel:
    """Test MoedlTrainer with MoE models."""
    
    def test_moe_model_training_step_no_lb(self, moe_model_config, training_args, tiny_dataset):
        """Test MoE model training step without load balancing."""
        model = MoedlForCausalLM(moe_model_config)
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
        )
        
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 5)),
            "labels": torch.randint(0, 1000, (2, 5)),
        }
        
        loss, outputs = trainer.compute_loss(
            model, batch, num_items_in_batch=2, return_outputs=True
        )
        
        assert loss is not None
        assert torch.isfinite(loss)
        assert outputs.router_logits is not None
        assert len(outputs.router_logits) == 2  # 2 layers
        assert outputs.aux_loss is None  # lb_coeff=0
    
    def test_moe_model_training_step_with_lb(self, moe_model_config, training_args, tiny_dataset):
        """Test MoE model training step with load balancing."""
        # Enable load balancing
        moe_model_config.lb_coeff = 0.01
        model = MoedlForCausalLM(moe_model_config)
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
        )
        
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 5)),
            "labels": torch.randint(0, 1000, (2, 5)),
        }
        
        loss, outputs = trainer.compute_loss(
            model, batch, num_items_in_batch=2, return_outputs=True
        )
        
        assert loss is not None
        assert torch.isfinite(loss)
        assert outputs.router_logits is not None
        assert outputs.aux_loss is not None  # lb_coeff > 0
        assert outputs.aux_loss >= 0
    
    def test_moe_model_wandb_logging_no_lb(self, moe_model_config, training_args, tiny_dataset):
        """Test wandb logging for MoE model without load balancing."""
        model = MoedlForCausalLM(moe_model_config)
        
        # Enable wandb reporting
        training_args.report_to = ["wandb"]
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
        )
        
        # Mock wandb
        mock_wandb = Mock()
        trainer._wb_handler = mock_wandb
        
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 5)),
            "labels": torch.randint(0, 1000, (2, 5)),
        }
        
        trainer.compute_loss(model, batch, num_items_in_batch=2)
        
        # Check that wandb.log was called
        mock_wandb.log.assert_called_once()
        log_dict = mock_wandb.log.call_args[0][0]
        
        # Should have expert load stats
        assert any(key.startswith("moe/top1/load/e") for key in log_dict.keys())
        # Should have 8 experts (e000 to e007)
        expert_keys = [k for k in log_dict.keys() if k.startswith("moe/top1/load/e")]
        assert len(expert_keys) == 8
        
        # lb_loss should be nan (lb_coeff=0)
        assert "train/lb_loss" in log_dict
        import math
        assert math.isnan(log_dict["train/lb_loss"])
    
    def test_moe_model_wandb_logging_with_lb(self, moe_model_config, training_args, tiny_dataset):
        """Test wandb logging for MoE model with load balancing."""
        moe_model_config.lb_coeff = 0.01
        model = MoedlForCausalLM(moe_model_config)
        
        training_args.report_to = ["wandb"]
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
        )
        
        mock_wandb = Mock()
        trainer._wb_handler = mock_wandb
        
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 5)),
            "labels": torch.randint(0, 1000, (2, 5)),
        }
        
        trainer.compute_loss(model, batch, num_items_in_batch=2)
        
        mock_wandb.log.assert_called_once()
        log_dict = mock_wandb.log.call_args[0][0]
        
        # Should have expert load stats
        expert_keys = [k for k in log_dict.keys() if k.startswith("moe/top1/load/e")]
        assert len(expert_keys) == 8
        
        # lb_loss should be a finite number
        assert "train/lb_loss" in log_dict
        assert isinstance(log_dict["train/lb_loss"], float)
        assert log_dict["train/lb_loss"] >= 0


class TestMoedlTrainerExpertStats:
    """Test expert statistics computation."""
    
    def test_get_expert_stats_shape(self, moe_model_config, training_args, tiny_dataset):
        """Test that get_expert_stats returns correct shapes."""
        model = MoedlForCausalLM(moe_model_config)
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
        )
        
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 5)),
            "labels": torch.randint(0, 1000, (2, 5)),
        }
        
        with torch.no_grad():
            outputs = model(**batch)
        
        count, frac = trainer.get_expert_stats(outputs.router_logits)
        
        # count shape: (L, K, E)
        assert count.shape == (2, 2, 8)  # 2 layers, 2 active experts, 8 total experts
        # frac shape: (L, K, E)
        assert frac.shape == (2, 2, 8)
        
        # Check that counts are integers
        assert count.dtype == torch.int64
        # Check that fractions are floats
        assert frac.dtype == torch.float32
    
    def test_get_expert_stats_values(self, moe_model_config, training_args, tiny_dataset):
        """Test that get_expert_stats computes correct values."""
        model = MoedlForCausalLM(moe_model_config)
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
        )
        
        batch_size, seq_len = 2, 5
        total_tokens = batch_size * seq_len
        
        batch = {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
            "labels": torch.randint(0, 1000, (batch_size, seq_len)),
        }
        
        with torch.no_grad():
            outputs = model(**batch)
        
        count, frac = trainer.get_expert_stats(outputs.router_logits)
        
        # For each layer and k slot, counts should sum to total_tokens
        for layer_idx in range(2):
            for k_idx in range(2):
                assert count[layer_idx, k_idx, :].sum().item() == total_tokens
                # Fractions should sum to 1.0
                assert torch.isclose(frac[layer_idx, k_idx, :].sum(), torch.tensor(1.0))
    
    def test_get_expert_stats_distribution(self, moe_model_config, training_args, tiny_dataset):
        """Test that expert statistics show reasonable distribution."""
        model = MoedlForCausalLM(moe_model_config)
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
        )
        
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 10)),  # Larger batch for better stats
            "labels": torch.randint(0, 1000, (2, 10)),
        }
        
        with torch.no_grad():
            outputs = model(**batch)
        
        count, frac = trainer.get_expert_stats(outputs.router_logits)
        
        # Each expert should get at least some tokens (or be close to uniform for top-1)
        # Just verify that not all tokens go to one expert
        for layer_idx in range(2):
            for k_idx in range(2):
                # At least 2 experts should be used
                experts_used = (count[layer_idx, k_idx, :] > 0).sum().item()
                assert experts_used >= 2, f"Too few experts used: {experts_used}"
    
    def test_get_expert_stats_top1_load(self, moe_model_config, training_args, tiny_dataset):
        """Test that top-1 load computation is correct."""
        model = MoedlForCausalLM(moe_model_config)
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
        )
        
        batch_size, seq_len = 2, 5
        num_items_in_batch = batch_size * seq_len
        
        batch = {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
            "labels": torch.randint(0, 1000, (batch_size, seq_len)),
        }
        
        with torch.no_grad():
            outputs = model(**batch)
        
        count, frac = trainer.get_expert_stats(outputs.router_logits)
        
        # Top-1 load per expert (summed across layers)
        top1_load = count[:, 0, :].sum(dim=0) / num_items_in_batch
        
        # Should have 8 values (one per expert)
        assert top1_load.shape == (8,)
        
        # All values should be between 0 and num_layers
        assert (top1_load >= 0).all()
        assert (top1_load <= 2).all()  # 2 layers


class TestMoedlTrainerIntegration:
    """Integration tests for MoedlTrainer."""
    
    def test_full_training_step_moe_with_lb(self, moe_model_config, training_args, tiny_dataset):
        """Test a full training step with MoE model and load balancing."""
        moe_model_config.lb_coeff = 0.01
        model = MoedlForCausalLM(moe_model_config)
        
        training_args.report_to = ["wandb"]
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
        )
        
        # Mock wandb
        mock_wandb = Mock()
        trainer._wb_handler = mock_wandb
        
        # Perform a training step
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 5)),
            "labels": torch.randint(0, 1000, (2, 5)),
        }
        
        loss, outputs = trainer.compute_loss(
            model, batch, num_items_in_batch=2, return_outputs=True
        )
        
        # Verify loss is computed
        assert loss is not None
        assert torch.isfinite(loss)
        
        # Verify outputs
        assert outputs.router_logits is not None
        assert outputs.aux_loss is not None
        
        # Verify wandb logging
        mock_wandb.log.assert_called_once()
        log_dict = mock_wandb.log.call_args[0][0]
        
        # Should have expert stats and lb_loss
        assert "train/lb_loss" in log_dict
        assert any(key.startswith("moe/top1/load/e") for key in log_dict.keys())
        
        # lb_loss should be positive
        assert log_dict["train/lb_loss"] > 0
    
    def test_return_outputs_flag(self, moe_model_config, training_args, tiny_dataset):
        """Test that return_outputs flag works correctly."""
        model = MoedlForCausalLM(moe_model_config)
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
        )
        
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 5)),
            "labels": torch.randint(0, 1000, (2, 5)),
        }
        
        # Test with return_outputs=False
        result = trainer.compute_loss(model, batch, num_items_in_batch=2, return_outputs=False)
        assert isinstance(result, torch.Tensor)
        
        # Test with return_outputs=True
        result = trainer.compute_loss(model, batch, num_items_in_batch=2, return_outputs=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        loss, outputs = result
        assert isinstance(loss, torch.Tensor)
        assert outputs is not None


class TestMoedlTrainerBiasAdjustment:
    """Test bias adjustment behavior in MoedlTrainer (lb_gamma mechanism)."""
    
    @pytest.fixture
    def moe_config_with_biasing(self):
        """Config for MoE with biasing enabled."""
        return MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            lb_gamma=0.01,  # Biasing enabled
        )
    
    @pytest.fixture
    def moe_config_no_biasing(self):
        """Config for MoE without biasing."""
        return MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            lb_gamma=0.0,  # Biasing disabled
        )
    
    def test_bias_untouched_when_lb_gamma_zero(self, moe_config_no_biasing, training_args, tiny_dataset):
        """Test that biases are NOT modified when lb_gamma=0."""
        model = MoedlForCausalLM(moe_config_no_biasing)
        
        # Note: when lb_gamma=0, no e_score_bias buffer is created
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
        )
        
        # Verify no e_score_bias buffers exist
        for name, module in model.named_modules():
            assert not hasattr(module, 'e_score_bias'), \
                f"Module {name} should not have e_score_bias when lb_gamma=0"
        
        # Training should work without errors
        trainer.train()
    
    def test_bias_touched_when_lb_gamma_nonzero(self, moe_config_with_biasing, tiny_dataset):
        """Test that biases ARE modified when lb_gamma>0."""
        model = MoedlForCausalLM(moe_config_with_biasing)
        
        # Record initial bias values
        initial_biases = {}
        for name, module in model.named_modules():
            if hasattr(module, 'e_score_bias'):
                initial_biases[name] = module.e_score_bias.clone()
        
        assert len(initial_biases) > 0, "Should have e_score_bias buffers when lb_gamma > 0"
        
        # Need wandb to enable bias adjustment
        training_args = TrainingArguments(
            output_dir="/tmp/test_moedl_bias",
            per_device_train_batch_size=2,
            max_steps=2,
            logging_steps=1,
            save_steps=999999,
            report_to=["wandb"],
        )
        
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
        )
        
        # Train for one step
        trainer.train()
        
        # Verify at least some biases changed (move initial to same device for comparison)
        bias_changed = False
        for name, module in model.named_modules():
            if hasattr(module, 'e_score_bias'):
                initial = initial_biases[name].to(module.e_score_bias.device)
                if not torch.allclose(module.e_score_bias, initial):
                    bias_changed = True
                    break
        
        assert bias_changed, "At least some biases should change when lb_gamma > 0"
    
    def test_bias_untouched_when_using_lb_coeff(self, training_args, tiny_dataset):
        """Test that biases are NOT created/modified when using lb_coeff (penalty method)."""
        config = MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            lb_coeff=0.01,  # Penalty method
            lb_gamma=0.0,    # Biasing disabled
        )
        model = MoedlForCausalLM(config)
        
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
        )
        
        # Verify no e_score_bias buffers exist
        for name, module in model.named_modules():
            assert not hasattr(module, 'e_score_bias'), \
                f"Module {name} should not have e_score_bias when using lb_coeff"
        
        # Should train without errors
        trainer.train()
    
    def test_bias_adjustment_direction(self, tiny_dataset):
        """Test that bias adjustment reduces load imbalance."""
        # Use stronger gamma for clearer signal
        config = MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            lb_gamma=0.1,  # Strong biasing
        )
        model = MoedlForCausalLM(config)
        
        # Manually bias expert 0 to be heavily favored
        for module in model.modules():
            if hasattr(module, 'e_score_bias'):
                module.e_score_bias[0, 0] = 5.0  # Heavily favor expert 0
        
        # Need wandb to enable bias adjustment
        training_args = TrainingArguments(
            output_dir="/tmp/test_moedl_bias",
            per_device_train_batch_size=2,
            max_steps=2,
            logging_steps=1,
            save_steps=999999,
            report_to=["wandb"],
        )
        
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
        )
        
        # Get initial bias for expert 0
        initial_bias_e0 = None
        for module in model.modules():
            if hasattr(module, 'e_score_bias'):
                initial_bias_e0 = module.e_score_bias[0, 0].item()
                break
        
        # Train - expert 0 will be overloaded, bias should decrease
        trainer.train()
        
        # Check bias for expert 0 decreased
        final_bias_e0 = None
        for module in model.modules():
            if hasattr(module, 'e_score_bias'):
                final_bias_e0 = module.e_score_bias[0, 0].item()
                break
        
        assert final_bias_e0 < initial_bias_e0, \
            f"Bias for overloaded expert should decrease (was {initial_bias_e0}, now {final_bias_e0})"
    
    def test_adjust_balancing_biases_computation(self, moe_config_with_biasing):
        """Test the bias adjustment computation logic directly."""
        model = MoedlForCausalLM(moe_config_with_biasing)
        
        trainer = MoedlTrainer(
            model=model,
            args=TrainingArguments(output_dir="/tmp", max_steps=1, report_to=[]),
            train_dataset=Dataset.from_dict({"input_ids": [[1, 2]], "labels": [[1, 2]]}),
        )
        
        # Create mock load per expert: layer 0 heavily loaded on expert 0
        # Shape: (num_layers, num_experts)
        E = moe_config_with_biasing.num_experts
        
        # Get device from model
        device = next(model.parameters()).device
        
        load_per_expert = torch.zeros((2, E), device=device)
        load_per_expert[0, 0] = 0.5  # Layer 0: expert 0 overloaded (50% vs ideal 12.5% for 8 experts)
        load_per_expert[0, 1:] = 0.5 / (E - 1)  # Distribute rest evenly
        load_per_expert[1, :] = 1.0 / E  # Layer 1: perfectly balanced
        
        # Record initial biases
        initial_biases = {}
        for name, module in trainer.moe_modules.items():
            initial_biases[name] = module.e_score_bias.clone()
        
        # Apply adjustment
        global_bias_sum = trainer.adjust_balancing_biases(load_per_expert)
        
        # Check that layer 0's expert 0 bias decreased
        layer_0_module = list(trainer.moe_modules.values())[0]
        assert layer_0_module.e_score_bias[0, 0] < initial_biases[list(trainer.moe_modules.keys())[0]][0, 0], \
            "Overloaded expert's bias should decrease"
        
        # Check that layer 1's biases changed minimally (already balanced)
        layer_1_module = list(trainer.moe_modules.values())[1]
        max_change = (layer_1_module.e_score_bias - initial_biases[list(trainer.moe_modules.keys())[1]]).abs().max()
        assert max_change < 0.02, "Balanced layer should have minimal bias changes"
    
    def test_get_expert_stats_with_biasing(self, moe_config_with_biasing, training_args, tiny_dataset):
        """Test that get_expert_stats works correctly with biasing enabled."""
        model = MoedlForCausalLM(moe_config_with_biasing)
        
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
        )
        
        # Do a forward pass
        batch = tiny_dataset[:2]
        input_ids = torch.tensor(batch["input_ids"])
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        # Get stats
        count, frac = trainer.get_expert_stats(outputs.router_logits)
        
        # Verify shapes
        L = moe_config_with_biasing.num_hidden_layers
        K = moe_config_with_biasing.num_active_experts
        E = moe_config_with_biasing.num_experts
        
        assert count.shape == (L, K, E), f"count shape should be (L, K, E), got {count.shape}"
        assert frac.shape == (L, K, E), f"frac shape should be (L, K, E), got {frac.shape}"
        
        # Verify frac sums to 1.0 per (layer, k) slot
        frac_sums = frac.sum(dim=-1)
        torch.testing.assert_close(
            frac_sums,
            torch.ones_like(frac_sums),
            rtol=1e-5,
            atol=1e-5,
            msg="Fraction per (layer, k) slot should sum to 1.0"
        )
    
    def test_dense_model_no_bias_adjustment(self, dense_model_config, training_args, tiny_dataset):
        """Test that dense models don't have bias adjustment logic."""
        model = MoedlForCausalLM(dense_model_config)
        
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
        )
        
        # Verify trainer recognizes this is not MoE
        assert not trainer.is_moe, "Dense model should not be recognized as MoE"
        assert trainer.moe_modules is None or len(trainer.moe_modules) == 0, \
            "Dense model should have no MoE modules"
        
        # Training should work without issues
        trainer.train()
    
    def test_wandb_logging_with_biasing(self, moe_config_with_biasing, tmp_path, tiny_dataset):
        """Test wandb logging includes lb_bias_global_sum when biasing is enabled."""
        model = MoedlForCausalLM(moe_config_with_biasing)
        
        # Mock wandb object
        mock_wandb = Mock()
        
        training_args = TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=2,
            max_steps=1,
            logging_steps=1,
            save_steps=999999,
            report_to=["wandb"],
        )
        
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
        )
        
        # Inject mock wandb directly through _wb_handler
        trainer._wb_handler = mock_wandb
        
        # Get device from model
        device = next(model.parameters()).device
        
        # Perform training step
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 5), device=device),
            "labels": torch.randint(0, 1000, (2, 5), device=device),
        }
        
        loss, outputs = trainer.compute_loss(
            model, batch, num_items_in_batch=2, return_outputs=True
        )
        
        # Verify wandb logging was called
        mock_wandb.log.assert_called()
        log_dict = mock_wandb.log.call_args[0][0]
        
        # Should have lb_bias_global_sum
        assert "moe/lb_bias_global_sum" in log_dict, "Should log lb_bias_global_sum when lb_gamma > 0"
        
        # lb_loss should be NaN (not using penalty method)
        assert "train/lb_loss" in log_dict
        import math
        assert math.isnan(log_dict["train/lb_loss"]), "lb_loss should be NaN when not using penalty method"


class TestMoedlTrainerCapacityFactor:
    """Test trainer behavior with capacity factor (token dropping)."""
    
    @pytest.fixture
    def moe_config_with_capacity(self):
        """Config for MoE with capacity limiting enabled."""
        return MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            capacity_factor=1.0,
        )
    
    def test_capacity_logging_disabled_when_cf_zero(self, moe_model_config, tmp_path, tiny_dataset):
        """Test that token_drop_ratio is NaN when capacity_factor=0."""
        model = MoedlForCausalLM(moe_model_config)
        
        mock_wandb = Mock()
        
        training_args = TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=2,
            max_steps=1,
            logging_steps=1,
            save_steps=999999,
            report_to=["wandb"],
        )
        
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
        )
        
        trainer._wb_handler = mock_wandb
        
        device = next(model.parameters()).device
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 5), device=device),
            "labels": torch.randint(0, 1000, (2, 5), device=device),
        }
        
        loss, outputs = trainer.compute_loss(
            model, batch, num_items_in_batch=2, return_outputs=True
        )
        
        mock_wandb.log.assert_called()
        log_dict = mock_wandb.log.call_args[0][0]
        
        assert "train/token_drop_ratio" in log_dict
        import math
        assert math.isnan(log_dict["train/token_drop_ratio"]), \
            "token_drop_ratio should be NaN when capacity_factor=0"
    
    def test_capacity_logging_enabled_when_cf_nonzero(self, moe_config_with_capacity, tmp_path, tiny_dataset):
        """Test that token_drop_ratio is logged when capacity_factor>0."""
        model = MoedlForCausalLM(moe_config_with_capacity)
        
        mock_wandb = Mock()
        
        training_args = TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=2,
            max_steps=1,
            logging_steps=1,
            save_steps=999999,
            report_to=["wandb"],
        )
        
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
        )
        
        trainer._wb_handler = mock_wandb
        
        device = next(model.parameters()).device
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 10), device=device),
            "labels": torch.randint(0, 1000, (2, 10), device=device),
        }
        
        loss, outputs = trainer.compute_loss(
            model, batch, num_items_in_batch=2, return_outputs=True
        )
        
        mock_wandb.log.assert_called()
        log_dict = mock_wandb.log.call_args[0][0]
        
        assert "train/token_drop_ratio" in log_dict
        # Should be a valid number (not NaN)
        import math
        assert not math.isnan(log_dict["train/token_drop_ratio"]), \
            "token_drop_ratio should be a number when capacity_factor>0"
        assert log_dict["train/token_drop_ratio"] >= 0.0, \
            "token_drop_ratio should be non-negative"
    
    def test_capacity_n_drop_counter_updates(self, moe_config_with_capacity, tiny_dataset):
        """Test that n_drop counters are tracked per layer."""
        model = MoedlForCausalLM(moe_config_with_capacity)
        
        training_args = TrainingArguments(
            output_dir="/tmp",
            per_device_train_batch_size=2,
            max_steps=1,
            report_to=[],
        )
        
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
        )
        
        # Check n_drop exists and is accessible
        assert trainer.moe_modules is not None
        for module in trainer.moe_modules.values():
            assert hasattr(module, 'n_drop')
            assert module.n_drop >= 0  # Non-negative
    
    def test_capacity_dense_model_no_drops(self, dense_model_config, tmp_path, tiny_dataset):
        """Test that dense models don't log token drops."""
        model = MoedlForCausalLM(dense_model_config)
        
        mock_wandb = Mock()
        
        training_args = TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=2,
            max_steps=1,
            logging_steps=1,
            save_steps=999999,
            report_to=["wandb"],
        )
        
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
        )
        
        trainer._wb_handler = mock_wandb
        
        device = next(model.parameters()).device
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 5), device=device),
            "labels": torch.randint(0, 1000, (2, 5), device=device),
        }
        
        loss = trainer.compute_loss(model, batch, num_items_in_batch=2)
        
        # Dense model shouldn't call wandb logging
        mock_wandb.log.assert_not_called()
    
    def test_capacity_with_lb_coeff(self, tmp_path, tiny_dataset):
        """Test capacity_factor logging with lb_coeff load balancing."""
        config = MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            capacity_factor=1.0,
            lb_coeff=0.01,  # Traditional load balancing
        )
        model = MoedlForCausalLM(config)
        
        mock_wandb = Mock()
        
        training_args = TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=2,
            max_steps=1,
            logging_steps=1,
            save_steps=999999,
            report_to=["wandb"],
        )
        
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
        )
        
        trainer._wb_handler = mock_wandb
        
        device = next(model.parameters()).device
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 10), device=device),
            "labels": torch.randint(0, 1000, (2, 10), device=device),
        }
        
        loss, outputs = trainer.compute_loss(
            model, batch, num_items_in_batch=2, return_outputs=True
        )
        
        mock_wandb.log.assert_called()
        log_dict = mock_wandb.log.call_args[0][0]
        
        # Should log both token_drop_ratio and load balancing metrics
        assert "train/token_drop_ratio" in log_dict
        assert "train/lb_loss" in log_dict
        import math
        assert not math.isnan(log_dict["train/token_drop_ratio"])
        assert not math.isnan(log_dict["train/lb_loss"])
        assert log_dict["train/lb_loss"] >= 0.0
    
    def test_capacity_with_lb_gamma(self, tmp_path, tiny_dataset):
        """Test capacity_factor logging with lb_gamma load balancing."""
        config = MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            capacity_factor=1.0,
            lb_gamma=0.01,  # Score bias load balancing
        )
        model = MoedlForCausalLM(config)
        
        mock_wandb = Mock()
        
        training_args = TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=2,
            max_steps=1,
            logging_steps=1,
            save_steps=999999,
            report_to=["wandb"],
        )
        
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
        )
        
        trainer._wb_handler = mock_wandb
        
        device = next(model.parameters()).device
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 10), device=device),
            "labels": torch.randint(0, 1000, (2, 10), device=device),
        }
        
        loss, outputs = trainer.compute_loss(
            model, batch, num_items_in_batch=2, return_outputs=True
        )
        
        mock_wandb.log.assert_called()
        log_dict = mock_wandb.log.call_args[0][0]
        
        # Should log both token_drop_ratio and bias adjustment metrics
        assert "train/token_drop_ratio" in log_dict
        assert "moe/lb_bias_global_sum" in log_dict
        import math
        assert not math.isnan(log_dict["train/token_drop_ratio"])
        assert not math.isnan(log_dict["moe/lb_bias_global_sum"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
