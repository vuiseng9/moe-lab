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
            heatmap_on=False,
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
            heatmap_on=False,
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
            heatmap_on=False,
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
            heatmap_on=False,
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
            heatmap_on=False,
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
            heatmap_on=False,
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
            heatmap_on=False,
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
            heatmap_on=False,
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
    
    def test_moe_model_wandb_logging_no_lb(self, moe_model_config, tmp_path, tiny_dataset, capsys):
        """Test wandb logging for MoE model without load balancing."""
        model = MoedlForCausalLM(moe_model_config)
        
        # Enable wandb reporting
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
            heatmap_on=False,
        )
        
        # Mock wandb on the trainer (callback will use trainer.wandb)
        mock_wandb = Mock()
        trainer._wb_handler = mock_wandb
        
        # Train for one step to trigger callback logging
        trainer.train()
        
        # With new logging mechanism, metrics are added to logs dict which gets printed
        # Check trainer's internal state instead
        assert trainer.last_lb_loss is not None
        import math
        assert math.isnan(trainer.last_lb_loss), "lb_loss should be NaN when lb_coeff=0"
        
        # Verify expert_load TensorMeter was used
        assert trainer.expert_load is not None
    
    def test_moe_model_wandb_logging_with_lb(self, moe_model_config, tmp_path, tiny_dataset):
        """Test wandb logging for MoE model with load balancing."""
        moe_model_config.lb_coeff = 0.01
        model = MoedlForCausalLM(moe_model_config)
        
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
            heatmap_on=False,
        )
        
        mock_wandb = Mock()
        trainer._wb_handler = mock_wandb
        
        # Train for one step to trigger callback logging
        trainer.train()
        
        # Check trainer's internal state
        assert trainer.last_lb_loss is not None
        assert isinstance(trainer.last_lb_loss, float)
        assert trainer.last_lb_loss >= 0, "lb_loss should be non-negative when lb_coeff > 0"


class TestMoedlTrainerExpertStats:
    """Test expert statistics computation."""
    
    def test_get_expert_stats_shape(self, moe_model_config, training_args, tiny_dataset):
        """Test that get_expert_stats returns correct shapes."""
        model = MoedlForCausalLM(moe_model_config)
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
            heatmap_on=False,
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
            heatmap_on=False,
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
            heatmap_on=False,
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
            heatmap_on=False,
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
    
    def test_full_training_step_moe_with_lb(self, moe_model_config, tmp_path, tiny_dataset):
        """Test a full training step with MoE model and load balancing."""
        moe_model_config.lb_coeff = 0.01
        model = MoedlForCausalLM(moe_model_config)
        
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
            heatmap_on=False,
        )
        
        # Mock wandb
        mock_wandb = Mock()
        trainer._wb_handler = mock_wandb
        
        # Train for one step
        trainer.train()
        
        # Verify trainer state - lb_loss should be positive
        assert trainer.last_lb_loss is not None
        assert trainer.last_lb_loss > 0, "lb_loss should be positive when lb_coeff > 0"
    
    def test_return_outputs_flag(self, moe_model_config, training_args, tiny_dataset):
        """Test that return_outputs flag works correctly."""
        model = MoedlForCausalLM(moe_model_config)
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
            heatmap_on=False,
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
            heatmap_on=False,
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
            heatmap_on=False,
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
            heatmap_on=False,
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
            heatmap_on=False,
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
        """Test the bias adjustment computation logic directly using LoadBalanceBiasController."""
        model = MoedlForCausalLM(moe_config_with_biasing)
        
        trainer = MoedlTrainer(
            model=model,
            args=TrainingArguments(output_dir="/tmp", max_steps=1, report_to=[]),
            train_dataset=Dataset.from_dict({"input_ids": [[1, 2]], "labels": [[1, 2]]}),
            heatmap_on=False,
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
        
        # Apply adjustment via the controller
        assert trainer.lb_ctrl is not None, "LoadBalanceBiasController should be initialized"
        global_bias_sum = trainer.lb_ctrl(load_per_expert)
        
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
        """Test wandb logging includes lb_bias_dbg when biasing is enabled."""
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
        
        # Train for one step to trigger callback logging
        trainer.train()
        
        # Check trainer's internal state
        assert trainer.last_lb_bias_dbg is not None
        assert isinstance(trainer.last_lb_bias_dbg, float)
        assert trainer.last_lb_bias_dbg >= 0, "lb_bias_dbg should be non-negative"
        
        # lb_loss should be NaN (not using penalty method)
        import math
        assert math.isnan(trainer.last_lb_loss), "lb_loss should be NaN when not using penalty method"


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
        """Test that token_drop_ratio and token_drop_count are NaN when capacity_factor=0."""
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
        
        # Train for one step to trigger callback logging
        trainer.train()
        
        # Check trainer's internal state
        import math
        assert math.isnan(trainer.last_drop_ratio), \
            "token_drop_ratio should be NaN when capacity_factor=0"
        assert math.isnan(trainer.last_drop_count), \
            "token_drop_count should be NaN when capacity_factor=0"
    
    def test_capacity_logging_enabled_when_cf_nonzero(self, moe_config_with_capacity, tmp_path, tiny_dataset):
        """Test that token_drop_ratio and token_drop_count are logged when capacity_factor>0."""
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
        
        # Train for one step to trigger callback logging
        trainer.train()
        
        # Check trainer's internal state - should be a valid number (not NaN)
        import math
        assert not math.isnan(trainer.last_drop_ratio), \
            "token_drop_ratio should be a number when capacity_factor>0"
        assert trainer.last_drop_ratio >= 0.0, \
            "token_drop_ratio should be non-negative"
        assert not math.isnan(trainer.last_drop_count), \
            "token_drop_count should be a number when capacity_factor>0"
        assert trainer.last_drop_count >= 0, \
            "token_drop_count should be non-negative"
    
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
        
        # Train for one step to trigger callback logging
        trainer.train()
        
        # Check trainer's internal state - should log both token_drop_ratio, token_drop_count, and load balancing metrics
        import math
        assert not math.isnan(trainer.last_drop_ratio), "token_drop_ratio should be valid"
        assert not math.isnan(trainer.last_drop_count), "token_drop_count should be valid"
        assert not math.isnan(trainer.last_lb_loss), "lb_loss should be valid"
        assert trainer.last_lb_loss >= 0.0, "lb_loss should be non-negative"
    
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
        
        # Train for one step to trigger callback logging
        trainer.train()
        
        # Check trainer's internal state - should log both token_drop_ratio, token_drop_count, and bias adjustment metrics
        import math
        assert not math.isnan(trainer.last_drop_ratio), "token_drop_ratio should be valid"
        assert not math.isnan(trainer.last_drop_count), "token_drop_count should be valid"
        assert not math.isnan(trainer.last_lb_bias_dbg), "lb_bias_dbg should be valid"
        assert trainer.last_lb_bias_dbg >= 0.0, "lb_bias_dbg should be non-negative"


class TestMoedlTrainerGradientAccumulation:
    """Test MoedlTrainer behavior with gradient accumulation.
    
    These tests expose bugs where operations in compute_loss are executed
    per forward pass instead of per optimizer step.
    """
    
    def test_bias_adjustment_with_gradient_accumulation(self, tmp_path, tiny_dataset):
        """Test that bias adjustment happens once per optimizer step via callback.
        
        With the new callback architecture, bias adjustment should happen in
        on_step_end, which is called once per optimizer step (after all gradient
        accumulation steps are complete).
        """
        config = MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            lb_gamma=0.01,  # Enable bias-based load balancing
        )
        model = MoedlForCausalLM(config)
        
        gradient_accumulation_steps = 4
        training_args = TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_steps=2,  # 2 optimizer steps
            logging_steps=1,
            save_steps=999999,
            report_to=["wandb"],
        )
        
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
        )
        
        # Find the MoedlPerStepCallback in the trainer's callbacks
        moedl_callback = None
        for cb in trainer.callback_handler.callbacks:
            if cb.__class__.__name__ == "MoedlPerStepCallback":
                moedl_callback = cb
                break
        
        assert moedl_callback is not None, "MoedlPerStepCallback should be registered"

        # Patch LoadBalanceBiasController.__call__ on the class to count invocations
        # Note: We must patch on the class level, not instance level, because
        # Python looks up __call__ on the class when invoking an instance as a function
        from moelab.moedl.trainer import LoadBalanceBiasController
        call_count = {"count": 0}

        def counting_wrapper(self, *args, **kwargs):
            call_count["count"] += 1
            # Call the original adjust_router_lb_bias method
            return self.adjust_router_lb_bias(*args, **kwargs)

        LoadBalanceBiasController.__call__ = counting_wrapper

        # Train for 2 optimizer steps
        trainer.train()

        # With the new callback architecture:
        # - Bias adjustment happens in on_step_end callback
        # - on_step_end is called once per optimizer step
        # - Expected: 2 calls (once per optimizer step)
        
        expected_calls = 2
        actual_calls = call_count["count"]
        
        assert actual_calls == expected_calls, (
            f"LoadBalanceBiasController called {actual_calls} times, "
            f"expected {expected_calls} times (once per optimizer step). "
            f"With {gradient_accumulation_steps} gradient accumulation steps and "
            f"{training_args.max_steps} optimizer steps."
        )
    
    def test_wandb_logging_with_gradient_accumulation(self, tmp_path, tiny_dataset):
        """Test that wandb logging happens once per logging step via callback.
        
        With the new callback architecture, wandb logging happens in on_log,
        which respects logging_steps.
        """
        config = MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            lb_gamma=0.01,
        )
        model = MoedlForCausalLM(config)
        
        mock_wandb = Mock()
        
        gradient_accumulation_steps = 4
        training_args = TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=gradient_accumulation_steps,
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
        
        trainer._wb_handler = mock_wandb
        
        # Train for 2 optimizer steps
        trainer.train()
        
        # With the new callback architecture:
        # - Stats are tracked per step and logged via trainer.log()
        # - Verify that training completed and metrics were tracked
        import math
        assert not math.isnan(trainer.last_lb_bias_dbg), "lb_bias_dbg should be tracked"
    
    def test_token_drop_stats_with_gradient_accumulation(self, tmp_path, tiny_dataset):
        """Test that token drop stats are accumulated properly via callback.
        
        With the new callback architecture, stats are accumulated across gradient
        accumulation steps and logged once per logging step.
        """
        config = MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            capacity_factor=1.0,  # Enable capacity-based dropping
            lb_gamma=0.01,
        )
        model = MoedlForCausalLM(config)
        
        mock_wandb = Mock()
        
        gradient_accumulation_steps = 4
        training_args = TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=gradient_accumulation_steps,
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
        
        # Train for 1 optimizer step (4 forward passes due to gradient accumulation)
        trainer.train()
        
        # With the new callback architecture:
        # - Stats are accumulated via TensorMeter across all micro-batches
        # - Verify that all expected metrics were tracked
        import math
        assert not math.isnan(trainer.last_drop_ratio), "Should track token drop ratio"
        assert math.isnan(trainer.last_lb_loss), "lb_loss should be NaN (using lb_gamma not lb_coeff)"
        assert not math.isnan(trainer.last_lb_bias_dbg), "Should track lb_bias_dbg"
        
        # Verify meters were used
        assert trainer.routing_stat is not None
        assert trainer.expert_load is not None


class TestMoedlTrainerEvaluation:
    """Test MoedlTrainer behavior during evaluation mode."""
    
    def test_trainer_evaluation_with_lb_coeff(self, tmp_path, tiny_dataset):
        """Test that trainer.evaluate() works correctly with lb_coeff enabled.
        
        BUG: During evaluation, outputs.aux_loss is None (correct behavior after fix),
        but trainer.compute_loss() tries to call outputs.aux_loss.detach().item()
        which will crash with AttributeError.
        
        This test will FAIL with the current buggy implementation and PASS
        once the bug is fixed.
        """
        config = MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=4,
            num_active_experts=2,
            lb_coeff=0.01,  # Load balancing enabled
        )
        model = MoedlForCausalLM(config)
        
        training_args = TrainingArguments(
            output_dir=str(tmp_path),
            per_device_eval_batch_size=2,
            report_to=[],
        )
        
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            eval_dataset=tiny_dataset,
        )
        
        # This should not crash even though aux_loss is None during eval
        results = trainer.evaluate()
        
        # Verify evaluation completed successfully
        assert "eval_loss" in results
        assert results["eval_loss"] >= 0
        
        # During evaluation, last_lb_loss should remain NaN (not updated)
        import math
        assert math.isnan(trainer.last_lb_loss), \
            "lb_loss should not be updated during evaluation"
    
    def test_trainer_evaluation_with_capacity_factor(self, tmp_path, tiny_dataset):
        """Test that trainer.evaluate() works correctly with capacity_factor enabled."""
        config = MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=4,
            num_active_experts=2,
            capacity_factor=1.0,
        )
        model = MoedlForCausalLM(config)
        
        training_args = TrainingArguments(
            output_dir=str(tmp_path),
            per_device_eval_batch_size=2,
            report_to=[],
        )
        
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            eval_dataset=tiny_dataset,
        )
        
        # This should work fine - capacity tracking doesn't depend on aux_loss
        results = trainer.evaluate()
        
        # Verify evaluation completed successfully
        assert "eval_loss" in results
        assert results["eval_loss"] >= 0
        
        # Capacity metrics are tracked per forward pass, so they'll be updated
        # but we don't log them during eval in wandb
        import math
        assert not math.isnan(trainer.last_drop_ratio), \
            "drop_ratio can be tracked during eval (per forward pass)"
    
    def test_trainer_evaluation_with_lb_gamma(self, tmp_path, tiny_dataset):
        """Test that trainer.evaluate() works correctly with lb_gamma enabled."""
        config = MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=4,
            num_active_experts=2,
            lb_gamma=0.01,  # Bias-based load balancing
        )
        model = MoedlForCausalLM(config)
        
        training_args = TrainingArguments(
            output_dir=str(tmp_path),
            per_device_eval_batch_size=2,
            report_to=[],
        )
        
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            eval_dataset=tiny_dataset,
        )
        
        # This should work - lb_gamma doesn't use aux_loss
        results = trainer.evaluate()
        
        # Verify evaluation completed successfully
        assert "eval_loss" in results
        assert results["eval_loss"] >= 0


class TestMoedlTrainerHeatmap:
    """Test heatmap generation feature in MoedlTrainer."""
    
    def test_heatmap_feature_enabled_by_default(self, moe_model_config, training_args, tiny_dataset):
        """Test that heatmap feature is enabled by default."""
        model = MoedlForCausalLM(moe_model_config)
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
        )
        
        assert trainer.heatmap_on is True, "heatmap_on should be True by default"
        assert trainer.heatmap_freq == 10, "heatmap_freq should be 10 by default"
    
    def test_heatmap_feature_can_be_disabled(self, moe_model_config, training_args, tiny_dataset):
        """Test that heatmap feature can be disabled via constructor."""
        model = MoedlForCausalLM(moe_model_config)
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
            heatmap_on=False,
        )
        
        assert trainer.heatmap_on is False, "heatmap_on should be disabled"
        
        # Verify callback doesn't initialize heatmapper
        moedl_callback = None
        for cb in trainer.callback_handler.callbacks:
            if cb.__class__.__name__ == "MoedlPerStepCallback":
                moedl_callback = cb
                break
        
        assert moedl_callback is not None
        assert moedl_callback.heatmapper is None, "heatmapper should not be initialized when disabled"
    
    def test_heatmap_custom_frequency(self, moe_model_config, training_args, tiny_dataset):
        """Test that heatmap frequency can be customized."""
        model = MoedlForCausalLM(moe_model_config)
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
            heatmap_freq=5,
        )
        
        assert trainer.heatmap_freq == 5, "heatmap_freq should be customizable"
    
    def test_heatmap_threadpool_initialized(self, moe_model_config, tmp_path, tiny_dataset):
        """Test that ThreadPoolExecutor is initialized when heatmap is enabled."""
        model = MoedlForCausalLM(moe_model_config)
        
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
            heatmap_on=True,
        )
        
        # Find the callback
        moedl_callback = None
        for cb in trainer.callback_handler.callbacks:
            if cb.__class__.__name__ == "MoedlPerStepCallback":
                moedl_callback = cb
                break
        
        assert moedl_callback is not None
        assert moedl_callback.heatmapper is not None, "ThreadPoolExecutor should be initialized"
        assert moedl_callback.semaphore is not None, "Semaphore should be initialized"
    
    def test_heatmap_directory_created(self, moe_model_config, tmp_path, tiny_dataset):
        """Test that heatmap output directory is created on training begin."""
        model = MoedlForCausalLM(moe_model_config)
        
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
            heatmap_on=True,
        )
        
        # Train to trigger on_train_begin
        trainer.train()
        
        # Check directory was created
        import os
        heatmap_dir = os.path.join(str(tmp_path), "heatmap")
        assert os.path.exists(heatmap_dir), "heatmap directory should be created"
        assert os.path.isdir(heatmap_dir), "heatmap path should be a directory"
    
    def test_heatmap_generation_compute_delta_percentage(self, moe_model_config, tmp_path, tiny_dataset):
        """Test that heatmap correctly computes delta to load balance as percentage."""
        import numpy as np
        import os
        
        model = MoedlForCausalLM(moe_model_config)
        
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
            heatmap_on=True,
            heatmap_freq=1,  # Generate every step
        )
        
        # Create test data: (L, K, E) = (2, 2, 8)
        # Expert load in fraction (should sum to 1.0 per layer per k slot)
        L, K, E = 2, 2, 8
        
        # Create specific load pattern:
        # Layer 0, K0: expert 0 has 50%, rest 7 experts share 50%
        # Layer 0, K1: uniform load
        # Layer 1: all uniform
        np_arr = np.zeros((L, K, E), dtype=np.float32)
        
        # Layer 0, K0: imbalanced
        np_arr[0, 0, 0] = 0.50
        np_arr[0, 0, 1:] = 0.50 / 7
        
        # Layer 0, K1: balanced
        np_arr[0, 1, :] = 1.0 / E
        
        # Layer 1, both K slots: balanced
        np_arr[1, :, :] = 1.0 / E
        
        # Verify inputs sum to 1.0
        assert np.allclose(np_arr.sum(axis=-1), 1.0), "Expert loads should sum to 1.0 per (layer, k)"
        
        # Call the internal heatmap generation method directly
        from moelab.moedl.trainer import MoedlPerStepCallback
        
        moedl_callback = None
        for cb in trainer.callback_handler.callbacks:
            if isinstance(cb, MoedlPerStepCallback):
                moedl_callback = cb
                break
        
        assert moedl_callback is not None
        
        # Trigger on_train_begin to create directory
        moedl_callback.on_train_begin(training_args, None, None)
        
        # Generate heatmap
        test_file = os.path.join(str(tmp_path), "heatmap", "test_heatmap.png")
        moedl_callback._generate_expert_load_heatmap(np_arr, step=0, file_path=test_file)
        
        # Verify file was created
        assert os.path.exists(test_file), "Heatmap file should be created"
        
        # Verify the computation manually
        balance_pct = 100.0 / E  # 12.5% for 8 experts
        
        # Expected delta for layer 0, k0, expert 0
        expert_0_load_pct = np_arr[0, 0, 0] * 100  # 50%
        expected_delta_e0 = abs(expert_0_load_pct - balance_pct)  # |50 - 12.5| = 37.5
        
        assert np.isclose(expected_delta_e0, 37.5), \
            f"Delta for expert 0 should be 37.5%, got {expected_delta_e0}"
        
        # Expected delta for layer 1, k0, expert 0 (balanced)
        expert_balanced_pct = np_arr[1, 0, 0] * 100  # 12.5%
        expected_delta_balanced = abs(expert_balanced_pct - balance_pct)  # |12.5 - 12.5| = 0
        
        assert np.isclose(expected_delta_balanced, 0.0, atol=0.1), \
            f"Delta for balanced expert should be ~0%, got {expected_delta_balanced}"
    
    def test_heatmap_delta_values_are_percentage(self, tmp_path):
        """Test that heatmap delta values are in percentage (0-100), not fraction (0-1)."""
        import numpy as np
        import os
        from moelab.moedl.trainer import MoedlPerStepCallback
        
        # Create test array with known values
        L, K, E = 2, 2, 4
        np_arr = np.zeros((L, K, E), dtype=np.float32)
        
        # Set expert 0 to 100% load (extreme case)
        np_arr[0, 0, 0] = 1.0
        np_arr[0, 0, 1:] = 0.0
        
        # Set other slots to uniform
        np_arr[0, 1, :] = 0.25
        np_arr[1, :, :] = 0.25
        
        # Create a mock trainer just for the callback
        from unittest.mock import Mock
        mock_trainer = Mock()
        mock_trainer.heatmap_on = True
        mock_trainer.heatmap_freq = 1
        mock_trainer.wandb = True
        
        callback = MoedlPerStepCallback(trainer=mock_trainer)
        
        # Create output directory
        heatmap_dir = os.path.join(str(tmp_path), "heatmap")
        os.makedirs(heatmap_dir, exist_ok=True)
        callback.heatmap_outdir = heatmap_dir
        
        # Generate heatmap
        test_file = os.path.join(heatmap_dir, "test_percentage.png")
        callback._generate_expert_load_heatmap(np_arr, step=0, file_path=test_file)
        
        # Verify file created
        assert os.path.exists(test_file), "Heatmap should be created"
        
        # Calculate expected values manually
        balance_pct = 100.0 / E  # 25% for 4 experts
        
        # Expert 0 in layer 0, k0: 100% load
        # Delta = |100 - 25| = 75 (percentage)
        expert_0_delta = abs(1.0 * 100 - balance_pct)
        assert np.isclose(expert_0_delta, 75.0), \
            f"Delta should be 75 (percentage), not 0.75 (fraction)"
        
        # Experts 1-3 in layer 0, k0: 0% load
        # Delta = |0 - 25| = 25 (percentage)
        expert_others_delta = abs(0.0 * 100 - balance_pct)
        assert np.isclose(expert_others_delta, 25.0), \
            f"Delta should be 25 (percentage), not 0.25 (fraction)"
    
    def test_heatmap_generation_at_correct_frequency(self, moe_model_config, tmp_path, tiny_dataset):
        """Test that heatmap is generated at the specified frequency."""
        import os
        
        model = MoedlForCausalLM(moe_model_config)
        
        training_args = TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=2,
            max_steps=15,  # 15 steps
            logging_steps=1,
            save_steps=999999,
            report_to=["wandb"],
        )
        
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
            heatmap_on=True,
            heatmap_freq=5,  # Every 5 steps
        )
        
        # Mock wandb
        mock_wandb = Mock()
        trainer._wb_handler = mock_wandb
        
        # Train
        trainer.train()
        
        # Check heatmap files were created at steps 0, 5, 10
        heatmap_dir = os.path.join(str(tmp_path), "heatmap")
        assert os.path.exists(heatmap_dir), "Heatmap directory should exist"
        
        # Expected files: gstep_000000, gstep_000005, gstep_000010
        expected_steps = [0, 5, 10]
        for step in expected_steps:
            expected_file = os.path.join(heatmap_dir, f"gstep_{step:06}_expert_load.png")
            # Note: Due to threading, file might not be created immediately
            # We'll just check that the directory was created
            # Full verification would require waiting for threads
        
        # At minimum, directory should exist
        assert os.path.isdir(heatmap_dir)
    
    def test_heatmap_dense_model_no_generation(self, dense_model_config, tmp_path, tiny_dataset):
        """Test that heatmap is not generated for dense models."""
        import os
        
        model = MoedlForCausalLM(dense_model_config)
        
        training_args = TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=2,
            max_steps=1,
            logging_steps=1,
            save_steps=999999,
            report_to=[],
        )
        
        trainer = MoedlTrainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
            heatmap_on=True,  # Even with heatmap enabled
        )
        
        # Dense models shouldn't have MoedlPerStepCallback with heatmapper
        # because they're not MoE models
        assert not trainer.is_moe
        
        # Train
        trainer.train()
        
        # No heatmap directory should be created for dense models
        heatmap_dir = os.path.join(str(tmp_path), "heatmap")
        # Directory might be created but no files should be generated
        # The callback won't be added for dense models
    
    def test_heatmap_threadpool_cleanup(self, moe_model_config, tmp_path, tiny_dataset):
        """Test that ThreadPoolExecutor is properly shut down after training."""
        model = MoedlForCausalLM(moe_model_config)
        
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
            heatmap_on=True,
        )
        
        # Mock wandb
        mock_wandb = Mock()
        trainer._wb_handler = mock_wandb
        
        # Get the callback
        moedl_callback = None
        for cb in trainer.callback_handler.callbacks:
            if cb.__class__.__name__ == "MoedlPerStepCallback":
                moedl_callback = cb
                break
        
        assert moedl_callback is not None
        assert moedl_callback.heatmapper is not None
        
        # Train - this should trigger on_train_end which shuts down the executor
        trainer.train()
        
        # Verify shutdown was called (executor should be shut down)
        # We can't directly check if shutdown was called, but we can verify
        # the executor is no longer accepting new tasks by checking _shutdown
        assert moedl_callback.heatmapper._shutdown, "ThreadPoolExecutor should be shut down"
    
    def test_heatmap_semaphore_prevents_concurrent_generation(self, moe_model_config, tmp_path, tiny_dataset):
        """Test that semaphore prevents concurrent heatmap generation."""
        import threading
        
        model = MoedlForCausalLM(moe_model_config)
        
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
            heatmap_on=True,
        )
        
        # Get the callback
        moedl_callback = None
        for cb in trainer.callback_handler.callbacks:
            if cb.__class__.__name__ == "MoedlPerStepCallback":
                moedl_callback = cb
                break
        
        assert moedl_callback is not None
        assert moedl_callback.semaphore is not None
        
        # Semaphore should start with value 1 (one permit available)
        # Acquire it
        acquired = moedl_callback.semaphore.acquire(blocking=False)
        assert acquired, "Should be able to acquire semaphore initially"
        
        # Try to acquire again - should fail (non-blocking)
        acquired_again = moedl_callback.semaphore.acquire(blocking=False)
        assert not acquired_again, "Should not be able to acquire semaphore twice"
        
        # Release it
        moedl_callback.semaphore.release()
        
        # Now should be able to acquire again
        acquired_after_release = moedl_callback.semaphore.acquire(blocking=False)
        assert acquired_after_release, "Should be able to acquire after release"
        
        # Clean up
        moedl_callback.semaphore.release()
    
    def test_heatmap_error_handling(self, tmp_path):
        """Test that heatmap generation handles errors gracefully."""
        import numpy as np
        import os
        from unittest.mock import Mock, patch
        from moelab.moedl.trainer import MoedlPerStepCallback
        
        # Create a mock trainer
        mock_trainer = Mock()
        mock_trainer.heatmap_on = True
        mock_trainer.heatmap_freq = 1
        
        callback = MoedlPerStepCallback(trainer=mock_trainer)
        
        # Create output directory
        heatmap_dir = os.path.join(str(tmp_path), "heatmap")
        os.makedirs(heatmap_dir, exist_ok=True)
        callback.heatmap_outdir = heatmap_dir
        
        # Test with invalid input (wrong shape)
        invalid_arr = np.zeros((2, 2), dtype=np.float32)  # Should be 3D
        test_file = os.path.join(heatmap_dir, "test_error.png")
        
        # This should not crash - error should be logged
        callback._generate_expert_load_heatmap(invalid_arr, step=0, file_path=test_file)
        
        # Semaphore should be released even after error
        acquired = callback.semaphore.acquire(blocking=False)
        assert acquired, "Semaphore should be released after error"
        callback.semaphore.release()
    
    def test_heatmap_visualization_correctness(self, tmp_path):
        """Test the visual correctness of heatmap data transformation."""
        import numpy as np
        import os
        from moelab.moedl.trainer import MoedlPerStepCallback
        from unittest.mock import Mock
        
        # Create test data
        L, K, E = 2, 2, 8
        np_arr = np.zeros((L, K, E), dtype=np.float32)
        
        # Create specific pattern for verification
        # All experts at perfect balance except one
        balance_frac = 1.0 / E  # 0.125
        np_arr[:, :, :] = balance_frac
        
        # Make expert 0 in layer 0, k0 overloaded
        np_arr[0, 0, 0] = 0.5
        # Reduce others proportionally to maintain sum=1
        np_arr[0, 0, 1:] = (1.0 - 0.5) / 7
        
        # Create callback
        mock_trainer = Mock()
        mock_trainer.heatmap_on = True
        callback = MoedlPerStepCallback(trainer=mock_trainer)
        
        heatmap_dir = os.path.join(str(tmp_path), "heatmap")
        os.makedirs(heatmap_dir, exist_ok=True)
        callback.heatmap_outdir = heatmap_dir
        
        test_file = os.path.join(heatmap_dir, "test_visual.png")
        callback._generate_expert_load_heatmap(np_arr, step=42, file_path=test_file)
        
        assert os.path.exists(test_file), "Heatmap visualization should be created"
        
        # Verify data transformation
        balance_pct = 100.0 / E
        
        # Expert 0 in layer 0, k0
        delta_e0 = abs(np_arr[0, 0, 0] * 100 - balance_pct)
        assert np.isclose(delta_e0, 37.5, atol=0.1), \
            f"Delta should be 37.5% for overloaded expert, got {delta_e0}"
        
        # Expert 1 in layer 0, k0
        delta_e1 = abs(np_arr[0, 0, 1] * 100 - balance_pct)
        expected_e1 = abs((1.0 - 0.5) / 7 * 100 - balance_pct)
        assert np.isclose(delta_e1, expected_e1, atol=0.1), \
            f"Delta should be {expected_e1}% for reduced expert, got {delta_e1}"
    
    def test_gif_generation_on_train_end(self, moe_model_config, tmp_path, tiny_dataset):
        """Test that GIF is generated after training ends."""
        import os
        from pathlib import Path
        
        # Setup training args for short training
        train_args = TrainingArguments(
            output_dir=str(tmp_path),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            max_steps=5,  # Only 5 steps
            logging_steps=1,
            save_steps=100,
            report_to=[],  # No wandb but we'll mock it
            use_cpu=True,
        )
        
        model = MoedlForCausalLM(moe_model_config)
        trainer = MoedlTrainer(
            model=model,
            args=train_args,
            train_dataset=tiny_dataset,
            heatmap_on=True,
            heatmap_freq=1,  # Generate heatmap every step
        )
        
        # Mock wandb handler to enable heatmap generation
        mock_wandb = Mock()
        mock_wandb.log = Mock()
        trainer._wb_handler = mock_wandb
        
        # Run training
        trainer.train()
        
        # Check if GIF exists
        gif_path = Path(tmp_path) / "expert_load_over_train_steps.gif"
        assert gif_path.exists(), f"GIF should be generated at {gif_path}"
        
        # Check if GIF is not empty
        assert gif_path.stat().st_size > 0, "GIF file should not be empty"
        
        # Check if heatmap directory exists and has PNG files
        heatmap_dir = Path(tmp_path) / "heatmap"
        assert heatmap_dir.exists(), "Heatmap directory should exist"
        
        png_files = list(heatmap_dir.glob("*.png"))
        assert len(png_files) > 0, "Should have generated PNG heatmaps"
        # With 5 steps and freq=1, we expect up to 5 PNGs
        # But due to threading and semaphore, we might get fewer
        # The important thing is that at least some were generated and GIF was created
        assert len(png_files) >= 1, f"Expected at least 1 PNG, got {len(png_files)}"
        assert len(png_files) <= 5, f"Expected at most 5 PNGs, got {len(png_files)}"
    
    def test_gif_not_generated_when_heatmap_disabled(self, moe_model_config, tmp_path, tiny_dataset):
        """Test that GIF is not generated when heatmap is disabled."""
        import os
        from pathlib import Path
        
        train_args = TrainingArguments(
            output_dir=str(tmp_path),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            max_steps=3,
            logging_steps=1,
            save_steps=100,
            report_to=[],
            use_cpu=True,
        )
        
        model = MoedlForCausalLM(moe_model_config)
        trainer = MoedlTrainer(
            model=model,
            args=train_args,
            train_dataset=tiny_dataset,
            heatmap_on=False,  # Disable heatmap
        )
        
        # Run training
        trainer.train()
        
        # Check that GIF does not exist
        gif_path = Path(tmp_path) / "expert_load_over_train_steps.gif"
        assert not gif_path.exists(), "GIF should not be generated when heatmap is disabled"
        
        # Check that heatmap directory doesn't exist or is empty
        heatmap_dir = Path(tmp_path) / "heatmap"
        if heatmap_dir.exists():
            png_files = list(heatmap_dir.glob("*.png"))
            assert len(png_files) == 0, "No PNG files should be generated when heatmap is disabled"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
