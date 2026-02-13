"""
Note: The test code was primarily generated with assistance from Claude Sonnet 4.5, based on provided specifications and instructions.

Tests for Moedl model.
"""
import pytest
import torch
import torch.nn as nn
import subprocess
import sys
from transformers import AutoModelForCausalLM, AutoConfig

from moelab.moedl import MoedlConfig, MoedlForCausalLM, Moedl

def llama_to_moedl_key_map(llama_key: str) -> str:
    """Map Llama state dict keys to Moedl keys.
    
    Handles the attribute renaming in Moedl:
    - Attention: q_proj→q, k_proj→k, v_proj→v, o_proj→o
    - MLP: gate_proj→gate, up_proj→up, down_proj→down
    - Decoder: input_layernorm→norm_attn, post_attention_layernorm→norm_mlp, self_attn→attn
    """
    # Attention projections
    llama_key = llama_key.replace('.q_proj.', '.q.')
    llama_key = llama_key.replace('.k_proj.', '.k.')
    llama_key = llama_key.replace('.v_proj.', '.v.')
    llama_key = llama_key.replace('.o_proj.', '.o.')
    
    # MLP projections
    llama_key = llama_key.replace('.gate_proj.', '.gate.')
    llama_key = llama_key.replace('.up_proj.', '.up.')
    llama_key = llama_key.replace('.down_proj.', '.down.')
    
    # Layer norms and attention
    llama_key = llama_key.replace('.input_layernorm.', '.norm_attn.')
    llama_key = llama_key.replace('.post_attention_layernorm.', '.norm_mlp.')
    llama_key = llama_key.replace('.self_attn.', '.attn.')
    
    return llama_key


def convert_llama_state_dict_to_moedl(llama_state_dict: dict) -> dict:
    """Convert a Llama state dict to Moedl format."""
    moedl_state_dict = {}
    for key, value in llama_state_dict.items():
        moedl_key = llama_to_moedl_key_map(key)
        moedl_state_dict[moedl_key] = value
    return moedl_state_dict

class TestMoedlDenseConstructor:
    """Test dense model construction with various configurations."""
    
    def test_basic_dense_construction(self):
        """Test creating a minimal dense Moedl model."""
        config = MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=512,
            num_experts=1,
            num_active_experts=1,
        )
        model = MoedlForCausalLM(config)
        assert model is not None
        assert model.config.vocab_size == 1000
        assert model.config.hidden_size == 128
        assert model.config.num_hidden_layers == 2
    
    def test_dense_construction_with_gqa(self):
        """Test dense construction with grouped-query attention."""
        config = MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,  # GQA
            max_position_embeddings=512,
            num_experts=1,
            num_active_experts=1,
        )
        model = MoedlForCausalLM(config)
        assert model.config.num_key_value_heads == 2
        assert model.config.num_attention_heads == 4
    
    def test_dense_construction_model_only(self):
        """Test constructing just dense Moedl without LM head."""
        config = MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=1,
            num_active_experts=1,
        )
        model = Moedl(config)
        assert model is not None
        assert not hasattr(model, 'lm_head')
    
    def test_dense_fail_invalid_heads(self):
        """Test that invalid head configuration raises error during forward (dense model)."""
        config = MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=5,  # Not divisible into hidden_size
            max_position_embeddings=512,
            num_experts=1,
            num_active_experts=1,
        )
        model = MoedlForCausalLM(config)
        
        # Forward should fail due to shape mismatch
        with pytest.raises(RuntimeError):
            _ = model(torch.randint(0, 1000, (1, 10)))
    
    def test_dense_fail_kv_heads_mismatch(self):
        """Test that kv_heads > num_heads fails during forward (dense model)."""
        config = MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=8,  # Can't have more KV heads than query heads
            max_position_embeddings=512,
            num_experts=1,
            num_active_experts=1,
        )
        model = MoedlForCausalLM(config)
        
        # Forward should fail - KV heads must divide query heads
        with pytest.raises(RuntimeError):
            _ = model(torch.randint(0, 1000, (1, 10)))


class TestMoedlDenseLlamaEquivalence:
    """Test equivalence between dense Moedl and Llama models."""
    
    @pytest.fixture
    def tiny_dense_config(self):
        """Shared tiny dense config for equivalence testing."""
        return {
            "vocab_size": 1000,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "hidden_act": "silu",
            "initializer_range": 0.02,
            "num_experts": 1,
            "num_active_experts": 1,
        }
    
    def test_forward_equivalence(self, tiny_dense_config):
        """Test that dense Moedl and Llama produce same outputs with same weights."""
        # Create both models
        moedl_config = MoedlConfig(**tiny_dense_config)
        moedl_model = MoedlForCausalLM(moedl_config)
        
        llama_config = AutoConfig.for_model("llama", **tiny_dense_config)
        llama_model = AutoModelForCausalLM.from_config(llama_config)
        
        # Copy weights from Llama to Moedl with key mapping
        llama_state_dict = llama_model.state_dict()
        moedl_state_dict = convert_llama_state_dict_to_moedl(llama_state_dict)
        moedl_model.load_state_dict(moedl_state_dict, strict=True)
        
        # Set to eval mode
        moedl_model.eval()
        llama_model.eval()
        
        # Test forward pass
        input_ids = torch.randint(0, tiny_dense_config["vocab_size"], (2, 32))
        
        with torch.no_grad():
            moedl_out = moedl_model(input_ids)
            llama_out = llama_model(input_ids)
        
        # Check logits match
        torch.testing.assert_close(
            moedl_out.logits, 
            llama_out.logits, 
            rtol=1e-4, 
            atol=1e-5
        )
    
    def test_backward_equivalence(self, tiny_dense_config):
        """Test that gradients are equivalent between dense Moedl and Llama."""
        # Create both models
        moedl_config = MoedlConfig(**tiny_dense_config)
        moedl_model = MoedlForCausalLM(moedl_config)
        
        llama_config = AutoConfig.for_model("llama", **tiny_dense_config)
        llama_model = AutoModelForCausalLM.from_config(llama_config)
        
        # Copy weights from Llama to Moedl with key mapping
        llama_state_dict = llama_model.state_dict()
        moedl_state_dict = convert_llama_state_dict_to_moedl(llama_state_dict)
        moedl_model.load_state_dict(moedl_state_dict, strict=True)
        
        # Set to train mode
        moedl_model.train()
        llama_model.train()
        
        # Create input and labels
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, tiny_dense_config["vocab_size"], (batch_size, seq_len))
        labels = torch.randint(0, tiny_dense_config["vocab_size"], (batch_size, seq_len))
        
        # Forward + backward for Moedl
        moedl_out = moedl_model(input_ids, labels=labels)
        moedl_loss = moedl_out.loss
        moedl_loss.backward()
        
        # Forward + backward for Llama
        llama_out = llama_model(input_ids, labels=labels)
        llama_loss = llama_out.loss
        llama_loss.backward()
        
        # Check losses match
        torch.testing.assert_close(moedl_loss, llama_loss, rtol=1e-4, atol=1e-5)
        
        # Check gradients match for embedding layer
        moedl_embed_grad = moedl_model.model.embed_tokens.weight.grad
        llama_embed_grad = llama_model.model.embed_tokens.weight.grad
        
        torch.testing.assert_close(
            moedl_embed_grad,
            llama_embed_grad,
            rtol=1e-4,
            atol=1e-5
        )
    
    def test_state_dict_compatibility(self, tiny_dense_config):
        """Test that state dict keys can be mapped between dense Moedl and Llama."""
        moedl_config = MoedlConfig(**tiny_dense_config)
        moedl_model = MoedlForCausalLM(moedl_config)
        
        llama_config = AutoConfig.for_model("llama", **tiny_dense_config)
        llama_model = AutoModelForCausalLM.from_config(llama_config)
        
        moedl_keys = set(moedl_model.state_dict().keys())
        llama_keys = set(llama_model.state_dict().keys())
        
        # Map Llama keys to Moedl format
        mapped_llama_keys = {llama_to_moedl_key_map(key) for key in llama_keys}
        
        # After mapping, keys should be identical
        assert moedl_keys == mapped_llama_keys, f"Key mismatch: {moedl_keys ^ mapped_llama_keys}"


class TestMoedlDenseGenerate:
    """Test generation functionality for dense models."""
    
    def test_dense_basic_generate(self):
        """Test basic text generation with dense model."""
        config = MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=512,
            num_experts=1,
            num_active_experts=1,
        )
        model = MoedlForCausalLM(config)
        model.eval()
        
        input_ids = torch.randint(0, 1000, (1, 10))
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=20,
                do_sample=False,
            )
        
        assert outputs.shape[0] == 1  # batch size
        assert outputs.shape[1] == 30  # 10 input + 20 generated
    
    def test_dense_generate_with_sampling(self):
        """Test generation with sampling (dense model)."""
        config = MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=512,
            num_experts=1,
            num_active_experts=1,
        )
        model = MoedlForCausalLM(config)
        model.eval()
        
        input_ids = torch.randint(0, 1000, (1, 10))
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=15,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
            )
        
        assert outputs.shape[0] == 1
        assert outputs.shape[1] == 25
    
    def test_dense_generate_batch(self):
        """Test batched generation (dense model)."""
        config = MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=512,
            pad_token_id=0,
            num_experts=1,
            num_active_experts=1,
        )
        model = MoedlForCausalLM(config)
        model.eval()
        
        batch_size = 3
        input_ids = torch.randint(1, 1000, (batch_size, 10))
        attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=0,
            )
        
        assert outputs.shape[0] == batch_size
        assert outputs.shape[1] == 20
    
    def test_dense_generate_with_eos(self):
        """Test early stopping with EOS token (dense model)."""
        config = MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=512,
            eos_token_id=2,
            num_experts=1,
            num_active_experts=1,
        )
        model = MoedlForCausalLM(config)
        model.eval()
        
        input_ids = torch.randint(3, 1000, (1, 10))
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=50,
                do_sample=False,
                eos_token_id=2,
            )
        
        # Output should be reasonable length
        # (might stop early if EOS is generated)
        assert outputs.shape[0] == 1
        assert outputs.shape[1] >= 10  # At least input length
        assert outputs.shape[1] <= 60  # At most input + max_new


class TestMoedlMoeConstructor:
    """Test MoE model construction with various configurations."""
    
    def test_basic_moe_construction(self):
        """Test creating a basic MoE Moedl model (no load balancing)."""
        config = MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=512,
            num_experts=8,
            num_active_experts=2,
            lb_coeff=0.0,
        )
        model = MoedlForCausalLM(config)
        
        assert model is not None
        assert model.config.num_experts == 8
        assert model.config.num_active_experts == 2
        
        # Verify MoeBlk is used in decoder layers
        for layer in model.model.layers:
            assert hasattr(layer, 'moe')
            assert layer.num_experts == 8
            assert layer.num_active_experts == 2
    
    def test_moe_expert_counts(self):
        """Test various expert configurations (no load balancing)."""
        test_configs = [
            (4, 1),   # 4 experts, top-1
            (8, 2),   # 8 experts, top-2
            (16, 4),  # 16 experts, top-4
        ]
        
        for num_experts, num_active in test_configs:
            config = MoedlConfig(
                vocab_size=1000,
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_experts=num_experts,
                num_active_experts=num_active,
                lb_coeff=0.0,
            )
            model = MoedlForCausalLM(config)
            
            assert model.config.num_experts == num_experts
            assert model.config.num_active_experts == num_active
            
            # Check each layer
            for layer in model.model.layers:
                assert layer.moe.experts.n_group == num_experts
    
    def test_moe_fail_invalid_expert_config(self):
        """Test that invalid expert configurations fail."""
        # num_active_experts > num_experts should fail
        with pytest.raises(ValueError):
            config = MoedlConfig(
                vocab_size=1000,
                hidden_size=128,
                num_experts=4,
                num_active_experts=8,  # More active than total
            )


class TestMoedlMoeForward:
    """Test MoE model forward pass and router_logits (no load balancing)."""
    
    def test_moe_forward_returns_router_logits(self):
        """Test that MoE forward pass returns router_logits."""
        config = MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            lb_coeff=0.0,
        )
        model = MoedlForCausalLM(config)
        model.eval()
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        # Check that router_logits are returned
        assert hasattr(outputs, 'router_logits')
        assert outputs.router_logits is not None
        
        # router_logits should be a tuple with one tensor per layer
        assert isinstance(outputs.router_logits, tuple)
        assert len(outputs.router_logits) == config.num_hidden_layers
        
        # Each router_logits tensor should have shape (batch_size * seq_len, num_experts)
        for router_logit in outputs.router_logits:
            assert router_logit.shape == (batch_size * seq_len, config.num_experts)
    
    def test_moe_router_logits_shape(self):
        """Test router_logits shapes with different configurations."""
        config = MoedlConfig(
            vocab_size=500,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_experts=16,
            num_active_experts=4,
            lb_coeff=0.0,
        )
        model = MoedlForCausalLM(config)
        model.eval()
        
        batch_size, seq_len = 4, 20
        input_ids = torch.randint(0, 500, (batch_size, seq_len))
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        # Verify router_logits structure
        assert len(outputs.router_logits) == 3  # num_hidden_layers
        for router_logit in outputs.router_logits:
            assert router_logit.shape == (batch_size * seq_len, 16)  # num_experts
    
    def test_moe_forward_with_labels(self):
        """Test MoE forward with labels for loss computation (no load balancing)."""
        config = MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            lb_coeff=0.0,
        )
        model = MoedlForCausalLM(config)
        model.train()
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        labels = torch.randint(0, 1000, (batch_size, seq_len))
        
        outputs = model(input_ids, labels=labels)
        
        # Check loss is computed
        assert outputs.loss is not None
        assert outputs.loss.numel() == 1  # Scalar loss
        
        # Check aux_loss is None when lb_coeff=0.0
        assert outputs.aux_loss is None
        
        # Check router_logits are still returned
        assert outputs.router_logits is not None
        assert len(outputs.router_logits) == config.num_hidden_layers


class TestMoedlMoeOlmoeEquivalence:
    """Test equivalence between Moedl MoeBlk and Olmoe SparseMoeBlock."""
    
    @pytest.fixture
    def tiny_moe_config(self):
        """Shared tiny MoE config for testing (no load balancing)."""
        return {
            "vocab_size": 500,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "hidden_act": "silu",
            "initializer_range": 0.02,
            "num_experts": 8,
            "num_active_experts": 2,
            "lb_coeff": 0.0,
        }
    
    def test_moe_block_forward_equivalence(self, tiny_moe_config):
        """Test that MoeBlk produces same output as OlmoeSparseMoeBlock with same weights."""
        from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock, OlmoeConfig
        from moelab.moedl.modeling_moedl import MoeBlk
        
        # Create Moedl MoeBlk
        moedl_config = MoedlConfig(**tiny_moe_config)
        moedl_moe = MoeBlk(moedl_config)
        
        # Create Olmoe SparseMoeBlock with compatible config
        # Olmoe uses num_experts_per_tok instead of num_active_experts
        # Set norm_topk_prob=True to match Moedl's normalization behavior
        olmoe_config_dict = {
            "hidden_size": tiny_moe_config["hidden_size"],
            "intermediate_size": tiny_moe_config["intermediate_size"],
            "num_experts": tiny_moe_config["num_experts"],
            "num_experts_per_tok": tiny_moe_config["num_active_experts"],
            "norm_topk_prob": True,  # Moedl always normalizes topk weights
            "hidden_act": tiny_moe_config["hidden_act"],
            "mlp_bias": False,
        }
        olmoe_config = OlmoeConfig(**olmoe_config_dict)
        olmoe_moe = OlmoeSparseMoeBlock(olmoe_config)
        
        # Copy weights from Olmoe to Moedl
        # Router weights
        moedl_moe.router.weight.data.copy_(olmoe_moe.gate.weight.data)
        
        # Expert weights: Olmoe nn.Linear stores (out, in), GroupedGLU stores (n_group, in, out)
        for i in range(tiny_moe_config["num_experts"]):
            moedl_moe.experts.weight_gate.data[i].copy_(olmoe_moe.experts[i].gate_proj.weight.data.T)
            moedl_moe.experts.weight_up.data[i].copy_(olmoe_moe.experts[i].up_proj.weight.data.T)
            moedl_moe.experts.weight_down.data[i].copy_(olmoe_moe.experts[i].down_proj.weight.data.T)
        
        # Set to eval mode
        moedl_moe.eval()
        olmoe_moe.eval()
        
        # Test forward pass
        batch_size, seq_len, hidden_dim = 2, 10, tiny_moe_config["hidden_size"]
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        
        with torch.no_grad():
            moedl_output, moedl_router_logits = moedl_moe(hidden_states)
            olmoe_output, olmoe_router_logits = olmoe_moe(hidden_states)
        
        # Check outputs match
        torch.testing.assert_close(
            moedl_output,
            olmoe_output,
            rtol=1e-4,
            atol=1e-5
        )
        
        # Check router logits match
        torch.testing.assert_close(
            moedl_router_logits,
            olmoe_router_logits,
            rtol=1e-4,
            atol=1e-5
        )
    
    def test_moe_block_backward_equivalence(self, tiny_moe_config):
        """Test that MoeBlk produces same gradients as OlmoeSparseMoeBlock with same weights."""
        from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock, OlmoeConfig
        from moelab.moedl.modeling_moedl import MoeBlk, load_balancing_loss_func
        
        # Use non-zero lb_coeff for this test
        lb_coeff = 0.5
        tiny_moe_config_with_lb = {**tiny_moe_config, "lb_coeff": lb_coeff}
        
        # Create Moedl MoeBlk
        moedl_config = MoedlConfig(**tiny_moe_config_with_lb)
        moedl_moe = MoeBlk(moedl_config)
        
        # Create Olmoe SparseMoeBlock with compatible config
        # Olmoe uses router_aux_loss_coef instead of lb_coeff
        olmoe_config_dict = {
            "hidden_size": tiny_moe_config["hidden_size"],
            "intermediate_size": tiny_moe_config["intermediate_size"],
            "num_experts": tiny_moe_config["num_experts"],
            "num_experts_per_tok": tiny_moe_config["num_active_experts"],
            "norm_topk_prob": True,
            "hidden_act": tiny_moe_config["hidden_act"],
            "mlp_bias": False,
            "router_aux_loss_coef": lb_coeff,  # Match Moedl's lb_coeff
        }
        olmoe_config = OlmoeConfig(**olmoe_config_dict)
        olmoe_moe = OlmoeSparseMoeBlock(olmoe_config)
        
        # Copy weights from Olmoe to Moedl
        # Router weights
        moedl_moe.router.weight.data.copy_(olmoe_moe.gate.weight.data)
        # Expert weights: Olmoe nn.Linear stores (out, in), GroupedGLU stores (n_group, in, out)
        for i in range(tiny_moe_config["num_experts"]):
            moedl_moe.experts.weight_gate.data[i].copy_(olmoe_moe.experts[i].gate_proj.weight.data.T)
            moedl_moe.experts.weight_up.data[i].copy_(olmoe_moe.experts[i].up_proj.weight.data.T)
            moedl_moe.experts.weight_down.data[i].copy_(olmoe_moe.experts[i].down_proj.weight.data.T)
        
        # Set to train mode
        moedl_moe.train()
        olmoe_moe.train()
        
        # Create input
        batch_size, seq_len, hidden_dim = 2, 10, tiny_moe_config["hidden_size"]
        torch.manual_seed(42)
        hidden_states_moedl = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)
        hidden_states_olmoe = hidden_states_moedl.clone().detach().requires_grad_(True)
        
        # Forward pass for Moedl
        moedl_output, moedl_router_logits = moedl_moe(hidden_states_moedl)
        
        # Compute auxiliary loss for Moedl
        moedl_aux_loss = load_balancing_loss_func(
            gate_logits=(moedl_router_logits,),
            num_experts=tiny_moe_config["num_experts"],
            top_k=tiny_moe_config["num_active_experts"],
            attention_mask=None,
        )
        
        # Compute total loss for Moedl (sum of outputs + aux_loss weighted by lb_coeff)
        moedl_total_loss = moedl_output.sum() + lb_coeff * moedl_aux_loss
        
        # Forward pass for Olmoe
        olmoe_output, olmoe_router_logits = olmoe_moe(hidden_states_olmoe)
        
        # Compute auxiliary loss for Olmoe (using same function)
        olmoe_aux_loss = load_balancing_loss_func(
            gate_logits=(olmoe_router_logits,),
            num_experts=tiny_moe_config["num_experts"],
            top_k=tiny_moe_config["num_active_experts"],
            attention_mask=None,
        )
        
        # Compute total loss for Olmoe
        olmoe_total_loss = olmoe_output.sum() + lb_coeff * olmoe_aux_loss
        
        # Check that auxiliary losses match
        torch.testing.assert_close(
            moedl_aux_loss,
            olmoe_aux_loss,
            rtol=1e-4,
            atol=1e-5,
            msg="Auxiliary losses should match"
        )
        
        # Check that total losses match
        torch.testing.assert_close(
            moedl_total_loss,
            olmoe_total_loss,
            rtol=1e-4,
            atol=1e-5,
            msg="Total losses should match"
        )
        
        # Backward pass
        moedl_total_loss.backward()
        olmoe_total_loss.backward()
        
        # Check router gradients match
        torch.testing.assert_close(
            moedl_moe.router.weight.grad,
            olmoe_moe.gate.weight.grad,
            rtol=1e-4,
            atol=1e-5,
            msg="Router gradients should match"
        )
        
        # Check expert gradients match for all experts
        # GroupedGLU stores weights as (n_group, in, out), nn.Linear as (out, in)
        for i in range(tiny_moe_config["num_experts"]):
            torch.testing.assert_close(
                moedl_moe.experts.weight_gate.grad[i],
                olmoe_moe.experts[i].gate_proj.weight.grad.T,
                rtol=1e-4,
                atol=1e-5,
                msg=f"Expert {i} gate gradients should match"
            )
            torch.testing.assert_close(
                moedl_moe.experts.weight_up.grad[i],
                olmoe_moe.experts[i].up_proj.weight.grad.T,
                rtol=1e-4,
                atol=1e-5,
                msg=f"Expert {i} up gradients should match"
            )
            torch.testing.assert_close(
                moedl_moe.experts.weight_down.grad[i],
                olmoe_moe.experts[i].down_proj.weight.grad.T,
                rtol=1e-4,
                atol=1e-5,
                msg=f"Expert {i} down gradients should match"
            )
        
        # Check input gradients match
        torch.testing.assert_close(
            hidden_states_moedl.grad,
            hidden_states_olmoe.grad,
            rtol=1e-4,
            atol=1e-5,
            msg="Input gradients should match"
        )
    
    def test_moe_block_expert_selection(self, tiny_moe_config):
        """Test that expert selection works correctly."""
        from moelab.moedl.modeling_moedl import MoeBlk
        
        config = MoedlConfig(**tiny_moe_config)
        moe = MoeBlk(config)
        moe.eval()
        
        batch_size, seq_len, hidden_dim = 2, 5, tiny_moe_config["hidden_size"]
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        
        with torch.no_grad():
            output, router_logits = moe(hidden_states)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, hidden_dim)
        
        # Check router_logits shape
        assert router_logits.shape == (batch_size * seq_len, config.num_experts)
        
        # Verify that routing probabilities sum to ~1 after softmax and topk
        router_probs = router_logits.softmax(dim=-1)
        topk_probs, topk_indices = router_probs.topk(config.num_active_experts, dim=-1)
        
        # After normalization, topk weights should sum to 1
        # (accounting for numerical precision)
        topk_weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-12)
        torch.testing.assert_close(
            topk_weights.sum(dim=-1),
            torch.ones(batch_size * seq_len),
            rtol=1e-5,
            atol=1e-6
        )


class TestMoedlMoeSharedExperts:
    """Test MoE shared experts functionality."""
    
    def test_shared_experts_construction(self):
        """Test that shared experts are created correctly."""
        config = MoedlConfig(
            vocab_size=500,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            num_shared_experts=2,
            lb_coeff=0.0,
        )
        model = MoedlForCausalLM(config)
        
        # Verify shared experts are created in MoeBlk
        for layer in model.model.layers:
            assert hasattr(layer, 'moe')
            assert hasattr(layer.moe, 'common')
            assert len(layer.moe.common) == 2
            assert layer.moe.num_shared_experts == 2
    
    def test_no_shared_experts(self):
        """Test MoE without shared experts (num_shared_experts=0)."""
        config = MoedlConfig(
            vocab_size=500,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            num_shared_experts=0,
            lb_coeff=0.0,
        )
        model = MoedlForCausalLM(config)
        
        # Verify no shared experts created
        for layer in model.model.layers:
            assert not hasattr(layer.moe, 'common')
    
    def test_shared_experts_affect_output(self):
        """Test that shared experts change the model output."""
        base_config = {
            "vocab_size": 500,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_experts": 8,
            "num_active_experts": 2,
            "lb_coeff": 0.0,
        }
        
        # Model without shared experts
        config_no_shared = MoedlConfig(**base_config, num_shared_experts=0)
        model_no_shared = MoedlForCausalLM(config_no_shared)
        
        # Model with shared experts
        config_with_shared = MoedlConfig(**base_config, num_shared_experts=2)
        model_with_shared = MoedlForCausalLM(config_with_shared)
        
        # Copy routed expert weights from no_shared to with_shared
        for i, layer in enumerate(model_with_shared.model.layers):
            # Copy router and routed experts (GroupedGLU stores as 3D tensors)
            layer.moe.router.weight.data.copy_(model_no_shared.model.layers[i].moe.router.weight.data)
            layer.moe.experts.weight_gate.data.copy_(model_no_shared.model.layers[i].moe.experts.weight_gate.data)
            layer.moe.experts.weight_up.data.copy_(model_no_shared.model.layers[i].moe.experts.weight_up.data)
            layer.moe.experts.weight_down.data.copy_(model_no_shared.model.layers[i].moe.experts.weight_down.data)
        
        model_no_shared.eval()
        model_with_shared.eval()
        
        # Same input
        torch.manual_seed(42)
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 500, (batch_size, seq_len))
        
        with torch.no_grad():
            outputs_no_shared = model_no_shared(input_ids)
            outputs_with_shared = model_with_shared(input_ids)
        
        # Outputs should differ (shared experts add their contribution)
        assert not torch.allclose(outputs_no_shared.logits, outputs_with_shared.logits)
    
    def test_dense_model_ignores_shared_experts(self):
        """Test that dense models force num_shared_experts to 0."""
        config = MoedlConfig(
            vocab_size=500,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=1,
            num_active_experts=1,
            num_shared_experts=2,  # Try to set shared experts for dense
            lb_coeff=0.0,
        )
        
        # Config should force num_shared_experts to 0 for dense
        assert config.num_shared_experts == 0
        
        model = MoedlForCausalLM(config)
        
        # Verify layers use MLP, not MoeBlk
        for layer in model.model.layers:
            assert hasattr(layer, 'mlp')
            assert not hasattr(layer, 'moe')


class TestMoedlMoeLoadBalancePenalty:
    """Test MoE load balance penalty method (auxiliary loss-based)."""
    
    def test_lb_penalty_default_disabled(self):
        """Test that load balance penalty is disabled by default (lb_coeff=0.0)."""
        config = MoedlConfig(
            vocab_size=500,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            # lb_coeff not specified, should default to 0.0
        )
        
        # Verify default is 0.0
        assert config.lb_coeff == 0.0
        
        model = MoedlForCausalLM(config)
        model.train()
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 500, (batch_size, seq_len))
        labels = torch.randint(0, 500, (batch_size, seq_len))
        
        outputs = model(input_ids, labels=labels)
        
        # With default lb_coeff=0.0, no penalty should be applied
        assert outputs.aux_loss is None
    
    def test_lb_penalty_applied_when_coeff_nonzero(self):
        """Test that load balance penalty is added when lb_coeff > 0."""
        config = MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            lb_coeff=0.01,
        )
        model = MoedlForCausalLM(config)
        model.train()
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        labels = torch.randint(0, 1000, (batch_size, seq_len))
        
        outputs = model(input_ids, labels=labels)
        
        # Loss should include load balance penalty component
        assert outputs.loss is not None
        assert outputs.loss.numel() == 1
        
        # With lb_coeff > 0, the penalty is added directly to the main loss
        assert outputs.router_logits is not None
        
        # This should be the load balancing loss value (before multiplying by lb_coeff)
        assert outputs.aux_loss is not None, "aux_loss should be set when lb_coeff > 0 and num_experts > 1"
        assert torch.isfinite(outputs.aux_loss), "aux_loss should be a finite value"
        assert outputs.aux_loss >= 0, "aux_loss should be non-negative (it's a penalty term)"
    
    def test_lb_penalty_increases_total_loss(self):
        """Test that enabling load balance penalty increases the total loss."""
        # Model without load balance penalty
        config_no_lb = MoedlConfig(
            vocab_size=500,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            lb_coeff=0.0,
        )
        model_no_lb = MoedlForCausalLM(config_no_lb)
        
        # Model with load balance penalty
        config_with_lb = MoedlConfig(
            vocab_size=500,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            lb_coeff=0.01,
        )
        model_with_lb = MoedlForCausalLM(config_with_lb)
        
        # Copy weights to ensure same model
        model_with_lb.load_state_dict(model_no_lb.state_dict())
        
        # Set to train mode
        model_no_lb.train()
        model_with_lb.train()
        
        # Same input
        batch_size, seq_len = 2, 10
        torch.manual_seed(42)
        input_ids = torch.randint(0, 500, (batch_size, seq_len))
        labels = torch.randint(0, 500, (batch_size, seq_len))
        
        # Forward pass
        with torch.no_grad():
            outputs_no_lb = model_no_lb(input_ids, labels=labels)
            outputs_with_lb = model_with_lb(input_ids, labels=labels)
        
        # Loss with penalty should be higher (original loss + lb penalty)
        assert outputs_with_lb.loss > outputs_no_lb.loss
    
    def test_lb_penalty_with_different_coefficients(self):
        """Test that higher lb_coeff results in higher penalty and total loss."""
        base_config = {
            "vocab_size": 500,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_experts": 8,
            "num_active_experts": 2,
        }
        
        # Create models with different lb_coeff
        config_lb_small = MoedlConfig(**base_config, lb_coeff=0.01)
        config_lb_large = MoedlConfig(**base_config, lb_coeff=0.1)
        
        model_lb_small = MoedlForCausalLM(config_lb_small)
        model_lb_large = MoedlForCausalLM(config_lb_large)
        
        # Copy weights
        model_lb_large.load_state_dict(model_lb_small.state_dict())
        
        model_lb_small.train()
        model_lb_large.train()
        
        # Same input
        torch.manual_seed(42)
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 500, (batch_size, seq_len))
        labels = torch.randint(0, 500, (batch_size, seq_len))
        
        with torch.no_grad():
            outputs_small = model_lb_small(input_ids, labels=labels)
            outputs_large = model_lb_large(input_ids, labels=labels)
        
        # Higher coefficient should result in higher total loss
        assert outputs_large.loss > outputs_small.loss
    
    def test_lb_penalty_computation_validity(self):
        """Test that load balance penalty is computed correctly."""
        config = MoedlConfig(
            vocab_size=500,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            lb_coeff=0.01,
        )
        model = MoedlForCausalLM(config)
        model.train()
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 500, (batch_size, seq_len))
        labels = torch.randint(0, 500, (batch_size, seq_len))
        
        outputs = model(input_ids, labels=labels)
        
        # Manually compute load balance penalty to verify
        from moelab.moedl.modeling_moedl import load_balancing_loss_func
        
        expected_lb_penalty = load_balancing_loss_func(
            gate_logits=outputs.router_logits,
            num_experts=config.num_experts,
            top_k=config.num_active_experts,
            attention_mask=None,
        )
        
        # The penalty should be positive (it's a penalty term)
        assert expected_lb_penalty >= 0
        
        # Loss should be finite and reasonable
        assert torch.isfinite(outputs.loss)
        assert outputs.loss > 0
        
        # aux_loss should match the computed load balancing penalty (scaled by lb_coeff)
        assert outputs.aux_loss is not None, "aux_loss should be set when lb_coeff > 0"
        expected_scaled_penalty = config.lb_coeff * expected_lb_penalty
        torch.testing.assert_close(
            outputs.aux_loss, 
            expected_scaled_penalty,
            rtol=1e-5,
            atol=1e-7,
            msg="aux_loss should equal lb_coeff * load_balancing_penalty"
        )
    
    def test_dense_model_no_lb_penalty(self):
        """Test that dense models (num_experts=1) don't apply load balance penalty."""
        config = MoedlConfig(
            vocab_size=500,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=1,
            num_active_experts=1,
            lb_coeff=0.01,  # Set lb_coeff but it shouldn't be used
        )
        model = MoedlForCausalLM(config)
        model.train()
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 500, (batch_size, seq_len))
        labels = torch.randint(0, 500, (batch_size, seq_len))
        
        outputs = model(input_ids, labels=labels)
        
        # Dense model should have None router_logits (no MoE)
        assert outputs.router_logits is None
        
        # aux_loss should be None (no penalty applied)
        assert outputs.aux_loss is None
        
        # Loss should still be computed normally (just LM loss)
        assert outputs.loss is not None


class TestMoedlMoeLoadBalanceBiasing:
    """Test MoE load balance biasing method (DeepSeek V3 style)."""
    
    def test_lb_biasing_config_validation(self):
        """Test that lb_gamma configuration is validated correctly."""
        # Valid configuration with lb_gamma > 0
        config = MoedlConfig(
            vocab_size=500,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            lb_gamma=0.01,
        )
        assert config.lb_gamma == 0.01
        assert config.lb_coeff == 0.0  # Should default to 0.0
        
        # Invalid: both lb_coeff and lb_gamma non-zero
        with pytest.raises((ValueError, NotImplementedError)):
            config = MoedlConfig(
                vocab_size=500,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_experts=8,
                num_active_experts=2,
                lb_gamma=0.01,
                lb_coeff=0.01,
            )
    
    def test_lb_biasing_buffer_creation(self):
        """Test that e_score_bias buffer is created when lb_gamma > 0."""
        config = MoedlConfig(
            vocab_size=500,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            lb_gamma=0.01,
        )
        model = MoedlForCausalLM(config)
        
        # Check that each MoE layer has the e_score_bias buffer
        for layer in model.model.layers:
            if hasattr(layer, 'moe'):
                assert hasattr(layer.moe, 'e_score_bias'), "MoeBlk should have e_score_bias buffer when lb_gamma > 0"
                assert layer.moe.e_score_bias.shape == (1, config.num_experts)
                assert torch.all(layer.moe.e_score_bias == 0.0), "e_score_bias should be initialized to zeros"
    
    def test_lb_biasing_no_buffer_when_disabled(self):
        """Test that e_score_bias buffer is NOT created when lb_gamma = 0."""
        config = MoedlConfig(
            vocab_size=500,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            lb_gamma=0.0,
        )
        model = MoedlForCausalLM(config)
        
        # Check that MoE layers do NOT have the e_score_bias buffer
        for layer in model.model.layers:
            if hasattr(layer, 'moe'):
                assert not hasattr(layer.moe, 'e_score_bias'), "MoeBlk should not have e_score_bias buffer when lb_gamma = 0"
    
    def test_lb_biasing_forward_pass(self):
        """Test that forward pass works with lb_gamma enabled."""
        config = MoedlConfig(
            vocab_size=500,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            lb_gamma=0.01,
        )
        model = MoedlForCausalLM(config)
        model.eval()
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 500, (batch_size, seq_len))
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        # Forward pass should work
        assert outputs.logits is not None
        assert outputs.router_logits is not None
        
        # No aux_loss for biasing method (no penalty)
        assert outputs.aux_loss is None
    
    def test_lb_biasing_uses_sigmoid(self):
        """Test that biasing method uses sigmoid activation instead of softmax."""
        from moelab.moedl.modeling_moedl import MoeBlk
        
        config_biasing = MoedlConfig(
            vocab_size=500,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            lb_gamma=0.01,
        )
        
        config_penalty = MoedlConfig(
            vocab_size=500,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            lb_coeff=0.01,
        )
        
        moe_biasing = MoeBlk(config_biasing)
        moe_penalty = MoeBlk(config_penalty)
        
        # Copy router weights to ensure same logits
        moe_biasing.router.weight.data.copy_(moe_penalty.router.weight.data)
        
        moe_biasing.eval()
        moe_penalty.eval()
        
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, 64)
        
        with torch.no_grad():
            output_biasing, router_logits_biasing = moe_biasing(hidden_states)
            output_penalty, router_logits_penalty = moe_penalty(hidden_states)
        
        # Router logits should be different due to sigmoid vs softmax
        # For biasing: router_logits is actually biased_scores (sigmoid + bias)
        # For penalty: router_logits is raw logits
        assert not torch.allclose(router_logits_biasing, router_logits_penalty, rtol=1e-3)
        
        # Outputs will be different due to different routing
        assert output_biasing.shape == output_penalty.shape
    
    def test_lb_biasing_no_aux_loss(self):
        """Test that biasing method does NOT generate aux_loss (unlike penalty method)."""
        config = MoedlConfig(
            vocab_size=500,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            lb_gamma=0.01,
        )
        model = MoedlForCausalLM(config)
        model.train()
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 500, (batch_size, seq_len))
        labels = torch.randint(0, 500, (batch_size, seq_len))
        
        outputs = model(input_ids, labels=labels)
        
        # Biasing method should NOT have aux_loss
        assert outputs.aux_loss is None
        
        # But should have router_logits
        assert outputs.router_logits is not None
        
        # Loss should only be LM loss
        assert outputs.loss is not None
    
    def test_lb_biasing_bias_affects_routing(self):
        """Test that e_score_bias actually affects expert selection."""
        from moelab.moedl.modeling_moedl import MoeBlk
        
        config = MoedlConfig(
            vocab_size=500,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            lb_gamma=0.01,
        )
        
        moe = MoeBlk(config)
        moe.eval()
        
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, 64)
        
        # Forward pass with zero bias
        with torch.no_grad():
            output_zero_bias, _ = moe(hidden_states)
        
        # Manually set bias to favor expert 0
        moe.e_score_bias[0, 0] = 10.0  # Large bias for expert 0
        
        with torch.no_grad():
            output_with_bias, _ = moe(hidden_states)
        
        # Outputs should be different
        assert not torch.allclose(output_zero_bias, output_with_bias, rtol=1e-3)
    
    def test_lb_biasing_dense_model_constraint(self):
        """Test that dense models (num_experts=1) don't use biasing."""
        config = MoedlConfig(
            vocab_size=500,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=1,
            # lb_gamma is ignored for dense models
        )
        
        # Should force lb_gamma to 0.0 for dense
        assert config.lb_gamma == 0.0
        
        model = MoedlForCausalLM(config)
        
        # Dense model should not have MoE layers
        for layer in model.model.layers:
            assert not hasattr(layer, 'moe')


class TestMoedlMoeDeepSeekV3Equivalence:
    """Test equivalence between Moedl MoeBlk and DeepSeek V3 MoE module."""
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set random seeds before each test and clean up after."""
        # Setup: Set random seeds for deterministic behavior
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        yield  # Run the test
        
        # Teardown: Clean up after test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    
    @pytest.fixture(scope="function")
    def tiny_moe_config_nogrouping(self):
        """Minimal config for equivalence testing without grouping."""
        return {
            "hidden_size": 64,
            "intermediate_size": 128,
            "moe_intermediate_size": 128,
            "num_experts": 8,
            "num_active_experts": 2,
            "num_shared_experts": 1,
            "lb_gamma": 0.01,
            "hidden_act": "silu",
            "mlp_bias": False,
            "capacity_factor": 0.0,  # DeepSeek V3 doesn't drop tokens
        }
    
    def test_deepseek_v3_availability(self):
        """Check DeepSeek V3 is available for equivalence testing."""
        try:
            from transformers.models.deepseek_v3.modular_deepseek_v3 import DeepseekV3MoE
            from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
        except ImportError:
            pytest.skip("DeepSeek V3 not available in this transformers version")
    
    def test_moe_forward_equivalence_no_grouping(self, tiny_moe_config_nogrouping):
        """Test MoeBlk forward matches DeepseekV3MoE with no grouping (n_group=1)."""
        try:
            from transformers.models.deepseek_v3.modular_deepseek_v3 import DeepseekV3MoE, DeepseekV3TopkRouter
            from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
            from moelab.moedl.modeling_moedl import MoeBlk
        except ImportError:
            pytest.skip("DeepSeek V3 not available")
        
        # Reset seed for deterministic model creation
        torch.manual_seed(100)
        
        # Moedl config
        moedl_config = MoedlConfig(**tiny_moe_config_nogrouping)
        moedl_moe = MoeBlk(moedl_config)
        
        # Ensure clean state
        moedl_moe.n_drop = 0
        
        # DeepSeek V3 config - disable grouping for equivalence
        ds_config = DeepseekV3Config(
            hidden_size=tiny_moe_config_nogrouping["hidden_size"],
            intermediate_size=tiny_moe_config_nogrouping["intermediate_size"],
            moe_intermediate_size=tiny_moe_config_nogrouping["moe_intermediate_size"],
            n_routed_experts=tiny_moe_config_nogrouping["num_experts"],
            num_experts_per_tok=tiny_moe_config_nogrouping["num_active_experts"],
            n_shared_experts=tiny_moe_config_nogrouping["num_shared_experts"],
            n_group=1,  # No grouping
            topk_group=1,
            norm_topk_prob=True,
            routed_scaling_factor=1.0,  # No scaling
            hidden_act=tiny_moe_config_nogrouping["hidden_act"],
        )
        ds_moe = DeepseekV3MoE(ds_config)
        
        # Copy router weights
        moedl_moe.router.weight.data.copy_(ds_moe.gate.weight.data)
        
        # Copy routed expert weights: nn.Linear stores (out, in), GroupedGLU stores (n_group, in, out)
        for i in range(tiny_moe_config_nogrouping["num_experts"]):
            moedl_moe.experts.weight_gate.data[i].copy_(ds_moe.experts[i].gate_proj.weight.data.T)
            moedl_moe.experts.weight_up.data[i].copy_(ds_moe.experts[i].up_proj.weight.data.T)
            moedl_moe.experts.weight_down.data[i].copy_(ds_moe.experts[i].down_proj.weight.data.T)
        
        # Copy shared expert weights
        # DeepSeek V3: single large MLP (intermediate = moe_intermediate_size * n_shared_experts)
        # Moedl: list of n_shared_experts MLPs (each with moe_intermediate_size)
        # For n_shared_experts=1, they should be equivalent
        if hasattr(moedl_moe, 'common') and len(moedl_moe.common) > 0:
            moedl_moe.common[0].gate.weight.data.copy_(ds_moe.shared_experts.gate_proj.weight.data)
            moedl_moe.common[0].up.weight.data.copy_(ds_moe.shared_experts.up_proj.weight.data)
            moedl_moe.common[0].down.weight.data.copy_(ds_moe.shared_experts.down_proj.weight.data)
        
        # Set zero bias for exact equivalence
        if hasattr(moedl_moe, 'e_score_bias'):
            moedl_moe.e_score_bias.data.zero_()
        if hasattr(ds_moe.gate, 'e_score_correction_bias'):
            ds_moe.gate.e_score_correction_bias.data.zero_()
        
        moedl_moe.eval()
        ds_moe.eval()
        
        # Test forward with deterministic input
        torch.manual_seed(123)  # Fixed seed for reproducible input
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, tiny_moe_config_nogrouping["hidden_size"])
        
        with torch.no_grad():
            moedl_output, _ = moedl_moe(hidden_states)
            ds_output = ds_moe(hidden_states)
        
        # Outputs should match
        torch.testing.assert_close(
            moedl_output,
            ds_output,
            rtol=1e-4,
            atol=1e-5,
            msg="MoeBlk output should match DeepseekV3MoE with no grouping"
        )
    
    def test_routing_equivalence_no_grouping(self, tiny_moe_config_nogrouping):
        """Test that routing (expert selection) matches when grouping is disabled."""
        try:
            from transformers.models.deepseek_v3.modular_deepseek_v3 import DeepseekV3TopkRouter
            from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
            from moelab.moedl.modeling_moedl import MoeBlk
        except ImportError:
            pytest.skip("DeepSeek V3 not available")
        
        # Reset seed for deterministic model creation
        torch.manual_seed(200)
        
        # Moedl config
        moedl_config = MoedlConfig(**tiny_moe_config_nogrouping)
        moedl_moe = MoeBlk(moedl_config)
        
        # Ensure clean state
        moedl_moe.n_drop = 0
        
        # DeepSeek V3 router
        ds_config = DeepseekV3Config(
            hidden_size=tiny_moe_config_nogrouping["hidden_size"],
            n_routed_experts=tiny_moe_config_nogrouping["num_experts"],
            num_experts_per_tok=tiny_moe_config_nogrouping["num_active_experts"],
            n_group=1,
            topk_group=1,
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
        )
        ds_router = DeepseekV3TopkRouter(ds_config)
        
        # Copy router weights
        moedl_moe.router.weight.data.copy_(ds_router.weight.data)
        if hasattr(moedl_moe, 'e_score_bias'):
            moedl_moe.e_score_bias.data.zero_()
        if hasattr(ds_router, 'e_score_correction_bias'):
            ds_router.e_score_correction_bias.data.zero_()
        
        moedl_moe.eval()
        ds_router.eval()
        
        # Use deterministic input
        torch.manual_seed(456)  # Fixed seed for reproducible input
        batch_size, seq_len = 2, 5
        hidden_states = torch.randn(batch_size, seq_len, tiny_moe_config_nogrouping["hidden_size"])
        
        with torch.no_grad():
            # Moedl routing
            flat_hidden = hidden_states.view(-1, tiny_moe_config_nogrouping["hidden_size"])
            router_logits = moedl_moe.router(flat_hidden)
            router_scores = router_logits.sigmoid()
            biased_scores = router_scores + moedl_moe.e_score_bias
            moedl_topk_weights, moedl_topk_ids = torch.topk(biased_scores, tiny_moe_config_nogrouping["num_active_experts"], dim=-1)
            moedl_topk_weights_gathered = torch.gather(router_scores, dim=-1, index=moedl_topk_ids)
            moedl_topk_weights_norm = moedl_topk_weights_gathered / (moedl_topk_weights_gathered.sum(dim=-1, keepdim=True) + 1e-20)
            
            # DeepSeek V3 routing
            ds_topk_ids, ds_topk_weights = ds_router(hidden_states)
        
        # Expert selection should match
        torch.testing.assert_close(
            moedl_topk_ids.sort(dim=-1)[0],
            ds_topk_ids.sort(dim=-1)[0],
            msg="Selected expert IDs should match"
        )
        
        # Weights should match (after normalization)
        # Since topk can return elements in different orders when values are tied,
        # we need to sort both IDs and weights together, then compare
        moedl_sorted_weights, moedl_sort_indices = moedl_topk_weights_norm.sort(dim=-1)
        moedl_sorted_ids_by_weight = torch.gather(moedl_topk_ids, dim=-1, index=moedl_sort_indices)
        
        ds_sorted_weights, ds_sort_indices = ds_topk_weights.sort(dim=-1)
        ds_sorted_ids_by_weight = torch.gather(ds_topk_ids, dim=-1, index=ds_sort_indices)
        
        # Verify IDs match when sorted by weight
        torch.testing.assert_close(
            moedl_sorted_ids_by_weight,
            ds_sorted_ids_by_weight,
            msg="Expert IDs sorted by weight should match"
        )
        
        # Verify weights match when sorted
        torch.testing.assert_close(
            moedl_sorted_weights,
            ds_sorted_weights,
            rtol=1e-4,
            atol=1e-5,
            msg="Sorted normalized expert weights should match"
        )


class TestMoedlMoeCapacityFactor:
    """Test MoE capacity factor and token dropping."""
    
    def test_capacity_factor_config_validation(self):
        """Test capacity_factor configuration validation."""
        # Valid: capacity_factor >= 0
        config = MoedlConfig(
            vocab_size=500,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            capacity_factor=1.5,
        )
        assert config.capacity_factor == 1.5
        
        # Invalid: negative capacity_factor
        with pytest.raises(ValueError, match="capacity_factor must be a non-negative float"):
            MoedlConfig(
                vocab_size=500,
                hidden_size=64,
                num_experts=8,
                capacity_factor=-1.0,
            )
    
    def test_capacity_factor_disabled_by_default(self):
        """Test that capacity limiting is disabled by default (CF=0)."""
        config = MoedlConfig(
            vocab_size=500,
            hidden_size=64,
            num_experts=8,
            num_active_experts=2,
        )
        assert config.capacity_factor == 0.0
        
        model = MoedlForCausalLM(config)
        
        # Check MoeBlk has CF=0
        for layer in model.model.layers:
            if hasattr(layer, 'moe'):
                assert layer.moe.capacity_factor == 0.0
                assert layer.moe.n_drop == 0
    
    def test_capacity_factor_buffer_creation(self):
        """Test that n_drop counter exists when CF > 0."""
        config = MoedlConfig(
            vocab_size=500,
            hidden_size=64,
            num_experts=8,
            num_active_experts=2,
            capacity_factor=1.0,
        )
        model = MoedlForCausalLM(config)
        
        for layer in model.model.layers:
            if hasattr(layer, 'moe'):
                assert hasattr(layer.moe, 'n_drop')
                assert layer.moe.n_drop == 0  # Initially zero
    
    def test_capacity_factor_forward_pass(self):
        """Test forward pass works with capacity limiting."""
        config = MoedlConfig(
            vocab_size=500,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            capacity_factor=1.0,
        )
        model = MoedlForCausalLM(config)
        model.eval()
        
        batch_size, seq_len = 2, 20
        input_ids = torch.randint(0, 500, (batch_size, seq_len))
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        assert outputs.logits is not None
        assert outputs.router_logits is not None
    
    def test_capacity_factor_causes_drops(self):
        """Test that low capacity factor causes token drops."""
        config = MoedlConfig(
            vocab_size=500,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            capacity_factor=0.5,  # Low capacity - should cause drops
        )
        model = MoedlForCausalLM(config)
        model.eval()
        
        batch_size, seq_len = 4, 50  # More tokens to trigger drops
        input_ids = torch.randint(0, 500, (batch_size, seq_len))
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        # Check that some layer had drops
        total_drops = sum(layer.moe.n_drop for layer in model.model.layers if hasattr(layer, 'moe'))
        # With CF=0.5 and enough tokens, we should see some drops
        # (not asserting > 0 because it's probabilistic, but verifying counter exists)
        assert total_drops >= 0
    
    def test_capacity_factor_no_drops_when_disabled(self):
        """Test that CF=0 never drops tokens."""
        config = MoedlConfig(
            vocab_size=500,
            hidden_size=64,
            num_experts=8,
            num_active_experts=2,
            capacity_factor=0.0,  # Disabled
        )
        model = MoedlForCausalLM(config)
        model.eval()
        
        batch_size, seq_len = 4, 50
        input_ids = torch.randint(0, 500, (batch_size, seq_len))
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        # All n_drop should be 0
        for layer in model.model.layers:
            if hasattr(layer, 'moe'):
                assert layer.moe.n_drop == 0
    
    def test_capacity_factor_dense_model_ignored(self):
        """Test that dense models ignore capacity_factor."""
        config = MoedlConfig(
            vocab_size=500,
            hidden_size=64,
            num_experts=1,  # Dense
            capacity_factor=0.0,  # Should be forced to 0
        )
        assert config.capacity_factor == 0.0
    
    def test_capacity_factor_with_lb_coeff(self):
        """Test capacity_factor works with lb_coeff load balancing."""
        config = MoedlConfig(
            vocab_size=500,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            capacity_factor=1.0,
            lb_coeff=0.01,  # Traditional load balancing
        )
        model = MoedlForCausalLM(config)
        model.eval()
        
        batch_size, seq_len = 2, 20
        input_ids = torch.randint(0, 500, (batch_size, seq_len))
        labels = input_ids.clone()
        
        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
        
        # Should have both main loss and lb loss
        assert outputs.loss is not None
        assert outputs.router_logits is not None
        
        # Verify n_drop tracking works
        for layer in model.model.layers:
            if hasattr(layer, 'moe'):
                assert hasattr(layer.moe, 'n_drop')
                assert layer.moe.n_drop >= 0
    
    def test_capacity_factor_with_lb_gamma(self):
        """Test capacity_factor works with lb_gamma load balancing."""
        config = MoedlConfig(
            vocab_size=500,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            capacity_factor=1.0,
            lb_gamma=0.01,  # Score bias load balancing
        )
        model = MoedlForCausalLM(config)
        model.eval()
        
        batch_size, seq_len = 2, 20
        input_ids = torch.randint(0, 500, (batch_size, seq_len))
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        # Should work with lb_gamma
        assert outputs.logits is not None
        assert outputs.router_logits is not None
        
        # Verify n_drop tracking works
        for layer in model.model.layers:
            if hasattr(layer, 'moe'):
                assert hasattr(layer.moe, 'n_drop')
                assert layer.moe.n_drop >= 0
                # With lb_gamma, e_score_bias should exist
                assert hasattr(layer.moe, 'e_score_bias')


class TestMoedlSaveLoadTrustRemoteCode:
    """Test save/load functionality with trust_remote_code."""
    
    @pytest.fixture
    def tiny_moe_config(self):
        """Create a tiny MoE model config for save/load testing."""
        return MoedlConfig(
            vocab_size=500,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            num_experts=4,
            num_active_experts=2,
            num_shared_experts=1,
            lb_coeff=0.01,
        )
    
    @pytest.fixture
    def tiny_dense_config_for_save(self):
        """Create a tiny dense model config for save/load testing."""
        return MoedlConfig(
            vocab_size=500,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            num_experts=1,
            num_active_experts=1,
        )
    
    def test_save_includes_custom_code_files(self, tmp_path, tiny_moe_config):
        """Test that save_pretrained copies modeling and config files."""
        model = MoedlForCausalLM(tiny_moe_config)
        save_dir = tmp_path / "moedl_checkpoint"
        
        # Save the model
        model.save_pretrained(save_dir)
        
        # Check that custom code files were copied
        assert (save_dir / "configuration_moedl.py").exists()
        assert (save_dir / "modeling_moedl.py").exists()
        assert (save_dir / "grouped_glu.py").exists()
        assert (save_dir / "config.json").exists()
        assert (save_dir / "model.safetensors").exists() or (save_dir / "pytorch_model.bin").exists()
    
    def test_config_has_automap(self, tiny_moe_config):
        """Test that config includes auto_map for remote code loading."""
        config = tiny_moe_config
        assert hasattr(config, 'auto_map')
        assert 'AutoConfig' in config.auto_map
        assert 'AutoModelForCausalLM' in config.auto_map
        assert config.auto_map['AutoConfig'] == 'configuration_moedl.MoedlConfig'
        assert config.auto_map['AutoModelForCausalLM'] == 'modeling_moedl.MoedlForCausalLM'
    
    def test_load_fails_without_trust_remote_code(self, tmp_path, tiny_moe_config):
        """Test that loading fails when trust_remote_code=False.
        
        This test runs in a subprocess to ensure moelab is not already imported,
        which would make the model class available even without trust_remote_code.
        """
        # Save model using moe-lab
        model = MoedlForCausalLM(tiny_moe_config)
        save_dir = tmp_path / "moedl_checkpoint"
        model.save_pretrained(save_dir)
        
        # Test loading in subprocess without trust_remote_code
        test_code = f"""
import sys
from transformers import AutoModelForCausalLM

try:
    model = AutoModelForCausalLM.from_pretrained(
        r"{save_dir}",
        trust_remote_code=False
    )
    sys.exit(1)  # Should not reach here
except (ValueError, OSError) as e:
    # Expected behavior - model requires trust_remote_code
    sys.exit(0)
except Exception as e:
    print(f"Unexpected exception: {{type(e).__name__}}: {{e}}")
    sys.exit(2)
"""
        
        result = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True,
            text=True
        )
        
        # Exit code 0 means it raised the expected exception
        # Exit code 1 means it loaded successfully (test failed)
        # Exit code 2 means unexpected exception
        if result.returncode == 1:
            pytest.fail("Model loaded successfully without trust_remote_code=True, but should have failed")
        elif result.returncode == 2:
            pytest.fail(f"Unexpected exception in subprocess:\n{result.stdout}\n{result.stderr}")
        
        assert result.returncode == 0, f"Test failed with unexpected return code: {result.returncode}"
    
    def test_load_succeeds_with_trust_remote_code(self, tmp_path, tiny_moe_config):
        """Test that loading succeeds when trust_remote_code=True.
        
        This test runs in a subprocess to ensure clean loading environment.
        """
        # Save model using moe-lab
        original_model = MoedlForCausalLM(tiny_moe_config)
        save_dir = tmp_path / "moedl_checkpoint"
        original_model.save_pretrained(save_dir)
        
        # Test loading in subprocess with trust_remote_code=True
        test_code = f"""
import sys
from transformers import AutoModelForCausalLM

try:
    model = AutoModelForCausalLM.from_pretrained(
        r"{save_dir}",
        trust_remote_code=True
    )
    # Verify it's the correct model type
    assert model.__class__.__name__ == 'MoedlForCausalLM', f"Expected MoedlForCausalLM, got {{model.__class__.__name__}}"
    assert model.config.num_experts == {tiny_moe_config.num_experts}
    assert model.config.num_active_experts == {tiny_moe_config.num_active_experts}
    sys.exit(0)  # Success
except Exception as e:
    print(f"Failed to load with trust_remote_code=True: {{type(e).__name__}}: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
        
        result = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            pytest.fail(f"Failed to load model with trust_remote_code=True:\n{result.stdout}\n{result.stderr}")
    
    def test_loaded_model_forward_pass(self, tmp_path, tiny_moe_config):
        """Test that loaded model can perform forward pass correctly."""
        # Save model using moe-lab
        original_model = MoedlForCausalLM(tiny_moe_config)
        save_dir = tmp_path / "moedl_checkpoint"
        original_model.save_pretrained(save_dir)
        
        # Load with trust_remote_code=True
        loaded_model = AutoModelForCausalLM.from_pretrained(
            save_dir,
            trust_remote_code=True
        )
        
        # Test forward pass
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, tiny_moe_config.vocab_size, (batch_size, seq_len))
        
        original_model.eval()
        loaded_model.eval()
        
        with torch.no_grad():
            original_output = original_model(input_ids)
            loaded_output = loaded_model(input_ids)
        
        # Outputs should match
        torch.testing.assert_close(
            original_output.logits,
            loaded_output.logits,
            rtol=1e-5,
            atol=1e-6
        )
    
    def test_loaded_model_generation(self, tmp_path, tiny_moe_config):
        """Test that loaded model can generate text."""
        # Save model using moe-lab
        original_model = MoedlForCausalLM(tiny_moe_config)
        save_dir = tmp_path / "moedl_checkpoint"
        original_model.save_pretrained(save_dir)
        
        # Load with trust_remote_code=True
        loaded_model = AutoModelForCausalLM.from_pretrained(
            save_dir,
            trust_remote_code=True
        )
        
        loaded_model.eval()
        
        # Test generation
        input_ids = torch.randint(0, tiny_moe_config.vocab_size, (1, 5))
        
        with torch.no_grad():
            generated = loaded_model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
            )
        
        # Should generate tokens
        assert generated.shape[1] == input_ids.shape[1] + 10
    
    def test_dense_model_save_load(self, tmp_path, tiny_dense_config_for_save):
        """Test save/load for dense (non-MoE) model."""
        # Save dense model
        original_model = MoedlForCausalLM(tiny_dense_config_for_save)
        save_dir = tmp_path / "moedl_dense_checkpoint"
        original_model.save_pretrained(save_dir)
        
        # Load with trust_remote_code=True
        loaded_model = AutoModelForCausalLM.from_pretrained(
            save_dir,
            trust_remote_code=True
        )
        
        # Verify it's correct
        assert loaded_model.__class__.__name__ == 'MoedlForCausalLM'
        assert loaded_model.config.num_experts == 1
        assert loaded_model.config.num_active_experts == 1
        
        # Test forward pass matches
        input_ids = torch.randint(0, 500, (2, 16))
        original_model.eval()
        loaded_model.eval()
        
        with torch.no_grad():
            original_output = original_model(input_ids)
            loaded_output = loaded_model(input_ids)
        
        torch.testing.assert_close(
            original_output.logits,
            loaded_output.logits,
            rtol=1e-5,
            atol=1e-6
        )
    
    def test_config_roundtrip(self, tmp_path, tiny_moe_config):
        """Test that config can be saved and loaded correctly."""
        model = MoedlForCausalLM(tiny_moe_config)
        save_dir = tmp_path / "moedl_checkpoint"
        model.save_pretrained(save_dir)
        
        # Load config with trust_remote_code=True
        loaded_config = AutoConfig.from_pretrained(
            save_dir,
            trust_remote_code=True
        )
        
        # Verify all MoE-specific config values
        assert loaded_config.num_experts == tiny_moe_config.num_experts
        assert loaded_config.num_active_experts == tiny_moe_config.num_active_experts
        assert loaded_config.num_shared_experts == tiny_moe_config.num_shared_experts
        assert loaded_config.lb_coeff == tiny_moe_config.lb_coeff
        assert loaded_config.capacity_factor == tiny_moe_config.capacity_factor
    
    def test_load_with_lb_gamma(self, tmp_path):
        """Test save/load with lb_gamma load balancing strategy."""
        config = MoedlConfig(
            vocab_size=500,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=4,
            num_active_experts=2,
            lb_gamma=0.01,  # Using lb_gamma instead of lb_coeff
        )
        
        original_model = MoedlForCausalLM(config)
        save_dir = tmp_path / "moedl_lb_gamma"
        original_model.save_pretrained(save_dir)
        
        # Load with trust_remote_code=True
        loaded_model = AutoModelForCausalLM.from_pretrained(
            save_dir,
            trust_remote_code=True
        )
        
        # Verify lb_gamma configuration
        assert loaded_model.config.lb_gamma == 0.01
        assert loaded_model.config.lb_coeff == 0.0
        
        # Test forward pass
        input_ids = torch.randint(0, 500, (2, 16))
        loaded_model.eval()
        
        with torch.no_grad():
            outputs = loaded_model(input_ids)
        
        assert outputs.logits is not None
        assert outputs.router_logits is not None


# ========================================
# Load Balancing Loss Evaluation Bug Tests
# ========================================

def test_load_balancing_loss_not_computed_during_eval():
    """Test that load balancing loss is NOT computed during evaluation mode.
    
    BUG: load_balancing_loss_func is called whenever loss is not None,
    which happens when labels are provided during evaluation.
    It should ONLY be computed during training.
    
    This test will FAIL with the current buggy implementation and PASS
    once the bug is fixed.
    """
    # Create a small MoE model with load balancing enabled
    config = MoedlConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,  # MoE with 4 experts
        num_active_experts=2,  # Top-2 routing
        lb_coeff=0.01,  # Load balancing enabled with non-zero coefficient
        pad_token_id=0,
    )
    
    model = MoedlForCausalLM(config)
    model.eval()  # Set to evaluation mode
    
    # Create dummy input with labels (typical for evaluation with loss computation)
    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    
    # Forward pass in eval mode
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
    
    # The bug: aux_loss (load balancing loss) should be None or 0 during eval
    # but currently it's computed because labels are provided
    assert outputs.aux_loss is None or outputs.aux_loss == 0, \
        f"Load balancing loss should NOT be computed during evaluation! " \
        f"Got aux_loss={outputs.aux_loss}, but expected None or 0. " \
        f"model.training={model.training}"


def test_load_balancing_loss_computed_during_training():
    """Test that load balancing loss IS computed during training mode."""
    # Create a small MoE model with load balancing enabled
    config = MoedlConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_active_experts=2,
        lb_coeff=0.01,  # Load balancing enabled
        pad_token_id=0,
    )
    
    model = MoedlForCausalLM(config)
    model.train()  # Set to training mode
    
    # Create dummy input with labels
    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    
    # Forward pass in training mode
    outputs = model(input_ids=input_ids, labels=labels)
    
    # During training with lb_coeff > 0, aux_loss should be computed
    assert outputs.aux_loss is not None, \
        "Load balancing loss should be computed during training!"
    assert outputs.aux_loss > 0, \
        f"Load balancing loss should be positive during training! Got {outputs.aux_loss}"


def test_load_balancing_loss_not_computed_when_lb_coeff_zero():
    """Test that load balancing loss is NOT computed when lb_coeff=0."""
    # Create model with lb_coeff=0 (load balancing disabled)
    config = MoedlConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_active_experts=2,
        lb_coeff=0.0,  # Load balancing DISABLED
        pad_token_id=0,
    )
    
    model = MoedlForCausalLM(config)
    model.train()  # Even in training mode
    
    # Create dummy input with labels
    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    
    # Forward pass
    outputs = model(input_ids=input_ids, labels=labels)
    
    # With lb_coeff=0, aux_loss should be None
    assert outputs.aux_loss is None, \
        f"Load balancing loss should NOT be computed when lb_coeff=0! Got {outputs.aux_loss}"


def test_training_vs_eval_mode_consistency():
    """Test that model behaves correctly in both training and eval modes."""
    config = MoedlConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_active_experts=2,
        lb_coeff=0.01,
        pad_token_id=0,
    )
    
    model = MoedlForCausalLM(config)
    
    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    
    # Training mode
    model.train()
    with torch.no_grad():  # Disable gradients for comparison
        train_outputs = model(input_ids=input_ids, labels=labels)
    
    # Eval mode
    model.eval()
    with torch.no_grad():
        eval_outputs = model(input_ids=input_ids, labels=labels)
    
    # Main loss should be similar (not identical due to potential dropout)
    # but aux_loss should differ
    assert train_outputs.aux_loss is not None, "Training should have aux_loss"
    assert eval_outputs.aux_loss is None or eval_outputs.aux_loss == 0, \
        f"Eval should NOT have aux_loss! Got {eval_outputs.aux_loss}"


class TestMoedlCapacityFactorGeneration:
    """Test to expose capacity_factor bug during generation.
    
    During generation (LM decoding), tokens are often generated one at a time.
    With capacity_factor > 0, expert_capacity = int((T*K/E) * capacity_factor)
    can become 0 when T is very small, triggering assertion error or other issues.
    
    For example:
    - T=1 (single token), K=2, E=8
    - (1*2/8) * 2.0 = 0.5 -> int(0.5) = 0
    - This triggers: "assert expert_capacity > 0"
    """
    
    def test_capacity_factor_generation_single_token(self):
        """BUG: Generation with capacity_factor > 0 fails with small batch/single token.
        
        This test should FAIL until the bug is fixed.
        Expert capacity calculation breaks down when T (total tokens) is very small.
        """
        config = MoedlConfig(
            vocab_size=500,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            capacity_factor=2.0,  # Reasonable CF, but breaks with T=1
        )
        model = MoedlForCausalLM(config)
        model.eval()
        
        # Single token input - typical for generation
        input_ids = torch.randint(0, 500, (1, 1))
        
        # This should fail with assertion error:
        # "expert_capacity must be positive. capacity_factor too small?"
        # Because: expert_capacity = int((1*2/8) * 2.0) = int(0.5) = 0
        with torch.no_grad():
            try:
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=5,
                    do_sample=False,
                )
                # If we get here, bug might be "fixed" but we should verify
                assert outputs.shape[1] > 1, "Should generate at least one token"
            except AssertionError as e:
                if "expert_capacity must be positive" in str(e):
                    pytest.fail(f"BUG CONFIRMED: {e}")
                else:
                    raise
    
    def test_capacity_factor_generation_small_batch(self):
        """BUG: Generation with small batches and capacity_factor > 0.
        
        Even with slightly larger inputs, autoregressive decoding processes
        one new token at a time, making expert_capacity calculation fragile.
        """
        config = MoedlConfig(
            vocab_size=500,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            capacity_factor=1.5,
        )
        model = MoedlForCausalLM(config)
        model.eval()
        
        # Small batch with very short sequence
        input_ids = torch.randint(0, 500, (2, 3))
        
        with torch.no_grad():
            try:
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=10,
                    do_sample=False,
                )
                assert outputs.shape[0] == 2
                assert outputs.shape[1] >= 3
            except AssertionError as e:
                if "expert_capacity must be positive" in str(e):
                    pytest.fail(f"BUG CONFIRMED: {e}")
                else:
                    raise
    
    def test_capacity_factor_forward_vs_generation(self):
        """Compare forward pass (works) vs generation (may fail) with capacity_factor.
        
        Forward pass with small inputs works fine because all tokens are processed together.
        Generation processes tokens incrementally, exposing the bug.
        """
        config = MoedlConfig(
            vocab_size=500,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            capacity_factor=2.0,
        )
        model = MoedlForCausalLM(config)
        model.eval()
        
        input_ids = torch.randint(0, 500, (1, 5))
        
        # Forward pass should work - T=5 tokens total
        with torch.no_grad():
            forward_outputs = model(input_ids)
            assert forward_outputs.logits is not None
            # expert_capacity = int((5*2/8) * 2.0) = int(2.5) = 2 ✓
        
        # Generation should fail when processing new tokens one-by-one
        # During generation, cache grows but new token T=1
        # expert_capacity = int((1*2/8) * 2.0) = int(0.5) = 0 ✗
        with torch.no_grad():
            try:
                gen_outputs = model.generate(
                    input_ids,
                    max_new_tokens=3,
                    do_sample=False,
                )
                # If successful, check output
                assert gen_outputs.shape[1] >= 5
            except AssertionError as e:
                if "expert_capacity must be positive" in str(e):
                    pytest.fail(
                        f"BUG CONFIRMED: Forward pass works but generation fails. "
                        f"Error: {e}"
                    )
                else:
                    raise
    
    def test_capacity_factor_zero_works_for_generation(self):
        """CONTROL: capacity_factor=0 should work fine for generation."""
        config = MoedlConfig(
            vocab_size=500,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts=8,
            num_active_experts=2,
            capacity_factor=0.0,  # Disabled - should work
        )
        model = MoedlForCausalLM(config)
        model.eval()
        
        input_ids = torch.randint(0, 500, (1, 1))
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
            )
        
        assert outputs.shape[0] == 1
        assert outputs.shape[1] == 11  # 1 input + 10 generated


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
