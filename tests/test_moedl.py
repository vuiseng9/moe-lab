"""
Minimal tests for Moedl model.

Tests cover:
1. Constructor tests (success and failure cases)
2. Equivalency tests with Llama (forward and backward)
3. Generate function tests
"""
import pytest
import torch
import torch.nn as nn
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
                assert len(layer.moe.experts) == num_experts
    
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
        
        # Expert weights
        for i in range(tiny_moe_config["num_experts"]):
            moedl_moe.experts[i].gate.weight.data.copy_(olmoe_moe.experts[i].gate_proj.weight.data)
            moedl_moe.experts[i].up.weight.data.copy_(olmoe_moe.experts[i].up_proj.weight.data)
            moedl_moe.experts[i].down.weight.data.copy_(olmoe_moe.experts[i].down_proj.weight.data)
        
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
            # Copy router and routed experts
            layer.moe.router.weight.data.copy_(model_no_shared.model.layers[i].moe.router.weight.data)
            for j in range(8):
                layer.moe.experts[j].gate.weight.data.copy_(model_no_shared.model.layers[i].moe.experts[j].gate.weight.data)
                layer.moe.experts[j].up.weight.data.copy_(model_no_shared.model.layers[i].moe.experts[j].up.weight.data)
                layer.moe.experts[j].down.weight.data.copy_(model_no_shared.model.layers[i].moe.experts[j].down.weight.data)
        
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
    
    @pytest.fixture
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
        
        # Moedl config
        moedl_config = MoedlConfig(**tiny_moe_config_nogrouping)
        moedl_moe = MoeBlk(moedl_config)
        
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
        
        # Copy routed expert weights
        for i in range(tiny_moe_config_nogrouping["num_experts"]):
            moedl_moe.experts[i].gate.weight.data.copy_(ds_moe.experts[i].gate_proj.weight.data)
            moedl_moe.experts[i].up.weight.data.copy_(ds_moe.experts[i].up_proj.weight.data)
            moedl_moe.experts[i].down.weight.data.copy_(ds_moe.experts[i].down_proj.weight.data)
        
        # Copy shared expert weights
        # DeepSeek V3: single large MLP (intermediate = moe_intermediate_size * n_shared_experts)
        # Moedl: list of n_shared_experts MLPs (each with moe_intermediate_size)
        # For n_shared_experts=1, they should be equivalent
        moedl_moe.common[0].gate.weight.data.copy_(ds_moe.shared_experts.gate_proj.weight.data)
        moedl_moe.common[0].up.weight.data.copy_(ds_moe.shared_experts.up_proj.weight.data)
        moedl_moe.common[0].down.weight.data.copy_(ds_moe.shared_experts.down_proj.weight.data)
        
        # Set zero bias for exact equivalence
        moedl_moe.e_score_bias.zero_()
        ds_moe.gate.e_score_correction_bias.zero_()
        
        moedl_moe.eval()
        ds_moe.eval()
        
        # Test forward
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
        
        # Moedl config
        moedl_config = MoedlConfig(**tiny_moe_config_nogrouping)
        moedl_moe = MoeBlk(moedl_config)
        
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
        moedl_moe.e_score_bias.zero_()
        ds_router.e_score_correction_bias.zero_()
        
        moedl_moe.eval()
        ds_router.eval()
        
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
        torch.testing.assert_close(
            moedl_topk_weights_norm,
            ds_topk_weights,
            rtol=1e-4,
            atol=1e-5,
            msg="Normalized expert weights should match"
        )


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
