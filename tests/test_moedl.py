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

class TestMoedlConstructor:
    """Test model construction with various configurations."""
    
    def test_basic_construction(self):
        """Test creating a minimal Moedl model."""
        config = MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=512,
        )
        model = MoedlForCausalLM(config)
        assert model is not None
        assert model.config.vocab_size == 1000
        assert model.config.hidden_size == 128
        assert model.config.num_hidden_layers == 2
    
    def test_construction_with_gqa(self):
        """Test construction with grouped-query attention."""
        config = MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,  # GQA
            max_position_embeddings=512,
        )
        model = MoedlForCausalLM(config)
        assert model.config.num_key_value_heads == 2
        assert model.config.num_attention_heads == 4
    
    def test_construction_model_only(self):
        """Test constructing just Moedl without LM head."""
        config = MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
        )
        model = Moedl(config)
        assert model is not None
        assert not hasattr(model, 'lm_head')
    
    def test_fail_invalid_heads(self):
        """Test that invalid head configuration raises error during forward."""
        config = MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=5,  # Not divisible into hidden_size
            max_position_embeddings=512,
        )
        model = MoedlForCausalLM(config)
        
        # Forward should fail due to shape mismatch
        with pytest.raises(RuntimeError):
            _ = model(torch.randint(0, 1000, (1, 10)))
    
    def test_fail_kv_heads_mismatch(self):
        """Test that kv_heads > num_heads fails during forward."""
        config = MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=8,  # Can't have more KV heads than query heads
            max_position_embeddings=512,
        )
        model = MoedlForCausalLM(config)
        
        # Forward should fail - KV heads must divide query heads
        with pytest.raises(RuntimeError):
            _ = model(torch.randint(0, 1000, (1, 10)))


class TestMoedlLlamaEquivalence:
    """Test equivalence between Moedl and Llama models."""
    
    @pytest.fixture
    def tiny_config(self):
        """Shared tiny config for testing."""
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
        }
    
    def test_forward_equivalence(self, tiny_config):
        """Test that Moedl and Llama produce same outputs with same weights."""
        # Create both models
        moedl_config = MoedlConfig(**tiny_config)
        moedl_model = MoedlForCausalLM(moedl_config)
        
        llama_config = AutoConfig.for_model("llama", **tiny_config)
        llama_model = AutoModelForCausalLM.from_config(llama_config)
        
        # Copy weights from Llama to Moedl with key mapping
        llama_state_dict = llama_model.state_dict()
        moedl_state_dict = convert_llama_state_dict_to_moedl(llama_state_dict)
        moedl_model.load_state_dict(moedl_state_dict, strict=True)
        
        # Set to eval mode
        moedl_model.eval()
        llama_model.eval()
        
        # Test forward pass
        input_ids = torch.randint(0, tiny_config["vocab_size"], (2, 32))
        
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
    
    def test_backward_equivalence(self, tiny_config):
        """Test that gradients are equivalent between Moedl and Llama."""
        # Create both models
        moedl_config = MoedlConfig(**tiny_config)
        moedl_model = MoedlForCausalLM(moedl_config)
        
        llama_config = AutoConfig.for_model("llama", **tiny_config)
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
        input_ids = torch.randint(0, tiny_config["vocab_size"], (batch_size, seq_len))
        labels = torch.randint(0, tiny_config["vocab_size"], (batch_size, seq_len))
        
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
    
    def test_state_dict_compatibility(self, tiny_config):
        """Test that state dict keys can be mapped between Moedl and Llama."""
        moedl_config = MoedlConfig(**tiny_config)
        moedl_model = MoedlForCausalLM(moedl_config)
        
        llama_config = AutoConfig.for_model("llama", **tiny_config)
        llama_model = AutoModelForCausalLM.from_config(llama_config)
        
        moedl_keys = set(moedl_model.state_dict().keys())
        llama_keys = set(llama_model.state_dict().keys())
        
        # Map Llama keys to Moedl format
        mapped_llama_keys = {llama_to_moedl_key_map(key) for key in llama_keys}
        
        # After mapping, keys should be identical
        assert moedl_keys == mapped_llama_keys, f"Key mismatch: {moedl_keys ^ mapped_llama_keys}"


class TestMoedlGenerate:
    """Test generation functionality."""
    
    def test_basic_generate(self):
        """Test basic text generation."""
        config = MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=512,
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
    
    def test_generate_with_sampling(self):
        """Test generation with sampling."""
        config = MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=512,
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
    
    def test_generate_batch(self):
        """Test batched generation."""
        config = MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=512,
            pad_token_id=0,
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
    
    def test_generate_with_eos(self):
        """Test early stopping with EOS token."""
        config = MoedlConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=512,
            eos_token_id=2,
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


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
