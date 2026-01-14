"""Moedl model configuration"""

from transformers import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation

class MoedlConfig(PretrainedConfig):

    model_type = "moedl"
    keys_to_ignore_at_inference = ["past_key_values", "aux_loss", "router_logits"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        # MoE specific
        num_experts=16,
        num_active_experts=4,
        num_shared_experts=0,
        lb_coeff=0.0,
        lb_gamma=0.0,
        capacity_factor=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        
        # MoE specific
        if num_experts <= 0 or num_active_experts <= 0 or num_shared_experts < 0:
            raise ValueError("num_experts and num_active_experts must be non-zero positive integers, and "
                             "num_shared_experts must be a non-negative integer.")
        elif num_experts > 1:
            self.num_experts = num_experts
            if num_active_experts > num_experts:
                raise ValueError("num_active_experts cannot be greater than num_experts.")
            if lb_coeff < 0.0:
                raise ValueError("lb_coeff must be a non-negative float.")
            if lb_gamma < 0.0:
                raise ValueError("load_balance_gamma must be a non-negative float.")
            if lb_coeff > 0.0 and lb_gamma > 0.0:
                raise NotImplementedError("Currently, load balance strategy via (1) penalty (2) biasing are mutually exclusive."
                                          "Either lb_coeff or lb_gamma can be set to non-zero, but not both. Maybe supported in future.")
            if capacity_factor < 0.0:
                raise ValueError("capacity_factor must be a non-negative float. Set to 0.0 to disable capacity limit.")
            self.num_active_experts = num_active_experts
            self.num_shared_experts = num_shared_experts
            self.lb_coeff = lb_coeff
            self.lb_gamma = lb_gamma
            self.capacity_factor = capacity_factor
        else:
            # dense MLP
            # while shared expert semantically closer to dense MLP,
            # we use num_experts = 1 and num_active_experts = 1 to represent dense MLP
            # we totally ignore num_shared_experts for dense, forcing it to be 0
            self.num_experts = 1  # Disable MoE when num_experts is set to 1
            self.num_active_experts = 1         
            self.num_shared_experts = 0
            self.lb_coeff = 0.0  
            self.lb_gamma = 0.0
            self.capacity_factor = 0.0

        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

__all__ = ["MoedlConfig"]