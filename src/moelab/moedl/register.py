from transformers import AutoConfig, AutoModelForCausalLM
from .configuration_moedl import MoedlConfig
from .modeling_moedl import MoedlForCausalLM


def register_moedl() -> None:
    AutoConfig.register(MoedlConfig.model_type, MoedlConfig)
    AutoModelForCausalLM.register(MoedlConfig, MoedlForCausalLM)

