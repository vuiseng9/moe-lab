from transformers import AutoConfig, AutoModelForCausalLM
from .modeling_olmoe import MoelabOlmoeForCausalLM, MoelabOlmoeConfig

def register_moelab_olmoe() -> None:
    AutoConfig.register(MoelabOlmoeConfig.model_type, MoelabOlmoeConfig)
    AutoModelForCausalLM.register(MoelabOlmoeConfig, MoelabOlmoeForCausalLM)