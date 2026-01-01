from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM


# NOTE: we keep all subclassing in this file for brevity
# since we only override minimally.


class MoelabDeepseekV3Config(DeepseekV3Config):
    model_type = "moelab_deepseek_v3"

    def __init__(self, **kwargs):
        load_balance_gamma: float = kwargs.pop("load_balance_gamma", 0.0)
        super().__init__(**kwargs)
        self.load_balance_gamma = load_balance_gamma


class MoelabDeepseekV3ForCausalLM(DeepseekV3ForCausalLM):
    config_class = MoelabDeepseekV3Config


def register_moelab_deepseek_v3() -> None:
    AutoConfig.register(MoelabDeepseekV3Config.model_type, MoelabDeepseekV3Config)
    AutoModelForCausalLM.register(MoelabDeepseekV3Config, MoelabDeepseekV3ForCausalLM)
