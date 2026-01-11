import transformers as tfmr
from packaging.version import Version

# NOTE: our tests and CI currently depend on this exact version
# just comment out this part if you want a quick test on different version

_REQUIRED = Version("4.57.3")
_INSTALLED = Version(tfmr.__version__)

if _INSTALLED != _REQUIRED:
    raise RuntimeError(f"moelab requires transformers=={_REQUIRED}, got {tfmr.__version__}")


from .olmoe.register import register_moelab_olmoe, MoelabOlmoeConfig
from .olmoe.trainer import OlmoeTrainer
from .deepseek_v3.register import register_moelab_deepseek_v3, MoelabDeepseekV3Config
from .deepseek_v3.trainer import DSv3Trainer

from .moedl.register import register_moedl

register_moelab_olmoe()
register_moelab_deepseek_v3()
register_moedl()

MOELAB_TRAINER_CLS = {
    MoelabDeepseekV3Config.model_type: DSv3Trainer,
    MoelabOlmoeConfig.model_type: OlmoeTrainer,
}
