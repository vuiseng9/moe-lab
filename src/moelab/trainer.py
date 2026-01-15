from transformers import Trainer


class MoelabTrainer(Trainer):
    """
    adds utilities only. Model-specific overrides live downstream.
    # TODO: revise design, brittle design
    #       wandb handle is only created if WandbCallback is used and during runtime
    #       if any downstream consumer of MoelabTrainer.wandb is called before that, 
    #       it will be None.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._wb_handler = None

    @property
    def wandb(self):
        """Return wandb module/handle from HF WandbCallback if present, else None.
        
        Usage: trainer.wandb.log() instead of Trainer.log() to prevent excessive stdout logging.
        Trainer.log() appends to the line per logging step.
        """
        if self._wb_handler is not None:
            return self._wb_handler

        for cb in getattr(self.callback_handler, "callbacks", []):
            if cb.__class__.__name__ == "WandbCallback":
                self._wb_handler = getattr(cb, "_wandb", None)
                return self._wb_handler

        return None
