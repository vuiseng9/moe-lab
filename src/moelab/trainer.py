from transformers import Trainer

class MoelabTrainer(Trainer):
    """
    adds utilities only. Model-specific overrides live downstream.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wb_handler = None

    def get_wandb_handler(self):
        """Return wandb module/handle from HF WandbCallback if present, else None."""
        # NOTE: the intent is to avoid Trainer.log to prevent excessive stdout logging,
        # user can use wb_handler.log instead.
        
        if self.wb_handler is not None:
            return self.wb_handler

        for cb in getattr(self.callback_handler, "callbacks", []):
            if cb.__class__.__name__ == "WandbCallback":
                self.wb_handler = getattr(cb, "_wandb", None)
                return self.wb_handler

        return None