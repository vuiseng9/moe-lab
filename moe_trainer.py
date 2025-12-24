import torch
from transformers import Trainer

class MOETrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wb_handler = None
    
    def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):

        loss, outputs = super().compute_loss(model, inputs, True, num_items_in_batch)
        if 'wandb' in self.args.report_to:
            if self.wb_handler is None:
                self.wb_handler = self.find_wandb_handler()

            tpe = outputs.tokens_per_expert.detach().to(torch.float32).cpu()
            K, E = tpe.shape
            log_dict = {f"moe/top{k+1}/e{e}": float(tpe[k, e]) for k in range(K) for e in range(E)}

            if getattr(self.model.config, 'enable_lbl', False):
                aux_loss = outputs.aux_loss.detach().to(torch.float32).cpu().item()
                log_dict.update({f"train/moe_aux_loss": aux_loss})

            self.wb_handler.log(log_dict)

        return (loss, outputs) if return_outputs else loss

    def find_wandb_handler(self):
        # we don't call self.log to avoid stdout blowing up with too much details
        for handler in self.callback_handler.callbacks:
            if handler.__class__.__name__ == "WandbCallback":
                return handler._wandb
        return None