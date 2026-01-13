import math
from moelab.trainer import MoelabTrainer
import torch

class MoedlTrainer(MoelabTrainer):
    """
    MoedlTrainer is a specialized trainer for Moedl type. 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_moe = None
        self._cfg = None

    @property
    def cfg(self):
        """Return the model config."""
        if self._cfg is None:
            model = self.model.module if hasattr(self.model, "module") else self.model
            self._cfg = model.config
        return self._cfg
    
    @property
    def is_moe(self):
        """Check if the underlying model has MoE layers (num_experts > 1)."""
        if self._is_moe is None:
            if not getattr(self.cfg, "num_experts", None):
                raise ValueError("Moedl config must have 'num_experts' attribute to determine MoE status.")
            
            self.E = self.cfg.num_experts
            self.K = self.cfg.num_active_experts
            self._is_moe = self.E > 1
        return self._is_moe

    def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):
        # Call HF Trainer's compute_loss
        # we force return_outputs=True to always get outputs
        # only filter during return based on return_outputs flag at the end
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        if "wandb" in self.args.report_to and self.wandb is not None:
            if self.is_moe:
                count, _ = self.get_expert_stats(outputs.router_logits)

                # only log top-1 and per model stats for brevity
                frac_per_e = (count[:, 0, :].sum(dim=0)/num_items_in_batch).tolist()

                log_dict = {
                    f"moe/top1/load/e{i:03d}": round(frac, 3) for i, frac in enumerate(frac_per_e)
                }

                if self.cfg.lb_coeff > 0:
                    # log load balancing loss
                    lb_loss = round(outputs.aux_loss.detach().item(), 6)
                else:
                    lb_loss = math.nan
                log_dict.update({f"train/lb_loss": lb_loss})

                self.wandb.log(log_dict)

        return (loss, outputs) if return_outputs else loss

    @torch.no_grad()
    def get_expert_stats(self, router_logits):
        # model wide aggregration per k per e

        K = self.cfg.num_active_experts
        E = self.cfg.num_experts

        _, k_ids = torch.stack(router_logits, dim=0).topk(k=K, dim=-1)

        L, T, _K  = k_ids.shape  # L layers, T tokens
        assert K == _K, "topk size mismatch"

        _, k_ids = torch.stack(router_logits, dim=0).topk(k=K, dim=-1)

        # count (L, K, E)
        # count is per layer per k slot per expert
        count = torch.zeros((L, K, E), device=k_ids.device, dtype=torch.int64)
        # k_ids (L, T, K) -> (L, K, T)
        k_ids = k_ids.permute(0, 2, 1)
        # scatter add 1 for each ids
        count.scatter_add_(dim=2, index=k_ids, src=torch.ones_like(k_ids, dtype=count.dtype))

        assert (count.sum(dim=-1) == T).all().item(), "k slot(s) sum mismatch with T"

        # Per k, routed tokens to all Es sum up to T tokens
        # divide count by T effectively gives proportion of tokens per expert for that k slot
        frac = count.float() / T
        
        return count, frac