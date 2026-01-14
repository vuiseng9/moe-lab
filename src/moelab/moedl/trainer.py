import math
import logging
import torch
from moelab.trainer import MoelabTrainer
from moelab.moedl import MoeBlk
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class MoedlTrainer(MoelabTrainer):
    """
    MoedlTrainer is a specialized trainer for Moedl model type. 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        model = self.model.module if hasattr(self.model, "module") else self.model
        self._cfg = model.config
        
        if not getattr(self._cfg, "num_experts", None):
            raise ValueError("Moedl config must have 'num_experts' attribute to determine MoE status.")
        
        self.L  = self.cfg.num_hidden_layers
        self.E  = self.cfg.num_experts
        self.K  = self.cfg.num_active_experts
        self.ES = self.cfg.num_shared_experts
        self.CF = self.cfg.capacity_factor
        self._is_moe = self.E > 1

        self.lb_ctrl = None
        self.moe_modules = None
        if self.is_moe:
            self.moe_modules = {}
            for name, module in model.named_modules():
                if isinstance(module, MoeBlk):
                    self.moe_modules[name] = module

            self.lb_ctrl = LoadBalanceBiasController(self.moe_modules, self.cfg.lb_gamma)


    @property
    def cfg(self):
        """Return the model config."""
        return self._cfg
    
    @property
    def is_moe(self):
        """Check if the underlying model has MoE layers (num_experts > 1)."""
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
                # num_items_in_batch == T distributed among K
                # since aggregration is per k slot over entire model
                # model-wide per k attend T * L layers
                denom = num_items_in_batch * len(outputs.router_logits)
                frac_per_e = (count[:, 0, :].sum(dim=0)/denom).tolist()

                log_dict = {
                    f"moe/top1/load/e{i:03d}": round(frac, 3) for i, frac in enumerate(frac_per_e)
                }

                if self.cfg.lb_coeff > 0:
                    # log load balancing loss
                    lb_loss = round(outputs.aux_loss.detach().item(), 6)
                else:
                    lb_loss = math.nan
                log_dict.update({f"train/lb_loss": lb_loss})

                if self.cfg.lb_gamma > 0:
                    # layer-wise load per expert, all k-slot combined
                    load_per_expert = count.sum(-2)/(num_items_in_batch*self.K)
                    lb_bias_global_sum = self.lb_ctrl(load_per_expert)
                else:
                    lb_bias_global_sum = math.nan
                log_dict.update({f"moe/lb_bias_global_sum": lb_bias_global_sum})

                drop_ratio = math.nan
                if self.CF > 0:
                    global_n_drop = sum([m.n_drop for m in self.moe_modules.values()])
                    # global_n_experts = self.E * self.L
                    # n_drop_per_e = global_n_drop / global_n_experts

                    global_n_routes = int(num_items_in_batch * self.L * self.K)
                    drop_ratio = round(global_n_drop / global_n_routes, 4)
                log_dict.update({f"train/token_drop_ratio": drop_ratio})

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
    
@dataclass
class LoadBalanceBiasController:
    moe_modules: dict
    gamma: float

    def __post_init__(self):
        if self.gamma == 0.0:
            logger.warning("LoadBalanceBiasController initialized with gamma=0.0, no effect will be applied.")

    def __call__(self, load_per_expert):
        return self.adjust_router_lb_bias(load_per_expert)
    
    @torch.no_grad()
    def adjust_router_lb_bias(self, load_per_expert):
        assert load_per_expert.shape[0] == len(self.moe_modules), "load_per_expert shape mismatch with num_layers"
        
        L, E = load_per_expert.shape
        balance_ratio = 1/E
        error = load_per_expert/balance_ratio - 1.0 # read as delta to 100% of ideal load

        global_bias_sum = 0.0 
        for l, module in enumerate(self.moe_modules.values()):
            module.e_score_bias.data -= self.gamma * error[l]
            global_bias_sum += module.e_score_bias.data.sum().item()
        
        return global_bias_sum # this is just for debug purpose, no other practical use