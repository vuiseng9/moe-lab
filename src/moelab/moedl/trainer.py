import math
import logging
import torch
from collections import deque
from dataclasses import dataclass
from moelab.moedl import MoeBlk
from moelab.trainer import MoelabTrainer
from moelab.utils import TensorMeter, pngs2gif
from transformers import TrainerCallback
from concurrent.futures import ThreadPoolExecutor
import threading

import os
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class MoedlTrainer(MoelabTrainer):
    """ MoedlTrainer is a specialized trainer for Moedl model type. """

    def __init__(self, *args, heatmap_on: bool = True, heatmap_freq: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.heatmap_on = heatmap_on
        self.heatmap_freq = heatmap_freq

        model = self.model.module if hasattr(self.model, "module") else self.model
        self._cfg = model.config
        
        if not getattr(self._cfg, "num_experts", None):
            raise ValueError("Moedl config must have 'num_experts' attribute to determine MoE status.")
        self._is_moe = self.cfg.num_experts > 1

        self.L  = self.cfg.num_hidden_layers
        self.E  = self.cfg.num_experts
        self.K  = self.cfg.num_active_experts
        self.ES = self.cfg.num_shared_experts
        self.CF = self.cfg.capacity_factor

        self.moe_modules = None
        self.lb_ctrl = None
        self.routing_stat = None
        self.expert_load = None
        self.last_lb_loss = math.nan
        self.last_drop_ratio = math.nan
        self.last_drop_count = math.nan
        self.last_lb_bias_dbg = math.nan
        self.moe_log_metrics = deque(maxlen=1)

        if self.is_moe:
            self.moe_modules = {}
            self.routing_stat = TensorMeter()  # shape (L, K, E), token count per expert per k slot per layer
            self.expert_load  = TensorMeter()  # shape (L, K, E), above denominated by total tokens T
            for name, module in model.named_modules():
                if isinstance(module, MoeBlk):
                    self.moe_modules[name] = module
            self.lb_ctrl = LoadBalanceBiasController(self.moe_modules, self.cfg.lb_gamma)
            self.add_callback(MoedlPerStepCallback(self))

    @property
    def cfg(self):
        """Return the model config."""
        return self._cfg
    
    @property
    def is_moe(self):
        """Check if the underlying model has MoE layers (num_experts > 1)."""
        return self._is_moe

    def log(self, logs: dict, start_time: float = None):
        # loss key presence indicates a training step log
        if 'loss' in logs and len(self.moe_log_metrics) > 0:
            logs.update(self.moe_log_metrics.pop())
        super().log(logs, start_time=start_time)

    def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):
        # Call HF Trainer's compute_loss
        # we force return_outputs=True to always return outputs
        # we need outputs for post-processing
        # final return is based on the input return_outputs flag
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        if self.is_moe:
            count, frac = self.get_expert_stats(outputs.router_logits)
            self.routing_stat.update(count.float())
            self.expert_load.update(frac) 
            # frac is expert load per k slot per layer
            # reduce over k (dim=-2) by mean (each k slot attends same total number of tokens)

            if self.model.training and self.cfg.lb_coeff > 0:
                self.last_lb_loss = round(outputs.aux_loss.detach().item(), 6)

            if self.CF > 0:
                global_n_drop = sum([m.n_drop for m in self.moe_modules.values()])
                global_n_routes = int(num_items_in_batch * self.L * self.K)
                self.last_drop_ratio = round(global_n_drop / global_n_routes, 4)
                self.last_drop_count = global_n_drop
                # If we need per expert instance drop stats, we can compute here:
                # global_n_experts = self.E * self.L
                # n_drop_per_e = global_n_drop / global_n_experts
            
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

        # count (L, K, E) is per layer per k slot per expert
        count = torch.zeros((L, K, E), device=k_ids.device, dtype=torch.int64)
        # k_ids (L, T, K) -> (L, K, T)
        k_ids = k_ids.permute(0, 2, 1)
        # scatter add 1 for each ids
        count.scatter_add_(dim=2, index=k_ids, src=torch.ones_like(k_ids, dtype=count.dtype))
        assert (count.sum(dim=-1) == T).all().item(), "k slot(s) sum mismatch with T"

        # For each k slot, routed tokens to all Es sum up to be T tokens
        # divide count by T effectively gives proportion of tokens per expert for every k slot
        frac = count.float() / T
        return count, frac


@dataclass
class LoadBalanceBiasController:
    """
    Proportional controller for biasing router to routing balance.
    Per DeepSeekV3 paper.
    """
    moe_modules: dict
    gamma: float

    def __post_init__(self):
        if self.gamma == 0.0:
            logger.warning("LoadBalanceBiasController initialized with gamma=0.0, no effect will be applied.")

    def __call__(self, layerwise_expert_load):
        return self.adjust_router_lb_bias(layerwise_expert_load)
    
    @torch.no_grad()
    def adjust_router_lb_bias(self, layerwise_expert_load):
        assert layerwise_expert_load.shape[0] == len(self.moe_modules), "layerwise_expert_load shape mismatch with num_layers"
        
        L, E = layerwise_expert_load.shape
        balance_ratio = 1/E
        error = layerwise_expert_load/balance_ratio - 1.0 # read as delta to 100% of ideal load

        dbg_bias = 0.0 
        for l, module in enumerate(self.moe_modules.values()):
            module.e_score_bias.data -= self.gamma * error[l]
            dbg_bias += module.e_score_bias.data.abs().sum().item() 
            # Caution - abs() to avoid negative bias cancelling out positive bias
            # therefore, just use this for debug purpose
        return dbg_bias


@dataclass
class MoedlPerStepCallback(TrainerCallback):
    # designed not to own any state buffers here, only "act"
    trainer: MoedlTrainer  # Reference to trainer to access attributes at runtime
        
    def __post_init__(self):
        # just for code readability
        self.moe_modules = self.trainer.moe_modules
        self.lb_ctrl = self.trainer.lb_ctrl
        self.routing_stat = self.trainer.routing_stat
        self.expert_load  = self.trainer.expert_load
        self.moe_log_metrics = self.trainer.moe_log_metrics
        # call order
        # callback.on_step_end() -> trainer.log() -> callback.on_log()
        # we avoid additional on_log callback here by
        # directly updating logs object in trainer.log()
        # we use a deque to cache the moe metrics during on_step_end.

        # Heatmap of expert load
        # designed to be non-blocking to minimize overhead on training step
        self.heatmapper = None
        self.heatmap_freq = None
        self.semaphore = threading.Semaphore(1)
        
        if self.trainer.heatmap_on:
            # one thread should be sufficient
            # only one busy at a time, skip generation if previous not done
            self.heatmapper = ThreadPoolExecutor(max_workers=1)
            self.heatmap_freq = self.trainer.heatmap_freq
            self.semaphore = threading.Semaphore(1) 
            # semaphore as a means of /skipping
            # heatmap generation if previous not done

    def on_train_begin(self, args, state, control, **kwargs):
        if self.heatmapper:
            self.heatmap_outdir = f"{args.output_dir}/heatmap"
            os.makedirs(self.heatmap_outdir, exist_ok=True)

    def on_step_end(self, args, state, control, **kwargs):
        # apply load balance bias adjustment (only if gamma > 0)
        if self.lb_ctrl.gamma > 0:
            self.trainer.last_lb_bias_dbg = self.lb_ctrl(self.expert_load.avg.mean(dim=-2))

        if self.trainer.wandb:
            # model-wide expert load
            # this is good high-level overview,
            # but may get smoothed out
            # check heatmap for detailed per layer per k slot view
            d = {}
            for i, frac in enumerate(self.expert_load.avg.mean(dim=-2).mean(dim=0).tolist()):
                d[f"moe/load/e{i:03d}"] = round(frac, 3)
            d[f"lb_loss"] = self.trainer.last_lb_loss
            d[f"token_drop_ratio"] = self.trainer.last_drop_ratio
            d[f"token_drop_count"] = self.trainer.last_drop_count
            d[f"lb_bias_dbg"] = self.trainer.last_lb_bias_dbg
            # Trainer.log will automatically prepend train/
            self.moe_log_metrics.append(d)
        
            # note that global_step has already been incremented 
            # prior to on step end
            gstep = state.global_step - 1
            if self.heatmapper and gstep % self.heatmap_freq == 0:
                if self.semaphore.acquire(blocking=False):
                    self.heatmapper.submit(
                        self._generate_expert_load_heatmap,
                        self.expert_load.avg.detach().cpu().numpy(),
                        gstep,
                        f"{self.heatmap_outdir}/gstep_{gstep:06}_expert_load.png"
                    )

        self.reset_meters()

    def reset_meters(self):
        self.routing_stat.reset()
        self.expert_load.reset()
    
    def on_train_end(self, args, state, control, **kwargs):
        if self.heatmapper:
            self.heatmapper.shutdown(wait=True)
            # limit to 15s, e.g. 1500 frames at 10 fps. Auto downsample inside pngs2gif.
            pngs2gif(folder=self.heatmap_outdir, 
                     output=f"{args.output_dir}/expert_load_over_train_steps.gif", 
                     downsample=None, max_len=15)

    def _generate_expert_load_heatmap(self, np_arr, step, file_path):
        try:
            # check numpy array and shape
            assert isinstance(np_arr, np.ndarray), "Input must be a numpy array"
            assert np_arr.ndim == 3, "Input tensor must be 3D (L, K, E)"
            
            L, K, E = np_arr.shape
            np_arr = np_arr.reshape(L*K, E)
            
            balance = 100 / E 

            abs_imbalance = np.abs(np_arr*100 - balance)

            fig, ax = plt.subplots(figsize=(8, 6))
            
            im = ax.imshow(abs_imbalance, cmap='Reds', vmin=0.0, vmax=50.0)
            
            for i in range(np_arr.shape[0]):
                for j in range(np_arr.shape[1]):
                    ax.text(j, i, f"{abs_imbalance[i, j]:.1f}",
                            ha="center", va="center", color="black", fontsize=8)
            
            y_labels = [f"l{l},k{k}" for l in range(L) for k in range(K)]
            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels(y_labels)
            ax.set_ylabel('Layer, K-slot')
            ax.set_xlabel('Expert Id')

            ax.set_title(f'Step {step}\n\nAbs. Delta from Uniform Expert Load ({balance:.1f}%)')
            fig.colorbar(im, ax=ax)
            plt.savefig(file_path)
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error generating expert load heatmap at step {step}: {e}")
        finally:
            # always release even error
            self.semaphore.release()