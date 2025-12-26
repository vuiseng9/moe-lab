import torch
from transformers import Trainer
from collections import OrderedDict

class OlmoeTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wb_handler = None
    
    def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):

        loss, outputs = super().compute_loss(model, inputs, True, num_items_in_batch)
        if 'wandb' in self.args.report_to:
            if self.wb_handler is None:
                self.wb_handler = self.find_wandb_handler()

            tpe = outputs.tokens_per_expert.detach().to(torch.float32)
            K, E = tpe.shape
            log_dict = {f"moe/top{k+1}/e{e}": float(tpe[k, e]) for k in range(K) for e in range(E)}

            if getattr(self.model.config, 'enable_lbl', False):
                aux_loss = outputs.aux_loss.detach().to(torch.float32).item()
                log_dict.update({f"train/moe_aux_loss": aux_loss})

            self.wb_handler.log(log_dict)

        return (loss, outputs) if return_outputs else loss

    def find_wandb_handler(self):
        # we don't call self.log to avoid stdout blowing up with too much details
        for handler in self.callback_handler.callbacks:
            if handler.__class__.__name__ == "WandbCallback":
                return handler._wandb
        return None


class DSv3Trainer(OlmoeTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.router_outputs = []  # stores (layer_name, topk_indices) tuples
        self.router_modules = OrderedDict()  # stores (layer_name, bias_tensor) tuples
        self._hooks_registered = False
        self._hook_handles = []

    def _register_router_hooks_and_biases(self):
        """Register forward hooks on all DeepseekV3TopkRouter modules to capture routing decisions."""
        if self._hooks_registered:
            return

        # Get the underlying model (handle potential wrapping like DDP)
        model = self.model
        if hasattr(model, 'module'):
            model = model.module

        self.lb_gamma = self.model.config.load_balance_gamma

        for name, module in model.named_modules():
            if module.__class__.__name__ == 'DeepseekV3TopkRouter':
                # hook to capture topk_indices per layer
                def make_hook(idx):
                    def router_hook(mod, inp, output):
                        topk_indices, _ = output
                        self.router_outputs.append((idx, topk_indices.detach()))
                    return router_hook
                handle = module.register_forward_hook(make_hook(name))
                self._hook_handles.append(handle)
                # capture bias buffer (assumed to exist)
                self.router_modules[name] = module
        
        self._hooks_registered = True


    def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):
        # Register hooks on first call (lazy initialization)
        if not self._hooks_registered:
            self._register_router_hooks_and_biases()

        self.router_outputs.clear()

        # Call Trainer's compute_loss (skip OlmoeTrainer)
        loss, outputs = super(OlmoeTrainer, self).compute_loss(model, inputs, True, num_items_in_batch)

        routed_frac_per_k, load_per_expert = self._compute_tokens_per_expert()
        
        if 'wandb' in self.args.report_to:
            if self.wb_handler is None:
                self.wb_handler = self.find_wandb_handler()

            log_dict = {}

            # Compute and log tokens per expert proportions
            if routed_frac_per_k is not None:
                K, E = routed_frac_per_k.shape
                for k in range(K):
                    for e in range(E):
                        log_dict[f"moe/top{k+1}/e{e}"] = float(routed_frac_per_k[k, e])

            # log only the first router's bias to avoid clutter
            bias = next(iter(self.router_modules.values())).e_score_correction_bias.tolist()
            log_dict.update({f"moe/r0_e{e}_bias": float(bias[e]) for e in range(len(bias))})

            if log_dict and self.wb_handler is not None:
                self.wb_handler.log(log_dict)

        self.adjust_router_biases(load_per_expert)
        return (loss, outputs) if return_outputs else loss


    def adjust_router_biases(self, load_per_expert):
        for r, load in load_per_expert.items():
            error = load - 1.0
            self.router_modules[r].e_score_correction_bias -=  self.lb_gamma * error


    def _compute_tokens_per_expert(self):
        """Aggregate routing statistics from captured hook outputs.
        Returns:
            tokens_per_expert: Tensor of shape [K, E] where K is top_k and E is num_experts,
                              containing the proportion of tokens routed to each expert.
        """
        if not self.router_outputs:
            return None

        # Get config from model
        model = self.model.module if hasattr(self.model, 'module') else self.model
        n_experts = model.config.n_routed_experts
        top_k = model.config.num_experts_per_tok

        # Aggregate counts across all layers
        # Shape: [K, E] - for each top-k slot, count how many tokens went to each expert
        total_counts = torch.zeros(top_k, n_experts, dtype=torch.float32).to(model.device)

        load_per_expert = OrderedDict()
        for router_name, topk_indices in self.router_outputs:
            tpe = torch.bincount(topk_indices.flatten(), minlength=n_experts).float()
            load_per_expert[router_name] = tpe / tpe.mean() # normalized load per expert

            # topk_indices shape: [num_tokens, top_k]
            for k in range(top_k):
                indices_k = topk_indices[:, k].view(-1)
                counts = torch.bincount(indices_k, minlength=n_experts).float()
                total_counts[k] += counts

        # Normalize to proportions per top-k slot
        row_sums = total_counts.sum(dim=1, keepdim=True)
        routed_frac_per_k = total_counts / row_sums # routed fraction per top-k slot per expert

        return routed_frac_per_k, load_per_expert


    def __del__(self):
        """Clean up hooks when trainer is deleted."""
        for handle in self._hook_handles:
            handle.remove()