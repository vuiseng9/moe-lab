from collections import OrderedDict
import torch

from moelab.trainer import MoelabTrainer


class DSv3Trainer(MoelabTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.router_outputs = []  # (layer_name, topk_indices) tuples
        self.router_modules = OrderedDict()  # (layer_name,  bias_tensor) tuples
        self._hooks_registered = False
        self._hook_handles = []
        self.lb_gamma = None

    def _register_router_hooks_and_make_module_map(self):
        """
        Register forward hooks on all DeepseekV3TopkRouter modules
        to capture routing decisions. and store mapping to bias tensors.
        """
        if self._hooks_registered:
            return

        # Get the underlying model (handle potential wrapping like DDP)
        model = self.model.module if hasattr(self.model, "module") else self.model
        self.lb_gamma = getattr(model.config, "load_balance_gamma", None)

        for name, module in model.named_modules():
            if module.__class__.__name__ == "DeepseekV3TopkRouter":

                def make_hook(layer_name):
                    def router_hook(mod, inp, output):
                        topk_indices, _ = output
                        self.router_outputs.append((layer_name, topk_indices.detach()))

                    return router_hook

                handle = module.register_forward_hook(make_hook(name))
                self._hook_handles.append(handle)

                # name to module mapping for bias adjustment later
                self.router_modules[name] = module

        self._hooks_registered = True

    def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):
        # Register hooks on first call (lazy initialization)
        if not self._hooks_registered:
            self._register_router_hooks_and_make_module_map()

        self.router_outputs.clear()

        # Call HF Trainer's compute_loss
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        routed_frac_per_k, load_per_expert = self._compute_tokens_per_expert()

        if "wandb" in self.args.report_to:
            if self.wb_handler is None:
                self.wb_handler = self.get_wandb_handler()

            if self.wb_handler is not None:
                log_dict = {}

                # Compute and log tokens per expert proportions
                if routed_frac_per_k is not None:
                    K, E = routed_frac_per_k.shape
                    for k in range(K):
                        for e in range(E):
                            log_dict[f"moe/top{k + 1}/e{e}"] = float(routed_frac_per_k[k, e])

                # log only the first router's bias to avoid clutter
                if self.router_modules:
                    bias = next(iter(self.router_modules.values())).e_score_correction_bias
                    bias_list = bias.detach().float().tolist()
                    log_dict.update(
                        {f"moe/r0_e{e}_bias": float(bias_list[e]) for e in range(len(bias_list))}
                    )

                self.wb_handler.log(log_dict)

        if load_per_expert is not None:
            self.adjust_router_biases(load_per_expert)

        return (loss, outputs) if return_outputs else loss

    @torch.no_grad()
    def adjust_router_biases(self, load_per_expert):
        if self.lb_gamma is None:
            return

        for r, load in load_per_expert.items():
            error = load - 1.0
            self.router_modules[r].e_score_correction_bias.data -= self.lb_gamma * error

    @torch.no_grad()
    def _compute_tokens_per_expert(self):
        """
        Aggregate routing statistics from captured hook outputs.
        Returns:
            tokens_per_expert: Tensor of shape [K, E] where K is top_k and E is num_experts,
                              containing the proportion of tokens routed to each expert.
        """
        if not self.router_outputs:
            return None, None

        # Get config from model
        model = self.model.module if hasattr(self.model, "module") else self.model
        n_experts = model.config.n_routed_experts
        top_k = model.config.num_experts_per_tok

        # Aggregate counts across all layers
        # Shape: [K, E] - for each top-k slot, count how many tokens went to each expert
        total_counts = torch.zeros(top_k, n_experts, dtype=torch.float32).to(model.device)

        load_per_expert = OrderedDict()
        for router_name, topk_indices in self.router_outputs:
            tpe = torch.bincount(topk_indices.flatten(), minlength=n_experts).float()
            load_per_expert[router_name] = tpe / tpe.mean()  # normalized load per expert

            # topk_indices shape: [num_tokens, top_k]
            for k in range(top_k):
                indices_k = topk_indices[:, k].view(-1)
                counts = torch.bincount(indices_k, minlength=n_experts).float()
                total_counts[k] += counts

        # Normalize to proportions per top-k slot
        row_sums = total_counts.sum(dim=1, keepdim=True)
        routed_frac_per_k = total_counts / row_sums  # routed fraction per top-k slot per expert

        return routed_frac_per_k, load_per_expert

    def __del__(self):
        """Clean up hooks when trainer is deleted."""
        for handle in self._hook_handles:
            handle.remove()
