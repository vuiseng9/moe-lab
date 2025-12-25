import torch
from transformers import Trainer

class OlmoeTrainer(Trainer):
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


class DSv3Trainer(OlmoeTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.router_outputs = []  # stores (layer_idx, topk_indices) tuples
        self._hooks_registered = False
        self._hook_handles = []

    def _register_router_hooks(self):
        """Register forward hooks on all DeepseekV3TopkRouter modules to capture routing decisions."""
        if self._hooks_registered:
            return

        # Get the underlying model (handle potential wrapping like DDP)
        model = self.model
        if hasattr(model, 'module'):
            model = model.module

        layer_idx = 0
        for name, module in model.named_modules():
            if module.__class__.__name__ == 'DeepseekV3TopkRouter':
                # hook to capture topk_indices per layer
                def make_hook(idx):
                    def router_hook(mod, inp, output):
                        topk_indices, _ = output
                        self.router_outputs.append((idx, topk_indices.detach().cpu()))
                    return router_hook
                
                handle = module.register_forward_hook(make_hook(layer_idx))
                self._hook_handles.append(handle)
                layer_idx += 1

        self._hooks_registered = True

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
        total_counts = torch.zeros(top_k, n_experts, dtype=torch.float32)

        for layer_idx, topk_indices in self.router_outputs:
            # topk_indices shape: [num_tokens, top_k]
            for k in range(top_k):
                indices_k = topk_indices[:, k].view(-1)
                counts = torch.bincount(indices_k, minlength=n_experts).float()
                total_counts[k] += counts

        # Normalize to proportions per top-k slot
        row_sums = total_counts.sum(dim=1, keepdim=True)
        tokens_per_expert = total_counts / (row_sums + 1e-10)

        return tokens_per_expert

    def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):
        # Register hooks on first call (lazy initialization)
        if not self._hooks_registered:
            self._register_router_hooks()

        # Clear previous router outputs
        self.router_outputs.clear()

        # Call Trainer's compute_loss (skip OlmoeTrainer)
        loss, outputs = super(OlmoeTrainer, self).compute_loss(model, inputs, True, num_items_in_batch)

        if 'wandb' in self.args.report_to:
            if self.wb_handler is None:
                self.wb_handler = self.find_wandb_handler()

            log_dict = {}

            # Compute and log tokens per expert proportions
            tpe = self._compute_tokens_per_expert()
            if tpe is not None:
                K, E = tpe.shape
                log_dict.update({f"moe/top{k+1}/e{e}": float(tpe[k, e]) for k in range(K) for e in range(E)})

            # Log bias per expert using existing buffers # NOTE: disable for now because it is always zero, not dynamic calibration 
            # bias_list = []
            # for n, b in model.named_buffers():
            #     if 'e_score_correction_bias' in n:
            #         bias_list.append(b)
            # if bias_list:
            #     bias_per_expert = torch.stack(bias_list, dim=0).mean(dim=0).tolist()
            #     log_dict.update({f"moe/e{e}_bias": float(bias_per_expert[e]) for e in range(len(bias_per_expert))})

            if log_dict and self.wb_handler is not None:
                self.wb_handler.log(log_dict)

        return (loss, outputs) if return_outputs else loss

    def __del__(self):
        """Clean up hooks when trainer is deleted."""
        for handle in self._hook_handles:
            handle.remove()