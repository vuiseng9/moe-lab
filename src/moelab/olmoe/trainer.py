from moelab.trainer import MoelabTrainer
import torch


class OlmoeTrainer(MoelabTrainer):
    def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):
        # Call HF Trainer's compute_loss
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        if "wandb" in self.args.report_to and self.wandb is not None:
            routed_frac_per_k = self.aggregate_routed_frac_per_expert(outputs.router_logits)

            log_dict = {
                f"moe/top{k + 1}/e{e}": float(routed_frac_per_k[k, e])
                for k in range(routed_frac_per_k.shape[0])
                for e in range(routed_frac_per_k.shape[1])
            }

            if model.config.capacity_factor > 0:
                # n_drop is total dropped tokens of a MoE layer, ie. dropped tokens of all experts in that layer
                global_n_drop = 0 # per model previous forward
                for blk in model.model.layers:
                    global_n_drop += blk.mlp.n_drop
                
                global_n_experts = model.config.num_experts * model.config.num_hidden_layers
                n_drop_per_e = global_n_drop / global_n_experts

                global_n_routed = num_items_in_batch * model.config.num_hidden_layers * model.config.num_experts_per_tok

                drop_ratio = global_n_drop / global_n_routed

                log_dict.update({
                    f"train/avg_drop_per_e": n_drop_per_e,
                    f"train/drop_ratio": drop_ratio})

            if getattr(self.model.config, "enable_lbloss", False):
                lbloss = outputs.aux_loss.detach().item()
                log_dict.update({f"train/moe_lbloss": lbloss})

            self.wandb.log(log_dict)

        return (loss, outputs) if return_outputs else loss

    @torch.no_grad()
    def aggregate_routed_frac_per_expert(self, router_logits):
        # model wide aggregration per k per e

        topk = self.model.config.num_experts_per_tok
        n_experts = self.model.config.num_experts

        _, topk_indices = torch.cat(router_logits, dim=0).topk(k=topk, dim=-1)

        total_counts = torch.zeros(topk, n_experts, dtype=torch.float32).to(self.model.device)

        for k in range(topk):
            indices_k = topk_indices[:, k].view(-1)
            total_counts[k] = torch.bincount(indices_k, minlength=n_experts).float()

        # Normalize to proportions per top-k slot
        row_sums = total_counts.sum(dim=1, keepdim=True)
        routed_frac_per_k = total_counts / row_sums  # routed fraction per top-k slot per expert

        return routed_frac_per_k
