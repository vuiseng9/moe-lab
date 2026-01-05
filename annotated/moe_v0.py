import torch
import torch.nn as nn


class MoE(nn.Module):
    def __init__(self, D, E, K, DFF):
        super(MoE, self).__init__()
        self.D = D          
        self.E = E
        self.DFF = DFF
        self.K = K
        
        # router (gating network)
        self.router = nn.Linear(D, E)
        
        # experts
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(D, DFF),
            nn.ReLU(),
            nn.Linear(DFF, D)
        ) for _ in range(E)])

    def extra_repr(self):
        return f"[K:E] = {self.K}:{self.E} , K activated out of E experts per token"
    
    def forward(self, x):
        # k = activated experts per token (top-K)
        # e = total experts

        B, L, D = x.size()
        T = B * L
        K = self.K
        E = self.E

        # flatten batch and sequence length into single token dimension: (T, D)
        token_list = x.view(T, D)

        # router: predict logits over all experts, then take top-K
        router_logits = self.router(token_list)           # (T, E)
        k_logits, k_ids = router_logits.topk(K, dim=-1)   # (T, K), (T, K)
        # element value of k_logits is the logits score
        # element value of k_ids is in [0, E-1], the eid

        # softmax over top-K logits = weightage of selected experts
        k_weights = k_logits.softmax(dim=-1)

        # Because K activated experts per token,
        # all tokens (B*L=T) will be dispatched to K experts
        # a naive implementation is to repeat tokens K times, from (T, D)
        #   to (T, K, D), then flatten to (T*K, D)
        # instead replication, we can use index to lookup the D features
        # we tag each k to the token (row) id
        # Build edge table for (T, K): each token appears K times, one per activated expert.
        tok_ids = torch.arange(T, device=x.device).unsqueeze(1).expand(T, K)  # (T, K)

        flat_tok_ids   = tok_ids.view(-1)      # (T*K,)
        flat_k_ids     = k_ids.view(-1)        # (T*K,)
        flat_k_weights = k_weights.view(-1)    # (T*K,)

        # Sort edges by expert id so each expert's edges are contiguous.
        # arg indices ordering by value (0 - E-1 expert ids)
        perm = torch.argsort(flat_k_ids) 
        
        # sort/reorder by perm
        sorted_k_ids     = flat_k_ids[perm]       # (T*K,)
        sorted_tok_ids   = flat_tok_ids[perm]     # (T*K,)
        sorted_k_weights = flat_k_weights[perm]   # (T*K,)

        # Count how many edges go to each expert (segment lengths).
        count_per_e = torch.bincount(sorted_k_ids, minlength=E)  # (E,)

        # Run each expert on its slice.
        expert_out_sorted = torch.zeros(T*K, D, device=x.device, dtype=x.dtype)

        start = 0
        for eid, ntok in enumerate(count_per_e):
            if ntok == 0:
                continue
            end = start + ntok
            x_per_e = token_list[sorted_tok_ids[start:end]]          # (ntok, D)
            expert_out_sorted[start:end] = self.experts[eid](x_per_e)   # (ntok, D)
            
            start = end
        
        # Apply routing weights (broadcast over D)
        expert_out_sorted = expert_out_sorted * sorted_k_weights[:, None]          # (T*K, D)
    
        # Unsort back to original edge order (token-major, then k)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(T * K, device=x.device)
        expert_out_edges = expert_out_sorted[inv_perm]                      # (T*K, D)

        # Combine K expert contributions back to tokens
        token_out = expert_out_edges.view(T, K, D).sum(dim=1)            # (T, D)

        return token_out.view(B, L, D)


# --------------------------------------------------------------------
# input (activation)
B = 64  # batch size
L = 256 # sequence length, # of tokens per sequence
D = 128 # token dimension (model dimension)

# router (a.k.a gating network)
E = 32 # number of experts
K = 8  # number of experts to select (activated), also the top-k value

# expert 
DFF = D*4 # expert dimension

moe_layer = MoE(D, E, K, DFF)

print(moe_layer)
# a batch of input sequences
x = torch.randn(B, L, D)

with torch.no_grad():
    output = moe_layer(x)  # shape: (B, L, D)

# with torch.no_grad():


print("")