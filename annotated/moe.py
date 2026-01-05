import torch
import torch.nn as nn

class GluMLP(nn.Module):
    def __init__(self, D, DFF, bias=False):
        super(GluMLP, self).__init__()
        self.gate = nn.Linear(D, DFF, bias=bias)
        self.up = nn.Linear(D, DFF,  bias=bias)
        self.act = nn.ReLU()
        self.down = nn.Linear(DFF, D, bias=bias)

    def forward(self, x):
        return self.down(self.act(self.gate(x)) * self.up(x))
    
class MoE(nn.Module):
    def __init__(self, D, E, K, DFF, bias=False):
        super(MoE, self).__init__()
        self.D = D       # token dimension (model dimension)       
        self.E = E       # number of experts
        self.K = K       # number of activated experts per token
        self.DFF = DFF   # expert dimension (feed-forward dimension)
        
        # router (gating network)
        self.router = nn.Linear(D, E, bias=bias)
        
        # experts
        self.experts = nn.ModuleList([GluMLP(D, DFF, bias=bias) for _ in range(E)])

    def extra_repr(self):
        return f"[K:E] = {self.K}:{self.E} , K activated out of E experts per token"
    
    def forward(self, x):
        # k = activated expert id
        # e = expert id
        # t = token id

        # batch size, sequence length, token dimension
        B, L, D = x.size()
        T = B * L
        K = self.K
        E = self.E

        # router: predict logits over all experts
        router_logits = self.router(x)    # (B, L, E)

        # NOTE: just for testing to compare against olmoe; to be removed
        router_logits = router_logits.softmax(dim=-1) # per olmoe, just for testing

        # Top-K expert selection per token
        k_logits, k_ids = router_logits.topk(K, dim=-1)   
        # element value of k_logits is the logits score
        # element value of k_ids is in [0, E-1], the eid

        # Reduce B, L into T to form single token dimension
        k_ids = k_ids.view(T, K) # where we compute T = B*L above
        # we keep k_logits as is and we will only need it at the end.

        # bookkeep (reverse mapping)
        # k_ids (T, K): for each token, row value of destination expert ids
        # t_ids (T, K): for each token, each element value capture source token id.
        #               It means reverse mapping of individual expert output to source token id.
        t_ids = torch.arange(T, device=x.device).unsqueeze(1).expand(T, K)  # (T, K)
        # arange gives each token an id from 0 to T-1
        # unsqueeze and expand make every destination expert id tagged with source token id
        
        # Now we want to group tokens per expert id
        # we can flatten k_ids to 1D, then sort by expert id
        # use argsort to get permutation indices with actually touching data
        # count how many tokens per expert id using bincount
        flat_k_ids = k_ids.view(-1)    # from (T, K) to (T*K,)
        
        perm = torch.argsort(flat_k_ids)
        # reversing the permutation, O(N)
        # alternatively, inv_perm = torch.argsort(perm)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.numel(), device=perm.device)

        # NOTE: in case you are worried like I am,
        # torch.bincount always lays out counts in ascending index order.
        ntok_per_e = torch.bincount(flat_k_ids, minlength=E)

        # Time to dispatch tokens for experts forward pass
        # we loop over each expert, look up / slice tokens for that expert
        end_offsets = torch.cumsum(ntok_per_e, dim=0)

        expert_outputs = torch.zeros(T*K, D, device=x.device, dtype=x.dtype)        
        for e in range(E):
            start = 0 if e == 0 else end_offsets[e - 1]
            offset = end_offsets[e]

            tids_of_e = perm[start:offset]  # indices of tokens for expert e
            # lookup idx to original input tensor
            tmp = tids_of_e//K
            bids = tmp // L
            lids = tmp % L
            # x_of_e = x[bids,lids,:]
            expert_outputs[start:offset] = self.experts[e](x[bids, lids, :])

        # putting it back to right order
        expert_outputs = expert_outputs[inv_perm].view(B, L, K, D)

        # compute softmax over k_logits and make dim friendly for broadcasting
        # k_weights = k_logits.softmax(dim=-1).unsqueeze(-1)   # (B, L, K) -> (B, L, K, 1)
        # NOTE: just for testing against olmoe version, k_logits was softmaxed above, need renormalize per k.
        # in standard MoE, the k_logits here are raw logits from router. 
        # we would just softmax-normalized per k directly.
        k_weights = k_logits / k_logits.sum(dim=-1, keepdim=True)
        k_weights = k_weights.unsqueeze(-1)   # (B, L, K) -> (B, L, K, 1)

        # weighted sum over K experts
        moe_outputs = (expert_outputs * k_weights).sum(dim=-2)  # reduce K dim

        return moe_outputs

if __name__ == "__main__":

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

    print("end.")