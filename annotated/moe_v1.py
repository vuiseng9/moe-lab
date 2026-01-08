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
        assert x.is_floating_point(), "MoE forward input must be floating point type."
        eps = torch.finfo(x.dtype).eps

        # prefix notation: 
        # k = activated expert id 
        # e = expert id 
        # t = token id 
        # batch size, sequence length, token dimension 
        B, L, D = x.size() 
        T = B * L 
        K = self.K 
        E = self.E 
        
        _x = x.view(T, D) # all tokens in a single list (dimension) 
        
        # router: predict logits over all experts 
        router_logits = self.router(_x) # (B, L, E) 
        
        # NOTE: just for testing to compare against olmoe; to be removed 
        # # typically, the normalization is post top-k selection 
        # per olmoe, just for testing against olmoe version
        router_scores = router_logits.softmax(dim=-1) 
        
        # # Top-K expert selection 
        k_scores, k_ids = router_scores.topk(K, dim=-1) 
        # element value of k_logits is the logits score 
        # element value of k_ids is in [0, E-1], the eid 

        # renormalize per K elemnts. 
        k_weights = k_scores / ( k_scores.sum(dim=-1, keepdim=True) + eps ) 

        # in some MoE, the k_logits are only softmax-normalized here.
        # unsqueeze for broadcast-friendly. 
        # k_weights = k_logits.softmax(dim=-1).unsqueeze(-1) # (T, K) -> (T, K, 1) 

        # initialize output buffer, per token, output dimension
        # accumulate per k slot (save buffer space by accumulating directly)
        moe_outputs = torch.zeros(T, D, device=x.device, dtype=x.dtype) 
        
        # loop over experts, 
        # get assigned tokens, get the corresponding weights
        # forward pass, 
        # accumulate to right token in expert_outputs
        for eid in range(E): 
            # tok_ids: indices of tokens to be attended by expert eid 
            # # which_k: for each token, which k slot it is from 0 to K 
            tok_ids, which_k = torch.where(k_ids == eid) 
            
            # LHS: write to the right token id, k slot in expert_outputs 
            # RHS: (1) get the expert module, (2) gather the tokens, (3) forward pass , 
            #      (4) multiply weights (5) accumulate
            expert = self.experts[eid]
            tokens = _x[tok_ids]  
            weights = k_weights[tok_ids, which_k].unsqueeze(-1)
            
            moe_outputs[tok_ids, :] += expert(tokens) * weights 

        return moe_outputs.reshape(B, L, D)


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