

from dataclasses import dataclass
from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
from moe_v1 import MoE
import torch

D = 128 # token dimension (model dimension)

# router (a.k.a gating network)
E = 32 # number of experts
K = 8  # number of experts to select (activated), also the top-k value

# expert 
DFF = D*4 # expert dimension

CF = -1
@dataclass
class MoeConfig:
    num_experts: int = E
    num_experts_per_tok: int = K
    norm_topk_prob: bool = True
    hidden_size: int = D
    intermediate_size: int = DFF
    hidden_act: str = "relu"
    capacity_factor: float = CF


cfg = MoeConfig()

olmoe = OlmoeSparseMoeBlock(cfg)

ours = MoE(D, E, K, DFF, bias=False)

B = 64  # batch size
L = 256 # sequence length, # of tokens per sequence

x = torch.randn(B, L, D)

@torch.no_grad()
def equate(w_dst, w_src):
    assert w_dst.shape == w_src.shape
    w_dst.copy_(w_src)
    # NOTE that w_dst = w_src would just rebind the reference, and rebinding at this level
    # meaning w_dst gets rebound locally only, not what we want.

equate(ours.router.weight, olmoe.gate.weight)
for e in range(ours.E):
    equate(ours.experts[e].gate.weight, olmoe.experts[e].gate_proj.weight)
    equate(ours.experts[e].up.weight,   olmoe.experts[e].up_proj.weight)
    equate(ours.experts[e].down.weight, olmoe.experts[e].down_proj.weight)

with torch.no_grad():
    out_olmoe_tuple = olmoe(x)  # shape: (B, L, D)
    out_olmoe = out_olmoe_tuple[0]
    out_ours = ours(x)    # shape: (B, L, D)
    
    olmoe_gate_logits = out_olmoe_tuple[1]
    ours_gate_logits = ours.router(x)
    assert torch.allclose(olmoe_gate_logits, ours_gate_logits.reshape(-1, E), atol=1e-6)

    eid=7
    one_of_olmoe = olmoe.experts[eid](x)
    one_of_ours = ours.experts[eid](x)
    assert torch.allclose(one_of_olmoe, one_of_ours, atol=1e-6)

    assert torch.allclose(out_olmoe, out_ours, atol=1e-6)
    # if ours.n_drop > 0:
    #     print(f"Total dropped tokens in ours: {ours.n_drop}")
    # else:
    #     assert torch.allclose(out_olmoe, out_ours, atol=1e-6)

print("end.")
