import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F, init
import math
from torch.amp import custom_fwd, custom_bwd

class GroupedMMFunc(torch.autograd.Function):
    """
    Wrap torch grouped_mm as autograd function.
    NOTE: F.grouped_mm does support autograd, but incompatible with torch.compile.
    Custom implementation this way is compatible with torch.compile.
    Using x, w, y instead of a, b, c for clarity mapping to expert linear layer.
    w is assumed to be in (n_group, oc, ic) shape, not ic by oc, therefore
    gemm 1 y      = dot(x, w.transpose(-2, -1))
    gemm 2 grad_x = dot(grad_y, w)
    gemm 3 grad_w = dot(grad_y.transpose(0, 1), x)
    """
    @staticmethod
    @custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)  # makes *incoming* tensors BF16 when autocast is enabled
    def forward(ctx, x, w, offs):
        ctx.save_for_backward(x, w, offs)
        y = F.grouped_mm(x, w.transpose(-2, -1), offs=offs, bias=None)
        return y 

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_y):
        # ensure grad_y is contiguous for backward, there is no guarantee
        grad_y = grad_y.contiguous()

        x, w, offs = ctx.saved_tensors

        if ctx.needs_input_grad[0]: 
            grad_x = F.grouped_mm(grad_y, w, offs=offs, bias=None)

        if ctx.needs_input_grad[1]: 
            grad_w = F.grouped_mm(grad_y.transpose(0, 1), x, offs=offs, bias=None)

        return grad_x, grad_w, None
    

class GroupedGLU(nn.Module):
    def __init__(self, n_group, d, dff, device=None, dtype=None):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.n_group = n_group
        self.d = d
        self.dff = dff
        self.weight_up = Parameter(
            torch.empty((n_group, dff, d), **factory_kwargs)
        )
        self.weight_gate = Parameter(
            torch.empty((n_group, dff, d), **factory_kwargs)
        )
        self.weight_down = Parameter(
            torch.empty((n_group, d, dff), **factory_kwargs)
        )
        self.reset_parameters()

    def extra_repr(self):
        return f"n_group={self.n_group}, d={self.d}, dff={self.dff}, biasless"

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight_up, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_gate, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_down, a=math.sqrt(5))

    def forward(self, x, offs):
        # x must be permuted and contigous
        z_gate = GroupedMMFunc.apply(x, self.weight_gate, offs)
        z_up   = GroupedMMFunc.apply(x, self.weight_up, offs)
        return GroupedMMFunc.apply(F.silu(z_gate) * z_up, self.weight_down, offs)

if __name__ == "__main__":
    n_group = 4
    d = 16
    dff = 64

    lengths = torch.tensor([4, 8, 7, 4], dtype=torch.int32, device="cuda")
    offs = torch.cumsum(lengths, dim=0).to(torch.int32) 
    
    model = GroupedGLU(n_group, d, dff, device="cuda")
    x = torch.randn(offs[-1], d, device="cuda")
    out = model(x, offs)
    print(out.shape)