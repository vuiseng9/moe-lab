import os
import torch
from .grouped_gemm_triton import group_gemm_fn

from torch.amp import custom_fwd, custom_bwd

def loop_gemm_fn(A_list, B_list, transA=False, transB=False):
    match (transA, transB):
        case (False, False):
            return [torch.mm(A, B) for A, B in zip(A_list, B_list)]
        case (False, True):
            return [torch.mm(A, B.T) for A, B in zip(A_list, B_list)]
        case (True, False):
            return [torch.mm(A.T, B) for A, B in zip(A_list, B_list)]
        case (True, True):
            return [torch.mm(A.T, B.T) for A, B in zip(A_list, B_list)]

EXPERT_GEMM_FN = {
    "triton": group_gemm_fn,
    "loop": loop_gemm_fn,
}

gemm_fn = EXPERT_GEMM_FN[os.getenv("MOEDL_GEMM_IMPL", "triton")]

class GroupedGemmFunc(torch.autograd.Function):
    """
    A autograded grouped gemm where backend is triton kernel supporting NN, TN, NT layout.
    Using X, W, Y instead of A, B, C for clarity mapping to expert linear layer.
    """
    @staticmethod
    @torch._dynamo.disable
    @custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)  # makes *incoming* tensors BF16 when autocast is enabled
    def forward(ctx, n_group, *XW_list):
        # Important: why this signature
        # 1. can only have one *args, so we can't have *X_list, *W_list
        # 2. Python containers (list/tuple/dict) are not treated as differentiable inputs,
        #    so using pure X_list, W_list will cause autograd to not recognize them ctx.needs_input_grad
        #    and thus we can't backpropagate gradients to them.
        # 3. save_for_backward only accepts tensors, so we need to unpack the list/tuple/dict using *XW_list
        # 4. return tuple instead of list, list is not tracked by autograd.

        X_list = XW_list[:n_group]
        W_list = XW_list[n_group:]
        
        Y_list = gemm_fn(X_list, W_list, transA=False, transB=True)  # gemm 1: dot(X, W.T))

        ctx.n_group = n_group
        ctx.save_for_backward(*XW_list)

        return tuple(Y_list)
    
    @staticmethod
    @torch._dynamo.disable
    @custom_bwd(device_type="cuda")
    def backward(ctx, *grad_Y_tuple):
        n_group = ctx.n_group
        X_list = ctx.saved_tensors[:n_group]
        W_list = ctx.saved_tensors[n_group:]

        need_grad_X = ctx.needs_input_grad[1 : 1+n_group]
        need_grad_W = ctx.needs_input_grad[1+n_group:]

        # Incoming grad tensors is not guaranteed to be contiguous
        # the triton kernel requires contiguous inputs
        # if it involves copy, shape is only (B, S, H)
        grad_Y_tuple = tuple(g.contiguous() for g in grad_Y_tuple)

        # gemm 2
        grad_X_list = [None] * n_group
        if any(need_grad_X):
            grad_X_list = gemm_fn(grad_Y_tuple, W_list, transA=False, transB=False) # gemm 2: dot(grad_Y, W))

        # gemm 3
        grad_W_list = [None] * n_group
        if any(need_grad_W):
            grad_W_list = gemm_fn(grad_Y_tuple, X_list, transA=True, transB=False) # gemm 3: dot(grad_Y.T, X)
              
        # because we are calling *XW_list in forward, 
        # so we need to return unpacked in backward, 
        # and we can use more than 1 unpacking here
        # unlike forward input sigature that only allows one *args
        return None, *grad_X_list, *grad_W_list
