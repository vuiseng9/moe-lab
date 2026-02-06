import torch
from .grouped_gemm_triton import group_gemm_fn
from torch.amp import custom_fwd, custom_bwd

class NaiveGroupedGemmFunc(torch.autograd.Function):
    """
    A naive autograded grouped gemm where backend is just calling torch.mm and loop over each group.
    Using X, W, Y instead of A, B, C for clarity mapping to expert linear layer.
    """
    @staticmethod
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
        Y_list = []
        for X, W in zip(X_list, W_list):
            Y_list.append(torch.mm(X, W.T))

        ctx.n_group = n_group
        ctx.save_for_backward(*XW_list)

        return tuple(Y_list)
    
    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, *grad_Y_tuple):
        n_group = ctx.n_group
        X_list = ctx.saved_tensors[:n_group]
        W_list = ctx.saved_tensors[n_group:]

        need_grad_X = ctx.needs_input_grad[1 : 1+n_group]
        need_grad_W = ctx.needs_input_grad[1+n_group:]

        # gemm 2
        grad_X_list = [None] * n_group
        if any(need_grad_X):
            for i, (grad_Y, W) in enumerate(zip(grad_Y_tuple, W_list)):
                grad_X_list[i] = torch.mm(grad_Y, W)

        # gemm 3
        grad_W_list = [None] * n_group
        if any(need_grad_W):
            for i, (grad_Y, X) in enumerate(zip(grad_Y_tuple, X_list)):
                grad_W_list[i] = torch.mm(grad_Y.T, X)

        # because we are calling *XW_list in forward, 
        # so we need to return unpacked in backward, 
        # and we can use more than 1 unpacking here
        # unlike forward input sigature that only allows one *args
        return None, *grad_X_list, *grad_W_list