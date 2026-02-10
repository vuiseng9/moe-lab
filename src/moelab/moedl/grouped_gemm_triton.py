from typing import Optional
import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9

def num_sms():
    if is_cuda():
        return torch.cuda.get_device_properties("cuda").multi_processor_count
    raise NotImplementedError("num_sms is only implemented for CUDA backend")


@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 84,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 84,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 64,
            'NUM_SM': num_sms(),
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 64,
            'NUM_SM': num_sms(),
        }),
    ],
    key=['group_size'],
)
@triton.jit
def grouped_matmul_kernel(
    # device tensor of matrices pointers
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # device tensor of gemm sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
    group_gemm_sizes,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    g_lds,
    # number of gemms
    group_size,
    # number of virtual SM
    transA: tl.constexpr,
    transB: tl.constexpr,
    IO_DTYPE: tl.constexpr,
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        # get the gemm size of the current problem
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        # iterate through the tiles in the current gemm problem
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            # pick up a tile from the current gemm problem
            k = gk
            m = gm
            n = gn
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(IO_DTYPE))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(IO_DTYPE))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(IO_DTYPE))
            # figure out tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # do regular gemm here
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            # a_ptrs are pointer grid of [am, ak] regardless
            # transA is achieved by reading the right address 
            if transA:
                # 1 row of base address strided by column, then broadcast + add column wise element 
                a_ptrs = a_ptr + offs_k[None, :] * lda + offs_am[:, None]
            else:
                # 1 column of base address strided by row, then broadcast + add row wise element 
                a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            # b_ptrs are pointer grid of [bk, bn] regardless
            # transB is achieved by reading the right address 
            if transB:
                b_ptrs = b_ptr + offs_bn[None, :] * ldb + offs_k[:, None]
            else:
                b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]

            mask_am = offs_am < m
            mask_bn = offs_bn < n

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                # hint to Triton compiler to do proper loop pipelining
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])
                # assume full tile for now
                # need to mask out the out-of-bound
                abs_k = offs_k + kk * BLOCK_SIZE_K

                amk_mask = (abs_k[None, :] < k) & mask_am[:, None]
                a = tl.load(a_ptrs, mask=amk_mask, other=0.0)
                
                bkn_mask = (abs_k[:, None] < k) & mask_bn[None, :]
                b = tl.load(b_ptrs, mask=bkn_mask, other=0.0)
                accumulator += tl.dot(a, b)
                if transA:
                    a_ptrs += BLOCK_SIZE_K * lda
                else:
                    a_ptrs += BLOCK_SIZE_K
                if transB:
                    b_ptrs += BLOCK_SIZE_K
                else:
                    b_ptrs += BLOCK_SIZE_K * ldb
            c = accumulator.to(IO_DTYPE)

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]

            # assumes full tile for now
            tl.store(c_ptrs, c, mask=mask_am[:, None] & mask_bn[None, :])

            # go to the next tile by advancing NUM_SM
            tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles

_TORCH_TO_TL = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.int8: tl.int8,
    torch.int32: tl.int32,
}

def group_gemm_fn(group_A, group_B, transA=False, transB=False):
    assert len(group_A) == len(group_B)
    group_size = len(group_A)

    A_addrs = []
    B_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    group_C = []
    io_dtype = _TORCH_TO_TL[group_A[0].dtype]
    dev = group_A[0].device
    for i in range(group_size):
        A = group_A[i]
        B = group_B[i]
        assert A.is_contiguous() and B.is_contiguous()
        if transA:
            K, M = A.shape
        else:
            M, K = A.shape
        if transB:
            assert B.shape[1] == K, f"Unmatched inner dimension K in group {i}"
            N, K = B.shape
        else:
            assert B.shape[0] == K, f"Unmatched inner dimension K in group {i}"
            K, N = B.shape
        C = torch.empty((M, N), device=dev, dtype=A.dtype)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [M, N, K]
        # NOTE A, B, C are all in row-major contigous even if logically requires transpose later
        # the ld is always dim 0 of stride, how we read the address per row
        g_lds += [A.stride(0), B.stride(0), C.stride(0)] 

    # note these are device tensors
    d_a_ptrs = torch.tensor(A_addrs, device=dev)
    d_b_ptrs = torch.tensor(B_addrs, device=dev)
    d_c_ptrs = torch.tensor(C_addrs, device=dev)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=dev)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=dev)
    # we use a fixed number of CTA, and it's auto-tunable
    grid = lambda META: (META['NUM_SM'], )
    grouped_matmul_kernel[grid](
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_g_sizes,
        d_g_lds,
        group_size,
        transA,
        transB,
        io_dtype
    )

    return group_C