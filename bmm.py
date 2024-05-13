import argparse
import sys
import torch
import triton
import triton.language as tl
import pytest


configs = [
    triton.Config(
    {
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
        "GROUP_M": GROUP_M,
        "waves_per_eu": waves_per_eu,
        "matrix_instr_nonkdim": matrix_instr_nonkdim,
        "kpack": kpack,
    },
    num_warps=num_warps,
    num_stages=num_stages,
    )
    for BLOCK_M in [16, 32, 64, 128, 256]
    for BLOCK_N in [16, 32, 64, 128, 256]
    for BLOCK_K in [16, 32, 64, 128]
    for GROUP_M in [4, 8]
    for matrix_instr_nonkdim in [16]
    for waves_per_eu in [0]
    for kpack in [2]
    for num_stages in [0]
    for num_warps in [8]
]
@triton.autotune(
    configs=configs,
    # configs=[
    #     triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,
    #                    'GROUP_M': 4, 'waves_per_eu': 0,
    #                    'matrix_instr_nonkdim': 16, 'kpack': 2},
    #                   num_stages=0, num_warps=8),
    # ],
    key=['M', 'N', 'K'],
    use_cuda_graph=True
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % args['BLOCK_K'] == 0,
})
@triton.jit
def triton_bmm(
    arg_A, arg_B, out_ptr0,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, EVEN_K: tl.constexpr):
    # GROUP_M : tl.constexpr = 8
    # EVEN_K : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    # BLOCK_M : tl.constexpr = 128
    # BLOCK_N : tl.constexpr = 32
    # BLOCK_K : tl.constexpr = 16

    A = arg_A
    B = arg_B

    stride_aq = M*K
    stride_am = K
    stride_ak = 1

    stride_bq = N*K
    stride_bk = N
    stride_bn = 1

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    idx_q = tl.program_id(1)  # batch dimension for BMM
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q*stride_aq)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q*stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_q = tl.program_id(1)  # batch dimension for BMM
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + (idx_m*N) + (idx_q*M*N)
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, mask.shape)), acc, mask)


def call_triton_bmm(a, b):
    assert a.shape[0] == b.shape[0]  # batch dim
    assert a.shape[2] == b.shape[1]  # N dim
    B, M, K = a.shape
    _, _, N = b.shape

    c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), B)
    triton_bmm[grid](
        a, b, c,  #
        M, N, K,  #
    )
    return c


@pytest.mark.parametrize("B, M, N, K, in_dtype, out_dtype",
[(*shape, in_dtype, out_dtype)
    for shape in [
        (32, 128, 21, 8),
        (128, 59, 40, 128),]
    for in_dtype, out_dtype in [
        ('bfloat16', 'bfloat16')]
]
)
def test_correctness(B, M, N, K, in_dtype, out_dtype):
    torch.manual_seed(0)
    a = torch.randn((B, M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((B, K, N), device='cuda', dtype=torch.float16)
    triton_output = call_triton_bmm(a, b)
    torch_output = torch.bmm(a, b)
    print(f"triton_output={triton_output}")
    print(f"torch_output={torch_output}")
    rtol = 0 if torch.version.hip is None else 1e-2
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
        assert torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol)


global verbose
verbose = False

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['B', 'M', 'N', 'K'],  # Argument names to use as an x-axis for the plot
        x_vals=[
            (3500, 128, 21, 8),
            (3500, 1024, 1, 8),
            (1024, 59, 40, 128),
            (1024, 128, 40, 59),
            (1024, 31, 16, 128),
            (1024, 128, 16, 31),
            (1024, 29, 4, 128),
        ],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['rocblas', 'triton'],
        # Label name for the lines
        line_names=["rocBLAS", "Triton"],
        # Line styles
        styles=[('green', '-'), ('blue', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))
def benchmark(B, M, N, K, provider):
    a = torch.randn((B, M, K), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((B, K, N), device='cuda', dtype=torch.bfloat16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'rocblas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.bmm(a, b), quantiles=quantiles)
    if provider == 'triton':
        with torch.cuda.stream(torch.cuda.Stream()):
            ms= triton.testing.do_bench_cudagraph(lambda: call_triton_bmm(a, b))
        # ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
        global verbose
        if verbose:
            print(f'SIZE: {M},{N},{K}   Best tuning config: ({triton_bmm.get_best_config()})')
            # print(f'SIZE: {M},{N},{K} TIME: {ms:.3f} ms, {min_ms:.3f} min_ms, {max_ms:.3f} max_ms')
    perf = lambda ms: 2 * B * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms)#, perf(max_ms), perf(min_ms)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="GEMM tutorial example",
        allow_abbrev=False,
    )

    parser.add_argument("-v", action='store_true', default=False, help="Print out the best tuning config")
    args = parser.parse_args()

    return args


def main():
    # assign to a global verbose var to indicate whether print
    # best tuning config
    global verbose
    args = parse_args()
    verbose=args.v
    benchmark.run(show_plots=False, print_data=True)


if __name__ == '__main__':
    sys.exit(main())