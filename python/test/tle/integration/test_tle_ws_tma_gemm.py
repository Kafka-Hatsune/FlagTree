# flagtree tle
"""
Smoke tests for warp-specialized TLE GEMM with explicit TMA completion barriers.

This intentionally keeps the GEMM to a single tile. The goal is to validate the
basic producer/consumer protocol:

- producer partition waits on "empty" barriers and issues TMA loads into smem
- TMA completion signals "full" barriers
- consumer partition waits on "full" barriers, computes one WGMMA, stores C
- consumer arrives on "empty" barriers to release the smem buffers
"""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _is_nvidia_backend() -> bool:
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda"


def _has_nvidia_hopper_gpu() -> bool:
    return _is_nvidia_backend() and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


pytestmark = pytest.mark.skipif(
    not _has_nvidia_hopper_gpu(),
    reason="warp-specialized TMA WGMMA GEMM requires NVIDIA Hopper (sm90+)",
)


@triton.jit
def _single_tile_tma_producer(
    a_desc,
    b_desc,
    a_smem,
    b_smem,
    a_empty,
    b_empty,
    a_full,
    b_full,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    phase: tl.constexpr = 0

    tle.gpu.barrier_wait(a_empty, phaseIdx=phase)
    tle.gpu.barrier_wait(b_empty, phaseIdx=phase)

    tle.gpu.copy(a_desc, a_smem, [BLOCK_M, BLOCK_K], [0, 0], barrier=a_full)
    tle.gpu.copy(b_desc, b_smem, [BLOCK_K, BLOCK_N], [0, 0], barrier=b_full)

    tle.gpu.barrier_wait(a_empty, phaseIdx=1)
    tle.gpu.barrier_wait(b_empty, phaseIdx=1)


@triton.jit
def _single_tile_gemm_consumer(
    c_ptr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    a_smem,
    b_smem,
    a_empty,
    b_empty,
    a_full,
    b_full,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    phase: tl.constexpr = 0

    tle.gpu.barrier_wait(a_full, phaseIdx=phase)
    tle.gpu.barrier_wait(b_full, phaseIdx=phase)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    acc = tle.gpu.wgmma(a_smem, b_smem, out_dtype=tl.float32)
    acc = tle.gpu.wgmma_wait(0, acc)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)

    tle.gpu.barrier_arrive(a_empty, phaseIdx=phase)
    tle.gpu.barrier_arrive(b_empty, phaseIdx=phase)


@triton.jit
def ws_tma_single_tile_gemm_kernel(
    a_desc,
    b_desc,
    c_ptr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    A_TILE_BYTES: tl.constexpr,
    B_TILE_BYTES: tl.constexpr,
):
    a_smem = tle.gpu.alloc(
        [BLOCK_M, BLOCK_K],
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
    )
    b_smem = tle.gpu.alloc(
        [BLOCK_K, BLOCK_N],
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
    )

    a_empty = tle.gpu.alloc_barrier(init=tle.gpu.READY)
    b_empty = tle.gpu.alloc_barrier(init=tle.gpu.READY)
    a_full = tle.gpu.alloc_barrier(expect_bytes=A_TILE_BYTES)
    b_full = tle.gpu.alloc_barrier(expect_bytes=B_TILE_BYTES)

    tle.gpu.warp_specialize(
        [
            (
                _single_tile_tma_producer,
                (
                    a_desc,
                    b_desc,
                    a_smem,
                    b_smem,
                    a_empty,
                    b_empty,
                    a_full,
                    b_full,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                ),
            ),
            (
                _single_tile_gemm_consumer,
                (
                    c_ptr,
                    stride_cm,
                    stride_cn,
                    a_smem,
                    b_smem,
                    a_empty,
                    b_empty,
                    a_full,
                    b_full,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                ),
            ),
        ],
        [4],
        [168],
    )


@triton.jit
def ws_tma_single_tile_gemm_kernel_async_tasks(
    a_desc,
    b_desc,
    c_ptr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    A_TILE_BYTES: tl.constexpr,
    B_TILE_BYTES: tl.constexpr,
):
    a_smem = tle.gpu.alloc(
        [BLOCK_M, BLOCK_K],
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
    )
    b_smem = tle.gpu.alloc(
        [BLOCK_K, BLOCK_N],
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
    )

    a_empty = tle.gpu.alloc_barrier(init=tle.gpu.READY)
    b_empty = tle.gpu.alloc_barrier(init=tle.gpu.READY)
    a_full = tle.gpu.alloc_barrier(expect_bytes=A_TILE_BYTES)
    b_full = tle.gpu.alloc_barrier(expect_bytes=B_TILE_BYTES)

    with tle.gpu.async_tasks():
        with tle.gpu.async_task("producer"):
            phase: tl.constexpr = 0

            tle.gpu.barrier_wait(a_empty, phaseIdx=phase)
            tle.gpu.barrier_wait(b_empty, phaseIdx=phase)

            tle.gpu.copy(a_desc, a_smem, [BLOCK_M, BLOCK_K], [0, 0], barrier=a_full)
            tle.gpu.copy(b_desc, b_smem, [BLOCK_K, BLOCK_N], [0, 0], barrier=b_full)

            tle.gpu.barrier_wait(a_empty, phaseIdx=1)
            tle.gpu.barrier_wait(b_empty, phaseIdx=1)

        with tle.gpu.async_task(num_warps=4, registers=168, name="consumer0"):
            phase: tl.constexpr = 0

            tle.gpu.barrier_wait(a_full, phaseIdx=phase)
            tle.gpu.barrier_wait(b_full, phaseIdx=phase)

            offs_m = tl.arange(0, BLOCK_M)
            offs_n = tl.arange(0, BLOCK_N)

            acc = tle.gpu.wgmma(a_smem, b_smem, out_dtype=tl.float32)
            acc = tle.gpu.wgmma_wait(0, acc)

            c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(c_ptrs, acc)

            tle.gpu.barrier_arrive(a_empty, phaseIdx=phase)
            tle.gpu.barrier_arrive(b_empty, phaseIdx=phase)


def ws_tma_single_tile_gemm(A, B, C, launch_num_warps, use_async_tasks=False):
    assert A.ndim == 2 and B.ndim == 2 and C.ndim == 2
    assert A.shape[1] == B.shape[0]
    assert C.shape == (A.shape[0], B.shape[1])
    assert A.dtype == torch.float16 and B.dtype == torch.float16 and C.dtype == torch.float32

    block_m, block_k = A.shape
    block_k_b, block_n = B.shape
    assert block_k == block_k_b

    from triton.tools.tensor_descriptor import TensorDescriptor

    a_desc = TensorDescriptor.from_tensor(A, block_shape=[block_m, block_k])
    b_desc = TensorDescriptor.from_tensor(B, block_shape=[block_k, block_n])
    kernel_fn = ws_tma_single_tile_gemm_kernel_async_tasks if use_async_tasks else ws_tma_single_tile_gemm_kernel
    return kernel_fn[(1, )](
        a_desc,
        b_desc,
        C,
        C.stride(0),
        C.stride(1),
        block_m,
        block_n,
        block_k,
        block_m * block_k * A.element_size(),
        block_k * block_n * B.element_size(),
        num_warps=launch_num_warps,
    )


class TestTLEWarpSpecializedTmaGemm:

    @pytest.mark.parametrize("use_async_tasks", [False, True])
    @pytest.mark.parametrize("launch_num_warps", [4, 8])
    def test_single_tile_producer_consumer_wgmma(self, use_async_tasks, launch_num_warps):
        torch.manual_seed(2026 + launch_num_warps + int(use_async_tasks))
        block_m, block_n, block_k = 64, 16, 16

        a = torch.randn(block_m, block_k, device="cuda", dtype=torch.float16).contiguous()
        b = torch.randn(block_k, block_n, device="cuda", dtype=torch.float16).contiguous()
        c = torch.empty((block_m, block_n), device="cuda", dtype=torch.float32).contiguous()

        kernel = ws_tma_single_tile_gemm(a, b, c, launch_num_warps, use_async_tasks=use_async_tasks)
        assert "tt.call" not in kernel.asm["ttgir"]

        expected = torch.matmul(a.float(), b.float())
        torch.testing.assert_close(c, expected, atol=2e-2, rtol=2e-2)
