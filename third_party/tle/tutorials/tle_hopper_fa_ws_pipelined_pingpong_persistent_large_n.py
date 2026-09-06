#!/usr/bin/env python3
"""Large-N persistent TLE attention experiment for Hopper.

K and V use ``tle.pipe`` instead of user-managed TMA completion barriers.  For
a non-power-of-two BLOCK_N, exact SMEM is split only along that fragment axis;
the full power-of-two HEAD_DIM remains inside every storage tile.  Ordinary TMA
lowering derives smaller hardware transaction boxes from those tiles.  The
default sweep compares BLOCK_N 64 and 80 for HEAD_DIM 256.
"""

from __future__ import annotations

import argparse
import re

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.tools.tensor_descriptor import TensorDescriptor

from tle_hopper_fa_ws_pipelined_pingpong_persistent import (
    AttentionProblem,
    DEVICE,
    _buf_phase,
    _compute_offsets,
    _next_power_of_2,
    alloc_fn,
    bench_ms,
    make_error_row,
    make_row,
    parse_problem,
    reference_attention,
    sdpa_attention,
    write_rows,
)


@triton.jit
def _attn_fwd_tle_ws_large_n_producer(
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    N_CTX,
    q_smem,
    k_writer,
    v_writer,
    q_empties,
    q_fulls,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    BM_SPLIT: tl.constexpr,
    CID: tl.constexpr,
):
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)
    num_pid_m = tl.cdiv(N_CTX, BLOCK_M)
    total_tiles = num_pid_m * Z * H

    tile_idx = prog_id
    tile_count = 0
    accum_cnt_kv = 0
    while tile_idx < total_tiles:
        start_m, off_hz, qo_offset_y, kv_offset_y = _compute_offsets(tile_idx, H, N_CTX, BLOCK_M)

        q_buf, q_phase_idx = _buf_phase(tile_count, NUM_BUFFERS_Q)
        q0_idx = q_buf
        q1_idx = q_buf + NUM_BUFFERS_Q

        tle.gpu.barrier_wait(q_empties[q0_idx], phaseIdx=q_phase_idx)
        tle.gpu.copy(
            desc_q,
            q_smem.slot(q0_idx),
            [BM_SPLIT, HEAD_DIM],
            [qo_offset_y, 0],
            barrier=q_fulls[q0_idx],
        )

        k_slot = k_writer.acquire(accum_cnt_kv)
        tle.gpu.copy(
            desc_k,
            k_slot.k,
            [BLOCK_N, HEAD_DIM],
            [kv_offset_y, 0],
        )
        k_writer.commit(accum_cnt_kv)

        tle.gpu.barrier_wait(q_empties[q1_idx], phaseIdx=q_phase_idx)
        tle.gpu.copy(
            desc_q,
            q_smem.slot(q1_idx),
            [BM_SPLIT, HEAD_DIM],
            [qo_offset_y + BM_SPLIT, 0],
            barrier=q_fulls[q1_idx],
        )

        v_slot = v_writer.acquire(accum_cnt_kv)
        tle.gpu.copy(
            desc_v,
            v_slot.v,
            [BLOCK_N, HEAD_DIM],
            [kv_offset_y, 0],
        )
        v_writer.commit(accum_cnt_kv)
        accum_cnt_kv += 1

        for kv_idx in range(BLOCK_N, N_CTX, BLOCK_N):
            kv_offset = kv_offset_y + kv_idx

            k_slot = k_writer.acquire(accum_cnt_kv)
            tle.gpu.copy(
                desc_k,
                k_slot.k,
                [BLOCK_N, HEAD_DIM],
                [kv_offset, 0],
            )
            k_writer.commit(accum_cnt_kv)

            v_slot = v_writer.acquire(accum_cnt_kv)
            tle.gpu.copy(
                desc_v,
                v_slot.v,
                [BLOCK_N, HEAD_DIM],
                [kv_offset, 0],
            )
            v_writer.commit(accum_cnt_kv)
            accum_cnt_kv += 1

        tile_idx += num_progs
        tile_count += 1


@triton.jit
def _attn_fwd_tle_ws_large_n_consumer(
    sm_scale,
    m_ptr,
    o_ptr,
    Z,
    H,
    N_CTX,
    q_smem,
    k_reader,
    v_reader,
    q_empties,
    q_fulls,
    ping_to_c0,
    ping_to_c1,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    BM_SPLIT: tl.constexpr,
    EARLY_CAST_P: tl.constexpr,
    CID: tl.constexpr,
):
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)
    num_pid_m = tl.cdiv(N_CTX, BLOCK_M)
    total_tiles = num_pid_m * Z * H
    consumer_idx: tl.constexpr = CID - 1

    if consumer_idx == 1:
        tle.gpu.barrier_arrive(ping_to_c0)

    tile_idx = prog_id
    tile_count = 0
    accum_cnt_kv = 0
    while tile_idx < total_tiles:
        start_m, off_hz, qo_offset_y, kv_offset_y = _compute_offsets(tile_idx, H, N_CTX, BLOCK_M)

        m_i = tl.zeros([BM_SPLIT], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BM_SPLIT], dtype=tl.float32) + 1.0
        acc = tl.zeros([BM_SPLIT, HEAD_DIM], dtype=tl.float32)
        qk_scale = sm_scale * 1.44269504

        q_buf, q_phase_idx = _buf_phase(tile_count, NUM_BUFFERS_Q)
        q_idx = q_buf + consumer_idx * NUM_BUFFERS_Q
        tle.gpu.barrier_wait(q_fulls[q_idx], phaseIdx=q_phase_idx)

        k_wait = k_reader.wait(accum_cnt_kv)
        if consumer_idx == 0:
            tle.gpu.barrier_wait(ping_to_c0)
        else:
            tle.gpu.barrier_wait(ping_to_c1)
        qk = tle.gpu.wgmma(
            q_smem.slot(q_idx),
            k_wait.slot.k,
            out_dtype=tl.float32,
            trans_b=True,
        )
        if consumer_idx == 0:
            tle.gpu.barrier_arrive(ping_to_c1)
        else:
            tle.gpu.barrier_arrive(ping_to_c0)
        qk = tle.gpu.wgmma_wait(0, qk)
        k_reader.release(accum_cnt_kv)

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        # Carry P across the overlapped QK/PV iteration in FP16.  Keeping the
        # FP32 value alive until the following PV consumes it substantially
        # increases register pressure for large active N.
        if EARLY_CAST_P:
            p = p.to(tl.float16)
        m_i = m_ij
        accum_cnt_kv += 1

        for _ in range(BLOCK_N, N_CTX, BLOCK_N):
            k_wait = k_reader.wait(accum_cnt_kv)

            if consumer_idx == 0:
                tle.gpu.barrier_wait(ping_to_c0)
            else:
                tle.gpu.barrier_wait(ping_to_c1)
            qk = tle.gpu.wgmma(
                q_smem.slot(q_idx),
                k_wait.slot.k,
                out_dtype=tl.float32,
                trans_b=True,
            )
            if consumer_idx == 0:
                tle.gpu.barrier_arrive(ping_to_c1)
            else:
                tle.gpu.barrier_arrive(ping_to_c0)

            v_iter = accum_cnt_kv - 1
            v_wait = v_reader.wait(v_iter)
            acc = tle.gpu.wgmma(p.to(tl.float16), v_wait.slot.v, acc)

            qk = tle.gpu.wgmma_wait(1, qk)
            k_reader.release(accum_cnt_kv)

            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = tl.math.exp2(qk)
            alpha = tl.math.exp2(m_i - m_ij)
            l_ij = tl.sum(p, 1)
            l_i = l_i * alpha + l_ij
            if EARLY_CAST_P:
                p = p.to(tl.float16)
            m_i = m_ij

            acc = tle.gpu.wgmma_wait(0, acc)
            v_reader.release(v_iter)
            acc = acc * alpha[:, None]
            accum_cnt_kv += 1

        v_iter = accum_cnt_kv - 1
        v_wait = v_reader.wait(v_iter)
        acc = tle.gpu.wgmma(p.to(tl.float16), v_wait.slot.v, acc)

        acc = tle.gpu.wgmma_wait(1, acc)
        tle.gpu.barrier_arrive(q_empties[q_idx], phaseIdx=q_phase_idx)

        acc = tle.gpu.wgmma_wait(0, acc)
        v_reader.release(v_iter)

        m_i += tl.math.log2(l_i)
        acc = acc / l_i[:, None]

        offs_m = start_m * BLOCK_M + consumer_idx * BM_SPLIT + tl.arange(0, BM_SPLIT)
        m_ptrs = m_ptr + off_hz * N_CTX + offs_m
        tl.store(m_ptrs, m_i)
        offs_n = tl.arange(0, HEAD_DIM)
        o_rows = qo_offset_y + consumer_idx * BM_SPLIT + tl.arange(0, BM_SPLIT)
        # A descriptor store would allocate one implicit BM_SPLIT x HEAD_DIM
        # SMEM epilogue tile per consumer.  Store the contiguous output from
        # registers so the explicit Q/K/V allocation remains the whole budget.
        tl.store(o_ptr + o_rows[:, None] * HEAD_DIM + offs_n[None, :], acc.to(tl.float16))

        tile_idx += num_progs
        tile_count += 1


@triton.jit
def _attn_fwd_tle_ws_pipelined_pingpong_persistent_large_n(
    sm_scale,
    M,
    O,
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
    NUM_MMA_WARPS: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    Q_STAGE_CAPACITY: tl.constexpr,
    KV_STAGE_CAPACITY: tl.constexpr,
    EARLY_CAST_P: tl.constexpr,
    CONSUMER_MAX_NREG: tl.constexpr,
):
    BM_SPLIT: tl.constexpr = BLOCK_M // NUM_MMA_GROUPS
    THREADS_IN_MMA_GROUPS: tl.constexpr = NUM_MMA_WARPS * 32

    q_smem = tle.gpu.alloc(
        [BM_SPLIT, HEAD_DIM],
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
        capacity=Q_STAGE_CAPACITY,
    )
    k_smem = tle.gpu.alloc(
        [BLOCK_N, HEAD_DIM],
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
        capacity=KV_STAGE_CAPACITY,
    )
    v_smem = tle.gpu.alloc(
        [BLOCK_N, HEAD_DIM],
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
        capacity=KV_STAGE_CAPACITY,
    )

    q_empties = tle.gpu.alloc_barriers(num_barriers=Q_STAGE_CAPACITY, arrive_count=1, init=tle.gpu.READY)
    q_fulls = tle.gpu.alloc_barriers(
        num_barriers=Q_STAGE_CAPACITY,
        arrive_count=1,
        expect_bytes=BM_SPLIT * HEAD_DIM * 2,
    )

    k_pipe = tle.pipe(
        capacity=KV_STAGE_CAPACITY,
        scope="cta",
        name="large_n_k",
        readers=("c0", "c1"),
        k=k_smem,
    )
    v_pipe = tle.pipe(
        capacity=KV_STAGE_CAPACITY,
        scope="cta",
        name="large_n_v",
        readers=("c0", "c1"),
        v=v_smem,
    )
    k_writer = k_pipe.writer()
    v_writer = v_pipe.writer()

    pingpong = tle.gpu.alloc_barriers(num_barriers=2, arrive_count=THREADS_IN_MMA_GROUPS)
    ping_to_c0 = pingpong[0]
    ping_to_c1 = pingpong[1]

    mma_warps: tl.constexpr = NUM_MMA_WARPS // NUM_MMA_GROUPS
    tle.gpu.warp_specialize(
        [
            (
                _attn_fwd_tle_ws_large_n_producer,
                (
                    Z,
                    H,
                    desc_q,
                    desc_k,
                    desc_v,
                    N_CTX,
                    q_smem,
                    k_writer,
                    v_writer,
                    q_empties,
                    q_fulls,
                    HEAD_DIM,
                    BLOCK_M,
                    BLOCK_N,
                    NUM_BUFFERS_Q,
                    BM_SPLIT,
                    0,
                ),
            ),
            (
                _attn_fwd_tle_ws_large_n_consumer,
                (
                    sm_scale,
                    M,
                    O,
                    Z,
                    H,
                    N_CTX,
                    q_smem,
                    k_pipe.reader("c0"),
                    v_pipe.reader("c0"),
                    q_empties,
                    q_fulls,
                    ping_to_c0,
                    ping_to_c1,
                    HEAD_DIM,
                    BLOCK_M,
                    BLOCK_N,
                    NUM_BUFFERS_Q,
                    BM_SPLIT,
                    EARLY_CAST_P,
                    1,
                ),
            ),
            (
                _attn_fwd_tle_ws_large_n_consumer,
                (
                    sm_scale,
                    M,
                    O,
                    Z,
                    H,
                    N_CTX,
                    q_smem,
                    k_pipe.reader("c1"),
                    v_pipe.reader("c1"),
                    q_empties,
                    q_fulls,
                    ping_to_c0,
                    ping_to_c1,
                    HEAD_DIM,
                    BLOCK_M,
                    BLOCK_N,
                    NUM_BUFFERS_Q,
                    BM_SPLIT,
                    EARLY_CAST_P,
                    2,
                ),
            ),
        ],
        [mma_warps, mma_warps],
        # 224 registers per consumer leaves 56 for the producer.  Both this
        # split and 232/40 are spill-free with the N80 storage-tile plan;
        # 224 is the faster setting for the default H100 benchmark.
        [CONSUMER_MAX_NREG, CONSUMER_MAX_NREG],
    )


def tle_attention_large_n(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    *,
    block_m: int = 128,
    block_n: int = 192,
    num_buffers_q: int = 1,
    num_buffers_kv: int = 2,
    num_mma_warps: int = 8,
    num_mma_groups: int = 2,
    producer_num_warps: int = 4,
    early_cast_p: bool = True,
    consumer_max_nreg: int = 224,
    return_kernel: bool = False,
    out: torch.Tensor | None = None,
    m_out: torch.Tensor | None = None,
):
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype == torch.float16 and k.dtype == torch.float16 and v.dtype == torch.float16
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert q.shape == k.shape == v.shape and q.ndim == 4
    assert num_mma_groups == 2 and block_m % num_mma_groups == 0
    assert producer_num_warps % 4 == 0

    z, h, n_ctx, head_dim = q.shape
    if head_dim not in (16, 32, 64, 128, 256):
        raise ValueError("HEAD_DIM must be one of 16, 32, 64, 128, 256")
    if block_n <= 0 or block_n > 256 or block_n % 16 != 0:
        raise ValueError("BLOCK_N must be a multiple of 16 no greater than 256")
    if num_buffers_kv <= 0:
        raise ValueError("num_buffers_kv must be positive")
    if consumer_max_nreg < 24 or consumer_max_nreg > 256 or consumer_max_nreg % 8:
        raise ValueError("consumer_max_nreg must be a multiple of 8 in [24, 256]")
    if n_ctx % block_m != 0 or n_ctx % block_n != 0:
        raise ValueError("N_CTX must be a multiple of BLOCK_M and BLOCK_N")

    triton.set_allocator(alloc_fn)
    o = torch.empty_like(q) if out is None else out
    m = torch.empty((z, h, n_ctx), device=q.device, dtype=torch.float32) if m_out is None else m_out
    if o.shape != q.shape or o.dtype != q.dtype or not o.is_contiguous():
        raise ValueError("out must be a contiguous tensor with the same shape and dtype as q")

    y_dim = z * h * n_ctx
    block_m_split = block_m // num_mma_groups
    kv_descriptor_block = [block_n, head_dim]
    if block_n & (block_n - 1):
        kv_descriptor_block = [block_n & -block_n, head_dim]

    desc_q = TensorDescriptor(q, shape=[y_dim, head_dim], strides=[head_dim, 1], block_shape=[block_m_split, head_dim])
    desc_k = TensorDescriptor(k, shape=[y_dim, head_dim], strides=[head_dim, 1], block_shape=kv_descriptor_block)
    desc_v = TensorDescriptor(v, shape=[y_dim, head_dim], strides=[head_dim, 1], block_shape=kv_descriptor_block)

    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    total_tiles = triton.cdiv(n_ctx, block_m) * z * h
    grid = (min(num_sms, total_tiles), )
    q_stage_capacity = _next_power_of_2(num_buffers_q * num_mma_groups)
    kv_stage_capacity = _next_power_of_2(num_buffers_kv)

    kernel = _attn_fwd_tle_ws_pipelined_pingpong_persistent_large_n[grid](
        sm_scale,
        m,
        o,
        z,
        h,
        desc_q,
        desc_k,
        desc_v,
        n_ctx,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        NUM_BUFFERS_Q=num_buffers_q,
        NUM_BUFFERS_KV=num_buffers_kv,
        NUM_MMA_WARPS=num_mma_warps,
        NUM_MMA_GROUPS=num_mma_groups,
        Q_STAGE_CAPACITY=q_stage_capacity,
        KV_STAGE_CAPACITY=kv_stage_capacity,
        EARLY_CAST_P=early_cast_p,
        CONSUMER_MAX_NREG=consumer_max_nreg,
        num_warps=producer_num_warps,
    )
    if return_kernel:
        return o, m, kernel
    return o


def _count_pattern(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text))


def kernel_stats(
    kernel,
    problem: AttentionProblem,
    block_n: int,
    num_buffers_kv: int,
    early_cast_p: bool,
    consumer_max_nreg: int,
) -> dict[str, object]:
    ttgir = kernel.asm.get("ttgir", "")
    ptx = kernel.asm.get("ptx", "")
    is_fragmented = bool(block_n & (block_n - 1))
    storage_tile_rows = block_n & -block_n if is_fragmented else None
    tma_box_cols = min(problem.head_dim, 64) if is_fragmented else problem.head_dim
    storage_tiles = block_n // storage_tile_rows if storage_tile_rows else ""
    tma_messages_per_tile = problem.head_dim // tma_box_cols if storage_tile_rows else 1
    tma_messages_per_stage = storage_tiles * tma_messages_per_tile if storage_tile_rows else 1
    carrier_rows = _next_power_of_2(block_n)
    return {
        "kv_iterations": problem.n_ctx // block_n,
        "kv_requested_buffers": num_buffers_kv,
        "kv_stage_capacity": _next_power_of_2(num_buffers_kv),
        "kv_storage_tile_rows": storage_tile_rows or "",
        "kv_storage_tile_cols": problem.head_dim if storage_tile_rows else "",
        "kv_storage_tiles_per_stage": storage_tiles,
        "kv_tma_box_rows": storage_tile_rows or block_n,
        "kv_tma_box_cols": tma_box_cols,
        "kv_tma_messages_per_storage_tile": tma_messages_per_tile,
        "kv_tma_messages_per_stage": tma_messages_per_stage,
        "kv_tma_bytes_per_message": (storage_tile_rows or block_n) * tma_box_cols * 2,
        "kv_carrier_rows": carrier_rows,
        "kv_exact_rows_saved_per_buffer": carrier_rows - block_n,
        "early_cast_p": early_cast_p,
        "consumer_max_nreg": consumer_max_nreg,
        "shared_bytes": getattr(kernel.metadata, "shared", ""),
        # TTGIR is captured before the backend Membar analysis runs.
        "pre_membar_ttgir_local_barriers": ttgir.count("ttg.local_barrier"),
        "ptx_bar_sync": _count_pattern(ptx, r"\bbar(?:rier)?\.sync\b"),
        "ptx_mbarrier_ops": _count_pattern(ptx, r"\bmbarrier\."),
        "ptx_tma_load_ops": _count_pattern(
            ptx, r"\bcp\.async\.bulk\.tensor\.\w+\.shared::cluster\.global"
        ),
        "ptx_wgmma_ops": _count_pattern(ptx, r"\bwgmma\.mma_async\."),
        "ttgir_len": len(ttgir),
        "ptx_len": len(ptx),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", action="append", type=parse_problem, default=[])
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", action="append", type=int, default=[])
    parser.add_argument("--num-buffers-kv", type=int, default=2)
    parser.add_argument("--late-cast-p", action="store_true")
    parser.add_argument("--consumer-max-nreg", type=int, default=224)
    parser.add_argument("--sm-scale", type=float, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--include-sdpa", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--continue-on-tle-error", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError("Hopper or newer CUDA GPU is required")

    problems = args.problem or [AttentionProblem(1, 8, 3840, 256)]
    block_ns = args.block_n or [64, 80]
    rows = []
    for problem in problems:
        q = torch.randn((problem.z, problem.h, problem.n_ctx, problem.head_dim), device=DEVICE, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        sm_scale = problem.head_dim**-0.5 if args.sm_scale is None else args.sm_scale
        ref = reference_attention(q, k, v, sm_scale) if args.check else None

        if args.include_sdpa:
            def run_sdpa():
                sdpa_attention(q, k, v, sm_scale)

            ms, p20, p80 = bench_ms(run_sdpa, args.warmup, args.rep, cuda_graph=args.cuda_graph)
            rows.append(make_row("SDPA", problem, ms, p20, p80, args.block_m, 0, cuda_graph=args.cuda_graph))

        for block_n in block_ns:
            try:
                o, m, kernel = tle_attention_large_n(
                    q,
                    k,
                    v,
                    sm_scale,
                    block_m=args.block_m,
                    block_n=block_n,
                    num_buffers_kv=args.num_buffers_kv,
                    early_cast_p=not args.late_cast_p,
                    consumer_max_nreg=args.consumer_max_nreg,
                    return_kernel=True,
                )
                torch.cuda.synchronize()
                max_abs_error = ""
                if ref is not None:
                    max_abs_error = f"{(o.float() - ref.float()).abs().max().item():.6f}"
                    torch.testing.assert_close(o, ref, atol=5e-2, rtol=5e-2)

                bench_o = torch.empty_like(q)
                bench_m = torch.empty((problem.z, problem.h, problem.n_ctx), device=q.device, dtype=torch.float32)

                def run():
                    tle_attention_large_n(
                        q,
                        k,
                        v,
                        sm_scale,
                        block_m=args.block_m,
                        block_n=block_n,
                        num_buffers_kv=args.num_buffers_kv,
                        early_cast_p=not args.late_cast_p,
                        consumer_max_nreg=args.consumer_max_nreg,
                        out=bench_o,
                        m_out=bench_m,
                    )

                ms, p20, p80 = bench_ms(run, args.warmup, args.rep, cuda_graph=args.cuda_graph)
                extra = kernel_stats(
                    kernel,
                    problem,
                    block_n,
                    args.num_buffers_kv,
                    not args.late_cast_p,
                    args.consumer_max_nreg,
                )
                extra.update({"max_abs_error": max_abs_error, "status": "ok"})
                rows.append(
                    make_row(
                        "flagtree.tle.fa3.ws_pipe_persistent_large_n",
                        problem,
                        ms,
                        p20,
                        p80,
                        args.block_m,
                        block_n,
                        extra,
                        cuda_graph=args.cuda_graph,
                    )
                )
            except Exception as exc:
                if not args.continue_on_tle_error:
                    raise
                rows.append(make_error_row("flagtree.tle.fa3.ws_pipe_persistent_large_n", problem,
                                           args.block_m, block_n, exc))

    write_rows(rows, args.out)


if __name__ == "__main__":
    main()
