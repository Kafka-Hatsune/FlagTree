# TLE async_task helper-call failure analysis

## Status

Branch: `tle-async-task-helper-call`

Last observed failing test:

```bash
python -m pytest python/test/tle/integration/test_tle_ws_tma_gemm.py
```

Observed behavior:

- `launch_num_warps=4` passes.
- `launch_num_warps=8` fails in `ConvertTritonToTritonGPU`.
- The failure happens before `triton-tle-inline-async-task-helpers`, so controlled inline cannot affect this failure yet.

Latest local mitigation commit:

```text
45d1800da Make TLE conversion operands warp-context aware
```

This commit has only been checked with local static checks. It still needs Linux/GPU rebuild and test validation.

## Error

The failing diagnostic is:

```text
Result has an invalid layout:
#ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16],
warpsPerCTA = [8, 1], order = [1, 0]}>.
Layout has 8 warps per CTA, but the context requires 4 warps per CTA.

acc = tle.gpu.wgmma(a_smem.slot(slot), b_smem.slot(slot), acc)
```

The IR already shows the intended async-task structure:

```mlir
partition0(...) num_warps(4) {
  tt.call @..._tle_async_task_..._consumer0(...)
  ttg.warp_return
}

tt.func private @..._consumer0(...) attributes {
  noinline = true,
  tle_async_task_helper,
  "ttg.num-warps" = 4 : i32
}
```

So the front-end helper-call shape is present, and the helper does carry `ttg.num-warps = 4`.

## Root Cause

The failing shape is caused by mixed warp contexts during `ConvertTritonToTritonGPU`.

There are two different type-conversion routes in play:

- Type-only conversion of `RankedTensorType` uses the pass-level launch `num_warps`.
- Value-aware conversion can use `lookupNumWarps(value)` / `maybeLookupNumWarps(op)` and therefore can find the helper or partition context.

For the failing case:

- Launch-level `num_warps = 8`.
- Consumer helper/partition `num_warps = 4`.
- Some tensor value in the consumer body is materialized with the type-only route, so it receives an 8-warps blocked layout.
- That value is then used inside a helper or partition context where `maybeLookupNumWarps(op)` requires 4 warps.
- The Triton tensor-layout verifier rejects the op before the inline pass can run.

In short:

```text
helper context says 4 warps
but a converted tensor operand/result was materialized as 8 warps
=> verifier fails during ConvertTritonToTritonGPU
```

## Why helper attr alone was insufficient

Adding `"ttg.num-warps" = 4` to the helper function is necessary but not sufficient.

The conversion patterns do not only create result types manually. They also receive adaptor operands from MLIR dialect conversion. If the framework needs to bridge an original value to a converted value, it can invoke target materialization using the type-only desired type. In this codebase, that type-only desired type is still derived from the launch-level converter, which is 8 warps in the failing test.

This means a pattern may see:

```text
original helper-local tensor: should be 4-warps
adaptor operand: already converted/materialized as 8-warps
```

If the pattern forwards `adaptor.getOperands()` unchanged, the 8-warps layout leaks into the helper body.

The most sensitive path is the loop-carried WGMMA accumulator:

```mlir
%for = scf.for ... iter_args(%acc = ...)
  %next = tle.wgmma ..., %acc
  %wait = tle.wgmma_wait %next
  scf.yield %wait
```

If the `scf.for` init arg, region arg, yield operand, or WGMMA accumulator operand is converted through the type-only route, the accumulator can become 8-warps even though the WGMMA op is in a 4-warps helper context.

## Why controlled inline cannot fix this failure

The planned pipeline order is:

```text
convert-triton-to-tritongpu
triton-tle-inline-async-task-helpers
...
```

The observed failure occurs while running `convert-triton-to-tritongpu`.

Therefore:

- The inline pass is never reached.
- The final desired IR shape is irrelevant until conversion produces legal helper IR.
- The conversion pass must be correct with non-inline helper calls.

Controlled inline is still useful for final IR shape and performance, but it is downstream of this failure.

## Old warp_specialize API comparison

The old API path uses:

```python
_generator.call_JitFunction(
    worker_fn,
    block_values,
    kwargs={},
    caller_context=WarpSpecializeCallerContext(worker_num_warps[idx]),
)
```

That path directly codegens worker functions through the existing JIT function-call mechanism with a caller context.

The async_task helper-call path is similar in intent, but not identical in generated shape:

- async_task synthesizes helper functions from captured `with` bodies.
- It passes captured liveins as synthetic helper arguments.
- The partition body contains a synthetic `tt.call @helper(...)`.
- The helper is later intended to be inlined by a TLE-only pass.

The old API not failing does not prove the conversion gap is absent. It more likely means the old tested shapes did not expose the same combination of:

- launch `num_warps=8`;
- worker/helper `num_warps=4`;
- TLE WGMMA accumulator loop-carried through `scf.for`;
- post-Convert helper inline requirement;
- synthetic capture arguments and partition call operands.

## Latest mitigation direction

The latest local commit attempts to make conversion value-aware more broadly:

- `tle.wgmma` and `tle.wgmma_wait` now use dedicated conversion patterns.
- WGMMA accumulator/result/input types are forced to match the op's contextual warp count.
- Generic conversion patterns coerce adaptor operands back to the original operand's contextual converted type.
- `scf.for`, `scf.yield`, `scf.while`, and `scf.condition` now realign loop-carried operands instead of blindly forwarding adaptor operands.
- Several custom Triton/TLE conversion patterns also realign operands before rebuilding ops.

The expected effect is to prevent type-only launch-level 8-warps materialization from leaking into 4-warps helper-local code.

## Remaining Uncertainties

The latest mitigation still needs validation in the Linux/GPU environment:

```bash
pip install -e . --no-build-isolation -v
python -m pytest python/test/tle/integration/test_tle_ws_tma_gemm.py
```

If the same error remains, the next thing to inspect is the exact converted IR around the first `tle.wgmma`:

- type of the initial zero accumulator;
- type of the first `tle.wgmma` result;
- type of the `scf.for` loop-carried accumulator block argument;
- type of the `scf.yield` operand;
- whether an unexpected `ttg.convert_layout` to `warpsPerCTA = [8, 1]` is inserted before WGMMA.

If the error changes, the new failure should be analyzed from its pass location first. The current known failure is specifically a Convert-stage 8-warps vs 4-warps layout mismatch.
