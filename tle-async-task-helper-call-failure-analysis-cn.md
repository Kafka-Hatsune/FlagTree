# TLE async_task helper-call 失败原因分析

## 当前状态

分支：`tle-async-task-helper-call`

最后一次已观测失败的测试：

```bash
python -m pytest python/test/tle/integration/test_tle_ws_tma_gemm.py
```

观测到的行为：

- `launch_num_warps=4` 通过。
- `launch_num_warps=8` 在 `ConvertTritonToTritonGPU` 阶段失败。
- 失败发生在 `triton-tle-inline-async-task-helpers` 之前，所以受控 inline pass 还没有机会运行。

当前本地最新 mitigation commit：

```text
45d1800da Make TLE conversion operands warp-context aware
```

这个 commit 目前只做过本地静态检查，还需要在 Linux/GPU 环境中重新构建和跑测试验证。

## 报错内容

失败诊断如下：

```text
Result has an invalid layout:
#ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16],
warpsPerCTA = [8, 1], order = [1, 0]}>.
Layout has 8 warps per CTA, but the context requires 4 warps per CTA.

acc = tle.gpu.wgmma(a_smem.slot(slot), b_smem.slot(slot), acc)
```

报错 IR 中已经能看到预期的 async-task helper-call 结构：

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

因此，前端已经生成了 helper-call 形态，helper 上也确实带有 `ttg.num-warps = 4`。

## 根因

失败原因是在 `ConvertTritonToTritonGPU` 中出现了混合 warp context。

当前 conversion 中同时存在两类类型转换路径：

- type-only conversion：只看 `RankedTensorType`，使用 pass-level launch `num_warps`。
- value-aware conversion：可以通过 `lookupNumWarps(value)` / `maybeLookupNumWarps(op)` 找到 helper 或 partition 的局部 warp context。

失败用例中：

- launch-level `num_warps = 8`。
- consumer helper / partition `num_warps = 4`。
- consumer body 中某些 tensor value 被 type-only 路径 materialize，因此拿到了 8-warps blocked layout。
- 这个 value 又被用在 helper 或 partition 的 4-warps context 里。
- Triton tensor-layout verifier 在 inline pass 之前就拒绝了这个 op。

简化后就是：

```text
helper context 要求 4 warps
但某个已转换 tensor operand/result 被 materialize 成 8 warps
=> ConvertTritonToTritonGPU 阶段 verifier 失败
```

## 为什么 helper attr 不够

给 helper `tt.func` 加 `"ttg.num-warps" = 4` 是必要条件，但不是充分条件。

conversion pattern 不只是自己手动创建 result type。它们还会从 MLIR dialect conversion 中拿到 adaptor operands。如果框架需要把 original value 桥接成 converted value，就可能调用 target materialization，而 target materialization 使用的是 type-only desired type。

在当前代码里，这个 type-only desired type 仍然来自 pass-level converter。对于失败测试，它就是 launch-level 8 warps。

于是某个 pattern 可能看到这种状态：

```text
original helper-local tensor: 应该是 4-warps layout
adaptor operand: 已经被转换或 materialize 成 8-warps layout
```

如果 pattern 直接转发 `adaptor.getOperands()`，8-warps layout 就会泄漏进 helper body。

最敏感的路径是 loop-carried WGMMA accumulator：

```mlir
%for = scf.for ... iter_args(%acc = ...)
  %next = tle.wgmma ..., %acc
  %wait = tle.wgmma_wait %next
  scf.yield %wait
```

如果 `scf.for` init arg、region arg、yield operand、或者 WGMMA accumulator operand 里任意一处走了 type-only conversion，accumulator 就可能变成 8-warps layout，即使 `tle.wgmma` 本身处在 4-warps helper context 中。

## 为什么受控 inline 解决不了这个失败

当前计划中的 pipeline 顺序是：

```text
convert-triton-to-tritongpu
triton-tle-inline-async-task-helpers
...
```

实际失败发生在 `convert-triton-to-tritongpu` 运行期间。

因此：

- inline pass 根本还没有运行。
- 最终 IR 是否 inline，在这个失败点之前没有影响。
- 非 inline helper-call 形态本身必须先能通过 `ConvertTritonToTritonGPU`。

受控 inline 对最终 IR 形态和潜在性能仍然有意义，但它位于这个失败点之后。

## 和旧 warp_specialize API 的差异

旧 API 路径使用：

```python
_generator.call_JitFunction(
    worker_fn,
    block_values,
    kwargs={},
    caller_context=WarpSpecializeCallerContext(worker_num_warps[idx]),
)
```

这条路径直接通过已有 JIT function-call 机制 codegen worker function，并传入 caller context。

async_task helper-call 路径目标类似，但生成形态并不完全一样：

- async_task 从 `with` body 合成 helper function。
- capture liveins 会作为 synthetic helper arguments 传入。
- partition body 中生成 synthetic `tt.call @helper(...)`。
- helper 后续计划由 TLE 专用 inline pass 展开。

旧 API 没有失败，并不代表 conversion 缺口不存在。更可能是旧 API 已测试形态没有同时暴露下面这组条件：

- launch `num_warps=8`；
- worker/helper `num_warps=4`；
- TLE WGMMA accumulator 通过 `scf.for` loop-carried；
- Convert 之后还要求 helper inline；
- synthetic capture arguments 和 partition call operands 共同参与类型转换。

## 当前 mitigation 方向

最新本地 commit 尝试把 conversion 做得更 value-aware：

- `tle.wgmma` 和 `tle.wgmma_wait` 改为专用 conversion pattern。
- WGMMA accumulator/result/input 类型强制按 op 所在 context 的 warp 数对齐。
- generic conversion pattern 会把 adaptor operand 校正回 original operand 的 contextual converted type。
- `scf.for`、`scf.yield`、`scf.while`、`scf.condition` 不再盲目转发 adaptor operands，而是重新对齐 loop-carried operands。
- 部分 custom Triton/TLE conversion pattern 在重建 op 前也会重新对齐 operands。

预期效果是阻止 launch-level 8-warps 的 type-only materialization 泄漏到 4-warps helper-local code 中。

## 仍需验证的问题

最新 mitigation 仍需要在 Linux/GPU 环境验证：

```bash
pip install -e . --no-build-isolation -v
python -m pytest python/test/tle/integration/test_tle_ws_tma_gemm.py
```

如果仍然出现同样错误，下一步应导出或定位第一个 `tle.wgmma` 附近的 converted IR，重点看：

- 初始 zero accumulator 的 type。
- 第一个 `tle.wgmma` result 的 type。
- `scf.for` loop-carried accumulator block argument 的 type。
- `scf.yield` operand 的 type。
- WGMMA 之前是否插入了非预期的 `ttg.convert_layout`，并且目标 layout 是 `warpsPerCTA = [8, 1]`。

如果错误变化，应优先根据新的失败 pass 位置重新分析。当前已知失败特指 Convert 阶段的 8-warps 与 4-warps layout mismatch。
