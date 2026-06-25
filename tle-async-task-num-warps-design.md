# TLE async_task num_warps 设计记录

这份文档记录三件事：

- 当前已有的 `num_warps` / layout 推导机制；
- 新的 inline `tle.gpu.async_task` API 设计；
- 为了让新设计正确工作，需要在 compiler conversion 里补的最小修改。

## 背景

Triton GPU tensor layout 会编码 CTA 内 warp 数。例如 blocked layout 里可能有：

```text
warpsPerCTA = [4, 1]
warpsPerCTA = [8, 1]
```

如果某个 op 位于 warp-specialized consumer partition 内，它的 tensor result
layout 应该服从 consumer partition 的 `num_warps`，而不一定等于 kernel launch
时的 `num_warps`。

触发这次问题的代码形态是：

```python
with tle.gpu.async_task(num_warps=4, registers=168, name="consumer0"):
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc = tle.gpu.wgmma(a_smem, b_smem, acc)
```

当 kernel 以 `num_warps=8` launch 时，consumer partition 仍然是
`num_warps(4)`。如果 `acc` 或 `tle.wgmma` 的 result 被 conversion 阶段分配成
8-warp layout，verifier 会正确报错：

```text
Layout has 8 warps per CTA, but the context requires 4 warps per CTA.
```

## 现有机制

现有机制里有两条相关路径。

第一条是 `TritonGPUTypeConverter` 的全局默认 warp 数。它在
`ConvertTritonToTritonGPU` pass 初始化时拿到 launch/module 的 `num_warps`。
type-only conversion 会使用这个值：

```cpp
convertType(type)
convertTypes(resultTypes, retTypes)
```

这类 API 只能看到 `Type`，例如：

```mlir
tensor<64x16xf32>
```

`Type` 本身没有 parent operation，也没有 parent region。因此 type-only
conversion 无法知道这个 tensor result 是否属于某个 consumer partition。

第二条是 TritonGPU dialect 已有的 contextual lookup：

```cpp
maybeLookupNumWarps(op)
lookupNumWarps(region)
```

这条路径本身是 partition-aware 的。如果 op 或 region 位于
`ttg.warp_specialize.partitions` 内，它会返回父 `ttg.warp_specialize` 上
`partitionNumWarps[idx]`；否则回退到周围 function/module 的 `ttg.num-warps`。

FlagTree/TLE 当前已经扩展了 value-aware conversion：

```cpp
convertType(value)
```

对 `Value` 来说，`getNumWarps(value)` 可以检查 defining op 或 block argument
所在 region，然后走上面的 contextual lookup。因此这条机制可以正确表达
partition-local layout，但前提是 conversion 调用点必须传入 `OpResult` /
`Value`，而不是只传 `Type`。

## MLIR Dialect Conversion 里的 Type、Value 和 Op

在 MLIR 中，`Type`、`Value` 和 `Operation` 是不同层次：

```text
Type      = SSA value 的静态类型
Value     = IR 里的某个 SSA value，可以追溯 defining op 或 block argument
Operation = 产生或使用 SSA value 的操作
```

例如：

```mlir
%cst = arith.constant dense<0.0> : tensor<64x16xf32>
```

这里：

```text
Operation = arith.constant
Value     = %cst
Type      = tensor<64x16xf32>
```

`Type` 是纯结构信息，可以被很多 op/result 共享。它不知道 `%cst` 是谁定义的，
也不知道 `%cst` 位于哪个 region。因此 `convertType(Type)` 只能做上下文无关的
默认转换。

`Value` 则带有 IR 上下文：

```text
%cst
  -> defining op = arith.constant
  -> parent region = partition0
  -> partition0 num_warps = 4
```

因此，当同一个 `tensor<64x16xf32>` 在不同 region 内需要不同 layout 时，
conversion 必须通过 `Value` 或 `OpResult` 找到上下文。

MLIR 的 design 不会默认把 op 和 type 绑定在一起，因为同一个 type 可能属于很多
value。真正需要上下文的 conversion 应该在 pattern 里显式选择：

```cpp
convertType(op.getType())    // type-only，保留原有默认行为
convertType(op.getResult())  // value-aware，可以读取 defining op / region
lookupNumWarps(op)           // op-aware，自定义 pattern 可直接使用
```

## 旧 warp_specialize API

旧 API 的形态是：

```python
tle.gpu.warp_specialize(
    ((producer_fn, producer_args), (consumer_fn, consumer_args)),
    worker_num_warps=[4],
    worker_num_regs=[168],
)
```

worker body 通过 JIT function call 生成。旧路径使用
`WarpSpecializeCallerContext(num_warps)` 给 worker callee function 设置：

```python
fn.set_attr("ttg.num-warps", builder.get_int32_attr(num_warps))
```

因此 worker callee 内部的 type-only conversion 也能看到 function-level
`ttg.num-warps = 4`。它能工作，是因为 worker body 被编译成了一个类似 callee
function 的上下文。

这个方案的问题是：worker 并不是真正的普通函数调用。它最终语义是同一个 CTA
里的 warp-specialized partition。call-based API 会引入额外的 capture/result
plumbing，也可能留下或依赖 `tt.call` 边界，直到后续 inline 成功。

## 新 inline async_task 设计

新的用户 API 直接暴露 producer/consumer partition 结构：

```python
with tle.gpu.async_tasks():
    with tle.gpu.async_task("producer"):
        ...
    with tle.gpu.async_task(num_warps=4, registers=168, name="consumer0"):
        ...
```

前端直接生成结构化 IR：

```mlir
ttg.warp_specialize(...) attributes {requestedRegisters = ...}
default {
  // producer
}
partition0(...) num_warps(4) {
  // consumer
}
```

这个模型更贴近 TritonGPU 上游的 region-based warp specialization：

- producer/consumer body 不需要经过 `tt.call` 边界；
- barrier、TMA、shared memory、WGMMA 协作协议都在同一个结构化 region 层级里；
- capture value 直接成为 `ttg.warp_specialize` operands / partition block args；
- partition-local `num_warps` 由 `partitionNumWarps` 表达。

## inline async_task 为什么暴露问题

inline API 会让 consumer ops 在 `ConvertTritonToTritonGPU` 阶段就已经位于
`ttg.warp_specialize.partitions` 内。

但很多 conversion pattern 仍沿用上游 type-only 写法：

```cpp
convertTypes(op->getResultTypes(), retTypes)
convertType(op.getType())
```

以 `tle.wgmma` 为例，如果它走 type-only conversion，TypeConverter 只看到：

```mlir
tensor<64x16xf32>
```

它看不到这个 result 属于 `partition0 num_warps(4)`，于是会使用 launch
`num_warps=8` 构造 layout。后续 verifier 使用 op context，可以正确查到该 op
位于 4-warp consumer partition，于是报 layout mismatch。

## 当前修复设计

当前修复方向是：保留上游式 contextual lookup 语义，只让 TLE result conversion
在 result 所属 op 位于 warp-specialize partition 内时传入 `OpResult`。如果不在
partition 内，仍然走原来的 type-only conversion。

### 1. 保留 value-aware TypeConverter 语义

`TritonGPUTypeConverter::getNumWarps(Value)` 仍然是 value layout 选择的入口：

- block argument：根据 owner region 调用 `lookupNumWarps(region)`；
- defining op：调用 `maybeLookupNumWarps(op)`；
- fallback：使用 TypeConverter 构造时的 launch/module `numWarps`。

这条机制已经能表达 partition 优先、module fallback，不需要再引入额外的
per-op `num-warps` attr 作为第二套真相来源。

### 2. 增加 TLE-only value-aware conversion helper

在 `lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp` 里增加一个
TLE-only helper。当前实现不再直接用
`getParentOfType<WarpSpecializePartitionsOp>()` 判断是否位于 partition 内，因为
这会绑定到当前 IR 的父 op 形态，容易在 clone、region 转移或 custom assembly
隐藏中间 op 时失效。新的 helper 改为复用
`TritonGPUTypeConverter::getNumWarps(Value)`，并和 converter 的全局
`getNumWarps()` 比较：

```cpp
static bool
needsTleContextualResultType(const TritonGPUTypeConverter *converter,
                             Value result) {
  return converter->getNumWarps(result) != converter->getNumWarps();
}
```

只有 value 所在上下文的 `num_warps` 和 launch/global `num_warps` 不同时，才调用
`convertType(Value)`；否则继续调用 `convertType(Type)`。这样普通路径更贴近上游
type-only 语义，而 partition-local 4-warps consumer 能拿到正确 layout。

这次 IR dump 还暴露了第二层问题：只修 result type 不够。MLIR dialect conversion
构造 adaptor operands 时，仍可能按照 type-only 目标类型插入 materialization。
在失败 IR 中，`scf.for` 的 loop-carried block arg 已经是 4-warps，但 adaptor 又
插入了：

```mlir
%acc_27 = ttg.convert_layout %acc_26
  : tensor<64x16xf32, #blocked1> -> tensor<64x16xf32, #blocked2>
```

其中 `#blocked1` 是 4-warps，`#blocked2` 是 launch 级 8-warps。随后内层
`tle.wgmma` 把 8-warps accumulator 当作 operand 使用，触发 verifier 报错。

因此 `GenericOpPattern` 的 TLE 分支也需要对 operands 做同样的 value-aware 目标
类型检查。如果 adaptor operand 的 tensor layout 和 `convertType(original Value)`
不一致，就插入一次 `ttg.convert_layout` 转回 value-aware 目标 layout：

```cpp
static LogicalResult
convertTleOperandTypes(const TritonGPUTypeConverter *converter,
                       ValueRange originalOperands,
                       SmallVectorImpl<Value> &operands,
                       ConversionPatternRewriter &rewriter) {
  ...
}
```

当前使用这些 helper 的位置包括：

- `GenericOpPattern`
- `TritonMapElementwisePattern`
- `SCFWhilePattern`

其他 TLE 分支里原先直接使用 `convertType(op.getResult())` 或
`convertType(result)` 的地方，也应通过 `convertTleResultType(...)` 走同样判断，
避免 partition 外行为偏离上游。

### 3. 保留 code_generator 的 caller_context 传递

inline async-task body 内部普通 op 应该通过 region lookup 获得 partition-local
`num_warps`。

但如果用户在 consumer task 内调用嵌套 JIT helper function，helper body 自身仍然
是 callee-like function，并不物理位于 consumer partition region 内。这个场景仍
需要 `WarpSpecializeCallerContext(task.num_warps)`。

因此 `python/triton/compiler/code_generator.py` 在访问 consumer `async_task` body
时传递 caller context。这个改动不是把新 API 退回旧 call-based API，而是保证
inline body 内部调用 JIT helper 时仍能继承 consumer 的 warp count。

## 为什么不优先采用 per-op num_warps attr

另一种方案是在 consumer partition 内每个 op 上打 attr，例如：

```mlir
{tle.num-warps = 4}
```

这个方案只有在 conversion 能拿到 op/value 时才有用：

```text
OpResult -> defining op -> tle.num-warps
```

它无法修复 type-only conversion：

```text
Type -> no parent op -> no attr to read
```

此外，per-op attr 会成为 `partitionNumWarps` 之外的第二套上下文缓存。后续 op
clone、move、inline、canonicalize 时都需要保证 attr 复制或刷新。当前更贴近
上游的做法是继续把 enclosing `ttg.warp_specialize` region 作为真相来源，并让
conversion 调用点传入 `OpResult`。

## 仍需审计的范围

当前 helper 修的是“给 op result 分配默认 layout”的路径，包括 TLE WGMMA 和大
多数 generic arith/math/triton ops。

仍需单独审计的是手工构造 layout 的 custom pattern。例如：

- `TritonDotPattern` 当前仍使用 `typeConverter->getNumWarps()`；
- `TritonSplitOpPattern` 当前用 converter 全局 warp count 构造
  `getDefaultBlockedEncoding(...)`；
- 其他直接调用 `getDefaultBlockedEncoding(...)` 的 custom pattern。

这些 pattern 如果未来允许出现在 consumer partition 内，应该改成使用 op/result
context，例如：

```cpp
lookupNumWarps(op)
```

或一个 value-aware helper，而不是引入新的全局 pass option。

## 验证

目标回归测试是：

```bash
python -m pytest python/test/tle/integration/test_tle_ws_tma_gemm.py
```

该测试同时覆盖 `launch_num_warps=4` 和 `launch_num_warps=8`，consumer task
固定为 `num_warps=4`。8-warp launch case 用来验证 consumer-local tensor
results 不会错误继承 launch warp count。
