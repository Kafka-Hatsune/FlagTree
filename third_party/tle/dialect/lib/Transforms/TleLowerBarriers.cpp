/*
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::triton::tle {

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

#define GEN_PASS_DEF_TRITONTLELOWERBARRIERS
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

static int64_t getI32Attr(Operation *op, StringRef name) {
  return op->getAttrOfType<IntegerAttr>(name).getInt();
}

static ttg::MemDescType getBarrierSlotType(ttg::MemDescType arrayTy) {
  auto context = arrayTy.getContext();
  auto ctaLayout = ttg::CTAEncodingAttr::getDefault(context, 1);
  Attribute slotEncoding =
      ttg::SwizzledSharedEncodingAttr::get(context, 1, 1, 1, {0}, ctaLayout);
  return ttg::MemDescType::get({1}, arrayTy.getElementType(), slotEncoding,
                               arrayTy.getMemorySpace(),
                               arrayTy.getMutableMemory());
}

static Value createBarrierSlot(OpBuilder &builder, Location loc, Value array,
                               int64_t index) {
  auto arrayTy = cast<ttg::MemDescType>(array.getType());
  auto slotTy = getBarrierSlotType(arrayTy);
  Value idx = builder.create<arith::ConstantIntOp>(loc, index, 32);
  return builder.create<ttg::MemDescIndexOp>(loc, slotTy, array, idx);
}

// Trace a barrier slot passed into an isolated warp-specialization partition
// back to its source allocation.  The partition capture is deliberately still
// present when this pass runs; canonicalization removes unused captures only
// after named barrier ops have been lowered.
static BarrierAllocOp findBarrierAlloc(Value value) {
  while (value) {
    if (auto view = value.getDefiningOp<ttg::MemDescIndexOp>()) {
      value = view.getSrc();
      continue;
    }
    auto blockArg = dyn_cast<BlockArgument>(value);
    if (!blockArg)
      return value.getDefiningOp<BarrierAllocOp>();
    auto partitions = dyn_cast<ttg::WarpSpecializePartitionsOp>(
        blockArg.getOwner()->getParentOp());
    if (!partitions)
      return {};
    auto warpSpecialize =
        dyn_cast<ttg::WarpSpecializeOp>(partitions->getParentOp());
    if (!warpSpecialize)
      return {};
    OperandRange captures = warpSpecialize.getExplicitCaptures();
    if (blockArg.getArgNumber() >= captures.size())
      return {};
    value = captures[blockArg.getArgNumber()];
  }
  return {};
}

// Participant TLE mbarrier arrivals have logical warp-partition semantics:
// every thread in the partition contributes a unit arrival. lookupNumWarps
// handles both the default warp-specialize region (the module's ttg.num-warps)
// and a worker partition (its partition-specific num_warps).
static int64_t getPartitionThreadCount(Operation *op, ModuleOp module) {
  return static_cast<int64_t>(ttg::lookupNumWarps(op)) *
         ttg::TritonGPUDialect::getThreadsPerWarp(module);
}

#if !defined(__HCU__)
static std::pair<Value, Value>
createNamedBarrierOperands(OpBuilder &builder, Location loc, Operation *op) {
  Value id =
      builder.create<arith::ConstantIntOp>(loc, getI32Attr(op, "named_id"), 32);
  Value threads = builder.create<arith::ConstantIntOp>(
      loc, getI32Attr(op, "named_num_threads"), 32);
  return {id, threads};
}
#endif

struct TritonTleLowerBarriers
    : public impl::TritonTleLowerBarriersBase<TritonTleLowerBarriers> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<BarrierWaitOp> waits;
    SmallVector<BarrierArriveOp> arrives;
    SmallVector<BarrierAllocOp> allocs;
    module.walk([&](Operation *op) {
      if (auto wait = dyn_cast<BarrierWaitOp>(op))
        waits.push_back(wait);
      else if (auto arrive = dyn_cast<BarrierArriveOp>(op))
        arrives.push_back(arrive);
      else if (auto alloc = dyn_cast<BarrierAllocOp>(op))
        allocs.push_back(alloc);
    });

    // Named barriers are hardware resources identified only by an integer id;
    // their memdesc operands are frontend SSA handles, not shared-memory
    // storage.  Remember which source allocations are named-only before
    // erasing the wait/arrive ops.  Otherwise a capture through
    // ttg.warp_specialize keeps the handle artificially live and the generic
    // allocation lowering emits an mbarrier.init into unrelated shared memory.
    llvm::DenseSet<Operation *> namedAllocs;
    llvm::DenseSet<Operation *> mbarrierAllocs;
    auto recordBackend = [&](Operation *op, Value barrier) -> LogicalResult {
      auto alloc = findBarrierAlloc(barrier);
      if (!alloc)
        return success();
      StringRef backend = op->getAttrOfType<StringAttr>("backend").getValue();
      if (backend == "named" && alloc.getArrivalMode() == "participant") {
        op->emitError("arrival_mode 'participant' does not support the named "
                      "barrier backend");
        return failure();
      }
      (backend == "named" ? namedAllocs : mbarrierAllocs)
          .insert(alloc.getOperation());
      return success();
    };
    for (BarrierWaitOp op : waits) {
      if (failed(recordBackend(op.getOperation(), op.getBarrier()))) {
        signalPassFailure();
        return;
      }
    }
    for (BarrierArriveOp op : arrives) {
      if (failed(recordBackend(op.getOperation(), op.getBarrier()))) {
        signalPassFailure();
        return;
      }
    }

    // A TLE BarrierAllocOp stores a logical expected-arrival count. Direct
    // phase-indexed participant arrivals require initialization with
    // logical_count * partition_threads. Elected and transaction/TMA barriers
    // retain their logical count.
    llvm::DenseMap<Operation *, int64_t> participantWidths;
    for (BarrierArriveOp op : arrives) {
      StringRef backend = op->getAttrOfType<StringAttr>("backend").getValue();
      if (backend != "mbarrier")
        continue;

      BarrierAllocOp alloc = findBarrierAlloc(op.getBarrier());
      if (!alloc) {
        op.emitOpError("cannot resolve direct mbarrier arrival mode from its "
                       "TLE barrier allocation");
        signalPassFailure();
        return;
      }
      if (alloc.getExpectBytesAttr()) {
        op.emitOpError("direct arrival cannot target a TMA transaction "
                       "barrier allocated with expect_bytes");
        signalPassFailure();
        return;
      }
      if (alloc.getArrivalMode() != "participant")
        continue;

      int64_t width = getPartitionThreadCount(op.getOperation(), module);
      auto [it, inserted] =
          participantWidths.try_emplace(alloc.getOperation(), width);
      if (!inserted && it->second != width) {
        op.emitOpError("all direct arrivals for one barrier allocation must "
                       "execute in partitions with the same thread count; "
                       "previous width was ")
            << it->second << ", current width is " << width;
        signalPassFailure();
        return;
      }
    }

    for (BarrierWaitOp op : waits) {
      OpBuilder builder(op);
      Location loc = op.getLoc();
      StringRef backend = op->getAttrOfType<StringAttr>("backend").getValue();
      if (backend == "mbarrier") {
        builder.create<ttng::WaitBarrierOp>(loc, op.getBarrier(),
                                            op.getPhase());
      } else {
#if defined(__HCU__)
        op.emitOpError("named barrier lowering is only supported on NVIDIA "
                       "backend");
        signalPassFailure();
        return;
#else
        auto [id, threads] =
            createNamedBarrierOperands(builder, loc, op.getOperation());
        builder.create<ttng::NamedBarrierWaitOp>(loc, id, threads);
#endif
      }
      op.erase();
    }

    for (BarrierArriveOp op : arrives) {
      OpBuilder builder(op);
      Location loc = op.getLoc();
      StringRef backend = op->getAttrOfType<StringAttr>("backend").getValue();
      if (backend == "mbarrier") {
        int64_t logicalCount = getI32Attr(op.getOperation(), "arrive_count");
        BarrierAllocOp alloc = findBarrierAlloc(op.getBarrier());
        if (alloc.getArrivalMode() == "participant") {
          int64_t participantCount = participantWidths.lookup(alloc);
          for (int64_t i = 0; i < logicalCount; ++i) {
            auto arrive = builder.create<ttng::ArriveBarrierOp>(
                loc, op.getBarrier(), static_cast<uint32_t>(participantCount));
            arrive.setReleaseFence(true);
            arrive.setParticipantArrive(true);
          }
        } else {
          builder.create<ttng::ArriveBarrierOp>(
              loc, op.getBarrier(), static_cast<uint32_t>(logicalCount));
        }
      } else {
#if defined(__HCU__)
        op.emitOpError("named barrier lowering is only supported on NVIDIA "
                       "backend");
        signalPassFailure();
        return;
#else
        auto [id, threads] =
            createNamedBarrierOperands(builder, loc, op.getOperation());
        builder.create<ttng::NamedBarrierArriveOp>(loc, id, threads);
#endif
      }
      op.erase();
    }

    for (BarrierAllocOp op : allocs) {
      OpBuilder builder(op);
      Location loc = op.getLoc();
      auto arrayTy = op.getResult().getType();
      bool onlyUnusedViews = true;
      SmallVector<ttg::MemDescIndexOp> deadViews;
      for (OpOperand &use : op.getResult().getUses()) {
        auto view = dyn_cast<ttg::MemDescIndexOp>(use.getOwner());
        if (!view || !view->use_empty()) {
          onlyUnusedViews = false;
          break;
        }
        deadViews.push_back(view);
      }
      if (onlyUnusedViews) {
        for (auto view : deadViews)
          view.erase();
        op.erase();
        continue;
      }

      Value alloc = builder.create<ttg::LocalAllocOp>(loc, arrayTy);
      bool namedOnly = namedAllocs.contains(op.getOperation()) &&
                       !mbarrierAllocs.contains(op.getOperation());
      int64_t numBarriers = getI32Attr(op.getOperation(), "num_barriers");
      int64_t arriveCount = getI32Attr(op.getOperation(), "arrive_count");
      if (auto it = participantWidths.find(op.getOperation());
          it != participantWidths.end()) {
        arriveCount *= it->second;
      }
      if (!namedOnly) {
        for (int64_t i = 0; i < numBarriers; ++i) {
          Value slot = createBarrierSlot(builder, loc, alloc, i);
          builder.create<ttng::InitBarrierOp>(
              loc, slot, static_cast<uint32_t>(arriveCount));
        }
      }
      op.getResult().replaceAllUsesWith(alloc);
      op.erase();
    }
  }
};

} // namespace

} // namespace mlir::triton::tle
