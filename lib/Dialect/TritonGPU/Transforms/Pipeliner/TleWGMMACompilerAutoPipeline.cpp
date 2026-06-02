#ifdef __TLE__
#include "TleWGMMAAnalysis.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include <algorithm>
#include <cassert>
#include <optional>

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton::gpu::detail {

static constexpr llvm::StringLiteral
    kTleExplicitWgmmaCommitAttr("tle.explicit_wgmma_commit");
static constexpr llvm::StringLiteral
    kTleWgmmaAccumulatorChainCAttr("tle.wgmma_accumulator_chain_c");

static void threadValuesThroughWait(ttng::WarpGroupDotWaitOp wait,
                                    MutableArrayRef<Value> values) {
  IRRewriter builder(wait.getContext());
  builder.setInsertionPoint(wait);

  size_t origNumOperands = wait.getNumOperands();
  SetVector<Value> newOperands(wait.getOperands().begin(),
                               wait.getOperands().end());
  assert(newOperands.size() == origNumOperands &&
         "Wait op has duplicate operands.");

  newOperands.insert(values.begin(), values.end());

  SmallVector<ttng::WarpGroupDotOp> asyncDots;
  for (Value v : values) {
    BackwardSliceOptions options;
    options.omitBlockArguments = true;
    options.filter = [&](Operation *op) {
      if (auto dot = dyn_cast<ttng::WarpGroupDotOp>(op)) {
        asyncDots.push_back(dot);
        return false;
      }
      return op->getBlock() == wait->getBlock();
    };
    SetVector<Operation *> slice;
    (void)getBackwardSlice(v, &slice, options);
  }

  for (ttng::WarpGroupDotOp dot : asyncDots) {
    for (Value operand : dot.getOperands()) {
      if (isa<ttg::MemDescType>(operand.getType()))
        newOperands.insert(operand);
    }
  }

  auto newWait = ttng::WarpGroupDotWaitOp::create(
      builder, wait.getLoc(), llvm::to_vector(newOperands), wait.getPendings());

  auto dominatedByNewWait = [&](OpOperand &operand) {
    auto opInThisBlock =
        newWait->getBlock()->findAncestorOpInBlock(*operand.getOwner());
    return opInThisBlock && newWait->isBeforeInBlock(opInThisBlock);
  };
  for (int i = 0; i < origNumOperands; i++) {
    Value operand = wait.getResult(i);
    if (!isa<ttg::MemDescType>(operand.getType()))
      operand.replaceAllUsesWith(newWait.getResult(i));
  }
  for (int i = origNumOperands; i < newOperands.size(); i++) {
    Value operand = newWait.getOperand(i);
    if (!isa<ttg::MemDescType>(operand.getType()))
      operand.replaceUsesWithIf(newWait.getResult(i), dominatedByNewWait);
  }
  wait->erase();
}

static Value getWarpGroupDotWaitSource(Value value) {
  while (auto result = dyn_cast<OpResult>(value)) {
    auto wait = dyn_cast<ttng::WarpGroupDotWaitOp>(result.getOwner());
    if (!wait)
      break;
    unsigned resultNo = result.getResultNumber();
    if (resultNo >= wait.getNumOperands())
      break;
    value = wait.getOperand(resultNo);
  }
  return value;
}

static Value skipNoopDefs(Value value) {
  while (Operation *def = value.getDefiningOp()) {
    if (!isNoop(def) || def->getNumOperands() != 1 || def->getNumResults() != 1)
      break;
    value = def->getOperand(0);
  }
  return value;
}

static ttng::WarpGroupDotOp getAccumulatorChainSourceDot(Value value) {
  return skipNoopDefs(value).getDefiningOp<ttng::WarpGroupDotOp>();
}

static bool valueDependsOn(Value value, Value root,
                           llvm::SmallDenseSet<Value, 8> &visited) {
  value = getWarpGroupDotWaitSource(value);
  if (!visited.insert(value).second)
    return false;
  if (value == root)
    return true;

  if (Operation *def = value.getDefiningOp()) {
    if (isNoop(def)) {
      for (Value operand : def->getOperands()) {
        if (valueDependsOn(operand, root, visited))
          return true;
      }
      return false;
    }

    if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
      auto result = cast<OpResult>(value);
      unsigned resultNo = result.getResultNumber();
      if (resultNo < ifOp.thenYield().getNumOperands() &&
          valueDependsOn(ifOp.thenYield().getOperand(resultNo), root, visited))
        return true;
      if (resultNo < ifOp.elseYield().getNumOperands() &&
          valueDependsOn(ifOp.elseYield().getOperand(resultNo), root, visited))
        return true;
    }
  }

  return false;
}

static bool valueDependsOn(Value value, Value root) {
  llvm::SmallDenseSet<Value, 8> visited;
  return valueDependsOn(value, root, visited);
}

static bool containsPendingDot(ArrayRef<PendingSharedWgmmaGroup> pendingGroups,
                               Operation *dotOp) {
  return llvm::any_of(pendingGroups, [&](const PendingSharedWgmmaGroup &group) {
    return llvm::any_of(group.dots, [&](ttng::WarpGroupDotOp dot) {
      return dot.getOperation() == dotOp;
    });
  });
}

static std::optional<unsigned>
findPendingGroupForValue(Value value,
                         ArrayRef<PendingSharedWgmmaGroup> pendingGroups) {
  for (auto indexed : llvm::enumerate(pendingGroups)) {
    for (ttng::WarpGroupDotOp dot : indexed.value().dots) {
      if (valueDependsOn(value, dot.getResult()))
        return indexed.index();
    }
  }
  return std::nullopt;
}

static bool
isAllowedAccumulatorChainUse(Operation *op, OpOperand &use,
                             const TleWgmmaScheduleAnalysis &analysis) {
  auto dot = dyn_cast<ttng::WarpGroupDotOp>(op);
  return dot && use.getOperandNumber() == 2 &&
         analysis.canReuseAccumulatorChainC(dot);
}

static std::optional<unsigned> findLastMaterializedPendingGroup(
    Operation *op, ArrayRef<PendingSharedWgmmaGroup> pendingGroups,
    const TleWgmmaScheduleAnalysis &analysis,
    SmallVectorImpl<Value> &materializedValues) {
  std::optional<unsigned> lastIndex;
  for (OpOperand &use : op->getOpOperands()) {
    Value operand = use.get();
    if (isa<ttg::MemDescType>(operand.getType()))
      continue;
    if (isAllowedAccumulatorChainUse(op, use, analysis))
      continue;

    std::optional<unsigned> index =
        findPendingGroupForValue(operand, pendingGroups);
    if (!index)
      continue;

    materializedValues.push_back(operand);
    if (!lastIndex || *lastIndex < *index)
      lastIndex = index;
  }
  return lastIndex;
}

static ttng::WarpGroupDotWaitOp
insertWgmmaWaitBefore(Operation *op, unsigned lastCompletedGroup,
                      SmallVectorImpl<PendingSharedWgmmaGroup> &pendingGroups,
                      ArrayRef<Value> forwardedValues) {
  assert(lastCompletedGroup < pendingGroups.size() &&
         "expected a valid pending group index");
  unsigned pendings = pendingGroups.size() - lastCompletedGroup - 1;
  IRRewriter builder(op->getContext());
  builder.setInsertionPoint(op);

  auto wait = ttng::WarpGroupDotWaitOp::create(builder, op->getLoc(),
                                               ArrayRef<Value>{}, pendings);

  SmallVector<Value, 4> waitOperands(forwardedValues.begin(),
                                     forwardedValues.end());
  if (waitOperands.empty()) {
    for (unsigned i = 0; i <= lastCompletedGroup; ++i) {
      assert(!pendingGroups[i].dots.empty() &&
             "pending WGMMA group must contain at least one dot");
      waitOperands.push_back(pendingGroups[i].dots.back().getResult());
    }
  }
  threadValuesThroughWait(wait, waitOperands);

  pendingGroups.erase(pendingGroups.begin(),
                      pendingGroups.begin() + lastCompletedGroup + 1);
  return wait;
}

static void
consumeExistingWait(ttng::WarpGroupDotWaitOp wait,
                    SmallVectorImpl<PendingSharedWgmmaGroup> &pendingGroups) {
  unsigned completedCount =
      pendingGroups.size() -
      std::min<unsigned>(pendingGroups.size(), wait.getPendings());
  pendingGroups.erase(pendingGroups.begin(),
                      pendingGroups.begin() + completedCount);
}

static void drainForMaterializedOperands(
    Operation *op, const TleWgmmaScheduleAnalysis &analysis,
    SmallVectorImpl<PendingSharedWgmmaGroup> &pendingGroups) {
  SmallVector<Value, 4> materializedValues;
  std::optional<unsigned> lastIndex = findLastMaterializedPendingGroup(
      op, pendingGroups, analysis, materializedValues);
  if (!lastIndex)
    return;
  insertWgmmaWaitBefore(op, *lastIndex, pendingGroups, materializedValues);
}

static void drainForLifetimeBoundary(
    Operation *op, const TlePipeResourceAnalysis &resources,
    SmallVectorImpl<PendingSharedWgmmaGroup> &pendingGroups) {
  if (!resources.isLifetimeBoundary(op))
    return;

  SmallVector<MemDescResource, 2> releasedResources =
      resources.getBoundaryReleasedResources(op);
  std::optional<unsigned> lastConflictIndex;
  for (auto indexed : llvm::enumerate(pendingGroups)) {
    if (resources.releasedResourcesMayAliasReads(releasedResources,
                                                 indexed.value().reads))
      lastConflictIndex = indexed.index();
  }
  if (!lastConflictIndex)
    return;

  insertWgmmaWaitBefore(op, *lastConflictIndex, pendingGroups,
                        /*forwardedValues=*/{});
}

static bool
canAppendToPendingGroup(ttng::WarpGroupDotOp dot,
                        ArrayRef<PendingSharedWgmmaGroup> pendingGroups,
                        const TleWgmmaScheduleAnalysis &analysis) {
  if (!analysis.canAppendToCurrentWgmmaCommitGroup(dot) ||
      pendingGroups.empty())
    return false;

  ttng::WarpGroupDotOp sourceDot = getAccumulatorChainSourceDot(dot.getC());
  if (!sourceDot)
    return false;
  const PendingSharedWgmmaGroup &tailGroup = pendingGroups.back();
  return llvm::is_contained(tailGroup.dots, sourceDot);
}

static void
recordPendingDot(ttng::WarpGroupDotOp dot,
                 const TlePipeResourceAnalysis &resources,
                 const TleWgmmaScheduleAnalysis &analysis,
                 SmallVectorImpl<PendingSharedWgmmaGroup> &pendingGroups) {
  dot.setIsAsync(true);

  if (analysis.canReuseAccumulatorChainC(dot))
    dot->setAttr(kTleWgmmaAccumulatorChainCAttr,
                 UnitAttr::get(dot.getContext()));

  if (!canAppendToPendingGroup(dot, pendingGroups, analysis))
    pendingGroups.push_back(PendingSharedWgmmaGroup{});

  pendingGroups.back().dots.push_back(dot);
  llvm::append_range(pendingGroups.back().reads,
                     resources.getDotReadResources(dot));
}

static void scheduleTleWgmmaWaitsInBlock(
    Block *block, const TlePipeResourceAnalysis &resources,
    const TleWgmmaScheduleAnalysis &analysis,
    SmallVectorImpl<PendingSharedWgmmaGroup> &pendingGroups) {
  for (Operation &bodyOp : llvm::make_early_inc_range(*block)) {
    Operation *op = &bodyOp;
    if (op->hasTrait<OpTrait::IsTerminator>())
      break;

    if (auto wait = dyn_cast<ttng::WarpGroupDotWaitOp>(op)) {
      consumeExistingWait(wait, pendingGroups);
      continue;
    }

    if (auto dot = dyn_cast<ttng::WarpGroupDotOp>(op)) {
      drainForMaterializedOperands(op, analysis, pendingGroups);
      recordPendingDot(dot, resources, analysis, pendingGroups);
      continue;
    }

    drainForMaterializedOperands(op, analysis, pendingGroups);
    drainForLifetimeBoundary(op, resources, pendingGroups);

    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      SmallVector<PendingSharedWgmmaGroup, 4> incomingPending(
          pendingGroups.begin(), pendingGroups.end());
      SmallVector<PendingSharedWgmmaGroup, 4> thenPending(
          incomingPending.begin(), incomingPending.end());
      SmallVector<PendingSharedWgmmaGroup, 4> elsePending(
          incomingPending.begin(), incomingPending.end());

      scheduleTleWgmmaWaitsInBlock(ifOp.thenBlock(), resources, analysis,
                                   thenPending);
      if (Block *elseBlock = ifOp.elseBlock())
        scheduleTleWgmmaWaitsInBlock(elseBlock, resources, analysis,
                                     elsePending);

      pendingGroups.clear();
      for (PendingSharedWgmmaGroup &pending : incomingPending) {
        bool remainsPending = llvm::any_of(pending.dots, [&](auto dot) {
          return containsPendingDot(thenPending, dot.getOperation()) ||
                 containsPendingDot(elsePending, dot.getOperation());
        });
        if (remainsPending)
          pendingGroups.push_back(pending);
      }
      continue;
    }
  }

  Operation *terminator = block->getTerminator();
  if (!terminator)
    return;
  drainForMaterializedOperands(terminator, analysis, pendingGroups);
  if (!pendingGroups.empty())
    insertWgmmaWaitBefore(terminator, pendingGroups.size() - 1, pendingGroups,
                          /*forwardedValues=*/{});
}

static void
markTleExplicitWgmmaCommitGroups(scf::ForOp forOp,
                                 const TleWgmmaScheduleAnalysis &analysis) {
  IRRewriter builder(forOp.getContext());
  SmallVector<ttng::WarpGroupDotOp, 8> dots;
  forOp.getBody()->walk([&](ttng::WarpGroupDotOp dot) {
    if (dot->getParentOfType<scf::ForOp>() == forOp)
      dots.push_back(dot);
  });

  for (ttng::WarpGroupDotOp dot : llvm::make_early_inc_range(dots)) {
    dot->setAttr(kTleExplicitWgmmaCommitAttr, builder.getUnitAttr());

    Operation *next = dot->getNextNode();
    if (next && isa<ttng::WarpGroupDotCommitOp>(next))
      continue;

    if (analysis.canDeferCommitToLaterDotC(dot))
      continue;

    builder.setInsertionPointAfter(dot);
    ttng::WarpGroupDotCommitOp::create(builder, dot.getLoc());
  }
}

void scheduleTleWgmmaCompilerAutoPipeline(scf::ForOp forOp) {
  TlePipeResourceAnalysis resources;
  TleWgmmaScheduleAnalysis analysis(forOp, resources);
  SmallVector<PendingSharedWgmmaGroup, 4> pendingGroups;

  scheduleTleWgmmaWaitsInBlock(forOp.getBody(), resources, analysis,
                               pendingGroups);
  markTleExplicitWgmmaCommitGroups(forOp, analysis);
}

} // namespace mlir::triton::gpu::detail
#endif // __TLE__
