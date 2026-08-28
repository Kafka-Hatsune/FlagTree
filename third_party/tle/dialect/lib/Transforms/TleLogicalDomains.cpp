/*
 * Copyright 2025- FlagOS Contributors
 * SPDX-License-Identifier: MIT
 */

#include "tle/dialect/include/Transforms/LogicalDomain.h"

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "tle/dialect/include/IR/ExactSMEM.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include <limits>
#include <numeric>

namespace mlir::triton::tle {
namespace ttg = mlir::triton::gpu;

namespace {
constexpr llvm::StringLiteral kLogicalAllocShape("tle.logical_alloc_shape");
constexpr llvm::StringLiteral kLogicalNonPowerAxis(
    "tle.logical_non_power_axis");
constexpr llvm::StringLiteral kStoragePlan("tle.storage_plan");

static RankedTensorType getTensorType(Value value) {
  return dyn_cast<RankedTensorType>(value.getType());
}

static SmallVector<int32_t> identityAxisMap(unsigned rank) {
  SmallVector<int32_t> result;
  for (unsigned i = 0; i < rank; ++i)
    result.push_back(i);
  return result;
}

static LogicalShape intersectShapes(ArrayRef<int64_t> lhs,
                                    ArrayRef<int64_t> rhs) {
  assert(lhs.size() == rhs.size());
  LogicalShape result;
  for (auto [a, b] : llvm::zip(lhs, rhs))
    result.push_back(std::min(a, b));
  return result;
}

static int64_t nextPowerOfTwo(int64_t value) {
  int64_t result = 1;
  while (result < value)
    result <<= 1;
  return result;
}

static int64_t largestPowerOfTwoDivisorNoGreaterThan(int64_t value,
                                                     int64_t upperBound) {
  assert(value > 0 && upperBound > 0);
  int64_t divisor = 1;
  while (divisor <= upperBound / 2 && value % (divisor * 2) == 0)
    divisor *= 2;
  return divisor;
}

static bool isGlobalPointerTensor(Value value) {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorType)
    return false;
  auto pointerType = dyn_cast<triton::PointerType>(tensorType.getElementType());
  return pointerType && pointerType.getAddressSpace() == 1;
}

static LogicalRootRewriteAction *findRootAction(LogicalDomainPlan &plan,
                                                Operation *root) {
  for (LogicalRootRewriteAction &action : plan.roots)
    if (action.alloc.getOperation() == root)
      return &action;
  return nullptr;
}

static LogicalResult
selectRootStorageTile(LogicalRootRewriteAction &root,
                      ArrayRef<int64_t> storageTile, Operation *anchor) {
  if (root.storageTileShape.empty()) {
    root.storageTileShape.assign(storageTile.begin(), storageTile.end());
    return success();
  }
  if (ArrayRef<int64_t>(root.storageTileShape) != storageTile)
    return anchor->emitOpError(
        "all producers of one logical root must use the same exact-SMEM "
        "storage tile");
  return success();
}

static LogicalDomainProvenance
mergeProvenance(LogicalDomainProvenance lhs, LogicalDomainProvenance rhs) {
  for (Operation *root : rhs.roots)
    if (!llvm::is_contained(lhs.roots, root))
      lhs.roots.push_back(root);
  for (Operation *seed : rhs.seeds)
    if (!llvm::is_contained(lhs.seeds, seed))
      lhs.seeds.push_back(seed);
  return lhs;
}

static bool mergeProvenanceInto(LogicalDomainProvenance &target,
                                const LogicalDomainProvenance &source) {
  size_t oldRootCount = target.roots.size();
  size_t oldSeedCount = target.seeds.size();
  target = mergeProvenance(std::move(target), source);
  return target.roots.size() != oldRootCount ||
         target.seeds.size() != oldSeedCount;
}

static FailureOr<LogicalReductionIdentity>
getReductionIdentity(triton::ReduceOp op) {
  Operation *combiner = op.getSingleCombiner();
  if (!combiner)
    return failure();
  return llvm::StringSwitch<FailureOr<LogicalReductionIdentity>>(
             combiner->getName().getStringRef())
      .Cases("arith.maxnumf", "arith.maximumf",
             LogicalReductionIdentity::NegativeInfinity)
      .Cases("arith.minnumf", "arith.minimumf",
             LogicalReductionIdentity::PositiveInfinity)
      .Cases("arith.addf", "arith.addi", LogicalReductionIdentity::Zero)
      .Cases("arith.mulf", "arith.muli", LogicalReductionIdentity::One)
      .Case("arith.andi", LogicalReductionIdentity::True)
      .Case("arith.ori", LogicalReductionIdentity::False)
      .Default(failure());
}

static bool validatePrefixShape(ArrayRef<int64_t> physical,
                                ArrayRef<int64_t> logical) {
  if (physical.size() != logical.size())
    return false;
  for (auto [p, l] : llvm::zip(physical, logical))
    if (l <= 0 || l > p)
      return false;
  return true;
}

static std::optional<int64_t>
getContiguousLinearPrefixSize(ArrayRef<int64_t> physical,
                              ArrayRef<int64_t> logical) {
  if (!validatePrefixShape(physical, logical))
    return std::nullopt;
  int restrictedAxis = -1;
  for (int i = 0, e = physical.size(); i < e; ++i) {
    if (logical[i] == physical[i])
      continue;
    if (restrictedAxis >= 0)
      return std::nullopt;
    restrictedAxis = i;
  }
  if (restrictedAxis >= 0) {
    for (int i = 0; i < restrictedAxis; ++i)
      if (logical[i] != 1)
        return std::nullopt;
    for (int i = restrictedAxis + 1, e = physical.size(); i < e; ++i)
      if (logical[i] != physical[i])
        return std::nullopt;
  }
  return std::accumulate(logical.begin(), logical.end(), int64_t{1},
                         std::multiplies<int64_t>());
}

static std::optional<LogicalShape>
representLinearPrefixAsRectangle(ArrayRef<int64_t> target,
                                 int64_t validElements) {
  int64_t suffix = 1;
  for (int axis = target.size() - 1; axis >= 0; --axis) {
    if (validElements % suffix == 0) {
      int64_t extent = validElements / suffix;
      if (extent > 0 && extent <= target[axis]) {
        LogicalShape result(target.size(), 1);
        result[axis] = extent;
        for (int i = axis + 1, e = target.size(); i < e; ++i)
          result[i] = target[i];
        return result;
      }
    }
    suffix *= target[axis];
  }
  return std::nullopt;
}

static std::optional<LogicalShape>
reshapeOnlyUnitDimensions(ArrayRef<int64_t> sourcePhysical,
                          ArrayRef<int64_t> sourceLogical,
                          ArrayRef<int64_t> targetPhysical) {
  SmallVector<int64_t> sourceNonUnitPhysical;
  SmallVector<int64_t> sourceNonUnitLogical;
  for (auto [physical, logical] :
       llvm::zip(sourcePhysical, sourceLogical)) {
    if (physical == 1) {
      if (logical != 1)
        return std::nullopt;
      continue;
    }
    sourceNonUnitPhysical.push_back(physical);
    sourceNonUnitLogical.push_back(logical);
  }
  SmallVector<int64_t> targetNonUnitPhysical;
  for (int64_t physical : targetPhysical)
    if (physical != 1)
      targetNonUnitPhysical.push_back(physical);
  if (sourceNonUnitPhysical != targetNonUnitPhysical)
    return std::nullopt;
  LogicalShape result;
  unsigned nextLogical = 0;
  for (int64_t physical : targetPhysical)
    result.push_back(physical == 1 ? 1 : sourceNonUnitLogical[nextLogical++]);
  return result;
}

} // namespace

FailureOr<SmallVector<int64_t, 2>>
selectLogicalSMEMStorageTileShape(ttg::MemDescType stageType,
                                  ArrayRef<int64_t> logicalShape) {
  if (stageType.getRank() != 2 || logicalShape.size() != 2)
    return failure();
  if (!isa<ttg::NVMMASharedEncodingAttr>(stageType.getEncoding()))
    return failure();

  bool fragmentedRows = !llvm::isPowerOf2_64(logicalShape[0]);
  bool fragmentedCols = !llvm::isPowerOf2_64(logicalShape[1]);
  if (fragmentedRows == fragmentedCols)
    return failure();
  unsigned fragmentAxis = fragmentedRows ? 0u : 1u;
  SmallVector<int64_t, 2> storageTile(logicalShape.begin(),
                                      logicalShape.end());
  storageTile[fragmentAxis] = largestPowerOfTwoDivisorNoGreaterThan(
      logicalShape[fragmentAxis], logicalShape[fragmentAxis]);
  if (storageTile[fragmentAxis] < kExactSMEMFragmentQuantum ||
      logicalShape[fragmentAxis] % storageTile[fragmentAxis] != 0)
    return failure();
  return storageTile;
}

namespace {

enum class LogicalDomainPhase : uint8_t { Propagate, Plan };

/// Internal compatibility facade for the existing transfer and planning
/// helpers. Tensor propagation itself is driven by TensorLogicalDomainAnalysis
/// below; this facade owns the finalized facts and the rewrite plan only.
class LogicalDomainContext {
public:
  explicit LogicalDomainContext(LogicalDomainPlan &plan);

  const MemDescLogicalState *lookupMemDesc(Value value) const;
  const TensorFragmentState *lookupTensor(Value value) const;
  bool mergeMemDesc(Value value, const MemDescLogicalState &state);
  bool hasRestrictedOperand(Operation *op) const;
  SmallVector<Value> takeChangedValues();
  LogicalResult validateTensorResult(Operation *op) const;

  LogicalResult processCandidateAlloc(Operation *op,
                                      LogicalDomainPhase phase);
  LogicalResult processMemDescIndex(Operation *op,
                                    LogicalDomainPhase phase);
  LogicalResult processLogicalTMACopy(Operation *op,
                                      LogicalDomainPhase phase);
  LogicalResult processLogicalPointerCopy(Operation *op,
                                          LogicalDomainPhase phase);
  LogicalResult processWarpSpecialize(Operation *op,
                                      LogicalDomainPhase phase);
  LogicalResult processPipe(Operation *op, LogicalDomainPhase phase);
  LogicalResult processMemDescUse(Operation *op, LogicalDomainPhase phase);
  LogicalResult processMemDescTranspose(Operation *op,
                                        LogicalDomainPhase phase);
  LogicalResult processWGMMA(Operation *op, LogicalDomainPhase phase);
  LogicalResult processWGMMAWait(Operation *op, LogicalDomainPhase phase);
  LogicalResult processSameShape(Operation *op, LogicalDomainPhase phase);
  LogicalResult processExpandDims(Operation *op, LogicalDomainPhase phase);
  LogicalResult processBroadcast(Operation *op, LogicalDomainPhase phase);
  LogicalResult processTranspose(Operation *op, LogicalDomainPhase phase);
  LogicalResult processReshape(Operation *op, LogicalDomainPhase phase);
  LogicalResult processCat(Operation *op, LogicalDomainPhase phase);
  LogicalResult processJoin(Operation *op, LogicalDomainPhase phase);
  LogicalResult processSplit(Operation *op, LogicalDomainPhase phase);
  LogicalResult processReduce(Operation *op, LogicalDomainPhase phase);
  LogicalResult processDot(Operation *op, LogicalDomainPhase phase);
  LogicalResult processStore(Operation *op, LogicalDomainPhase phase);
  LogicalResult processAtomicRMW(Operation *op, LogicalDomainPhase phase);
  LogicalResult processRejected(Operation *op, LogicalDomainPhase phase,
                                StringRef reason);

  LogicalResult emitError(Operation *op, unsigned operandIndex,
                          const Twine &reason) const;

private:
  void markChanged(Value value);

  LogicalDomainPlan &plan;
  SmallVector<Value, 8> changedValues;
};

enum class LogicalBehavior : uint8_t {
  CandidateAlloc,
  MemDescIndex,
  LogicalTMACopy,
  LogicalPointerCopy,
  WarpSpecialize,
  Pipe,
  MemDescUse,
  MemDescTranspose,
  WGMMA,
  WGMMAWait,
  ExpandDims,
  Broadcast,
  Transpose,
  Reshape,
  Cat,
  Join,
  Split,
  Reduce,
  Dot,
  Store,
  AtomicRMW,
  RejectGather,
  RejectHistogram,
  RejectAtomicCAS,
  RejectLoad,
  RejectEscape,
  RejectScan,
};

static LogicalResult dispatchLogicalBehavior(LogicalDomainContext &context,
                                             Operation *op,
                                             LogicalDomainPhase phase,
                                             LogicalBehavior behavior) {
  switch (behavior) {
  case LogicalBehavior::CandidateAlloc:
    return context.processCandidateAlloc(op, phase);
  case LogicalBehavior::MemDescIndex:
    return context.processMemDescIndex(op, phase);
  case LogicalBehavior::LogicalTMACopy:
    return context.processLogicalTMACopy(op, phase);
  case LogicalBehavior::LogicalPointerCopy:
    return context.processLogicalPointerCopy(op, phase);
  case LogicalBehavior::WarpSpecialize:
    return context.processWarpSpecialize(op, phase);
  case LogicalBehavior::Pipe:
    return context.processPipe(op, phase);
  case LogicalBehavior::MemDescUse:
    return context.processMemDescUse(op, phase);
  case LogicalBehavior::MemDescTranspose:
    return context.processMemDescTranspose(op, phase);
  case LogicalBehavior::WGMMA:
    return context.processWGMMA(op, phase);
  case LogicalBehavior::WGMMAWait:
    return context.processWGMMAWait(op, phase);
  case LogicalBehavior::ExpandDims:
    return context.processExpandDims(op, phase);
  case LogicalBehavior::Broadcast:
    return context.processBroadcast(op, phase);
  case LogicalBehavior::Transpose:
    return context.processTranspose(op, phase);
  case LogicalBehavior::Reshape:
    return context.processReshape(op, phase);
  case LogicalBehavior::Cat:
    return context.processCat(op, phase);
  case LogicalBehavior::Join:
    return context.processJoin(op, phase);
  case LogicalBehavior::Split:
    return context.processSplit(op, phase);
  case LogicalBehavior::Reduce:
    return context.processReduce(op, phase);
  case LogicalBehavior::Dot:
    return context.processDot(op, phase);
  case LogicalBehavior::Store:
    return context.processStore(op, phase);
  case LogicalBehavior::AtomicRMW:
    return context.processAtomicRMW(op, phase);
  case LogicalBehavior::RejectGather:
    return context.processRejected(
        op, phase, "gather does not preserve a rectangular logical prefix");
  case LogicalBehavior::RejectHistogram:
    return context.processRejected(
        op, phase, "histogram observes values outside the logical prefix");
  case LogicalBehavior::RejectAtomicCAS:
    return context.processRejected(
        op, phase, "atomic CAS is not a supported logical-domain terminal");
  case LogicalBehavior::RejectLoad:
    return context.processRejected(
        op, phase,
        "restricted values cannot participate in a load address or mask");
  case LogicalBehavior::RejectEscape:
    return context.processRejected(
        op, phase, "logical-domain value escapes the early TTIR function");
  case LogicalBehavior::RejectScan:
    return context.processRejected(
        op, phase,
        "attention fragment v1 does not support scan or exclusive_cumsum");
  }
  llvm_unreachable("unknown logical-domain behavior");
}

/// Tensor logical domains form a sparse, forward single-fragment lattice.
/// Full means every physical coordinate is valid. Fragment carries the only
/// axis whose physical tail is invalid. Conflict rejects joins or transforms
/// that would require tracking more than one restricted axis.
class TensorFact {
public:
  enum class Kind : uint8_t { Uninitialized, Full, Fragment, Conflict };

  TensorFact() = default;

  static TensorFact getPessimisticValueState(Value) { return getFull(); }

  static TensorFact getFull() {
    TensorFact fact;
    fact.kind = Kind::Full;
    return fact;
  }

  static TensorFact getFragment(Value value, int32_t axis,
                                int64_t logicalExtent,
                                LogicalDomainProvenance provenance) {
    auto type = getTensorType(value);
    if (!type || axis < 0 || axis >= type.getRank() || logicalExtent <= 0 ||
        logicalExtent > type.getShape()[axis])
      return getConflict();
    if (logicalExtent == type.getShape()[axis])
      return getFull();
    TensorFact fact;
    fact.kind = Kind::Fragment;
    fact.state.axis = axis;
    fact.state.logicalExtent = logicalExtent;
    fact.state.provenance = std::move(provenance);
    return fact;
  }

  static TensorFact getPrefix(Value value, ArrayRef<int64_t> logicalShape,
                              LogicalDomainProvenance provenance) {
    auto type = getTensorType(value);
    if (!type || !validatePrefixShape(type.getShape(), logicalShape))
      return getConflict();
    std::optional<int32_t> fragmentAxis;
    for (auto [axis, extents] : llvm::enumerate(
             llvm::zip(type.getShape(), logicalShape))) {
      auto [physical, logical] = extents;
      if (physical == logical)
        continue;
      if (fragmentAxis)
        return getConflict();
      fragmentAxis = static_cast<int32_t>(axis);
    }
    if (!fragmentAxis)
      return getFull();
    return getFragment(value, *fragmentAxis, logicalShape[*fragmentAxis],
                       std::move(provenance));
  }

  static TensorFact getConflict() {
    TensorFact fact;
    fact.kind = Kind::Conflict;
    return fact;
  }

  Kind getKind() const { return kind; }
  bool isFragment() const { return kind == Kind::Fragment; }
  const TensorFragmentState &getState() const {
    assert(isFragment());
    return state;
  }

  static TensorFact join(const TensorFact &lhs, const TensorFact &rhs) {
    if (lhs.kind == Kind::Uninitialized)
      return rhs;
    if (rhs.kind == Kind::Uninitialized)
      return lhs;
    if (lhs.kind == Kind::Conflict || rhs.kind == Kind::Conflict)
      return getConflict();
    if (lhs.kind == Kind::Full)
      return rhs;
    if (rhs.kind == Kind::Full)
      return lhs;
    if (lhs.state.axis != rhs.state.axis)
      return getConflict();
    TensorFact result = lhs;
    result.state.logicalExtent =
        std::min(lhs.state.logicalExtent, rhs.state.logicalExtent);
    result.state.provenance = mergeProvenance(
        std::move(result.state.provenance), rhs.state.provenance);
    return result;
  }

  bool operator==(const TensorFact &other) const {
    if (kind != other.kind)
      return false;
    if (kind != Kind::Fragment)
      return true;
    return state.axis == other.state.axis &&
           state.logicalExtent == other.state.logicalExtent &&
           state.provenance.roots == other.state.provenance.roots &&
           state.provenance.seeds == other.state.provenance.seeds;
  }

  void print(raw_ostream &os) const {
    switch (kind) {
    case Kind::Uninitialized:
      os << "uninitialized";
      return;
    case Kind::Full:
      os << "full";
      return;
    case Kind::Conflict:
      os << "conflict";
      return;
    case Kind::Fragment:
      os << "fragment<axis=" << state.axis
         << ", extent=" << state.logicalExtent << '>';
      return;
    }
    llvm_unreachable("unknown tensor fact");
  }

private:
  Kind kind = Kind::Uninitialized;
  TensorFragmentState state;
};

using TensorLattice = dataflow::Lattice<TensorFact>;

static const TensorFragmentState *
getFragmentState(ArrayRef<const TensorLattice *> operands, unsigned index) {
  if (index >= operands.size() || !operands[index]->getValue().isFragment())
    return nullptr;
  return &operands[index]->getValue().getState();
}

static LogicalShape materializeLogicalShape(Value value,
                                            const TensorFragmentState &state) {
  auto type = getTensorType(value);
  assert(type && state.axis >= 0 && state.axis < type.getRank());
  LogicalShape logicalShape(type.getShape());
  logicalShape[state.axis] = state.logicalExtent;
  return logicalShape;
}

static TensorFact
inferSameShapeFact(Operation *op, Value result,
                   ArrayRef<const TensorLattice *> operands) {
  auto resultType = getTensorType(result);
  if (!resultType)
    return TensorFact::getFull();
  std::optional<int32_t> fragmentAxis;
  int64_t logicalExtent = 0;
  LogicalDomainProvenance provenance;
  for (auto [operandValue, operand] : llvm::zip(op->getOperands(), operands)) {
    const TensorFact &fact = operand->getValue();
    if (fact.getKind() == TensorFact::Kind::Conflict)
      return TensorFact::getConflict();
    if (!fact.isFragment())
      continue;
    auto operandType = getTensorType(operandValue);
    if (!operandType || operandType.getShape() != resultType.getShape())
      continue;
    const TensorFragmentState &state = fact.getState();
    if (fragmentAxis && *fragmentAxis != state.axis)
      return TensorFact::getConflict();
    if (!fragmentAxis) {
      fragmentAxis = state.axis;
      logicalExtent = state.logicalExtent;
    } else {
      logicalExtent = std::min(logicalExtent, state.logicalExtent);
    }
    provenance = mergeProvenance(std::move(provenance), state.provenance);
  }
  if (!fragmentAxis)
    return TensorFact::getFull();
  return TensorFact::getFragment(result, *fragmentAxis, logicalExtent,
                                 std::move(provenance));
}

static bool isGenericElementwiseTransfer(Operation *op) {
  return op->getNumRegions() == 0 &&
         op->hasTrait<OpTrait::Elementwise>() && isMemoryEffectFree(op);
}

static bool isMapElementwiseTransfer(Operation *op) {
  return isa<triton::MapElementwiseOp>(op) && isMemoryEffectFree(op);
}

static SmallVector<TensorFact, 2>
inferTensorFacts(Operation *op,
                 ArrayRef<const TensorLattice *> operands,
                 const DenseMap<Value, MemDescLogicalState> &memdescs) {
  SmallVector<TensorFact, 2> inferred(op->getNumResults(),
                                      TensorFact::getFull());
  auto setResult = [&](unsigned index, TensorFact fact) {
    if (index < inferred.size())
      inferred[index] = std::move(fact);
  };
  auto inferSameShape = [&] {
    for (auto [index, result] : llvm::enumerate(op->getResults()))
      setResult(index, inferSameShapeFact(op, result, operands));
  };

  if (auto dot = dyn_cast<WGMMAOp>(op)) {
    auto resultType = getTensorType(dot.getD());
    if (!resultType || resultType.getRank() != 2)
      return inferred;
    std::optional<int32_t> fragmentAxis;
    int64_t logicalExtent = 0;
    LogicalDomainProvenance provenance;
    bool conflict = false;
    auto mergeOutputFragment = [&](int32_t axis, int64_t extent,
                                   const LogicalDomainProvenance &source) {
      if (extent == resultType.getShape()[axis])
        return;
      if (fragmentAxis && *fragmentAxis != axis) {
        conflict = true;
        return;
      }
      if (!fragmentAxis) {
        fragmentAxis = axis;
        logicalExtent = extent;
      } else {
        logicalExtent = std::min(logicalExtent, extent);
      }
      provenance = mergeProvenance(std::move(provenance), source);
    };
    auto bIt = memdescs.find(dot.getB());
    if (bIt != memdescs.end() && bIt->second.logicalShape.size() == 2 &&
        bIt->second.logicalShape[1] != bIt->second.physicalShape[1])
      mergeOutputFragment(1, bIt->second.logicalShape[1],
                          bIt->second.provenance);
    if (const TensorFragmentState *aState =
            getFragmentState(operands, 0)) {
      if (aState->axis == 0)
        mergeOutputFragment(0, aState->logicalExtent, aState->provenance);
      else if (aState->axis != 1)
        conflict = true;
    }
    if (const TensorFragmentState *cState =
            getFragmentState(operands, 2)) {
      if (cState->axis < 0 || cState->axis >= 2)
        conflict = true;
      else
        mergeOutputFragment(cState->axis, cState->logicalExtent,
                            cState->provenance);
    }
    if (conflict) {
      setResult(0, TensorFact::getConflict());
    } else if (fragmentAxis) {
      mergeProvenanceInto(provenance, {nullptr, dot});
      setResult(0, TensorFact::getFragment(dot.getD(), *fragmentAxis,
                                           logicalExtent,
                                           std::move(provenance)));
    }
    return inferred;
  }

  if (isa<WGMMAWaitOp>(op)) {
    inferSameShape();
    return inferred;
  }

  if (auto expand = dyn_cast<triton::ExpandDimsOp>(op)) {
    if (const TensorFragmentState *state =
            getFragmentState(operands, 0)) {
      int32_t resultAxis =
          state->axis >= expand.getAxis() ? state->axis + 1 : state->axis;
      setResult(0, TensorFact::getFragment(
                       expand.getResult(), resultAxis, state->logicalExtent,
                       state->provenance));
    }
    return inferred;
  }

  if (auto broadcast = dyn_cast<triton::BroadcastOp>(op)) {
    if (const TensorFragmentState *state =
            getFragmentState(operands, 0)) {
      auto srcType = getTensorType(broadcast.getSrc());
      auto resultType = getTensorType(broadcast.getResult());
      if (srcType && resultType && srcType.getRank() == resultType.getRank() &&
          srcType.getShape()[state->axis] ==
              resultType.getShape()[state->axis])
        setResult(0, TensorFact::getFragment(
                         broadcast.getResult(), state->axis,
                         state->logicalExtent, state->provenance));
    }
    return inferred;
  }

  if (auto transpose = dyn_cast<triton::TransOp>(op)) {
    if (const TensorFragmentState *state =
            getFragmentState(operands, 0)) {
      auto resultAxis = llvm::find(transpose.getOrder(), state->axis);
      if (resultAxis != transpose.getOrder().end())
        setResult(0, TensorFact::getFragment(
                         transpose.getResult(),
                         static_cast<int32_t>(
                             std::distance(transpose.getOrder().begin(),
                                           resultAxis)),
                         state->logicalExtent, state->provenance));
    }
    return inferred;
  }

  if (auto reshape = dyn_cast<triton::ReshapeOp>(op)) {
    if (const TensorFragmentState *state =
            getFragmentState(operands, 0)) {
      auto srcType = getTensorType(reshape.getSrc());
      auto resultType = getTensorType(reshape.getResult());
      std::optional<LogicalShape> shape;
      LogicalShape sourceLogical =
          materializeLogicalShape(reshape.getSrc(), *state);
      if (srcType && resultType && !reshape.getAllowReorder()) {
        if (srcType.getShape() == resultType.getShape()) {
          shape = sourceLogical;
        } else {
          shape = reshapeOnlyUnitDimensions(srcType.getShape(),
                                            sourceLogical,
                                            resultType.getShape());
          if (!shape)
            if (auto prefix = getContiguousLinearPrefixSize(
                    srcType.getShape(), sourceLogical))
              shape = representLinearPrefixAsRectangle(resultType.getShape(),
                                                       *prefix);
        }
      }
      if (shape)
        setResult(0, TensorFact::getPrefix(reshape.getResult(), *shape,
                                           state->provenance));
    }
    return inferred;
  }

  if (auto cat = dyn_cast<triton::CatOp>(op)) {
    const TensorFragmentState *lhs = getFragmentState(operands, 0);
    const TensorFragmentState *rhs = getFragmentState(operands, 1);
    auto lhsType = getTensorType(cat.getLhs());
    auto rhsType = getTensorType(cat.getRhs());
    auto resultType = getTensorType(cat.getResult());
    if ((lhs || rhs) && lhsType && rhsType && resultType &&
        resultType.getRank() > 0 &&
        lhsType.getRank() == resultType.getRank() &&
        rhsType.getRank() == resultType.getRank()) {
      LogicalShape left =
          lhs ? materializeLogicalShape(cat.getLhs(), *lhs)
              : LogicalShape(lhsType.getShape());
      LogicalShape right =
          rhs ? materializeLogicalShape(cat.getRhs(), *rhs)
              : LogicalShape(rhsType.getShape());
      LogicalShape shape(resultType.getShape());
      for (unsigned i = 0; i + 1 < shape.size(); ++i)
        shape[i] = std::min(left[i], right[i]);
      unsigned axis = shape.size() - 1;
      shape[axis] = left[axis] == lhsType.getShape()[axis]
                        ? lhsType.getShape()[axis] + right[axis]
                        : left[axis];
      setResult(
          0, TensorFact::getPrefix(
                 cat.getResult(), shape,
                 mergeProvenance(lhs ? lhs->provenance
                                     : LogicalDomainProvenance{},
                                 rhs ? rhs->provenance
                                     : LogicalDomainProvenance{})));
    }
    return inferred;
  }

  if (auto join = dyn_cast<triton::JoinOp>(op)) {
    const TensorFragmentState *lhs = getFragmentState(operands, 0);
    const TensorFragmentState *rhs = getFragmentState(operands, 1);
    auto lhsType = getTensorType(join.getLhs());
    auto rhsType = getTensorType(join.getRhs());
    if ((lhs || rhs) && lhsType && rhsType &&
        lhsType.getShape() == rhsType.getShape()) {
      LogicalShape left =
          lhs ? materializeLogicalShape(join.getLhs(), *lhs)
              : LogicalShape(lhsType.getShape());
      LogicalShape right =
          rhs ? materializeLogicalShape(join.getRhs(), *rhs)
              : LogicalShape(rhsType.getShape());
      LogicalShape shape = intersectShapes(left, right);
      shape.push_back(2);
      setResult(
          0, TensorFact::getPrefix(
                 join.getResult(), shape,
                 mergeProvenance(lhs ? lhs->provenance
                                     : LogicalDomainProvenance{},
                                 rhs ? rhs->provenance
                                     : LogicalDomainProvenance{})));
    }
    return inferred;
  }

  if (auto split = dyn_cast<triton::SplitOp>(op)) {
    if (const TensorFragmentState *state =
            getFragmentState(operands, 0)) {
      LogicalShape sourceLogical =
          materializeLogicalShape(split.getSrc(), *state);
      if (!sourceLogical.empty() && sourceLogical.back() == 2) {
        LogicalShape shape = std::move(sourceLogical);
        shape.pop_back();
        for (auto [index, result] : llvm::enumerate(split.getResults()))
          setResult(index, TensorFact::getPrefix(result, shape,
                                                  state->provenance));
      }
    }
    return inferred;
  }

  if (auto reduce = dyn_cast<triton::ReduceOp>(op)) {
    int32_t axis = reduce.getAxis();
    std::optional<LogicalShape> reducedShape;
    LogicalDomainProvenance provenance;
    for (unsigned index = 0; index < reduce.getSrcs().size(); ++index) {
      const TensorFragmentState *state = getFragmentState(operands, index);
      if (!state)
        continue;
      LogicalShape shape =
          materializeLogicalShape(reduce.getSrcs()[index], *state);
      if (axis < 0 || axis >= static_cast<int32_t>(shape.size()))
        continue;
      shape.erase(shape.begin() + axis);
      reducedShape = reducedShape
                         ? std::optional<LogicalShape>(
                               intersectShapes(*reducedShape, shape))
                         : std::optional<LogicalShape>(std::move(shape));
      provenance =
          mergeProvenance(std::move(provenance), state->provenance);
    }
    if (reducedShape) {
      for (auto [index, result] : llvm::enumerate(reduce.getResults())) {
        if (!getTensorType(result))
          continue;
        setResult(index, TensorFact::getPrefix(result, *reducedShape,
                                               provenance));
      }
    }
    return inferred;
  }

  if (isa<triton::DotOp, triton::DotScaledOp>(op)) {
    if (op->getNumOperands() < 3 || op->getNumResults() != 1)
      return inferred;
    const TensorFragmentState *aState = getFragmentState(operands, 0);
    const TensorFragmentState *bState = getFragmentState(operands, 1);
    const TensorFragmentState *cState = getFragmentState(operands, 2);
    auto aType = getTensorType(op->getOperand(0));
    auto bType = getTensorType(op->getOperand(1));
    auto resultType = getTensorType(op->getResult(0));
    if ((!aState && !bState && !cState) || !aType || !bType || !resultType ||
        aType.getRank() != 2 || bType.getRank() != 2 ||
        resultType.getRank() != 2)
      return inferred;
    std::optional<int32_t> resultAxis;
    int64_t resultExtent = 0;
    LogicalDomainProvenance provenance;
    auto mergeResultFragment = [&](int32_t axis, int64_t extent,
                                   const LogicalDomainProvenance &source) {
      if (resultAxis && *resultAxis != axis)
        return false;
      if (!resultAxis) {
        resultAxis = axis;
        resultExtent = extent;
      } else {
        resultExtent = std::min(resultExtent, extent);
      }
      provenance = mergeProvenance(std::move(provenance), source);
      return true;
    };
    bool valid = true;
    if (aState && aState->axis == 0)
      valid &= mergeResultFragment(0, aState->logicalExtent,
                                   aState->provenance);
    if (bState && bState->axis == 1)
      valid &= mergeResultFragment(1, bState->logicalExtent,
                                   bState->provenance);
    if (cState)
      valid &= mergeResultFragment(cState->axis, cState->logicalExtent,
                                   cState->provenance);
    if (!valid)
      setResult(0, TensorFact::getConflict());
    else if (resultAxis)
      setResult(0, TensorFact::getFragment(op->getResult(0), *resultAxis,
                                           resultExtent,
                                           std::move(provenance)));
    return inferred;
  }

  if (isMapElementwiseTransfer(op) || isGenericElementwiseTransfer(op))
    inferSameShape();
  return inferred;
}

class TensorLogicalDomainAnalysis final
    : public dataflow::SparseForwardDataFlowAnalysis<TensorLattice> {
  using Base = dataflow::SparseForwardDataFlowAnalysis<TensorLattice>;

public:
  TensorLogicalDomainAnalysis(
      DataFlowSolver &solver,
      const DenseMap<Value, MemDescLogicalState> &memdescs)
      : Base(solver), memdescs(memdescs) {}

  LogicalResult visitOperation(
      Operation *op, ArrayRef<const TensorLattice *> operands,
      ArrayRef<TensorLattice *> results) override {
    SmallVector<TensorFact, 2> inferred =
        inferTensorFacts(op, operands, memdescs);
    for (auto [result, fact] : llvm::zip(results, inferred))
      propagateIfChanged(result, result->join(fact));
    return success();
  }

private:
  void setToEntryState(TensorLattice *lattice) override {
    propagateIfChanged(
        lattice,
        lattice->join(TensorFact::getPessimisticValueState(
            lattice->getAnchor())));
  }

  const DenseMap<Value, MemDescLogicalState> &memdescs;
};

} // namespace

LogicalDomainContext::LogicalDomainContext(LogicalDomainPlan &plan)
    : plan(plan) {}

const MemDescLogicalState *
LogicalDomainContext::lookupMemDesc(Value value) const {
  auto it = plan.memdescs.find(value);
  return it == plan.memdescs.end() ? nullptr : &it->second;
}
const TensorFragmentState *
LogicalDomainContext::lookupTensor(Value value) const {
  auto it = plan.tensors.find(value);
  return it == plan.tensors.end() ? nullptr : &it->second;
}

bool LogicalDomainContext::mergeMemDesc(Value value,
                                        const MemDescLogicalState &candidate) {
  auto type = dyn_cast<ttg::MemDescType>(value.getType());
  if (!type || !validatePrefixShape(type.getShape(), candidate.logicalShape))
    return false;
  auto [it, inserted] = plan.memdescs.try_emplace(value, candidate);
  if (inserted) {
    markChanged(value);
    return true;
  }
  LogicalShape merged =
      intersectShapes(it->second.logicalShape, candidate.logicalShape);
  bool provenanceChanged =
      mergeProvenanceInto(it->second.provenance, candidate.provenance);
  if (merged == it->second.logicalShape && !provenanceChanged)
    return false;
  it->second.logicalShape = std::move(merged);
  markChanged(value);
  return true;
}

bool LogicalDomainContext::hasRestrictedOperand(Operation *op) const {
  return llvm::any_of(op->getOperands(), [&](Value value) {
    return lookupTensor(value) || lookupMemDesc(value);
  });
}

LogicalResult
LogicalDomainContext::validateTensorResult(Operation *operation) const {
  std::optional<unsigned> restrictedOperand;
  for (auto [index, operand] : llvm::enumerate(operation->getOperands()))
    if (lookupTensor(operand)) {
      restrictedOperand = index;
      break;
    }
  if (!restrictedOperand)
    return success();
  if (llvm::any_of(operation->getResults(),
                   [&](Value result) { return lookupTensor(result); }))
    return success();
  return emitError(operation, *restrictedOperand,
                   "logical-domain transfer produced no restricted tensor "
                   "result");
}
void LogicalDomainContext::markChanged(Value value) {
  changedValues.push_back(value);
}

SmallVector<Value> LogicalDomainContext::takeChangedValues() {
  SmallVector<Value> result;
  result.swap(changedValues);
  return result;
}

LogicalResult LogicalDomainContext::emitError(Operation *op,
                                              unsigned operandIndex,
                                              const Twine &reason) const {
  InFlightDiagnostic diag = op->emitOpError();
  diag << "logical-domain operand " << operandIndex << " rejected: " << reason;
  if (operandIndex >= op->getNumOperands())
    return failure();
  Value value = op->getOperand(operandIndex);
  if (const TensorFragmentState *state = lookupTensor(value)) {
    auto type = getTensorType(value);
    LogicalShape logicalShape = materializeLogicalShape(value, *state);
    diag << "; physical=" << type.getShape()
         << ", logical=" << logicalShape;
    for (Operation *root : state->provenance.roots)
      diag.attachNote(root->getLoc())
          << "logical domain root is here";
    for (Operation *seed : state->provenance.seeds) {
      if (llvm::is_contained(state->provenance.roots, seed))
        continue;
      diag.attachNote(seed->getLoc())
          << "logical tensor seed is here";
    }
  } else if (const MemDescLogicalState *state = lookupMemDesc(value)) {
    diag << "; physical=" << state->physicalShape
         << ", logical=" << state->logicalShape;
    for (Operation *root : state->provenance.roots)
      diag.attachNote(root->getLoc())
          << "logical domain root is here";
  }
  return failure();
}

LogicalResult
LogicalDomainContext::processCandidateAlloc(Operation *operation,
                                            LogicalDomainPhase phase) {
  auto alloc = cast<ttg::LocalAllocOp>(operation);
  auto storageAttr = alloc->getAttrOfType<StringAttr>(kStoragePlan);
  if (!storageAttr) {
    if (phase == LogicalDomainPhase::Plan && hasRestrictedOperand(alloc))
      return processRejected(
          alloc, phase,
          "ordinary local allocation cannot consume a restricted logical "
          "tensor");
    return success();
  }
  if (storageAttr.getValue() != "candidate")
    return alloc.emitOpError("unknown TLE logical storage plan");
  auto logicalAttr = alloc->getAttrOfType<DenseI64ArrayAttr>(kLogicalAllocShape);
  auto axisAttr = alloc->getAttrOfType<IntegerAttr>(kLogicalNonPowerAxis);
  if (!logicalAttr || !axisAttr)
    return alloc.emitOpError("has incomplete logical candidate metadata");
  auto type = alloc.getType();
  ArrayRef<int64_t> logical = logicalAttr.asArrayRef();
  int64_t fragmentAxis = axisAttr.getInt();
  if (logical.size() != 3 || type.getRank() != 3 || fragmentAxis < 1 ||
      fragmentAxis >= 3)
    return alloc.emitOpError(
        "logical candidate requires an explicit capacity and a rank-2 payload");
  int64_t capacity = logical[0], rows = logical[1], cols = logical[2];
  if (capacity <= 0 ||
      capacity > std::numeric_limits<int32_t>::max())
    return alloc.emitOpError(
        "logical capacity must fit a positive i32 stage index");
  if (llvm::isPowerOf2_64(logical[fragmentAxis]) ||
      llvm::any_of(llvm::enumerate(logical.drop_front()),
                   [&](auto indexedExtent) {
                     int64_t rootAxis = indexedExtent.index() + 1;
                     return rootAxis != fragmentAxis &&
                            !llvm::isPowerOf2_64(indexedExtent.value());
                   }))
    return alloc.emitOpError(
        "logical candidate metadata must identify its only non-power-of-two "
        "payload axis");
  if (rows <= 0 || cols <= 0)
    return alloc.emitOpError("logical payload extents must be positive");
  if (logical[fragmentAxis] % 16 != 0)
    return alloc.emitOpError(
        "logical fragment extent must be a multiple of 16");
  if (!isSupportedExactSMEMElementType(type.getElementType()))
    return alloc.emitOpError(
        "logical candidate requires a Hopper WGMMA-compatible element type");
  if (!type.getMutableMemory() ||
      !isa<ttg::SharedMemorySpaceAttr>(type.getMemorySpace()))
    return alloc.emitOpError(
        "logical candidate requires mutable shared-memory storage");
  if (alloc.getSrc())
    return alloc.emitOpError(
        "logical candidate does not support an initializer");
  LogicalShape expected(logical.begin(), logical.end());
  expected[fragmentAxis] = nextPowerOfTwo(expected[fragmentAxis]);
  if (type.getShape() != ArrayRef<int64_t>(expected) ||
      type.getAllocShape() != ArrayRef<int64_t>(expected))
    return alloc.emitOpError(
        "logical candidate carrier type does not match its padded shape");
  if (phase == LogicalDomainPhase::Propagate) {
    MemDescLogicalState state;
    state.physicalShape = expected;
    state.logicalShape.assign(logical.begin(), logical.end());
    state.axisMap = identityAxisMap(logical.size());
    state.provenance = {alloc, alloc};
    mergeMemDesc(alloc.getResult(), state);
  } else if (!findRootAction(plan, alloc)) {
    LogicalRootRewriteAction action;
    action.alloc = alloc;
    action.logicalShape.assign(logical.begin(), logical.end());
    plan.roots.push_back(std::move(action));
  }
  return success();
}

LogicalResult
LogicalDomainContext::processMemDescIndex(Operation *operation,
                                          LogicalDomainPhase phase) {
  auto index = cast<ttg::MemDescIndexOp>(operation);
  const MemDescLogicalState *source = lookupMemDesc(index.getSrc());
  if (!source)
    return success();
  auto rootType = dyn_cast<ttg::MemDescType>(index.getSrc().getType());
  auto stageType = dyn_cast<ttg::MemDescType>(index.getType());
  if (!rootType || !stageType || source->logicalShape.size() != 3)
    return emitError(index, 0,
                     "logical stage requires capacity plus a rank-2 payload");
  int64_t capacity = source->logicalShape[0];
  SmallVector<int64_t> expectedShape(source->physicalShape.begin() + 1,
                                     source->physicalShape.end());
  auto nvmma = dyn_cast<ttg::NVMMASharedEncodingAttr>(
      stageType.getEncoding());
  if (stageType.getShape() != ArrayRef<int64_t>(expectedShape) ||
      stageType.getAllocShape() != ArrayRef<int64_t>(expectedShape) ||
      stageType.getElementType() != rootType.getElementType() ||
      stageType.getMemorySpace() != rootType.getMemorySpace() ||
      !stageType.getMutableMemory() || !nvmma ||
      nvmma.getElementBitWidth() !=
          stageType.getElementType().getIntOrFloatBitWidth())
    return emitError(
        index, 0,
        "planned tiled stage requires a mutable NVMMAShared carrier whose "
        "bitwidth matches the candidate storage");
  if (!index.getIndex().getType().isInteger(32))
    return emitError(index, 1, "planned tiled stage index must be i32");
  if (auto constant =
          index.getIndex().getDefiningOp<arith::ConstantIntOp>()) {
    int64_t stage = constant.value();
    if (stage < 0 || stage >= capacity)
      return emitError(index, 1,
                       "planned tiled stage index exceeds capacity");
  }
  if (phase == LogicalDomainPhase::Propagate) {
    MemDescLogicalState state = *source;
    state.physicalShape.erase(state.physicalShape.begin());
    state.logicalShape.erase(state.logicalShape.begin());
    state.axisMap.erase(state.axisMap.begin());
    state.isStage = true;
    mergeMemDesc(index.getResult(), state);
  } else {
    auto *root = findRootAction(plan, source->provenance.primaryRoot());
    if (!root)
      return emitError(index, 0, "candidate root has no storage action");
    if (!llvm::is_contained(root->stages, index))
      root->stages.push_back(index);
  }
  return success();
}

LogicalResult
LogicalDomainContext::processLogicalTMACopy(Operation *operation,
                                            LogicalDomainPhase phase) {
  auto copy = cast<ttg::TMACopyOp>(operation);
  for (auto [index, operand] : llvm::enumerate(operation->getOperands())) {
    const MemDescLogicalState *state = lookupMemDesc(operand);
    if (!state)
      continue;
    if (index != 1)
      return emitError(copy, index,
                       "logical TMA supports only global-to-SMEM direction");
#ifndef __HCU__
    if (copy.getBarrier())
      return emitError(copy, index,
                       "logical TMA does not support an explicit barrier");
#endif
    auto descType = dyn_cast<triton::TensorDescType>(copy.getSrc().getType());
    auto dstType = dyn_cast<ttg::MemDescType>(copy.getDst().getType());
    if (!descType || !dstType || state->logicalShape.size() != 2 ||
        dstType.getRank() != 2)
      return emitError(copy, index,
                       "logical TMA requires a rank-2 stage and tensor "
                       "descriptor source");
    if (!state->isStage || state->transposed ||
        !copy.getDst().getDefiningOp<ttg::MemDescIndexOp>())
      return emitError(copy, index,
                       "logical TMA destination must be a direct stage view");
    RankedTensorType blockType = descType.getSignlessBlockType();
    ArrayRef<int64_t> blockShape = blockType.getShape();
    if (blockShape.size() < 2 ||
        llvm::any_of(blockShape.drop_back(2),
                     [](int64_t extent) { return extent != 1; }))
      return emitError(copy, index,
                       "logical TMA descriptor block must have only unit "
                       "leading dimensions");
    FailureOr<SmallVector<int64_t, 2>> selectedStorageTile =
        selectLogicalSMEMStorageTileShape(dstType, state->logicalShape);
    if (failed(selectedStorageTile))
      return emitError(copy, index,
                       "logical stage has no exact power-of-two storage tile "
                       "of at least 16 along its fragment axis");
    SmallVector<int64_t, 2> storageTile = std::move(*selectedStorageTile);
    if (blockShape.take_back(2) != ArrayRef<int64_t>(storageTile)) {
      InFlightDiagnostic diag = copy.emitOpError(
          "logical-domain operand 1 rejected: descriptor-driven exact-SMEM "
          "storage tile does not match the selected heuristic; expected ");
      diag << storageTile << ", got " << blockShape.take_back(2);
      return failure();
    }
    auto logical =
        copy->getAttrOfType<DenseI64ArrayAttr>(kLogicalCopyShapeAttr);
    if (!logical || logical.size() != blockShape.size() ||
        llvm::any_of(logical.asArrayRef().drop_back(2),
                     [](int64_t extent) { return extent != 1; }) ||
        logical.asArrayRef().take_back(2) !=
            ArrayRef<int64_t>(state->logicalShape))
      return emitError(copy, index,
                       "logical copy shape must match the descriptor rank and "
                       "end in the stage shape");
    if (copy.getIndices().size() != blockShape.size())
      return emitError(copy, index,
                       "logical TMA coordinate count must match descriptor "
                       "rank");
    if (blockType.getElementType() != dstType.getElementType())
      return emitError(copy, index,
                       "logical TMA source and destination element types "
                       "must match");
    if (phase == LogicalDomainPhase::Plan) {
      auto *root = findRootAction(plan, state->provenance.primaryRoot());
      if (!root)
        return emitError(copy, index, "candidate root has no storage action");
      if (failed(selectRootStorageTile(*root, storageTile, copy)))
        return failure();
      if (!llvm::is_contained(root->copies, copy))
        root->copies.push_back(copy);
    }
  }
  return success();
}

LogicalResult
LogicalDomainContext::processLogicalPointerCopy(Operation *operation,
                                                LogicalDomainPhase phase) {
  auto pointers = cast<LocalPointersOp>(operation);
  const MemDescLogicalState *state = lookupMemDesc(pointers.getSrc());
  if (!state)
    return success();
  if (phase == LogicalDomainPhase::Propagate)
    return success();

  auto srcType = dyn_cast<ttg::MemDescType>(pointers.getSrc().getType());
  auto ptrType = dyn_cast<RankedTensorType>(pointers.getResult().getType());
  if (!state->isStage || state->transposed ||
      !pointers.getSrc().getDefiningOp<ttg::MemDescIndexOp>() || !srcType ||
      !ptrType || state->logicalShape.size() != 2 || srcType.getRank() != 2)
    return emitError(
        pointers, 0,
        "logical pointer copy requires a direct rank-2 stage destination");

  auto logical =
      pointers->getAttrOfType<DenseI64ArrayAttr>(kLogicalCopyShapeAttr);
  if (!logical ||
      logical.asArrayRef() != ArrayRef<int64_t>(state->logicalShape))
    return emitError(
        pointers, 0,
        "logical pointer copy shape must match the logical stage shape");
  if (ptrType.getShape() != srcType.getShape() ||
      ptrType.getShape() != ArrayRef<int64_t>(state->physicalShape) ||
      pointers.getIndices().size() != 2)
    return emitError(
        pointers, 0,
        "logical pointer copy must address the complete padded stage carrier");
  for (Value index : pointers.getIndices()) {
    auto indexType = dyn_cast<RankedTensorType>(index.getType());
    if (!indexType || indexType.getShape() != ptrType.getShape())
      return emitError(
          pointers, 0,
          "logical pointer copy indices must cover the padded stage carrier");
  }

  if (!pointers.getResult().hasOneUse())
    return emitError(
        pointers, 0,
        "logical pointer copy local pointers must feed exactly one store");
  auto store =
      dyn_cast<triton::StoreOp>(*pointers.getResult().getUsers().begin());
  if (!store || store.getPtr() != pointers.getResult() || store.getMask() ||
      !store.getBoundaryCheck().empty())
    return emitError(
        pointers, 0,
        "logical pointer copy requires one unmasked full-carrier store");

  auto load = store.getValue().getDefiningOp<triton::LoadOp>();
  auto loadType = dyn_cast<RankedTensorType>(store.getValue().getType());
  if (!load || !load->hasOneUse() || load.getIsVolatile() ||
      !load.getBoundaryCheck().empty() || load.getPadding() || !loadType ||
      !isGlobalPointerTensor(load.getPtr()) ||
      loadType.getShape() != ptrType.getShape() ||
      loadType.getElementType() != srcType.getElementType())
    return emitError(
        pointers, 0,
        "logical pointer copy requires a direct non-volatile global load "
        "whose carrier shape and element type match the stage");

  FailureOr<SmallVector<int64_t, 2>> selectedStorageTile =
      selectLogicalSMEMStorageTileShape(srcType, state->logicalShape);
  if (failed(selectedStorageTile))
    return emitError(
        pointers, 0,
        "logical stage has no exact power-of-two storage tile of at least 16 "
        "along its fragment axis");

  auto *root = findRootAction(plan, state->provenance.primaryRoot());
  if (!root)
    return emitError(pointers, 0, "candidate root has no storage action");
  if (failed(selectRootStorageTile(*root, *selectedStorageTile, pointers)))
    return failure();
  if (llvm::none_of(root->pointerCopies, [&](const auto &action) {
        return action.pointers == pointers;
      }))
    root->pointerCopies.push_back({pointers, store, load});
  return success();
}

LogicalResult LogicalDomainContext::processPipe(Operation *operation,
                                                LogicalDomainPhase phase) {
  for (auto [index, operand] : llvm::enumerate(operation->getOperands())) {
    const MemDescLogicalState *state = lookupMemDesc(operand);
    if (!state)
      continue;
    auto names = operation->getAttrOfType<ArrayAttr>("field_names");
    if (!names || index >= names.size())
      return emitError(operation, index,
                       "candidate memdesc is not a pipe field operand");
    if (phase == LogicalDomainPhase::Plan && !state->isStage) {
      auto *root = findRootAction(plan, state->provenance.primaryRoot());
      if (!root)
        return emitError(operation, index,
                         "candidate root has no storage action");
      OpOperand *use = &operation->getOpOperand(index);
      if (llvm::none_of(root->memdescUses, [&](const auto &action) {
            return action.use == use;
          }))
        root->memdescUses.push_back({use, true});
    }
  }
  return success();
}

LogicalResult LogicalDomainContext::processWarpSpecialize(
    Operation *operation, LogicalDomainPhase phase) {
  auto warpSpecialize = cast<ttg::WarpSpecializeOp>(operation);
  for (auto [index, capture] :
       llvm::enumerate(warpSpecialize.getExplicitCaptures())) {
    const MemDescLogicalState *state = lookupMemDesc(capture);
    if (!state)
      continue;
    for (Region *partition : warpSpecialize.getPartitionRegions()) {
      if (index >= partition->getNumArguments())
        return emitError(warpSpecialize, index,
                         "candidate capture has no partition argument");
      Value argument = partition->getArgument(index);
      if (argument.getType() != capture.getType())
        return emitError(warpSpecialize, index,
                         "candidate capture and partition argument types "
                         "must match");
      if (phase == LogicalDomainPhase::Propagate)
        mergeMemDesc(argument, *state);
    }
    if (phase == LogicalDomainPhase::Plan && !state->isStage) {
      auto *root = findRootAction(plan, state->provenance.primaryRoot());
      if (!root)
        return emitError(warpSpecialize, index,
                         "candidate root has no storage action");
      OpOperand *use = &operation->getOpOperand(index);
      if (llvm::none_of(root->memdescUses, [&](const auto &action) {
            return action.use == use;
          }))
        root->memdescUses.push_back({use, false});
    }
  }
  return success();
}

LogicalResult
LogicalDomainContext::processMemDescUse(Operation *operation,
                                        LogicalDomainPhase phase) {
  for (auto [index, operand] : llvm::enumerate(operation->getOperands())) {
    const MemDescLogicalState *state = lookupMemDesc(operand);
    if (!state)
      continue;
    if (phase == LogicalDomainPhase::Plan && !state->isStage) {
      auto *root =
          findRootAction(plan, state->provenance.primaryRoot());
      if (!root)
        return emitError(operation, index,
                         "candidate root has no storage action");
      OpOperand *use = &operation->getOpOperand(index);
      if (llvm::none_of(root->memdescUses, [&](const auto &action) {
            return action.use == use;
          }))
        root->memdescUses.push_back({use, false});
    }
  }
  return success();
}

LogicalResult
LogicalDomainContext::processMemDescTranspose(Operation *operation,
                                              LogicalDomainPhase phase) {
  auto view = cast<MemDescWGMMAViewOp>(operation);
  const MemDescLogicalState *source = lookupMemDesc(view.getSrc());
  if (!source)
    return success();
  if (source->transposed)
    return emitError(view, 0,
                     "nested descriptor transpose is unsupported");
  if (view.getOrder() != ArrayRef<int32_t>({1, 0}))
    return emitError(view, 0,
                     "descriptor transpose must use order [1, 0]");
  if (phase == LogicalDomainPhase::Propagate) {
    if (source->logicalShape.size() != 2)
      return emitError(view, 0, "WGMMA descriptor view requires rank two");
    MemDescLogicalState result = *source;
    std::swap(result.physicalShape[0], result.physicalShape[1]);
    std::swap(result.logicalShape[0], result.logicalShape[1]);
    std::swap(result.axisMap[0], result.axisMap[1]);
    result.transposed = true;
    mergeMemDesc(view.getResult(), result);
  }
  return success();
}

static LogicalResult validatePlannedWGMMA(
    WGMMAOp dot, std::optional<int64_t> activeN,
    std::optional<int64_t> activeK, const MemDescLogicalState *candidateB) {
  auto aTensorType = dyn_cast<RankedTensorType>(dot.getA().getType());
  auto aMemDescType = dyn_cast<ttg::MemDescType>(dot.getA().getType());
  auto bType = dyn_cast<ttg::MemDescType>(dot.getB().getType());
  auto cType = getTensorType(dot.getC());
  if ((!aTensorType && !aMemDescType) || !bType || !cType)
    return dot.emitOpError(
        "planned logical extent requires ranked WGMMA carriers");

  ArrayRef<int64_t> aShape =
      aTensorType ? aTensorType.getShape() : aMemDescType.getShape();
  ArrayRef<int64_t> bShape = bType.getShape();
  Type aElementType = aTensorType ? aTensorType.getElementType()
                                  : aMemDescType.getElementType();
  Type bElementType = bType.getElementType();
  std::optional<int64_t> instructionK =
      getWGMMAInstructionK(aElementType);
  if (!instructionK || !isSupportedWGMMATypeCombination(
                           aElementType, bElementType,
                           cType.getElementType(), dot.getInputPrecision()))
    return dot.emitOpError(
        "planned logical extent has an unsupported Hopper WGMMA operand, "
        "accumulator, or input precision combination");
  if (candidateB && (candidateB->logicalShape.size() != 2 ||
                     candidateB->physicalShape.size() != 2))
    return dot.emitOpError(
        "planned logical SMEM operand must have rank two");

  auto isDirectCandidateB = [&] {
    return dot.getB().getDefiningOp<ttg::MemDescIndexOp>() != nullptr;
  };
  auto isTransposedCandidateB = [&] {
    auto view = dot.getB().getDefiningOp<MemDescWGMMAViewOp>();
    return view &&
           view.getSrc().getDefiningOp<ttg::MemDescIndexOp>() != nullptr;
  };
  auto hasCandidateBTopology = [&] {
    return candidateB &&
           (candidateB->transposed ? isTransposedCandidateB()
                                   : isDirectCandidateB());
  };

  if (!activeN && !activeK && candidateB) {
    if (!candidateB->transposed &&
        !supportsWGMMAOperandTranspose(bElementType))
      return dot.emitOpError(
          "TF32, FP8, and int8 tiled B require a transposed logical view so "
          "WGMMA consumes a column-major descriptor without PTX transpose "
          "operands");
    if (!candidateB->transposed && aTensorType && isDirectCandidateB())
      return success();
    if (candidateB->transposed && aMemDescType &&
        candidateB->logicalShape[1] == candidateB->physicalShape[1] &&
        isTransposedCandidateB())
      return success();
    return dot.emitOpError(
        "planned tiled SMEM operand requires active_n, active_k, or a "
        "full-shape compatible WGMMA form");
  }
  if (!activeN && !activeK)
    return success();
  if (activeN && activeK)
    return dot.emitOpError(
        "planned active_n and active_k cannot be combined");
  if (aShape.size() != 2 || bShape.size() != 2 || cType.getRank() != 2)
    return dot.emitOpError(
        "planned active extent requires rank-two WGMMA carriers");
  if (!isSupportedActiveWGMMATypeCombination(
          aElementType, bElementType, cType.getElementType(),
          dot.getInputPrecision()))
    return dot.emitOpError(
        "planned active extent requires a supported Hopper WGMMA type "
        "combination with a 32-bit accumulator");
  if (candidateB && !candidateB->transposed &&
      !supportsWGMMAOperandTranspose(bElementType))
    return dot.emitOpError(
        "TF32, FP8, and int8 tiled B require a transposed logical view so "
        "WGMMA consumes a column-major descriptor without PTX transpose "
        "operands");
  if (aShape[0] != 64)
    return dot.emitOpError(
        "planned active extent currently requires physical M=64");

  if (activeN) {
    if (*activeN <= 0 || *activeN % 8 != 0 || *activeN > bShape[1])
      return dot.emitOpError(
          "planned active_n must be a positive multiple of 8 within N");
    int64_t maxInstructionN = aElementType.isInteger(8) ? 224 : 256;
    if (bShape[1] > maxInstructionN)
      return dot.emitOpError(
          "planned active_n physical N carrier exceeds the WGMMA type limit");
    if (dot->hasAttr("tle.wgmma_accumulator_chain_c"))
      return dot.emitOpError(
          "planned active_n does not support an accumulator chain");
    if (!aMemDescType)
      return dot.emitOpError(
          "planned active_n requires a shared-memory A operand");
    if (candidateB) {
      if (!hasCandidateBTopology() || candidateB->logicalShape.size() != 2 ||
          *activeN != candidateB->logicalShape[1] ||
          bShape[1] != nextPowerOfTwo(candidateB->logicalShape[1]) ||
          candidateB->logicalShape[0] != candidateB->physicalShape[0] ||
          bShape[0] != candidateB->physicalShape[0] ||
          aShape[1] != candidateB->physicalShape[0])
        return dot.emitOpError(
            "planned active_n carrier does not match the logical SMEM stage");
    }
    return success();
  }

  if (*activeK <= 0 || *activeK % *instructionK != 0 ||
      *activeK > aShape[1])
    return dot.emitOpError(
        "planned active_k must be a positive multiple of the operand type's "
        "WGMMA instruction K within the physical carrier");
  if (aShape[1] % *instructionK != 0) {
    return dot.emitOpError(
        "planned active_k requires physical K divisible by the WGMMA "
        "instruction K");
  }
  int64_t physicalSplits = aShape[1] / *instructionK;
  if ((physicalSplits & (physicalSplits - 1)) != 0)
    return dot.emitOpError(
        "planned active_k physical carrier must contain a power-of-two "
        "number of WGMMA K instructions");
  if (candidateB) {
    if (!hasCandidateBTopology() || candidateB->logicalShape.size() != 2 ||
        *activeK != candidateB->logicalShape[0] ||
        aShape[1] != nextPowerOfTwo(candidateB->logicalShape[0]) ||
        bShape[0] != aShape[1] ||
        candidateB->logicalShape[1] != candidateB->physicalShape[1] ||
        bShape[1] != candidateB->physicalShape[1])
      return dot.emitOpError(
          "planned active_k carrier does not match the logical SMEM stage");
  }
  return success();
}

LogicalResult LogicalDomainContext::processWGMMA(Operation *operation,
                                                 LogicalDomainPhase phase) {
  assert(phase == LogicalDomainPhase::Plan);
  auto dot = cast<WGMMAOp>(operation);
  const MemDescLogicalState *aMemDescState = lookupMemDesc(dot.getA());
  const MemDescLogicalState *bState = lookupMemDesc(dot.getB());
  const TensorFragmentState *aState = lookupTensor(dot.getA());
  if (aMemDescState)
    return emitError(dot, 0,
                     "candidate SMEM is not supported as WGMMA operand A");
  LogicalWGMMAAction action{dot, std::nullopt};
  std::optional<int64_t> activeK;
  unsigned activeKOperand = 0;
  int32_t activeKAxis = 1;
  if (bState) {
    auto *root = findRootAction(plan, bState->provenance.primaryRoot());
    if (!root)
      return emitError(dot, 1, "candidate root has no storage action");
    root->reachesWGMMA = true;
    bool restrictedK =
        bState->logicalShape[0] != bState->physicalShape[0];
    bool restrictedN =
        bState->logicalShape[1] != bState->physicalShape[1];
    if (restrictedK && restrictedN)
      return emitError(dot, 1,
                       "candidate WGMMA B has multiple fragment axes");
    if (restrictedN) {
      int64_t logicalN = bState->logicalShape[1];
      if (dot.getActiveK())
        return emitError(dot, 1, "candidate B N fragment cannot use active_k");
      if (dot.getActiveN() && *dot.getActiveN() != logicalN)
        return emitError(
            dot, 1,
            "explicit active_n disagrees with candidate B logical N");
      action.activeN = logicalN;
    } else if (restrictedK) {
      if (dot.getActiveN())
        return emitError(dot, 1, "candidate B K fragment cannot use active_n");
      int64_t logicalK = bState->logicalShape[0];
      if (dot.getActiveK() && *dot.getActiveK() != logicalK)
        return emitError(
            dot, 1,
            "explicit active_k disagrees with candidate B logical K");
      activeK = logicalK;
      activeKOperand = 1;
      activeKAxis = 0;
    }
  }
  if (aState) {
    auto aType = getTensorType(dot.getA());
    if (!aType || aType.getRank() != 2)
      return emitError(dot, 0, "restricted WGMMA A must have rank two");
    if (aState->axis < 0 || aState->axis >= 2)
      return emitError(dot, 0, "restricted WGMMA A axis is out of range");
    if (aState->axis == 1) {
      int64_t physicalK = aType.getShape()[1];
      int64_t logicalK = aState->logicalExtent;
      auto bType = dyn_cast<ttg::MemDescType>(dot.getB().getType());
      if (!bType || bType.getRank() != 2 || bType.getShape()[0] != physicalK)
        return emitError(dot, 0,
                         "A and B physical contraction extents disagree");
      std::optional<int64_t> instructionK =
          getWGMMAInstructionK(aType.getElementType());
      if (!instructionK || logicalK <= 0 ||
          logicalK % *instructionK != 0)
        return emitError(dot, 0,
                         "inferred active_k must be a positive multiple of "
                         "the operand type's WGMMA instruction K");
      if (dot.getActiveN() || action.activeN)
        return emitError(dot, 0,
                         "active_n and active_k cannot be combined");
      if (bState && bState->logicalShape[0] !=
                        bState->physicalShape[0] &&
          bState->logicalShape[0] != logicalK)
        return emitError(dot, 1,
                         "K/V logical extents disagree at the PV WGMMA join");
      if (dot.getActiveK() && *dot.getActiveK() != logicalK)
        return emitError(dot, 0,
                         "explicit active_k disagrees with logical A K");
      activeK = logicalK;
      activeKOperand = 0;
      activeKAxis = 1;
    }
  }
  if (failed(validatePlannedWGMMA(dot, action.activeN, activeK, bState)))
    return failure();
  if (action.activeN)
    plan.wgmmas.push_back(action);
  if (activeK)
    plan.folds.push_back(
        {dot, activeKOperand, activeKAxis, *activeK,
         LogicalFragmentFoldMechanism::WGMMAActiveK,
         LogicalReductionIdentity::Zero});
  if (lookupTensor(dot.getD()))
    return validateTensorResult(operation);
  return success();
}

LogicalResult
LogicalDomainContext::processWGMMAWait(Operation *operation,
                                       LogicalDomainPhase phase) {
  assert(phase == LogicalDomainPhase::Plan);
  return validateTensorResult(operation);
}

LogicalResult
LogicalDomainContext::processSameShape(Operation *operation,
                                       LogicalDomainPhase phase) {
  assert(phase == LogicalDomainPhase::Plan);
  for (auto [operandIndex, operand] :
       llvm::enumerate(operation->getOperands())) {
    const TensorFragmentState *state = lookupTensor(operand);
    if (!state)
      continue;
    auto operandType = getTensorType(operand);
    bool transferred = llvm::any_of(operation->getResults(), [&](Value result) {
      auto resultType = getTensorType(result);
      return operandType && resultType &&
             resultType.getShape() == operandType.getShape() &&
             lookupTensor(result);
    });
    if (!transferred)
      return emitError(operation, operandIndex,
                       "same-shape model could not transfer the restricted "
                       "operand to a tensor result");
  }
  return success();
}

LogicalResult
LogicalDomainContext::processExpandDims(Operation *operation,
                                        LogicalDomainPhase phase) {
  assert(phase == LogicalDomainPhase::Plan);
  return validateTensorResult(operation);
}

LogicalResult
LogicalDomainContext::processBroadcast(Operation *operation,
                                       LogicalDomainPhase phase) {
  assert(phase == LogicalDomainPhase::Plan);
  return validateTensorResult(operation);
}

LogicalResult
LogicalDomainContext::processTranspose(Operation *operation,
                                       LogicalDomainPhase phase) {
  assert(phase == LogicalDomainPhase::Plan);
  return validateTensorResult(operation);
}

LogicalResult LogicalDomainContext::processReshape(Operation *operation,
                                                   LogicalDomainPhase phase) {
  assert(phase == LogicalDomainPhase::Plan);
  auto reshape = cast<triton::ReshapeOp>(operation);
  const TensorFragmentState *state = lookupTensor(reshape.getSrc());
  if (!state)
    return success();
  auto srcType = getTensorType(reshape.getSrc());
  auto resultType = getTensorType(reshape.getResult());
  LogicalShape sourceLogical =
      materializeLogicalShape(reshape.getSrc(), *state);
  std::optional<LogicalShape> resultShape;
  if (reshape.getAllowReorder())
    return emitError(
        reshape, 0,
        "allow_reorder reshape may mix logical-prefix coordinates");
  if (srcType.getShape() == resultType.getShape()) {
    resultShape = sourceLogical;
  } else {
    resultShape = reshapeOnlyUnitDimensions(
        srcType.getShape(), sourceLogical, resultType.getShape());
    if (!resultShape) {
      auto prefix = getContiguousLinearPrefixSize(srcType.getShape(),
                                                  sourceLogical);
      if (prefix)
        resultShape =
            representLinearPrefixAsRectangle(resultType.getShape(), *prefix);
    }
  }
  if (!resultShape)
    return emitError(reshape, 0,
                     "reshape cannot prove a rectangular target prefix");
  return validateTensorResult(operation);
}

LogicalResult LogicalDomainContext::processCat(Operation *operation,
                                               LogicalDomainPhase phase) {
  assert(phase == LogicalDomainPhase::Plan);
  return validateTensorResult(operation);
}

LogicalResult LogicalDomainContext::processJoin(Operation *operation,
                                                LogicalDomainPhase phase) {
  assert(phase == LogicalDomainPhase::Plan);
  return validateTensorResult(operation);
}

LogicalResult LogicalDomainContext::processSplit(Operation *operation,
                                                 LogicalDomainPhase phase) {
  assert(phase == LogicalDomainPhase::Plan);
  auto split = cast<triton::SplitOp>(operation);
  const TensorFragmentState *state = lookupTensor(split.getSrc());
  if (!state)
    return success();
  auto sourceType = getTensorType(split.getSrc());
  if (!sourceType ||
      (state->axis == sourceType.getRank() - 1 &&
       state->logicalExtent != sourceType.getShape().back()))
    return emitError(split, 0,
                     "split requires the joined minor dimension to be full");
  return validateTensorResult(operation);
}

LogicalResult LogicalDomainContext::processReduce(Operation *operation,
                                                  LogicalDomainPhase phase) {
  assert(phase == LogicalDomainPhase::Plan);
  auto reduce = cast<triton::ReduceOp>(operation);
  for (auto [index, operand] : llvm::enumerate(reduce.getSrcs())) {
    const TensorFragmentState *state = lookupTensor(operand);
    if (!state)
      continue;
    int32_t axis = reduce.getAxis();
    auto type = getTensorType(operand);
    if (!type || axis < 0 || axis >= type.getRank())
      return emitError(reduce, index, "reduction axis is out of range");
    if (axis != state->axis)
      return emitError(
          reduce, index,
          "attention fragment reduction must consume the fragment axis");
    if (reduce.getNumOperands() != 1 || reduce.getNumResults() != 1)
      return emitError(reduce, index,
                       "restricted reduction supports one input and result");
    FailureOr<LogicalReductionIdentity> identity = getReductionIdentity(reduce);
    if (failed(identity))
      return emitError(reduce, index,
                       "reduction combiner has no supported tail identity");
    if (type.getEncoding())
      return emitError(reduce, index,
                       "strict early-TTIR analysis rejects encoded tensors");
    plan.folds.push_back(
        {reduce, static_cast<unsigned>(index), axis, state->logicalExtent,
         LogicalFragmentFoldMechanism::IdentityMask, *identity});
  }
  return success();
}

LogicalResult LogicalDomainContext::processDot(Operation *operation,
                                               LogicalDomainPhase phase) {
  assert(phase == LogicalDomainPhase::Plan);
  if (operation->getNumOperands() < 3 || operation->getNumResults() != 1)
    return processRejected(operation, phase, "malformed dot operation");
  const TensorFragmentState *aState = lookupTensor(operation->getOperand(0));
  const TensorFragmentState *bState = lookupTensor(operation->getOperand(1));
  const TensorFragmentState *cState = lookupTensor(operation->getOperand(2));
  if (!aState && !bState && !cState)
    return success();
  auto aType = getTensorType(operation->getOperand(0));
  auto bType = getTensorType(operation->getOperand(1));
  auto resultType = getTensorType(operation->getResult(0));
  if (!aType || !bType || !resultType || aType.getRank() != 2 ||
      bType.getRank() != 2 || resultType.getRank() != 2)
    return processRejected(operation, phase,
                           "logical dot currently requires rank-two tensors");
  if ((aState && aState->axis == 1) || (bState && bState->axis == 0))
    return emitError(operation, aState ? 0 : 1,
                     "ordinary tt.dot requires a full contraction K");
  if (isa<triton::DotScaledOp>(operation))
    for (unsigned i = 3; i < operation->getNumOperands(); ++i)
      if (lookupTensor(operation->getOperand(i)))
        return emitError(operation, i,
                         "dot_scaled scale operands must be full-shape");
  return validateTensorResult(operation);
}

LogicalResult LogicalDomainContext::processStore(Operation *operation,
                                                 LogicalDomainPhase phase) {
  auto store = cast<triton::StoreOp>(operation);
  for (auto [index, operand] : llvm::enumerate(operation->getOperands())) {
    const TensorFragmentState *state = lookupTensor(operand);
    if (!state)
      continue;
    if (index != 1 || !isGlobalPointerTensor(store.getPtr()))
      return emitError(store, index,
                       "only a value stored through global pointers is a "
                       "logical-domain terminal");
    if (phase == LogicalDomainPhase::Plan) {
      auto valueType = getTensorType(store.getValue());
      if (!valueType || valueType.getEncoding())
        return emitError(store, index,
                         "strict early-TTIR store requires an unencoded tensor");
      plan.guards.push_back(
          {store, 1,
           store.getMask() ? std::optional<unsigned>(2) : std::nullopt,
           state->axis, state->logicalExtent});
    }
  }
  return success();
}

LogicalResult
LogicalDomainContext::processAtomicRMW(Operation *operation,
                                       LogicalDomainPhase phase) {
  auto atomic = cast<triton::AtomicRMWOp>(operation);
  for (auto [index, operand] : llvm::enumerate(operation->getOperands())) {
    const TensorFragmentState *state = lookupTensor(operand);
    if (!state)
      continue;
    if (index != 1 || !isGlobalPointerTensor(atomic.getPtr()))
      return emitError(atomic, index,
                       "only an atomic value through global pointers is a "
                       "logical-domain terminal");
    if (!atomic.getResult().use_empty())
      return emitError(
          atomic, index,
          "attention fragment atomic_rmw requires an unused result");
    if (phase == LogicalDomainPhase::Plan) {
      auto valueType = getTensorType(atomic.getVal());
      if (!valueType || valueType.getEncoding())
        return emitError(atomic, index,
                         "strict early-TTIR atomic requires an unencoded tensor");
      plan.guards.push_back(
          {atomic, 1,
           atomic.getMask() ? std::optional<unsigned>(2) : std::nullopt,
           state->axis, state->logicalExtent});
    }
  }
  return success();
}

LogicalResult LogicalDomainContext::processRejected(
    Operation *operation, LogicalDomainPhase phase, StringRef reason) {
  if (phase == LogicalDomainPhase::Propagate)
    return success();
  for (auto [index, operand] : llvm::enumerate(operation->getOperands()))
    if (lookupTensor(operand) || lookupMemDesc(operand))
      return emitError(operation, index, reason);
  return success();
}

namespace {

static std::optional<LogicalBehavior>
getExplicitLogicalBehavior(Operation *op) {
  return llvm::TypeSwitch<Operation *, std::optional<LogicalBehavior>>(op)
      .Case<ttg::LocalAllocOp>(
          [](auto) { return LogicalBehavior::CandidateAlloc; })
      .Case<ttg::MemDescIndexOp>(
          [](auto) { return LogicalBehavior::MemDescIndex; })
      .Case<ttg::TMACopyOp>(
          [](auto) { return LogicalBehavior::LogicalTMACopy; })
      .Case<LocalPointersOp>([](LocalPointersOp op) {
        return op->hasAttr(kLogicalCopyShapeAttr)
                   ? LogicalBehavior::LogicalPointerCopy
                   : LogicalBehavior::RejectEscape;
      })
      .Case<ttg::WarpSpecializeOp>(
          [](auto) { return LogicalBehavior::WarpSpecialize; })
      .Case<MemDescWGMMAViewOp>(
          [](auto) { return LogicalBehavior::MemDescTranspose; })
      .Case<WGMMAOp>([](auto) { return LogicalBehavior::WGMMA; })
      .Case<WGMMAWaitOp>([](auto) { return LogicalBehavior::WGMMAWait; })
      .Case<ExclusiveCumsumOp>([](auto) { return LogicalBehavior::RejectScan; })
      .Case<PipeCreateOp, PipeWriterAcquireOp, PipeWriterCommitOp,
            PipeWriterCloseOp, PipeReaderWaitOp, PipeReaderReleaseOp>(
          [](auto) { return LogicalBehavior::Pipe; })
      .Case<WGMMASharedOperandFenceOp>(
          [](auto) { return LogicalBehavior::MemDescUse; })
      .Case<triton::ExpandDimsOp>(
          [](auto) { return LogicalBehavior::ExpandDims; })
      .Case<triton::BroadcastOp>(
          [](auto) { return LogicalBehavior::Broadcast; })
      .Case<triton::TransOp>([](auto) { return LogicalBehavior::Transpose; })
      .Case<triton::ReshapeOp>([](auto) { return LogicalBehavior::Reshape; })
      .Case<triton::CatOp>([](auto) { return LogicalBehavior::Cat; })
      .Case<triton::JoinOp>([](auto) { return LogicalBehavior::Join; })
      .Case<triton::SplitOp>([](auto) { return LogicalBehavior::Split; })
      .Case<triton::ReduceOp>([](auto) { return LogicalBehavior::Reduce; })
      .Case<triton::ScanOp>([](auto) { return LogicalBehavior::RejectScan; })
      .Case<triton::DotOp, triton::DotScaledOp>(
          [](auto) { return LogicalBehavior::Dot; })
      .Case<triton::StoreOp>([](auto) { return LogicalBehavior::Store; })
      .Case<triton::AtomicRMWOp>(
          [](auto) { return LogicalBehavior::AtomicRMW; })
      .Case<triton::GatherOp>(
          [](auto) { return LogicalBehavior::RejectGather; })
      .Case<triton::HistogramOp>(
          [](auto) { return LogicalBehavior::RejectHistogram; })
      .Case<triton::AtomicCASOp>(
          [](auto) { return LogicalBehavior::RejectAtomicCAS; })
      .Case<triton::LoadOp>([](auto) { return LogicalBehavior::RejectLoad; })
      .Case<ExtractTileOp, InsertTileOp, triton::CallOp, triton::ReturnOp>(
          [](auto) { return LogicalBehavior::RejectEscape; })
      .Default([](Operation *) -> std::optional<LogicalBehavior> {
        return std::nullopt;
      });
}

static bool hasRestrictedMemDescOperand(const LogicalDomainContext &context,
                                        Operation *op) {
  return llvm::any_of(op->getOperands(),
                      [&](Value value) { return context.lookupMemDesc(value); });
}

static bool isTensorControlFlow(Operation *op) {
  return isa<RegionBranchOpInterface, RegionBranchTerminatorOpInterface,
             BranchOpInterface>(op);
}

static bool containsOperand(OperandRange range, unsigned operandNumber) {
  if (range.empty())
    return false;
  unsigned begin = range.getBeginOperandIndex();
  return operandNumber >= begin && operandNumber < begin + range.size();
}

static bool isForwardedControlFlowOperand(Operation *op,
                                          unsigned operandNumber) {
  if (auto branch = dyn_cast<RegionBranchOpInterface>(op)) {
    SmallVector<RegionSuccessor> successors;
    branch.getSuccessorRegions(RegionBranchPoint::parent(), successors);
    for (RegionSuccessor successor : successors)
      if (containsOperand(
              branch.getEntrySuccessorOperands(RegionBranchPoint(successor)),
              operandNumber))
        return true;
  }
  if (auto terminator = dyn_cast<RegionBranchTerminatorOpInterface>(op)) {
    SmallVector<Attribute> operandAttrs(op->getNumOperands());
    SmallVector<RegionSuccessor> successors;
    terminator.getSuccessorRegions(operandAttrs, successors);
    for (RegionSuccessor successor : successors)
      if (containsOperand(
              terminator.getSuccessorOperands(RegionBranchPoint(successor)),
              operandNumber))
        return true;
  }
  if (auto branch = dyn_cast<BranchOpInterface>(op)) {
    for (unsigned index = 0; index < op->getNumSuccessors(); ++index)
      if (containsOperand(
              branch.getSuccessorOperands(index).getForwardedOperands(),
              operandNumber))
        return true;
  }
  return false;
}

static LogicalResult propagateMemDescDomains(ModuleOp module,
                                             LogicalDomainContext &context) {
  WalkResult seeded = module.walk([&](ttg::LocalAllocOp alloc) {
    if (failed(context.processCandidateAlloc(
            alloc, LogicalDomainPhase::Propagate)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (seeded.wasInterrupted())
    return failure();

  SmallVector<Value> worklist;
  DenseSet<Value> queued;
  auto enqueueChanged = [&] {
    for (Value value : context.takeChangedValues())
      if (queued.insert(value).second)
        worklist.push_back(value);
  };
  enqueueChanged();

  for (size_t next = 0; next < worklist.size(); ++next) {
    Value value = worklist[next];
    queued.erase(value);
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      LogicalResult result = success();
      if (isa<ttg::MemDescIndexOp>(user))
        result = context.processMemDescIndex(user,
                                             LogicalDomainPhase::Propagate);
      else if (isa<MemDescWGMMAViewOp>(user))
        result = context.processMemDescTranspose(
            user, LogicalDomainPhase::Propagate);
      else if (isa<ttg::WarpSpecializeOp>(user))
        result = context.processWarpSpecialize(
            user, LogicalDomainPhase::Propagate);
      if (failed(result))
        return failure();
    }
    enqueueChanged();
  }
  return success();
}

static LogicalResult collectTensorFacts(ModuleOp module,
                                        DataFlowSolver &solver,
                                        LogicalDomainPlan &plan) {
  bool failedCollection = false;
  auto collect = [&](Value value) {
    if (failedCollection)
      return;
    const TensorLattice *lattice = solver.lookupState<TensorLattice>(value);
    if (!lattice)
      return;
    const TensorFact &fact = lattice->getValue();
    if (fact.getKind() == TensorFact::Kind::Conflict) {
      Operation *owner = value.getDefiningOp();
      if (!owner)
        owner = cast<BlockArgument>(value).getOwner()->getParentOp();
      owner->emitOpError(
          "logical-domain dataflow would require multiple fragment axes");
      failedCollection = true;
      return;
    }
    if (fact.isFragment())
      plan.tensors.try_emplace(value, fact.getState());
  };

  module.walk([&](Operation *op) {
    for (Value result : op->getResults())
      collect(result);
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          collect(argument);
  });
  return failure(failedCollection);
}

static LogicalResult planOperation(LogicalDomainContext &context,
                                   Operation *op) {
  if (std::optional<LogicalBehavior> behavior =
          getExplicitLogicalBehavior(op))
    return dispatchLogicalBehavior(context, op, LogicalDomainPhase::Plan,
                                   *behavior);

  if (!context.hasRestrictedOperand(op))
    return success();

  // Shared-memory facts are intentionally not propagated through generic
  // operations or control flow: changing their physical representation
  // requires an explicit storage rewrite action.
  if (hasRestrictedMemDescOperand(context, op))
    return context.processRejected(
        op, LogicalDomainPhase::Plan,
        "operation has no logical-domain transfer semantics");

  if (isMapElementwiseTransfer(op) || isGenericElementwiseTransfer(op))
    return context.processSameShape(op, LogicalDomainPhase::Plan);

  // SparseForwardDataFlowAnalysis owns RegionBranch/CFG joins.  These ops do
  // not need a pass-specific model as long as only tensor facts cross them.
  if (isTensorControlFlow(op)) {
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      if (!context.lookupTensor(operand))
        continue;
      if (!isForwardedControlFlowOperand(op, index))
        return context.emitError(
            op, index,
            "restricted tensor is not forwarded across this control-flow "
            "edge");
    }
    return success();
  }

  return context.processRejected(
      op, LogicalDomainPhase::Plan,
      "operation has no logical-domain transfer semantics");
}

} // namespace

FailureOr<LogicalDomainPlan> analyzeLogicalDomains(ModuleOp module) {
  LogicalDomainPlan plan;
  plan.module = module;
  LogicalDomainContext context(plan);

  // Memdesc facts are settled first because WGMMA tensor transfer reads its B
  // descriptor.  Keeping this map immutable while the tensor solver runs
  // gives the solver a complete dependency boundary.
  if (failed(propagateMemDescDomains(module, context)))
    return failure();
  if (plan.memdescs.empty())
    return plan;

  std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
  solver->load<TensorLogicalDomainAnalysis>(plan.memdescs);
  WalkResult analyzed =
      module.walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (op->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
            failed(solver->initializeAndRun(op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      });
  if (analyzed.wasInterrupted() ||
      failed(collectTensorFacts(module, *solver, plan)))
    return failure();

  // Materialize every storage root before planning consumers, independent of
  // their nesting or traversal order.
  WalkResult roots = module.walk([&](ttg::LocalAllocOp alloc) {
    if (failed(context.processCandidateAlloc(alloc,
                                             LogicalDomainPhase::Plan)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (roots.wasInterrupted())
    return failure();

  WalkResult planned = module.walk([&](Operation *op) {
    if (failed(planOperation(context, op)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (planned.wasInterrupted())
    return failure();

  for (const LogicalRootRewriteAction &root : plan.roots) {
    if (root.stages.empty()) {
      root.alloc->emitOpError("logical domain has no stage views");
      return failure();
    }
    if ((root.copies.empty() && root.pointerCopies.empty()) ||
        !root.reachesWGMMA) {
      root.alloc->emitOpError(
          "logical domain must reach both an exact producer and WGMMA");
      return failure();
    }
  }
  return plan;
}

namespace {
struct PredicateCacheEntry {
  Block *block;
  LogicalShape physicalShape;
  int32_t axis;
  int64_t logicalExtent;
  Value predicate;
};

static Value createLogicalPredicate(
    OpBuilder &builder, Operation *anchor, RankedTensorType tensorType,
    int32_t axis, int64_t logicalExtent,
    SmallVectorImpl<PredicateCacheEntry> &cache) {
  assert(!tensorType.getEncoding() && "strict TTIR predicate must be unencoded");
  for (const PredicateCacheEntry &entry : cache) {
    Operation *predicateOp = entry.predicate.getDefiningOp();
    if (entry.block == builder.getInsertionBlock() &&
        ArrayRef<int64_t>(entry.physicalShape) == tensorType.getShape() &&
        entry.axis == axis &&
        entry.logicalExtent == logicalExtent && predicateOp &&
        predicateOp->isBeforeInBlock(anchor))
      return entry.predicate;
  }
  int32_t rank = tensorType.getRank();
  int64_t physicalExtent = tensorType.getShape()[axis];
  auto rangeType = RankedTensorType::get({physicalExtent}, builder.getI32Type());
  Value expanded = triton::MakeRangeOp::create(
      builder, anchor->getLoc(), rangeType, 0, physicalExtent);
  LogicalShape expandedShape{physicalExtent};
  for (int32_t dim = 0; dim < rank; ++dim) {
    if (dim == axis)
      continue;
    expandedShape.insert(expandedShape.begin() + dim, 1);
    expanded = triton::ExpandDimsOp::create(
        builder, anchor->getLoc(),
        RankedTensorType::get(expandedShape, builder.getI32Type()), expanded,
        dim);
  }
  Value extent = arith::ConstantIntOp::create(
      builder, anchor->getLoc(), builder.getI32Type(), logicalExtent);
  Value extentTensor = triton::SplatOp::create(
      builder, anchor->getLoc(), expanded.getType(), extent);
  Value predicate = arith::CmpIOp::create(
      builder, anchor->getLoc(), arith::CmpIPredicate::slt, expanded,
      extentTensor);
  if (expandedShape != tensorType.getShape())
    predicate = triton::BroadcastOp::create(
        builder, anchor->getLoc(),
        RankedTensorType::get(tensorType.getShape(), builder.getI1Type()),
        predicate);
  cache.push_back({builder.getInsertionBlock(),
                   LogicalShape(tensorType.getShape().begin(),
                                tensorType.getShape().end()),
                   axis, logicalExtent, predicate});
  return predicate;
}

static Value createIdentityTensor(OpBuilder &builder, Location loc,
                                  RankedTensorType type,
                                  LogicalReductionIdentity identity) {
  Type elementType = type.getElementType();
  Attribute scalar;
  if (auto floatType = dyn_cast<FloatType>(elementType)) {
    double value = 0.0;
    switch (identity) {
    case LogicalReductionIdentity::NegativeInfinity:
      value = -std::numeric_limits<double>::infinity();
      break;
    case LogicalReductionIdentity::PositiveInfinity:
      value = std::numeric_limits<double>::infinity();
      break;
    case LogicalReductionIdentity::One:
      value = 1.0;
      break;
    case LogicalReductionIdentity::Zero:
      break;
    default:
      llvm_unreachable("boolean identity used with float tensor");
    }
    scalar = FloatAttr::get(floatType, value);
  } else {
    auto integerType = cast<IntegerType>(elementType);
    APInt value(integerType.getWidth(), 0);
    if (identity == LogicalReductionIdentity::True)
      value = APInt::getAllOnes(integerType.getWidth());
    else if (identity == LogicalReductionIdentity::One)
      value = APInt(integerType.getWidth(), 1);
    scalar = IntegerAttr::get(integerType, value);
  }
  return arith::ConstantOp::create(builder, loc,
                                   DenseElementsAttr::get(type, scalar));
}

static void replaceFollowingUses(Value original, Value replacement,
                                 Operation *anchor) {
  auto replace = [&](Operation *op) {
    if (isa<triton::ReduceOp>(op))
      return;
    for (OpOperand &operand : op->getOpOperands())
      if (operand.get() == original)
        operand.set(replacement);
  };
  for (Operation *following = anchor->getNextNode(); following;
       following = following->getNextNode()) {
    replace(following);
    following->walk([&](Operation *nested) {
      if (nested != following)
        replace(nested);
    });
  }
}
} // namespace

void applyLogicalTensorActions(LogicalDomainPlan &plan) {
  for (LogicalWGMMAAction &action : plan.wgmmas) {
    if (action.activeN && !action.op.getActiveN())
      action.op->setAttr("active_n", IntegerAttr::get(
                                         IntegerType::get(
                                             plan.module.getContext(), 32),
                                         *action.activeN));
  }
  SmallVector<PredicateCacheEntry> predicateCache;
  for (LogicalFragmentFoldAction &action : plan.folds) {
    if (action.mechanism ==
        LogicalFragmentFoldMechanism::WGMMAActiveK) {
      auto dot = cast<WGMMAOp>(action.op);
      if (!dot.getActiveK())
        dot->setAttr("active_k", IntegerAttr::get(
                                     IntegerType::get(
                                         plan.module.getContext(), 32),
                                     action.logicalExtent));
      continue;
    }
    Value operand = action.op->getOperand(action.operandIndex);
    auto type = cast<RankedTensorType>(operand.getType());
    OpBuilder builder(action.op);
    Value predicate = createLogicalPredicate(
        builder, action.op, type, action.axis, action.logicalExtent,
        predicateCache);
    Value identity = createIdentityTensor(builder, action.op->getLoc(), type,
                                          action.identity);
    Value masked = arith::SelectOp::create(builder, action.op->getLoc(),
                                           predicate, operand, identity);
    action.op->setOperand(action.operandIndex, masked);
    if (action.identity == LogicalReductionIdentity::NegativeInfinity)
      replaceFollowingUses(operand, masked, action.op);
  }
  for (LogicalFragmentGuardAction &action : plan.guards) {
    Value value = action.op->getOperand(action.valueOperand);
    auto type = cast<RankedTensorType>(value.getType());
    OpBuilder builder(action.op);
    Value logicalMask = createLogicalPredicate(
        builder, action.op, type, action.axis, action.logicalExtent,
        predicateCache);
    if (action.maskOperand) {
      Value userMask = action.op->getOperand(*action.maskOperand);
      logicalMask = arith::AndIOp::create(builder, action.op->getLoc(),
                                          userMask, logicalMask);
    }
    if (auto store = dyn_cast<triton::StoreOp>(action.op))
      store.getMaskMutable().assign(logicalMask);
    else
      cast<triton::AtomicRMWOp>(action.op).getMaskMutable().assign(logicalMask);
  }
}

} // namespace mlir::triton::tle
