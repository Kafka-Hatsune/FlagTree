/*
 * Copyright 2025- FlagOS Contributors
 * SPDX-License-Identifier: MIT
 */

#include "tle/dialect/include/Transforms/LogicalDomain.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "tle/dialect/include/IR/ExactSMEM.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::triton::tle {
namespace ttg = mlir::triton::gpu;

#define GEN_PASS_DEF_TRITONTLEPLANLOGICALDOMAINS
#include "tle/dialect/include/Transforms/Passes.h.inc"

// Implemented with the tensor actions so it can reuse the predicate/tail
// materialization helpers without exposing them as public APIs.
void applyLogicalTensorActions(LogicalDomainPlan &plan);

namespace {
constexpr llvm::StringLiteral kTiledFields("tiled_smem_fields");

static void addTiledPipeField(Operation *op, unsigned fieldIndex,
                              OpBuilder &builder) {
  SmallVector<int32_t> fields;
  if (auto old = op->getAttrOfType<DenseI32ArrayAttr>(kTiledFields))
    fields.assign(old.asArrayRef().begin(), old.asArrayRef().end());
  if (!llvm::is_contained(fields, static_cast<int32_t>(fieldIndex)))
    fields.push_back(fieldIndex);
  llvm::sort(fields);
  op->setAttr(kTiledFields, builder.getDenseI32ArrayAttr(fields));
}

static Value addOffset(OpBuilder &builder, Location loc, Value base,
                       int64_t delta) {
  if (delta == 0)
    return base;
  auto intType = cast<IntegerType>(base.getType());
  Value constant = arith::ConstantIntOp::create(
      builder, loc, delta, intType.getWidth());
  return arith::AddIOp::create(builder, loc, base, constant);
}

static void applyRootRewrite(LogicalRootRewriteAction &action) {
  ttg::LocalAllocOp oldAlloc = action.alloc;
  auto oldType = oldAlloc.getType();
  int64_t capacity = action.logicalShape[0];
  int64_t rows = action.logicalShape[1];
  int64_t cols = action.logicalShape[2];
  assert(action.storageTileShape.size() == 2 &&
         "validated root must select a storage tile");
  int64_t storageTileRows = action.storageTileShape[0];
  int64_t storageTileCols = action.storageTileShape[1];
  unsigned fragmentAxis = storageTileRows != rows ? 0u : 1u;
  int64_t fragmentExtent = fragmentAxis == 0 ? rows : cols;
  int64_t fragmentTileExtent =
      fragmentAxis == 0 ? storageTileRows : storageTileCols;
  int64_t tilesPerStage = fragmentExtent / fragmentTileExtent;
  SmallVector<int64_t> exactShape{capacity * tilesPerStage, storageTileRows,
                                  storageTileCols};
  SmallVector<int64_t> storageTileShape{storageTileRows, storageTileCols};
  assert(!action.stages.empty() &&
         "validated root must have at least one stage view");
  auto carrierEncoding = cast<ttg::NVMMASharedEncodingAttr>(
      action.stages.front().getType().getEncoding());
  auto storageTileEncoding = getExactSMEMStorageTileEncoding(
      carrierEncoding, storageTileShape, oldType.getElementType());

  OpBuilder builder(oldAlloc);
  auto ctaLayout = ttg::CTAEncodingAttr::getDefault(oldAlloc.getContext(), 2);
  auto exactEncoding = ttg::SwizzledSharedEncodingAttr::get(
      oldAlloc.getContext(), 8, 1, 1, {1, 0}, ctaLayout);
  auto exactType = ttg::MemDescType::get(
      exactShape, oldType.getElementType(), exactEncoding,
      oldType.getMemorySpace(), oldType.getMutableMemory(), exactShape);
  auto exactAlloc = ttg::LocalAllocOp::create(builder, oldAlloc.getLoc(),
                                               exactType, Value());
  exactAlloc->setAttr(kExactSMEMShapeAttr,
                      builder.getDenseI64ArrayAttr(action.logicalShape));
  if (IntegerAttr alignment = oldAlloc.getAlignmentAttr())
    exactAlloc.setAlignmentAttr(alignment);

  for (LogicalMemDescUseAction &memdescUse : action.memdescUses) {
    OpOperand *use = memdescUse.use;
    unsigned fieldIndex = use->getOperandNumber();
    if (use->get() == oldAlloc.getResult()) {
      use->set(exactAlloc);
      if (auto warpSpecialize =
              dyn_cast<ttg::WarpSpecializeOp>(use->getOwner()))
        for (Region *partition : warpSpecialize.getPartitionRegions())
          partition->getArgument(fieldIndex).setType(exactType);
    }
  }
  for (LogicalMemDescUseAction &memdescUse : action.memdescUses) {
    OpOperand *use = memdescUse.use;
    unsigned fieldIndex = use->getOperandNumber();
    assert(use->get().getType() == exactType &&
           "forwarded exact-SMEM use must have rewritten storage type");
    if (memdescUse.markTiledPipeField)
      addTiledPipeField(use->getOwner(), fieldIndex, builder);
  }

  for (ttg::MemDescIndexOp stage : action.stages) {
    OpBuilder stageBuilder(stage);
    Value tileCount = arith::ConstantIntOp::create(
        stageBuilder, stage.getLoc(), stageBuilder.getI32Type(),
        tilesPerStage);
    Value stageBase = arith::MulIOp::create(
        stageBuilder, stage.getLoc(), stage.getIndex(), tileCount);
    auto storageTileType = ttg::MemDescType::get(
        storageTileShape, oldType.getElementType(), storageTileEncoding,
        oldType.getMemorySpace(),
        oldType.getMutableMemory(), storageTileShape);
    Value storage = stage.getSrc();
    if (storage == oldAlloc.getResult())
      storage = exactAlloc;
    assert(storage.getType() == exactType &&
           "exact-SMEM stage must index its rewritten storage value");
    Value storageTile = ttg::MemDescIndexOp::create(
        stageBuilder, stage.getLoc(), storageTileType, storage, stageBase);
    auto carrier = ttg::MemDescReinterpretOp::create(
        stageBuilder, stage.getLoc(), stage.getType(), storageTile);
    carrier->setAttr(kExactSMEMStageAttr, stageBuilder.getUnitAttr());
    stage.getResult().replaceAllUsesWith(carrier);
  }

  for (ttg::TMACopyOp copy : action.copies) {
    OpBuilder copyBuilder(copy);
    ExactSMEMStage stage = getExactSMEMStage(copy.getDst());
    assert(stage && "validated logical TMA destination must be a stage view");
    auto stageType = stage.getType();
    auto storageTileType = ttg::MemDescType::get(
        storageTileShape, stageType.getElementType(),
        stage.atom.getType().getEncoding(), stageType.getMemorySpace(),
        stageType.getMutableMemory(),
        storageTileShape);
    for (int64_t tile = 0; tile < tilesPerStage; ++tile) {
      Value flatIndex = addOffset(copyBuilder, copy.getLoc(),
                                  stage.atom.getIndex(), tile);
      auto storageTile = ttg::MemDescIndexOp::create(
          copyBuilder, copy.getLoc(), storageTileType, stage.atom.getSrc(),
          flatIndex);
      storageTile->setAttr(kExactSMEMTileAttr,
                           copyBuilder.getI32IntegerAttr(tile));
      SmallVector<Value> indices(copy.getIndices().begin(),
                                 copy.getIndices().end());
      unsigned coordinate = indices.size() - 2 + fragmentAxis;
      indices[coordinate] = addOffset(copyBuilder, copy.getLoc(),
                                      indices[coordinate],
                                      tile * fragmentTileExtent);
#ifdef __HCU__
      auto tiledCopy = ttg::TMACopyOp::create(copyBuilder, copy.getLoc(),
                                              copy.getSrc(), storageTile,
                                              indices);
#else
      auto tiledCopy = ttg::TMACopyOp::create(
          copyBuilder, copy.getLoc(), copy.getSrc(), storageTile, indices,
          Value(), IntegerAttr());
#endif
      int64_t elementBytes =
          oldType.getElementType().getIntOrFloatBitWidth() / 8;
      tiledCopy->setAttr(
          kLogicalTMACopyBytesAttr,
          copyBuilder.getI64IntegerAttr(storageTileRows * storageTileCols *
                                        elementBytes));
    }
  }

  for (ttg::TMACopyOp copy : action.copies)
    copy.erase();
  for (ttg::MemDescIndexOp stage : action.stages)
    stage.erase();
  oldAlloc.erase();
}

struct TritonTlePlanLogicalDomains
    : public impl::TritonTlePlanLogicalDomainsBase<
          TritonTlePlanLogicalDomains> {
  void runOnOperation() override {
    FailureOr<LogicalDomainPlan> plan = analyzeLogicalDomains(getOperation());
    if (failed(plan)) {
      signalPassFailure();
      return;
    }
    applyLogicalDomainPlan(std::move(*plan));
  }
};
} // namespace

void applyLogicalDomainPlan(LogicalDomainPlan &&plan) {
  for (LogicalRootRewriteAction &root : plan.roots)
    applyRootRewrite(root);
  applyLogicalTensorActions(plan);
}

} // namespace mlir::triton::tle
