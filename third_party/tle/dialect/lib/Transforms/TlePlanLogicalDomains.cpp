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
constexpr int64_t kHopperThreadsPerWarp = 32;
constexpr int64_t kCpAsyncMaxTransactionBytes = 16;
constexpr int64_t kNVMMACoreRows = 8;

static SmallVector<int64_t, 2> selectLogicalPointerCopyMicroTile(
    Operation *copy, ArrayRef<int64_t> storageTileShape,
    ttg::NVMMASharedEncodingAttr storageEncoding, Type elementType) {
  assert(storageTileShape.size() == 2 &&
         "logical pointer-copy storage tile must be rank two");
  assert(elementType.isIntOrFloat() &&
         "exact-SMEM element type must have a bit width");

  // Size one copy wave for the participating producer threads.  The later
  // async-copy legality pass still selects 4/8/16-byte transactions from
  // AxisInfo; this is only the rectangular work partition presented to it.
  int64_t numWarps = ttg::maybeLookupNumWarps(copy).value_or(4);
  int64_t elementBits = elementType.getIntOrFloatBitWidth();
  int64_t targetElements =
      numWarps * kHopperThreadsPerWarp * kCpAsyncMaxTransactionBytes * 8 /
      elementBits;

  unsigned contiguousAxis = storageEncoding.getTransposed() ? 0u : 1u;
  unsigned outerAxis = 1u - contiguousAxis;
  int64_t swizzleBytes =
      std::max<int64_t>(storageEncoding.getSwizzlingByteWidth(), 16);
  int64_t swizzleElements = swizzleBytes * 8 / elementBits;
  int64_t minContiguous =
      std::min(storageTileShape[contiguousAxis], swizzleElements);
  int64_t minOuter = std::min(storageTileShape[outerAxis], kNVMMACoreRows);

  // Both storage extents are powers of two, so enumerating their divisors is
  // cheap.  Maximize useful work within one producer wave while keeping at
  // least one complete NVMMA core.  For equal areas prefer one swizzle span;
  // this avoids the larger register layout and code size seen when 16x128 or
  // 16x256 fp16 copies exceed the useful work of one four-warp wave.
  SmallVector<int64_t, 2> best{minOuter, minContiguous};
  if (contiguousAxis == 0)
    std::swap(best[0], best[1]);
  int64_t bestElements = best[0] * best[1];
  bool bestFits = bestElements <= targetElements;
  for (int64_t rows = 1; rows <= storageTileShape[0]; rows *= 2) {
    for (int64_t cols = 1; cols <= storageTileShape[1]; cols *= 2) {
      SmallVector<int64_t, 2> candidate{rows, cols};
      if (candidate[contiguousAxis] < minContiguous ||
          candidate[outerAxis] < minOuter)
        continue;
      int64_t elements = rows * cols;
      bool fits = elements <= targetElements;
      if (fits != bestFits) {
        if (!fits)
          continue;
      } else if (fits && elements < bestElements) {
        continue;
      } else if (!fits && elements > bestElements) {
        continue;
      }
      if (fits != bestFits || elements != bestElements ||
          candidate[contiguousAxis] < best[contiguousAxis]) {
        best = candidate;
        bestElements = elements;
        bestFits = fits;
      }
    }
  }
  return best;
}

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

static Value createStaticExtractTile(OpBuilder &builder, Location loc,
                                     Value value, int64_t linearTile,
                                     ArrayRef<int64_t> tileShape) {
  if (!value)
    return {};
  Value index = arith::ConstantIntOp::create(builder, loc, builder.getI32Type(),
                                             linearTile);
  return ExtractTileOp::create(builder, loc, value, index, tileShape)
      .getResult();
}

static Value createFullTileIndex(OpBuilder &builder, Location loc,
                                 ArrayRef<int64_t> tileShape, unsigned axis,
                                 int64_t offset) {
  auto rangeType =
      RankedTensorType::get({tileShape[axis]}, builder.getI32Type());
  Value index = triton::MakeRangeOp::create(builder, loc, rangeType, offset,
                                            offset + tileShape[axis]);
  SmallVector<int64_t, 2> expandedShape{tileShape[axis]};
  unsigned expandAxis = axis == 0 ? 1u : 0u;
  expandedShape.insert(expandedShape.begin() + expandAxis, 1);
  index = triton::ExpandDimsOp::create(
      builder, loc, RankedTensorType::get(expandedShape, builder.getI32Type()),
      index, expandAxis);
  if (expandedShape != tileShape)
    index = triton::BroadcastOp::create(
        builder, loc, RankedTensorType::get(tileShape, builder.getI32Type()),
        index);
  return index;
}

static void applyLogicalPointerCopy(LogicalPointerCopyAction &action,
                                    ArrayRef<int64_t> storageTileShape,
                                    int64_t tilesPerStage,
                                    unsigned fragmentAxis) {
  LocalPointersOp pointers = action.pointers;
  triton::StoreOp store = action.store;
  triton::LoadOp load = action.load;
  ExactSMEMStage stage = getExactSMEMStage(pointers.getSrc());
  assert(stage && "validated logical pointer destination must be a stage");

  auto carrierType = cast<RankedTensorType>(load.getPtr().getType());
  ArrayRef<int64_t> carrierShape = carrierType.getShape();
  auto storageEncoding = cast<ttg::NVMMASharedEncodingAttr>(
      stage.atom.getType().getEncoding());
  SmallVector<int64_t, 2> microTileShape = selectLogicalPointerCopyMicroTile(
      store, storageTileShape, storageEncoding,
      stage.atom.getType().getElementType());
  int64_t storageRowTiles = storageTileShape[0] / microTileShape[0];
  int64_t storageColTiles = storageTileShape[1] / microTileShape[1];
  int64_t sourceGridCols = carrierShape[1] / microTileShape[1];
  auto originalPtrType = cast<RankedTensorType>(pointers.getResult().getType());
  auto microPtrType =
      RankedTensorType::get(microTileShape, originalPtrType.getElementType(),
                            originalPtrType.getEncoding());

  OpBuilder builder(store);
  Location loc = store.getLoc();
  for (int64_t fragmentTile = 0; fragmentTile < tilesPerStage; ++fragmentTile) {
    Value flatIndex =
        addOffset(builder, loc, stage.atom.getIndex(), fragmentTile);
    auto storageTile = ttg::MemDescIndexOp::create(
        builder, loc, stage.atom.getType(), stage.atom.getSrc(), flatIndex);
    storageTile->setAttr(kExactSMEMTileAttr,
                         builder.getI32IntegerAttr(fragmentTile));

    for (int64_t storageRowTile = 0; storageRowTile < storageRowTiles;
         ++storageRowTile) {
      for (int64_t storageColTile = 0; storageColTile < storageColTiles;
           ++storageColTile) {
        int64_t sourceRowTile =
            fragmentAxis == 0
                ? fragmentTile * storageRowTiles + storageRowTile
                : storageRowTile;
        int64_t sourceColTile =
            fragmentAxis == 1
                ? fragmentTile * storageColTiles + storageColTile
                : storageColTile;
        int64_t sourceLinearTile =
            sourceRowTile * sourceGridCols + sourceColTile;
        Value tiledPtr = createStaticExtractTile(
            builder, loc, load.getPtr(), sourceLinearTile, microTileShape);
        Value tiledMask = createStaticExtractTile(
            builder, loc, load.getMask(), sourceLinearTile, microTileShape);
        Value tiledOther = createStaticExtractTile(
            builder, loc, load.getOther(), sourceLinearTile, microTileShape);
        Value tiledLoad = triton::LoadOp::create(
            builder, loc, tiledPtr, tiledMask, tiledOther, load.getCache(),
            load.getEvict(), load.getIsVolatile(), load.getFlagtreeHintsAttr());

        SmallVector<Value, 2> indices;
        indices.push_back(createFullTileIndex(
            builder, loc, microTileShape, 0,
            storageRowTile * microTileShape[0]));
        indices.push_back(createFullTileIndex(
            builder, loc, microTileShape, 1,
            storageColTile * microTileShape[1]));
        Value tiledPointers = LocalPointersOp::create(
            builder, loc, microPtrType, storageTile, indices);
        triton::StoreOp::create(builder, loc, tiledPointers, tiledLoad,
                                store.getCache(), store.getEvict());
      }
    }
  }

  store.erase();
  pointers.erase();
  load.erase();
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

  for (LogicalPointerCopyAction &copy : action.pointerCopies)
    applyLogicalPointerCopy(copy, storageTileShape, tilesPerStage,
                            fragmentAxis);

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
