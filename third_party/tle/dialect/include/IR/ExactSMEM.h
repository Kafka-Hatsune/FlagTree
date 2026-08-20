/*
 * Copyright 2025- FlagOS Contributors
 * SPDX-License-Identifier: MIT
 */

#ifndef TRITON_TLE_IR_EXACT_SMEM_H_
#define TRITON_TLE_IR_EXACT_SMEM_H_

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"
#include <limits>
#include <optional>

namespace mlir::triton::tle {

namespace ttg = mlir::triton::gpu;

inline constexpr llvm::StringLiteral
    kExactSMEMShapeAttr("tle.exact_smem_shape");
inline constexpr llvm::StringLiteral
    kExactSMEMStageAttr("tle.exact_smem_stage");
inline constexpr llvm::StringLiteral
    kExactSMEMTileAttr("tle.exact_smem_tile");

inline constexpr int64_t kExactSMEMRowQuantum = 16;
inline constexpr int64_t kExactSMEMColQuantum = 64;

/// Element types accepted by the Hopper exact-SMEM/WGMMA path.  Keep this in
/// sync with Triton's WGMMA operand lowering rather than treating exact SMEM
/// as inherently 16-bit storage.
inline bool isSupportedExactSMEMElementType(Type type) {
  return type.isF16() || type.isBF16() || type.isF32() ||
         type.isInteger(8) ||
         isa<Float8E5M2Type, Float8E4M3FNType>(type);
}

inline bool isWGMMAFloat8ElementType(Type type) {
  return isa<Float8E5M2Type, Float8E4M3FNType>(type);
}

/// Hopper WGMMA consumes 256 operand bits along K per instruction.  This is
/// the same rule used by Triton's mmaVersionToInstrShape(v3, ...).
inline std::optional<int64_t> getWGMMAInstructionK(Type type) {
  if (!isSupportedExactSMEMElementType(type))
    return std::nullopt;
  return 256 / type.getIntOrFloatBitWidth();
}

inline bool supportsWGMMAOperandTranspose(Type type) {
  return type.isF16() || type.isBF16();
}

/// Re-run Triton's NVMMAShared swizzle selection for the exact storage tile.
/// A carrier may use a wider swizzle than a tile whose contiguous fragment
/// axis is smaller, so inheriting the carrier encoding is not generally
/// valid.
inline ttg::NVMMASharedEncodingAttr getExactSMEMStorageTileEncoding(
    ttg::NVMMASharedEncodingAttr carrierEncoding,
    ArrayRef<int64_t> storageTileShape, Type elementType) {
  SmallVector<unsigned, 2> order =
      carrierEncoding.getTransposed() ? SmallVector<unsigned, 2>{0, 1}
                                      : SmallVector<unsigned, 2>{1, 0};
  return ttg::NVMMASharedEncodingAttr::get(
      carrierEncoding.getContext(), storageTileShape, order,
      carrierEncoding.getCTALayout(), elementType,
      carrierEncoding.getFp4Padded());
}

/// Active logical extents currently keep one 32-bit accumulator element per
/// register.  This covers float WGMMA with f32 accumulation and integer WGMMA
/// with i32 accumulation while preserving the existing active-N carrier code.
inline bool isSupportedWGMMATypeCombination(
    Type aType, Type bType, Type accumulatorType,
    triton::InputPrecision inputPrecision) {
  if (aType.isF16() || aType.isBF16())
    return aType == bType &&
           (aType.isF16()
                ? accumulatorType.isF16() || accumulatorType.isF32()
                : accumulatorType.isF32());
  if (aType.isF32())
    return bType.isF32() && accumulatorType.isF32() &&
           inputPrecision == triton::InputPrecision::TF32;
  if (isWGMMAFloat8ElementType(aType))
    return isWGMMAFloat8ElementType(bType) &&
           (accumulatorType.isF16() || accumulatorType.isF32());
  if (aType.isInteger(8))
    return bType.isInteger(8) && accumulatorType.isInteger(32);
  return false;
}

inline bool isSupportedActiveWGMMATypeCombination(
    Type aType, Type bType, Type accumulatorType,
    triton::InputPrecision inputPrecision) {
  return accumulatorType.getIntOrFloatBitWidth() == 32 &&
         isSupportedWGMMATypeCombination(aType, bType, accumulatorType,
                                         inputPrecision);
}

struct ExactSMEMRoot {
  ttg::LocalAllocOp alloc;
  int64_t capacity = 0;
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t atomRows = 0;
  int64_t atomCols = 0;

  explicit operator bool() const { return static_cast<bool>(alloc); }
  int64_t getAtomRows() const { return atomRows; }
  int64_t getAtomCols() const { return atomCols; }
  int64_t getRowTiles() const { return rows / atomRows; }
  int64_t getColTiles() const { return cols / atomCols; }
  int64_t getTilesPerStage() const { return getRowTiles() * getColTiles(); }
};

struct ExactSMEMStage {
  ttg::MemDescReinterpretOp view;
  ttg::MemDescIndexOp atom;
  ExactSMEMRoot root;
  Value stage;
  std::optional<int64_t> staticStage;
  bool transposed = false;

  explicit operator bool() const { return static_cast<bool>(view); }
  Value getSrc() { return root.alloc.getResult(); }
  Value getStage() const { return stage; }
  std::optional<int64_t> getStaticStage() const { return staticStage; }
  int64_t getCapacity() const { return root.capacity; }
  int64_t getRows() const { return root.rows; }
  int64_t getCols() const { return root.cols; }
  int64_t getAtomRows() const { return root.getAtomRows(); }
  int64_t getAtomCols() const { return root.getAtomCols(); }
  int64_t getLogicalK() const { return transposed ? root.cols : root.rows; }
  int64_t getLogicalN() const { return transposed ? root.rows : root.cols; }
  int64_t getCarrierK() const {
    return static_cast<int64_t>(llvm::PowerOf2Ceil(getLogicalK()));
  }
  int64_t getCarrierN() const {
    return static_cast<int64_t>(llvm::PowerOf2Ceil(getLogicalN()));
  }
  ttg::MemDescType getType() { return view.getType(); }
  Operation *getOperation() { return view.getOperation(); }
};

struct ExactSMEMTile {
  ttg::MemDescIndexOp view;
  ExactSMEMRoot root;
  Value stage;
  std::optional<int64_t> staticStage;
  int64_t tile = 0;

  explicit operator bool() const { return static_cast<bool>(view); }
  Value getSrc() { return root.alloc.getResult(); }
  Value getStage() const { return stage; }
  std::optional<int64_t> getStaticStage() const { return staticStage; }
  int64_t getTile() const { return tile; }
  int64_t getCapacity() const { return root.capacity; }
  int64_t getRows() const { return root.rows; }
  int64_t getCols() const { return root.cols; }
  int64_t getAtomRows() const { return root.getAtomRows(); }
  int64_t getAtomCols() const { return root.getAtomCols(); }
  ttg::MemDescType getType() { return view.getType(); }
  Operation *getOperation() { return view.getOperation(); }
};

inline std::optional<int64_t> getExactSMEMConstant(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantIntOp>())
    return constant.value();
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value();
  return std::nullopt;
}

inline ExactSMEMRoot getExactSMEMRoot(Value value) {
  auto alloc = value.getDefiningOp<ttg::LocalAllocOp>();
  if (!alloc)
    return {};
  auto shape = alloc->getAttrOfType<DenseI64ArrayAttr>(kExactSMEMShapeAttr);
  auto type = alloc.getType();
  if (!shape || shape.size() != 3 || type.getRank() != 3)
    return {};
  ArrayRef<int64_t> values = shape.asArrayRef();
  ArrayRef<int64_t> storageShape = type.getShape();
  return {alloc, values[0], values[1], values[2], storageShape[1],
          storageShape[2]};
}

struct ExactSMEMStageIndex {
  Value value;
  std::optional<int64_t> constant;
};

inline std::optional<ExactSMEMStageIndex>
matchExactSMEMStage(Value flatIndex, int64_t tilesPerStage,
                    int64_t tileOffset = 0) {
  Value stageBase = flatIndex;
  if (auto add = stageBase.getDefiningOp<arith::AddIOp>()) {
    if (getExactSMEMConstant(add.getLhs()) == tileOffset)
      stageBase = add.getRhs();
    else if (getExactSMEMConstant(add.getRhs()) == tileOffset)
      stageBase = add.getLhs();
    else
      return std::nullopt;
  } else if (tileOffset != 0) {
    // Canonicalization may fold stage * tilesPerStage + tileOffset to one
    // constant. Recover the logical stage without mutating the IR.
    auto flattened = getExactSMEMConstant(stageBase);
    if (!flattened || *flattened < tileOffset ||
        (*flattened - tileOffset) % tilesPerStage != 0)
      return std::nullopt;
    return ExactSMEMStageIndex{
        stageBase, (*flattened - tileOffset) / tilesPerStage};
  }

  auto mul = stageBase.getDefiningOp<arith::MulIOp>();
  if (mul) {
    Value stage;
    if (getExactSMEMConstant(mul.getLhs()) == tilesPerStage)
      stage = mul.getRhs();
    else if (getExactSMEMConstant(mul.getRhs()) == tilesPerStage)
      stage = mul.getLhs();
    if (stage)
      return ExactSMEMStageIndex{stage, getExactSMEMConstant(stage)};
  }
  if (tilesPerStage == 1)
    return ExactSMEMStageIndex{stageBase,
                               getExactSMEMConstant(stageBase)};
  if (auto flattened = getExactSMEMConstant(stageBase)) {
    if (*flattened >= 0 && *flattened % tilesPerStage == 0)
      return ExactSMEMStageIndex{stageBase,
                                 *flattened / tilesPerStage};
  }
  return std::nullopt;
}

inline ExactSMEMStage getExactSMEMStage(Value value) {
  bool transposed = false;
  if (auto transpose = value.getDefiningOp<MemDescWGMMAViewOp>()) {
    if (transpose.getOrder() != ArrayRef<int32_t>({1, 0}))
      return {};
    value = transpose.getSrc();
    transposed = true;
  }
  auto view = value.getDefiningOp<ttg::MemDescReinterpretOp>();
  if (!view || !view->hasAttr(kExactSMEMStageAttr))
    return {};
  auto atom = view.getSrc().getDefiningOp<ttg::MemDescIndexOp>();
  if (!atom)
    return {};
  ExactSMEMRoot root = getExactSMEMRoot(atom.getSrc());
  if (!root)
    return {};
  std::optional<ExactSMEMStageIndex> stage =
      matchExactSMEMStage(atom.getIndex(), root.getTilesPerStage());
  if (!stage)
    return {};
  return {view, atom, root, stage->value, stage->constant, transposed};
}

inline ExactSMEMTile getExactSMEMTile(Value value) {
  auto view = value.getDefiningOp<ttg::MemDescIndexOp>();
  if (!view)
    return {};
  auto tileAttr = view->getAttrOfType<IntegerAttr>(kExactSMEMTileAttr);
  if (!tileAttr)
    return {};
  ExactSMEMRoot root = getExactSMEMRoot(view.getSrc());
  if (!root)
    return {};
  int64_t tile = tileAttr.getInt();
  std::optional<ExactSMEMStageIndex> stage = matchExactSMEMStage(
      view.getIndex(), root.getTilesPerStage(), tile);
  if (!stage)
    return {};
  return {view, root, stage->value, stage->constant, tile};
}

inline LogicalResult verifyExactSMEMRoot(Operation *anchor,
                                         ExactSMEMRoot root) {
  if (!root)
    return anchor->emitOpError("has invalid exact-SMEM root metadata");
  if (root.capacity <= 0 ||
      root.capacity > std::numeric_limits<int32_t>::max())
    return anchor->emitOpError("expects positive i32 exact-SMEM capacity");
  if (root.rows <= 0 || root.rows > 128 ||
      root.rows % kExactSMEMRowQuantum != 0)
    return anchor->emitOpError(
        "expects exact-SMEM rows to be a multiple of 16 no greater than 128");
  if (root.cols <= 0 || root.cols > 256 ||
      root.cols % kExactSMEMColQuantum != 0)
    return anchor->emitOpError(
        "expects exact-SMEM cols to be a multiple of 64 no greater than 256");
  if (root.atomRows < kExactSMEMRowQuantum || root.atomRows > root.rows ||
      !llvm::isPowerOf2_64(root.atomRows) ||
      root.rows % root.atomRows != 0 ||
      root.atomCols < kExactSMEMColQuantum || root.atomCols > root.cols ||
      !llvm::isPowerOf2_64(root.atomCols) ||
      root.cols % root.atomCols != 0)
    return anchor->emitOpError(
        "expects a power-of-two exact-SMEM atom that divides the logical "
        "stage and is at least 16x64");

  auto type = root.alloc.getType();
  SmallVector<int64_t> expected{root.capacity * root.getTilesPerStage(),
                                root.atomRows, root.atomCols};
  if (type.getShape() != ArrayRef<int64_t>(expected) ||
      type.getAllocShape() != ArrayRef<int64_t>(expected))
    return anchor->emitOpError("expects exact-SMEM root shape ") << expected;
  if (!type.getMutableMemory() ||
      !isa<ttg::SharedMemorySpaceAttr>(type.getMemorySpace()))
    return anchor->emitOpError("expects mutable shared exact-SMEM storage");
  if (!isSupportedExactSMEMElementType(type.getElementType()))
    return anchor->emitOpError(
        "expects a Hopper WGMMA-compatible exact-SMEM element type");
  auto encoding = dyn_cast<ttg::SharedEncodingTrait>(type.getEncoding());
  if (!encoding ||
      cast<ttg::LayoutEncodingTrait>(encoding).getRank() != 2)
    return anchor->emitOpError(
        "expects exact-SMEM root to use a rank-2 shared encoding");
  return success();
}

inline LogicalResult verifyExactSMEMStage(Operation *anchor,
                                          ExactSMEMStage stage) {
  if (!stage)
    return anchor->emitOpError("has malformed exact-SMEM stage coordinates");
  if (failed(verifyExactSMEMRoot(anchor, stage.root)))
    return failure();
  if (!stage.stage.getType().isInteger(32))
    return anchor->emitOpError("expects exact-SMEM stage index to be i32");
  if (stage.staticStage)
    if (*stage.staticStage < 0 || *stage.staticStage >= stage.root.capacity)
      return anchor->emitOpError("static exact-SMEM stage exceeds capacity");

  auto atomType = stage.atom.getType();
  SmallVector<int64_t> atomShape{stage.getAtomRows(), stage.getAtomCols()};
  if (atomType.getShape() != ArrayRef<int64_t>(atomShape) ||
      atomType.getAllocShape() != ArrayRef<int64_t>(atomShape))
    return anchor->emitOpError(
        "expects exact-SMEM stage base to be one selected atom");
  auto type = stage.getType();
  SmallVector<int64_t> expected{
      static_cast<int64_t>(llvm::PowerOf2Ceil(stage.root.rows)),
      static_cast<int64_t>(llvm::PowerOf2Ceil(stage.root.cols))};
  if (type.getShape() != ArrayRef<int64_t>(expected) ||
      type.getAllocShape() != ArrayRef<int64_t>(expected))
    return anchor->emitOpError("expects exact-SMEM carrier shape ") << expected;
  if (type.getElementType() != atomType.getElementType() ||
      type.getMemorySpace() != atomType.getMemorySpace() ||
      !type.getMutableMemory())
    return anchor->emitOpError(
        "expects exact-SMEM carrier to preserve mutable storage type");
  auto nvmma = dyn_cast<ttg::NVMMASharedEncodingAttr>(type.getEncoding());
  unsigned elementBitWidth = type.getElementType().getIntOrFloatBitWidth();
  if (!nvmma || nvmma.getElementBitWidth() != elementBitWidth)
    return anchor->emitOpError(
        "expects exact-SMEM carrier NVMMAShared bitwidth to match its "
        "element type");
  return success();
}

inline LogicalResult verifyExactSMEMTile(Operation *anchor,
                                         ExactSMEMTile tile) {
  if (!tile)
    return anchor->emitOpError("has malformed exact-SMEM tile coordinates");
  if (failed(verifyExactSMEMRoot(anchor, tile.root)))
    return failure();
  if (!tile.stage.getType().isInteger(32))
    return anchor->emitOpError("expects exact-SMEM stage index to be i32");
  if (tile.staticStage)
    if (*tile.staticStage < 0 || *tile.staticStage >= tile.root.capacity)
      return anchor->emitOpError(
          "static exact-SMEM tile stage exceeds capacity");
  if (tile.tile < 0 || tile.tile >= tile.root.getTilesPerStage())
    return anchor->emitOpError("exact-SMEM tile index exceeds stage");
  auto type = tile.getType();
  SmallVector<int64_t> atomShape{tile.getAtomRows(), tile.getAtomCols()};
  if (type.getShape() != ArrayRef<int64_t>(atomShape) ||
      type.getAllocShape() != ArrayRef<int64_t>(atomShape))
    return anchor->emitOpError(
        "expects exact-SMEM tile to match the selected atom shape");
  auto nvmma = dyn_cast<ttg::NVMMASharedEncodingAttr>(type.getEncoding());
  unsigned elementBitWidth = type.getElementType().getIntOrFloatBitWidth();
  if (!nvmma || nvmma.getElementBitWidth() != elementBitWidth)
    return anchor->emitOpError(
        "expects exact-SMEM tile NVMMAShared bitwidth to match its element "
        "type");
  return success();
}

} // namespace mlir::triton::tle

#endif // TRITON_TLE_IR_EXACT_SMEM_H_
