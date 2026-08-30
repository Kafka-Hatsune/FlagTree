/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
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

#include "MMAHelpers.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::NVIDIA;

using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::MemDescType;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingTrait;

#ifdef __TLE__
static constexpr llvm::StringLiteral
    kTleExplicitWgmmaCommitAttr("tle.explicit_wgmma_commit");
static constexpr llvm::StringLiteral
    kTleWgmmaAccumulatorChainCAttr("tle.wgmma_accumulator_chain_c");
static constexpr llvm::StringLiteral
    kTleWgmmaActiveNAttr("tle.wgmma_active_n");
static constexpr llvm::StringLiteral
    kTleWgmmaActiveKAttr("tle.wgmma_active_k");
static constexpr llvm::StringLiteral
    kTleTiledSMEMOperandBAttr("tle.tiled_smem_operand_b");
static constexpr llvm::StringLiteral
    kTleTiledSMEMLogicalRowsAttr("tle.tiled_smem_logical_rows");
static constexpr llvm::StringLiteral
    kTleTiledSMEMLogicalColsAttr("tle.tiled_smem_logical_cols");
static constexpr llvm::StringLiteral
    kTleTiledSMEMStorageTileShapeAttr(
        "tle.tiled_smem_storage_tile_shape");

struct TleWgmmaOperand {
  Value value;
  std::optional<int64_t> descriptorImm;
};

static LinearLayout
getTleTiledSMEMLoaderLayout(triton::gpu::MemDescType bTensorTy,
                            int64_t logicalRows, int64_t logicalCols,
                            int64_t storageTileRows,
                            int64_t storageTileCols) {
  auto shape = bTensorTy.getShape();
  auto nvmma = cast<triton::gpu::NVMMASharedEncodingAttr>(
      bTensorTy.getEncoding());
  bool transposed = nvmma.getTransposed();
  int64_t physicalRows = llvm::PowerOf2Ceil(logicalRows);
  int64_t physicalCols = llvm::PowerOf2Ceil(logicalCols);
  SmallVector<int64_t> expectedShape =
      transposed ? SmallVector<int64_t>{physicalCols, physicalRows}
                 : SmallVector<int64_t>{physicalRows, physicalCols};
  assert(shape == ArrayRef<int64_t>(expectedShape) &&
         "tiled SMEM B must be a direct or transposed carrier");

  SmallVector<int64_t> storageTileShape =
      transposed
          ? SmallVector<int64_t>{storageTileCols, storageTileRows}
          : SmallVector<int64_t>{storageTileRows, storageTileCols};
  SmallVector<unsigned, 2> storageTileOrder =
      transposed ? SmallVector<unsigned, 2>{0, 1}
                 : SmallVector<unsigned, 2>{1, 0};
  auto storageTileEncoding =
      triton::gpu::NVMMASharedEncodingAttr::get(
          bTensorTy.getContext(), storageTileShape, storageTileOrder,
          nvmma.getCTALayout(), bTensorTy.getElementType(),
          nvmma.getFp4Padded());
  auto storageTileType = triton::gpu::MemDescType::get(
      storageTileShape, bTensorTy.getElementType(), storageTileEncoding,
      bTensorTy.getMemorySpace(), bTensorTy.getMutableMemory());
  LinearLayout layout = toLinearLayout(storageTileType).pseudoinvert();

  return layout;
}

static DotOpMmaSmemLoader buildTleTiledSMEMLoader(
    Location loc, RewriterBase &rewriter,
    triton::gpu::MemDescType bTensorTy, Value smemBase,
    int64_t logicalRows, int64_t logicalCols, int64_t storageTileRows,
    int64_t storageTileCols, ArrayRef<unsigned> instrShape,
    RankedTensorType mmaType) {
  LinearLayout layout = getTleTiledSMEMLoaderLayout(
      bTensorTy, logicalRows, logicalCols, storageTileRows, storageTileCols);
  auto mmaEncoding = cast<triton::gpu::NvidiaMmaEncodingAttr>(
      mmaType.getEncoding());
  RankedTensorType descriptorMmaType = mmaType;
  if (instrShape[1] != mmaEncoding.getInstrShape()[1]) {
    SmallVector<unsigned> descriptorInstrShape(mmaEncoding.getInstrShape());
    descriptorInstrShape[1] = instrShape[1];
    auto descriptorMmaEncoding = triton::gpu::NvidiaMmaEncodingAttr::get(
        mmaType.getContext(), mmaEncoding.getVersionMajor(),
        mmaEncoding.getVersionMinor(), mmaEncoding.getWarpsPerCTA(),
        mmaEncoding.getCTALayout(), descriptorInstrShape);
    SmallVector<int64_t> descriptorMmaShape(mmaType.getShape());
    descriptorMmaShape[1] = instrShape[1];
    descriptorMmaType = RankedTensorType::get(
        descriptorMmaShape, mmaType.getElementType(), descriptorMmaEncoding);
  }
  unsigned elementBitWidth = bTensorTy.getElementTypeBitWidth();
  return DotOpMmaSmemLoader::build(
      loc, rewriter, layout, elementBitWidth, smemBase, instrShape,
      /*MNdim=*/1, /*mmaVersion=*/3, descriptorMmaType);
}

static bool usesTlePanelMajorWGMMAAtoms(
    triton::gpu::MemDescType bTensorTy, int64_t logicalRows,
    int64_t logicalCols, int64_t storageTileRows,
    int64_t storageTileCols) {
  Type elementType = bTensorTy.getElementType();
  return isa<Float16Type, BFloat16Type>(elementType) &&
         storageTileRows == 16 && storageTileCols == 64 &&
         logicalRows % storageTileRows == 0 &&
         logicalCols % storageTileCols == 0;
}

// Build the matrix descriptor used by the historical native-N80 path, but on
// the current exact-SMEM root/view representation. Physical atoms are ordered
// [K/N64 panel][logical N/K16 fragment]. The descriptor's leading stride then
// crosses the atom array instead of aliasing the next K64 panel as another N16
// fragment.
static DotOpMmaSmemLoader buildTlePanelMajorSMEMLoader(
    Location loc, RewriterBase &rewriter,
    triton::gpu::MemDescType bTensorTy, Value smemBase,
    int64_t logicalRows, int64_t logicalCols, int64_t storageTileRows,
    int64_t storageTileCols) {
  assert(usesTlePanelMajorWGMMAAtoms(
             bTensorTy, logicalRows, logicalCols, storageTileRows,
             storageTileCols) &&
         "panel-major descriptor requires 16x64 f16/bf16 atoms");
  auto nvmma = cast<triton::gpu::NVMMASharedEncodingAttr>(
      bTensorTy.getEncoding());
  bool transposed = nvmma.getTransposed();
  int64_t elementBytes = bTensorTy.getElementTypeBitWidth() / 8;
  int64_t atomBytes = storageTileRows * storageTileCols * elementBytes;
  int64_t rowTiles = logicalRows / storageTileRows;

  SMEMDescriptor smemDescriptor{};
  smemDescriptor.descriptor = 0;
  smemDescriptor.leadDimensionBaseOffset =
      transposed ? 1 : rowTiles * atomBytes / 16;
  smemDescriptor.strideDimensionBaseOffset = 64;
  smemDescriptor.swizzlingMode = 1;
  MMASMEMDescriptor descriptor{/*descriptor=*/smemDescriptor,
                               /*swizzlingByteWidth=*/128,
                               /*bitwidth=*/16,
                               /*transposed=*/transposed,
                               /*fp4Padded=*/false};

  TritonLLVMOpBuilder b(loc, rewriter);
  Value sharedByteAddress = b.ptrtoint(rewriter.getI32Type(), smemBase);
  Value baseSrcb128 = b.lshr(sharedByteAddress, b.i32_val(4));
  Value baseb128 =
      b.zext(rewriter.getI64Type(), b.and_(baseSrcb128, b.i32_val(0x3FFF)));
  return DotOpMmaSmemLoader(
      descriptor, baseb128,
      getTleTiledSMEMLoaderLayout(bTensorTy, logicalRows, logicalCols,
                                  storageTileRows, storageTileCols));
}
#endif

triton::nvgpu::WGMMAEltType getMmaRetType(Value d) {
  auto dTy = cast<RankedTensorType>(d.getType()).getElementType();
  if (dTy.isF32()) {
    return triton::nvgpu::WGMMAEltType::f32;
  } else if (dTy.isF16()) {
    return triton::nvgpu::WGMMAEltType::f16;
  } else if (dTy.isInteger(32)) {
    return triton::nvgpu::WGMMAEltType::s32;
  } else {
    llvm::report_fatal_error("Unsupported mma result type found");
  }
}

triton::nvgpu::WGMMAEltType getMmaOperandType(Value a, bool allowTF32) {
  auto aTy = cast<triton::gpu::TensorOrMemDesc>(a.getType()).getElementType();
  if (aTy.isF16()) {
    return triton::nvgpu::WGMMAEltType::f16;
  } else if (aTy.isBF16()) {
    return triton::nvgpu::WGMMAEltType::bf16;
  } else if (aTy.isF32() && allowTF32) {
    return triton::nvgpu::WGMMAEltType::tf32;
  } else if (aTy.isInteger(8)) {
    return triton::nvgpu::WGMMAEltType::s8;
  } else if (llvm::isa<Float8E5M2Type>(aTy)) {
    return triton::nvgpu::WGMMAEltType::e5m2;
  } else if (llvm::isa<Float8E4M3FNType>(aTy)) {
    return triton::nvgpu::WGMMAEltType::e4m3;
  } else {
    llvm::report_fatal_error("Unsupported mma operand type found");
  }
}

// Return a vector of Value of the accumulator start at startIndex and pack the
// values into 32bits in case the accumulator is fp16.
//
// `elements` contains all loaded register values for operand A.
// This consists of operand A for possibly multiple wgmma instructions.
// For each wgmma, each warp in a warp group feeds a single "warp matrix"
// Each warp matrix consists of 2x2 "quads".
// Each thread holds several elements in each quad. Right before a wgmma,
// the sum of bitwidth of
// the elements in each quad should add up to 32.
//
// These values are stored unrolled in `elements`.
// The ordering of dimensions is as follows:
// batch (only 1 batch for Hopper currently)
// matM (m-index of the "warp matrix")
// matK (k-index of the "warp matrix")
// quadK (k-index of the "quad" in the core matrix)
// quadM (m-index of the "quad" in the core matrix)
// vecIdx (index of the element in the quad; this is always along the k-dim)
//
// This ordering is decided when a tensor in DotOpEnc is lowered into llvm.
// For WGMMA this happens in both SharedToDotOperand and MMAToDotOperand.
// Thus, both lowerings must obey this above ordering for the below code to be
// correct.
llvm::SmallVector<Value> loadReg(ConversionPatternRewriter &rewriter,
                                 Location loc,
                                 const SmallVector<Value> &elements,
                                 int startIndex, int numElements,
                                 Operation *insertBefore) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(insertBefore);

  if (!elements[0].getType().isIntOrFloat() ||
      elements[0].getType().getIntOrFloatBitWidth() >= 32) {
    llvm::SmallVector<Value> mmaOut(numElements);
    for (int i = 0; i < numElements; ++i)
      mmaOut[i] = elements[startIndex + i];
    return mmaOut;
  }
  Type elementType = elements[0].getType();
  int numElemsPer32Bits = 32 / elementType.getIntOrFloatBitWidth();

  // For FP16 and BF16 we need to pack accumulator into 32-bit integers.
  int num32BitValues = numElements / numElemsPer32Bits;
  llvm::SmallVector<Value> mmaOut(num32BitValues);
  Type packTy = vec_ty(elementType, numElemsPer32Bits);
  for (int i = 0; i < num32BitValues; ++i) {
    Value pack = LLVM::UndefOp::create(rewriter, loc, packTy);
    for (int j = 0; j < numElemsPer32Bits; ++j) {
      Value element = elements[startIndex + i * numElemsPer32Bits + j];
      pack = b.insert_element(packTy, pack, element, b.i32_val(j));
    }
    pack = b.bitcast(pack, rewriter.getIntegerType(32));
    mmaOut[i] = pack;
  }
  return mmaOut;
}

// If the accumulator is fp16 unpack it from 32-bit integers.
SmallVector<Value> unpackAccumulator(ConversionPatternRewriter &rewriter,
                                     Location loc,
                                     const SmallVector<Value> &packed,
                                     RankedTensorType tensorTy) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  if (!tensorTy.getElementType().isF16())
    return packed;
  // For fp16 the accumulator is pack into 32-bit integers so we need to unpack
  // it.
  SmallVector<Value> results;
  for (Value elem : packed) {
    elem = b.bitcast(elem, vec_ty(rewriter.getF16Type(), 2));
    results.push_back(
        b.extract_element(rewriter.getF16Type(), elem, b.i32_val(0)));
    results.push_back(
        b.extract_element(rewriter.getF16Type(), elem, b.i32_val(1)));
  }
  return results;
}

static Value faddAccumulate(ConversionPatternRewriter &rewriter, Location loc,
                            Value a, Value b) {
  int numEl = cast<LLVM::LLVMStructType>(a.getType()).getBody().size();
  Value newStruct = LLVM::UndefOp::create(rewriter, loc, a.getType());
  for (int i = 0; i < numEl; ++i) {
    Value lhs = LLVM::ExtractValueOp::create(rewriter, loc, a, i);
    Value rhs = LLVM::ExtractValueOp::create(rewriter, loc, b, i);
    Value add = LLVM::FAddOp::create(rewriter, loc, lhs, rhs);
    newStruct = LLVM::InsertValueOp::create(rewriter, loc, newStruct, add, i);
  }
  return newStruct;
}

static SmallVector<Value> emitWait(ConversionPatternRewriter &rewriter,
                                   Location loc, SmallVector<Value> acc,
                                   int pendings) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Type> types(acc.size(), acc[0].getType());
  auto structTy =
      LLVM::LLVMStructType::getLiteral(rewriter.getContext(), types);
  Value llvmStruct = LLVM::UndefOp::create(rewriter, loc, structTy);
  int i = 0;
  for (Value v : acc) {
    llvmStruct = b.insert_val(structTy, llvmStruct, v, i++);
  }
  Value res = triton::nvgpu::WGMMAWaitGroupOp::create(rewriter, loc, llvmStruct,
                                                      pendings);
  SmallVector<Value> results;
  for (int i = 0; i < acc.size(); ++i) {
    results.push_back(b.extract_val(types[0], res, i));
  }
  return results;
}

#ifdef __TLE__
LogicalResult convertDot(const LLVMTypeConverter *typeConverter,
                         ConversionPatternRewriter &rewriter, Location loc,
                         Operation *op, Value a, Value b, Value c, Value d,
                         Value useCOperand, Value loadedA, Value loadedB,
                         Value loadedC, bool allowTF32,
                         bool needsPartialAccumulator,
                         uint32_t maxNumImpreciseAcc, bool sync, Value thread) {
  auto tb = TritonLLVMOpBuilder(loc, rewriter);
  auto aTensorTy = cast<triton::gpu::TensorOrMemDesc>(a.getType());
  auto bTensorTy = cast<triton::gpu::MemDescType>(b.getType());
  auto dTensorTy = cast<RankedTensorType>(d.getType());
  bool aInShared = isa<SharedEncodingTrait>(aTensorTy.getEncoding());
  auto mmaEncoding = cast<NvidiaMmaEncodingAttr>(dTensorTy.getEncoding());
  bool tiledSMEMOperandB = op->hasAttr(kTleTiledSMEMOperandBAttr);
  std::optional<SharedMemoryObject> smemObjA;
  Value baseA;
  if (aInShared) {
    baseA = getOffsetedBase(loadedA, cast<MemDescType>(aTensorTy),
                            typeConverter, rewriter, loc);
  }
  Value baseB;
  if (tiledSMEMOperandB) {
    auto llvmElemTy = typeConverter->convertType(bTensorTy.getElementType());
    baseB = LLVM::getSharedMemoryObjectFromStruct(loc, loadedB, llvmElemTy,
                                                  rewriter)
                .getBase();
  } else {
    baseB = getOffsetedBase(loadedB, cast<MemDescType>(bTensorTy),
                            typeConverter, rewriter, loc);
  }
  auto dShapePerCTA = getShapePerCTA(dTensorTy);
  auto instrMNK = mmaEncoding.getInstrShape();
  unsigned physicalN = instrMNK[1];
  unsigned physicalAccSize = 2 * (physicalN / 4);
  unsigned M = 4 * instrMNK[0];
  unsigned N = physicalN;
  unsigned K = instrMNK[2];
  IntegerAttr activeNAttr =
      op->getAttrOfType<IntegerAttr>(kTleWgmmaActiveNAttr);
  IntegerAttr activeKAttr =
      op->getAttrOfType<IntegerAttr>(kTleWgmmaActiveKAttr);
  if (activeNAttr && activeKAttr)
    return op->emitOpError(
        "tle.wgmma_active_n and tle.wgmma_active_k cannot be combined");
  IntegerAttr tiledSMEMLogicalRowsAttr =
      op->getAttrOfType<IntegerAttr>(kTleTiledSMEMLogicalRowsAttr);
  IntegerAttr tiledSMEMLogicalColsAttr =
      op->getAttrOfType<IntegerAttr>(kTleTiledSMEMLogicalColsAttr);
  DenseI32ArrayAttr tiledSMEMStorageTileShapeAttr =
      op->getAttrOfType<DenseI32ArrayAttr>(
          kTleTiledSMEMStorageTileShapeAttr);
  if (tiledSMEMOperandB &&
      (!tiledSMEMLogicalRowsAttr || !tiledSMEMLogicalColsAttr ||
       !tiledSMEMStorageTileShapeAttr ||
       tiledSMEMStorageTileShapeAttr.size() != 2))
    return op->emitOpError(
        "tiled SMEM operand B requires logical extents and a two-dimensional "
        "storage tile shape");
  int64_t tiledSMEMStorageTileRows = 0;
  int64_t tiledSMEMStorageTileCols = 0;
  if (tiledSMEMOperandB) {
    ArrayRef<int32_t> storageTileShape =
        tiledSMEMStorageTileShapeAttr.asArrayRef();
    tiledSMEMStorageTileRows = storageTileShape[0];
    tiledSMEMStorageTileCols = storageTileShape[1];
    if (tiledSMEMStorageTileRows <= 0 || tiledSMEMStorageTileCols <= 0 ||
        tiledSMEMLogicalRowsAttr.getInt() % tiledSMEMStorageTileRows != 0 ||
        tiledSMEMLogicalColsAttr.getInt() % tiledSMEMStorageTileCols != 0)
      return op->emitOpError(
          "tiled SMEM storage tile must divide its logical extents");
  }
  if (tiledSMEMOperandB && (activeNAttr || activeKAttr)) {
    auto nvmma = cast<triton::gpu::NVMMASharedEncodingAttr>(
        bTensorTy.getEncoding());
    int64_t logicalK = nvmma.getTransposed()
                           ? tiledSMEMLogicalColsAttr.getInt()
                           : tiledSMEMLogicalRowsAttr.getInt();
    int64_t logicalN = nvmma.getTransposed()
                           ? tiledSMEMLogicalRowsAttr.getInt()
                           : tiledSMEMLogicalColsAttr.getInt();
    if (activeKAttr && activeKAttr.getInt() != logicalK)
      return op->emitOpError(
          "tle.wgmma_active_k must equal the tiled SMEM logical K extent");
    if (activeNAttr && activeNAttr.getInt() != logicalN)
      return op->emitOpError(
          "tle.wgmma_active_n must equal the tiled SMEM logical N extent");
  }
  if (activeNAttr) {
    assert(activeNAttr.getInt() > 0 && activeNAttr.getInt() % 8 == 0 &&
           activeNAttr.getInt() <= physicalN &&
           "active_n verifier must restrict codegen to a positive multiple "
           "of 8 within one physical N carrier");
    N = activeNAttr.getInt();
  }
  unsigned wgmmaAccSize = 2 * (N / 4);
  bool zeroAcc = isZeroConst(c);
  if (activeNAttr && op->hasAttr(kTleWgmmaAccumulatorChainCAttr))
    return op->emitOpError(
        "active_n does not support tle.wgmma_accumulator_chain_c");
  bool reuseAccumulatorChainC =
      !zeroAcc && op->hasAttr(kTleWgmmaAccumulatorChainCAttr);
  auto warpSize = mmaEncoding.getWarpsPerCTA();
  auto shapePerCTATile = SmallVector<unsigned>{instrMNK[0] * warpSize[0],
                                               physicalN * warpSize[1]};
  unsigned mmaSizeM = shapePerCTATile[0];
  unsigned mmaSizeN = shapePerCTATile[1];
  unsigned mmaSizeK = instrMNK[2];
  int numRepM = ceil<unsigned>(dShapePerCTA[0], mmaSizeM);
  int numRepN = ceil<unsigned>(dShapePerCTA[1], mmaSizeN);
  int physicalNumRepK =
      ceil<unsigned>(aTensorTy.getShape()[1], mmaSizeK);
  int activeNumRepK = physicalNumRepK;
  if (activeKAttr) {
    int64_t physicalK = aTensorTy.getShape()[1];
    int64_t activeK = activeKAttr.getInt();
    if (physicalK % mmaSizeK != 0 || activeK <= 0 || activeK > physicalK ||
        activeK % mmaSizeK != 0)
      return op->emitOpError(
          "tle.wgmma_active_k must select complete WGMMA K instructions "
          "within the physical carrier");
    activeNumRepK = activeK / mmaSizeK;
  }
  if (reuseAccumulatorChainC && (numRepM != 1 || numRepN != 1))
    return op->emitOpError(
        "cannot reuse WGMMA accumulator chain C for multi-tile results");
  DotOpMmaSmemLoader aLoader;
  SmallVector<Value> structA;
  auto warpGroups = {warpSize[0] / 4, warpSize[1]};
  bool transA = false;
  if (aInShared) {
    aLoader =
        DotOpMmaSmemLoader::build(loc, rewriter, cast<MemDescType>(aTensorTy),
                                  baseA, {M, K}, 0, 3, false, dTensorTy);
    transA = aLoader.getDescriptor().transposed;
  } else {
    structA = unpackLLElements(loc, loadedA, rewriter);
  }
  DotOpMmaSmemLoader bLoader;
  bool tiledSMEMTransposed = false;
  bool panelMajorTiledSMEM = false;
  unsigned tiledSMEMTileK = 0;
  unsigned tiledSMEMTileN = 0;
  if (tiledSMEMOperandB) {
    auto nvmma = cast<triton::gpu::NVMMASharedEncodingAttr>(
        bTensorTy.getEncoding());
    tiledSMEMTransposed = nvmma.getTransposed();
    tiledSMEMTileK = tiledSMEMTransposed ? tiledSMEMStorageTileCols
                                        : tiledSMEMStorageTileRows;
    tiledSMEMTileN = tiledSMEMTransposed ? tiledSMEMStorageTileRows
                                        : tiledSMEMStorageTileCols;
    if (activeNAttr &&
        (tiledSMEMTileN < 8 || tiledSMEMTileN % 8 != 0))
      return op->emitOpError(
          "active-N tiled SMEM requires storage tiles containing complete "
          "WGMMA N8 groups");
    if (activeKAttr &&
        (tiledSMEMTileK < mmaSizeK || tiledSMEMTileK % mmaSizeK != 0))
      return op->emitOpError(
          "active-K tiled SMEM requires storage tiles containing complete "
          "WGMMA K instructions");
    panelMajorTiledSMEM = usesTlePanelMajorWGMMAAtoms(
        bTensorTy, tiledSMEMLogicalRowsAttr.getInt(),
        tiledSMEMLogicalColsAttr.getInt(), tiledSMEMStorageTileRows,
        tiledSMEMStorageTileCols);
    if (panelMajorTiledSMEM) {
      bLoader = buildTlePanelMajorSMEMLoader(
          loc, rewriter, bTensorTy, baseB,
          tiledSMEMLogicalRowsAttr.getInt(),
          tiledSMEMLogicalColsAttr.getInt(), tiledSMEMStorageTileRows,
          tiledSMEMStorageTileCols);
    } else {
      // The generic layout remains a correctness fallback for exact stages
      // that do not use the native panel-major 16x64 atom contract.
      unsigned descriptorN = activeNAttr ? tiledSMEMTileN : physicalN;
      bLoader = buildTleTiledSMEMLoader(
          loc, rewriter, bTensorTy, baseB,
          tiledSMEMLogicalRowsAttr.getInt(),
          tiledSMEMLogicalColsAttr.getInt(), tiledSMEMStorageTileRows,
          tiledSMEMStorageTileCols, {K, descriptorN}, dTensorTy);
    }
  } else {
    bLoader = DotOpMmaSmemLoader::build(
        loc, rewriter, bTensorTy, baseB, {K, physicalN}, 1, 3, false,
        dTensorTy);
  }
  bool transB = !bLoader.getDescriptor().transposed;

  SmallVector<Value> fc;
  if (!reuseAccumulatorChainC || activeNAttr)
    fc = unpackLLElements(loc, loadedC, rewriter);

  triton::nvgpu::WGMMAEltType eltTypeC = getMmaRetType(d);
  triton::nvgpu::WGMMAEltType eltTypeA = getMmaOperandType(a, allowTF32);
  triton::nvgpu::WGMMAEltType eltTypeB = getMmaOperandType(b, allowTF32);

  bool supportsTransposeOperands =
      (eltTypeA == triton::nvgpu::WGMMAEltType::f16 &&
       eltTypeB == triton::nvgpu::WGMMAEltType::f16) ||
      (eltTypeA == triton::nvgpu::WGMMAEltType::bf16 &&
       eltTypeB == triton::nvgpu::WGMMAEltType::bf16);
  if (!supportsTransposeOperands && (transA || transB))
    return op->emitOpError(
        "TF32, FP8, and int8 WGMMA require row-major A and column-major B "
        "descriptors because their PTX instructions have no transpose "
        "operands");

  triton::nvgpu::WGMMALayout layoutA = transA ? triton::nvgpu::WGMMALayout::col
                                              : triton::nvgpu::WGMMALayout::row;
  triton::nvgpu::WGMMALayout layoutB = transB ? triton::nvgpu::WGMMALayout::row
                                              : triton::nvgpu::WGMMALayout::col;

  // A panel-major atom array is descriptor-contiguous across active N. Other
  // exact layouts retain the established per-storage-fragment lowering.
  unsigned wgmmaInstructionN =
      activeNAttr && !panelMajorTiledSMEM ? tiledSMEMTileN : N;
  unsigned activeNFragments =
      activeNAttr ? N / wgmmaInstructionN : 1;
  unsigned instructionAccSize = 2 * (wgmmaInstructionN / 4);

  auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
  // TLE kernels often carry extra live values around WGMMA regions. Keep all
  // ordinary register prep that feeds the first WGMMA outside the fence-to-MMA
  // window so ptxas sees a clean WGMMA pipeline stage.
  // Materialize the initial C accumulator before wgmma.fence.  Otherwise LLVM
  // may leave struct packing/copies between wgmma.fence and the first
  // wgmma.mma_async, which ptxas treats as accumulator definitions inside the
  // WGMMA pipeline stage.
  SmallVector<Type> accTypes;
  SmallVector<Value> initialAccumulators;
  SmallVector<Value> initialUseC;
  for (int m = 0; m < numRepM; ++m) {
    for (int n = 0; n < numRepN; ++n) {
      for (unsigned fragment = 0; fragment < activeNFragments; ++fragment) {
        Value d;
        Value useC;
        LLVM::LLVMStructType accTy;
        if (reuseAccumulatorChainC && !activeNAttr) {
          accTy = cast<LLVM::LLVMStructType>(loadedC.getType());
          d = loadedC;
          useC = tb.i1_val(1);
        } else {
          llvm::SmallVector<Value> mmaOut = loadReg(
              rewriter, loc, fc,
              (m * numRepN + n) * physicalAccSize +
                  fragment * instructionAccSize,
              instructionAccSize, op);
          llvm::SmallVector<Type> elemTypes;
          for (Value accEl : mmaOut)
            elemTypes.push_back(accEl.getType());
          accTy = LLVM::LLVMStructType::getLiteral(rewriter.getContext(),
                                                   elemTypes);
          useC = tb.i1_val(0);
          if (!zeroAcc) {
            d = packLLElements(loc, typeConverter, mmaOut, rewriter, accTy);
            useC = tb.i1_val(1);
          }
        }
        if (useCOperand)
          useC = tb.and_(useC, useCOperand);
        accTypes.push_back(accTy);
        initialAccumulators.push_back(d);
        initialUseC.push_back(useC);
      }
    }
  }

  // Keep shared-memory descriptors near their WGMMA use. Treating them as
  // ordinary pure integer SSA lets LLVM hoist them across the full dot region,
  // which can make ptxas spill descriptor registers under high pressure.
  auto buildRegisterA = [&](int m, int k,
                            Operation *insertBefore) -> TleWgmmaOperand {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(insertBefore);
    auto aDotOpEnc = cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
    assert(aDotOpEnc.getKWidth() == 32 / aTensorTy.getElementTypeBitWidth());

    unsigned regASize = (instrMNK[0] * instrMNK[2]) / 32;
    llvm::SmallVector<Value> regA =
        loadReg(rewriter, loc, structA, (m * physicalNumRepK + k) * regASize,
                regASize, insertBefore);
    auto regATy = LLVM::LLVMStructType::getLiteral(
        rewriter.getContext(),
        SmallVector<Type>(regA.size(), regA[0].getType()));
    return {packLLElements(loc, typeConverter, regA, rewriter, regATy),
            std::nullopt};
  };

  auto buildSharedA = [&](int m, int k) -> TleWgmmaOperand {
    LocalizedSMEMDescriptor desc = aLoader.localizedSmemLoad(m, k);
    return {desc.baseb128, desc.descriptorImm};
  };

  auto buildSharedB = [&](int k, int n) -> TleWgmmaOperand {
    if (!tiledSMEMOperandB) {
      LocalizedSMEMDescriptor desc = bLoader.localizedSmemLoad(k, n);
      return {desc.baseb128, desc.descriptorImm};
    }

    int64_t elementBytes = bTensorTy.getElementTypeBitWidth() / 8;
    int64_t tileBytes = tiledSMEMStorageTileRows *
                        tiledSMEMStorageTileCols * elementBytes;
    if (panelMajorTiledSMEM) {
      if (n != 0)
        llvm::report_fatal_error(
            "panel-major tiled SMEM expects one physical WGMMA N tile");
      LocalizedSMEMDescriptor desc;
      int64_t tileOffset = 0;
      if (tiledSMEMTransposed) {
        // QK: K advances inside a K64 panel. Each next panel skips all N16
        // row fragments so one N80 descriptor observes a contiguous N axis.
        int64_t rowTiles = tiledSMEMLogicalRowsAttr.getInt() /
                           tiledSMEMStorageTileRows;
        int localK = k % tiledSMEMTileK;
        desc = bLoader.localizedSmemLoad(localK, 0);
        tileOffset = (k / tiledSMEMTileK) * rowTiles;
      } else {
        // PV: each active K16 slice selects the next row atom. The descriptor
        // leading stride spans all row atoms to reach the next N64 panel.
        desc = bLoader.localizedSmemLoad(0, 0);
        tileOffset = k / tiledSMEMTileK;
      }
      int64_t tileByteOffset = tileOffset * tileBytes;
      assert(tileByteOffset % 16 == 0 &&
             "panel-major tile must use WGMMA descriptor units");
      if (tileByteOffset != 0)
        desc.baseb128 =
            tb.add(desc.baseb128, tb.i64_val(tileByteOffset / 16));
      return {desc.baseb128, desc.descriptorImm};
    }

    int64_t storageTile = 0;
    int localK = k;
    int localN = n;
    if (activeNAttr) {
      storageTile = n / tiledSMEMTileN;
      localN = n % tiledSMEMTileN;
    } else if (activeKAttr) {
      storageTile = k / tiledSMEMTileK;
      localK = k % tiledSMEMTileK;
    }
    LocalizedSMEMDescriptor desc =
        bLoader.localizedSmemLoad(localK, localN);
    int64_t tileByteOffset = storageTile * tileBytes;
    assert(tileByteOffset % 16 == 0 &&
           "storage tile must use WGMMA descriptor units");
    if (tileByteOffset != 0)
      desc.baseb128 =
          tb.add(desc.baseb128, tb.i64_val(tileByteOffset / 16));
    return {desc.baseb128, desc.descriptorImm};
  };

  auto setDescriptorAttrs = [&](triton::nvgpu::WGMMAOp mmaOp,
                                const TleWgmmaOperand &a,
                                const TleWgmmaOperand &b) {
    if (a.descriptorImm)
      mmaOp->setAttr(nvgpu::kTleWgmmaOperandADescImmAttr,
                     rewriter.getI64IntegerAttr(*a.descriptorImm));
    if (b.descriptorImm)
      mmaOp->setAttr(nvgpu::kTleWgmmaOperandBDescImmAttr,
                     rewriter.getI64IntegerAttr(*b.descriptorImm));
  };

  TleWgmmaOperand firstA;
  TleWgmmaOperand firstB;
  if (numRepM > 0 && numRepN > 0 && activeNumRepK > 0) {
    // The first WGMMA reuses operands materialized before the fence; later
    // descriptors are still localized at their individual WGMMA use sites.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    if (aInShared) {
      firstA = buildSharedA(/*m=*/0, /*k=*/0);
    } else {
      firstA = buildRegisterA(/*m=*/0, /*k=*/0, op);
    }
    firstB = buildSharedB(/*k=*/0, /*n=*/0);
  }

  Operation *startSequence = op;
  if (!reuseAccumulatorChainC)
    startSequence = NVVM::WgmmaFenceAlignedOp::create(rewriter, loc);
  SmallVector<Value> mmaResults;
  unsigned tileIdx = 0;
  for (int m = 0; m < numRepM; ++m) {
    for (int n = 0; n < numRepN; ++n) {
      for (unsigned fragment = 0; fragment < activeNFragments; ++fragment) {
        auto accTy = accTypes[tileIdx];
        Value d = initialAccumulators[tileIdx];
        Value useC = initialUseC[tileIdx++];
        uint32_t numLowPrecisionAcc = 0;
        Value partialAcc;
        for (int k = 0; k < activeNumRepK; ++k) {
          Value a;
          TleWgmmaOperand aOperand;
          TleWgmmaOperand bOperand;
          bool isFirstWgmma =
              m == 0 && n == 0 && fragment == 0 && k == 0;
          if (aInShared) {
            aOperand = isFirstWgmma
                           ? firstA
                           : buildSharedA(m * mmaSizeM, k * mmaSizeK);
            a = aOperand.value;
          } else {
            if (isFirstWgmma) {
              aOperand = firstA;
              a = aOperand.value;
            } else {
              auto aDotOpEnc =
                  cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
              assert(aDotOpEnc.getKWidth() ==
                     32 / aTensorTy.getElementTypeBitWidth());

              unsigned regASize = (instrMNK[0] * instrMNK[2]) / 32;
              llvm::SmallVector<Value> regA =
                  loadReg(rewriter, loc, structA,
                          (m * physicalNumRepK + k) * regASize, regASize,
                          startSequence);
              auto regATy = LLVM::LLVMStructType::getLiteral(
                  rewriter.getContext(),
                  SmallVector<Type>(regA.size(), regA[0].getType()));
              a = packLLElements(loc, typeConverter, regA, rewriter, regATy);
              aOperand = {a, std::nullopt};
            }
          }
          int nOffset = n * mmaSizeN + fragment * wgmmaInstructionN;
          bOperand = isFirstWgmma
                         ? firstB
                         : buildSharedB(k * mmaSizeK, nOffset);
          auto b = bOperand.value;
          numLowPrecisionAcc += K;
          // If using native accumulation would cause use to do more low
          // precision accumulation than allowed, use a separate accumulator.
          bool requireAddAccumulator =
              needsPartialAccumulator &&
              (numLowPrecisionAcc >= maxNumImpreciseAcc ||
               k == activeNumRepK - 1);
          Value mmaAcc = needsPartialAccumulator ? partialAcc : d;
          auto mmaOp = triton::nvgpu::WGMMAOp::create(
              rewriter, loc, accTy, a, b, useC, mmaAcc, M,
              wgmmaInstructionN, K, eltTypeC, eltTypeA, eltTypeB, layoutA,
              layoutB);
          setDescriptorAttrs(mmaOp, aOperand, bOperand);
          mmaAcc = mmaOp;
          useC = tb.i1_val(1);
          if (needsPartialAccumulator)
            partialAcc = mmaAcc;
          else
            d = mmaAcc;
          if (requireAddAccumulator) {
            d = d ? faddAccumulate(rewriter, loc, d, partialAcc) : partialAcc;
            numLowPrecisionAcc = 0;
            partialAcc = Value();
          }
        }
        auto acc = unpackLLElements(loc, d, rewriter);
        for (int i = 0; i < acc.size(); ++i)
          mmaResults.push_back(acc[i]);
      }
    }
  }
  if (!op->hasAttr(kTleExplicitWgmmaCommitAttr))
    NVVM::WgmmaGroupSyncAlignedOp::create(rewriter, loc);

  if (sync)
    mmaResults = emitWait(rewriter, loc, mmaResults, 0);

  if (activeNAttr) {
    // The logical-domain contract makes the carrier suffix invalid. Retain
    // the physical result type by inserting only the valid prefixes into an
    // undef carrier. This neither reads C's dead suffix nor adds it to the
    // asynchronous wait group.
    assert(dTensorTy.getElementTypeBitWidth() == 32 &&
           "active_n verifier must restrict the accumulator to 32 bits");
    unsigned physicalResultSize =
        numRepM * numRepN * physicalAccSize;
    auto structTy = LLVM::LLVMStructType::getLiteral(
        mmaEncoding.getContext(),
        SmallVector<Type>(physicalResultSize, dTensorTy.getElementType()));
    Value res = LLVM::UndefOp::create(rewriter, loc, structTy);
    unsigned activeOffset = 0;
    unsigned physicalOffset = 0;
    for (int m = 0; m < numRepM; ++m) {
      for (int n = 0; n < numRepN; ++n) {
        for (unsigned i = 0; i < wgmmaAccSize; ++i)
          res = tb.insert_val(structTy, res, mmaResults[activeOffset++],
                              physicalOffset + i);
        physicalOffset += physicalAccSize;
      }
    }
    assert(activeOffset == mmaResults.size() &&
           "active WGMMA accumulator size must match emitted tiles");
    rewriter.replaceOp(op, res);
    return success();
  }

  SmallVector<Value> results =
      unpackAccumulator(rewriter, loc, mmaResults, dTensorTy);

  // replace with new packed result
  Type structTy = LLVM::LLVMStructType::getLiteral(
      mmaEncoding.getContext(),
      SmallVector<Type>(results.size(), dTensorTy.getElementType()));
  auto res = packLLElements(loc, typeConverter, results, rewriter, structTy);
  rewriter.replaceOp(op, res);
  return success();
}

#else

LogicalResult convertDot(const LLVMTypeConverter *typeConverter,
                         ConversionPatternRewriter &rewriter, Location loc,
                         Operation *op, Value a, Value b, Value c, Value d,
                         Value useCOperand, Value loadedA, Value loadedB,
                         Value loadedC, bool allowTF32,
                         bool needsPartialAccumulator,
                         uint32_t maxNumImpreciseAcc, bool sync, Value thread) {
  auto tb = TritonLLVMOpBuilder(loc, rewriter);
  auto aTensorTy = cast<triton::gpu::TensorOrMemDesc>(a.getType());
  auto bTensorTy = cast<triton::gpu::MemDescType>(b.getType());
  auto dTensorTy = cast<RankedTensorType>(d.getType());
  bool aInShared = isa<SharedEncodingTrait>(aTensorTy.getEncoding());
  auto mmaEncoding = cast<NvidiaMmaEncodingAttr>(dTensorTy.getEncoding());
  std::optional<SharedMemoryObject> smemObjA;
  Value baseA;
  if (aInShared) {
    baseA = getOffsetedBase(loadedA, cast<MemDescType>(aTensorTy),
                            typeConverter, rewriter, loc);
  }
  auto baseB = getOffsetedBase(loadedB, cast<MemDescType>(bTensorTy),
                               typeConverter, rewriter, loc);
  auto dShapePerCTA = getShapePerCTA(dTensorTy);
  auto instrMNK = mmaEncoding.getInstrShape();
  auto accSize = 2 * (instrMNK[1] / 4);
  unsigned M = 4 * instrMNK[0];
  unsigned N = instrMNK[1];
  unsigned K = instrMNK[2];
  bool zeroAcc = isZeroConst(c);
  auto warpSize = mmaEncoding.getWarpsPerCTA();
  auto shapePerCTATile = SmallVector<unsigned>{instrMNK[0] * warpSize[0],
                                               instrMNK[1] * warpSize[1]};
  unsigned mmaSizeM = shapePerCTATile[0];
  unsigned mmaSizeN = shapePerCTATile[1];
  unsigned mmaSizeK = instrMNK[2];
  int numRepM = ceil<unsigned>(dShapePerCTA[0], mmaSizeM);
  int numRepN = ceil<unsigned>(dShapePerCTA[1], mmaSizeN);
  int numRepK = ceil<unsigned>(aTensorTy.getShape()[1], mmaSizeK);
  DotOpMmaSmemLoader aLoader;
  SmallVector<Value> structA;
  auto warpGroups = {warpSize[0] / 4, warpSize[1]};
  bool transA = false;
  if (aInShared) {
    aLoader =
        DotOpMmaSmemLoader::build(loc, rewriter, cast<MemDescType>(aTensorTy),
                                  baseA, {M, K}, 0, 3, false, dTensorTy);
    transA = aLoader.getDescriptor().transposed;
  } else {
    structA = unpackLLElements(loc, loadedA, rewriter);
  }
  DotOpMmaSmemLoader bLoader = DotOpMmaSmemLoader::build(
      loc, rewriter, bTensorTy, baseB, {K, N}, 1, 3, false, dTensorTy);
  bool transB = !bLoader.getDescriptor().transposed;

  auto fc = unpackLLElements(loc, loadedC, rewriter);

  triton::nvgpu::WGMMAEltType eltTypeC = getMmaRetType(d);
  triton::nvgpu::WGMMAEltType eltTypeA = getMmaOperandType(a, allowTF32);
  triton::nvgpu::WGMMAEltType eltTypeB = getMmaOperandType(b, allowTF32);

  triton::nvgpu::WGMMALayout layoutA = transA ? triton::nvgpu::WGMMALayout::col
                                              : triton::nvgpu::WGMMALayout::row;
  triton::nvgpu::WGMMALayout layoutB = transB ? triton::nvgpu::WGMMALayout::row
                                              : triton::nvgpu::WGMMALayout::col;

  auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
  Operation *startSequence = NVVM::WgmmaFenceAlignedOp::create(rewriter, loc);
  SmallVector<Value> mmaResults;
  for (int m = 0; m < numRepM; ++m) {
    for (int n = 0; n < numRepN; ++n) {
      llvm::SmallVector<Value> mmaOut =
          loadReg(rewriter, loc, fc, (m * numRepN + n) * accSize, accSize,
                  startSequence);
      llvm::SmallVector<Type> elemTypes;
      for (Value accEl : mmaOut)
        elemTypes.push_back(accEl.getType());
      auto accTy =
          LLVM::LLVMStructType::getLiteral(rewriter.getContext(), elemTypes);
      Value d;
      Value useC = tb.i1_val(0);
      if (!zeroAcc) {
        d = packLLElements(loc, typeConverter, mmaOut, rewriter, accTy);
        useC = tb.i1_val(1);
      }
      if (useCOperand)
        useC = tb.and_(useC, useCOperand);
      uint32_t numLowPrecisionAcc = 0;
      Value partialAcc;
      for (int k = 0; k < numRepK; ++k) {
        Value a;
        if (aInShared) {
          a = aLoader.smemLoad(m * mmaSizeM, k * mmaSizeK, rewriter, loc);
        } else {
          auto aDotOpEnc =
              cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
          assert(aDotOpEnc.getKWidth() ==
                 32 / aTensorTy.getElementTypeBitWidth());

          unsigned regASize = (instrMNK[0] * instrMNK[2]) / 32;
          llvm::SmallVector<Value> regA =
              loadReg(rewriter, loc, structA, (m * numRepK + k) * regASize,
                      regASize, startSequence);
          auto regATy = LLVM::LLVMStructType::getLiteral(
              rewriter.getContext(),
              SmallVector<Type>(regA.size(), regA[0].getType()));
          a = packLLElements(loc, typeConverter, regA, rewriter, regATy);
        }
        auto b = bLoader.smemLoad(k * mmaSizeK, n * mmaSizeN, rewriter, loc);
        numLowPrecisionAcc += K;
        // If using native accumulation would cause use to do more low precion
        // accumulation than allowed do a separate allocation.
        bool requireAddAccumulator =
            needsPartialAccumulator &&
            (numLowPrecisionAcc >= maxNumImpreciseAcc || k == numRepK - 1);
        Value mmaAcc = needsPartialAccumulator ? partialAcc : d;
        mmaAcc = triton::nvgpu::WGMMAOp::create(
            rewriter, loc, accTy, a, b, useC, mmaAcc, M, N, K, eltTypeC,
            eltTypeA, eltTypeB, layoutA, layoutB);
        useC = tb.i1_val(1);
        if (needsPartialAccumulator)
          partialAcc = mmaAcc;
        else
          d = mmaAcc;
        // If we need accumulate separately to have higher precision, insert
        // adds.
        if (requireAddAccumulator) {
          d = d ? faddAccumulate(rewriter, loc, d, partialAcc) : partialAcc;
          numLowPrecisionAcc = 0;
          partialAcc = Value();
        }
      }
      auto acc = unpackLLElements(loc, d, rewriter);
      for (int i = 0; i < acc.size(); ++i) {
        mmaResults.push_back(acc[i]);
      }
    }
  }
  NVVM::WgmmaGroupSyncAlignedOp::create(rewriter, loc);

  if (sync)
    mmaResults = emitWait(rewriter, loc, mmaResults, 0);

  SmallVector<Value> results =
      unpackAccumulator(rewriter, loc, mmaResults, dTensorTy);

  // replace with new packed result
  Type structTy = LLVM::LLVMStructType::getLiteral(
      mmaEncoding.getContext(),
      SmallVector<Type>(results.size(), dTensorTy.getElementType()));
  auto res = packLLElements(loc, typeConverter, results, rewriter, structTy);
  rewriter.replaceOp(op, res);
  return success();
}

#endif // __TLE__

LogicalResult convertWGMMA(triton::nvidia_gpu::WarpGroupDotOp op,
                           triton::nvidia_gpu::WarpGroupDotOp::Adaptor adaptor,
                           const LLVMTypeConverter *typeConverter,
                           ConversionPatternRewriter &rewriter, Value thread) {
  return convertDot(typeConverter, rewriter, op.getLoc(), op.getOperation(),  //
                    op.getA(), op.getB(), op.getC(), op.getD(), op.getUseC(), //
                    adaptor.getA(), adaptor.getB(), adaptor.getC(),           //
                    op.getInputPrecision() == InputPrecision::TF32,
                    op.needsPartialAccumulator(), op.getMaxNumImpreciseAcc(),
                    !op.getIsAsync(), thread);
}
