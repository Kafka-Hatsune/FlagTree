#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#ifdef __TLE__
#include "tle/dialect/include/IR/Dialect.h"
#include <optional>
#endif
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTTRITONTOTRITONGPU
#include "triton/Conversion/TritonToTritonGPU/Passes.h.inc"
} // namespace mlir::triton

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

// pass named attrs (e.g., tt.contiguity) from Triton to Triton
static void addNamedAttrs(Operation *op, DictionaryAttr dictAttrs) {
  for (const NamedAttribute attr : dictAttrs.getValue())
    if (!op->hasAttr(attr.getName()))
      op->setAttr(attr.getName(), attr.getValue());
}

#ifdef __TLE__
static Type convertTleValueType(const TritonGPUTypeConverter *converter,
                                Value value) {
  assert(converter && "expected a TritonGPU type converter");
  Type type = value.getType();
  int contextualNumWarps = converter->getNumWarps(value);
  if (auto tensorType = dyn_cast<RankedTensorType>(type))
    return converter->convertRankedTensorType(tensorType, contextualNumWarps);

  if (auto ptrType = dyn_cast<triton::PointerType>(type)) {
    auto pointeeTensorType =
        dyn_cast<RankedTensorType>(ptrType.getPointeeType());
    if (pointeeTensorType)
      return triton::PointerType::get(
          converter->convertRankedTensorType(pointeeTensorType,
                                             contextualNumWarps),
          ptrType.getAddressSpace());
  }

  return converter->convertType(type);
}

static Type convertTleResultType(const TritonGPUTypeConverter *converter,
                                 Value result) {
  return convertTleValueType(converter, result);
}

static LogicalResult
convertTleResultTypes(const TritonGPUTypeConverter *converter,
                      ResultRange results, SmallVectorImpl<Type> &resultTypes) {
  for (OpResult result : results) {
    Type resultType = convertTleResultType(converter, result);
    if (!resultType)
      return failure();
    resultTypes.push_back(resultType);
  }
  return success();
}

static LogicalResult
convertTleRegionTypes(Region *region, const TritonGPUTypeConverter *converter,
                      ConversionPatternRewriter &rewriter) {
  assert(converter && "expected a TritonGPU type converter");
  if (region->empty())
    return success();

  Block &entry = region->front();
  TypeConverter::SignatureConversion conversion(entry.getNumArguments());
  for (unsigned i = 0, e = entry.getNumArguments(); i < e; ++i) {
    Type newArgType = convertTleValueType(converter, entry.getArgument(i));
    if (!newArgType)
      return failure();
    conversion.addInputs(i, newArgType);
  }
  return rewriter.convertRegionTypes(region, *converter, &conversion);
}

static FailureOr<RankedTensorType>
convertTleTensorResultTypeForOp(const TritonGPUTypeConverter *converter,
                                Operation *op, Value result) {
  auto resultType = dyn_cast<RankedTensorType>(result.getType());
  if (!resultType)
    return failure();

  if (std::optional<int> contextualNumWarps = maybeLookupNumWarps(op)) {
    auto unencodedType =
        RankedTensorType::get(resultType.getShape(),
                              resultType.getElementType());
    return converter->convertRankedTensorType(unencodedType,
                                              *contextualNumWarps);
  }

  auto convertedType =
      dyn_cast_or_null<RankedTensorType>(convertTleResultType(converter,
                                                              result));
  if (!convertedType)
    return failure();
  return convertedType;
}

static FailureOr<Value>
coerceTleTensorValueToType(ConversionPatternRewriter &rewriter, Location loc,
                           Value value, RankedTensorType targetType) {
  if (value.getType() == targetType)
    return value;

  if (auto convert = value.getDefiningOp<triton::gpu::ConvertLayoutOp>()) {
    if (convert.getSrc().getType() == targetType)
      return convert.getSrc();
  }

  if (!isa<RankedTensorType>(value.getType()))
    return failure();
  return triton::gpu::ConvertLayoutOp::create(rewriter, loc, targetType,
                                              value)
      .getResult();
}

static FailureOr<Value>
coerceTleValueToType(ConversionPatternRewriter &rewriter, Location loc,
                     Value value, Type targetType) {
  if (value.getType() == targetType)
    return value;
  if (auto targetTensorType = dyn_cast<RankedTensorType>(targetType))
    return coerceTleTensorValueToType(rewriter, loc, value,
                                      targetTensorType);
  return failure();
}

static FailureOr<Value>
coerceTleOperandToOriginalType(ConversionPatternRewriter &rewriter,
                               Location loc,
                               const TritonGPUTypeConverter *converter,
                               Value operand, Value original) {
  Type targetType = convertTleValueType(converter, original);
  if (!targetType)
    return failure();
  return coerceTleValueToType(rewriter, loc, operand, targetType);
}

static LogicalResult
coerceTleOperandsToOriginalTypes(ConversionPatternRewriter &rewriter,
                                 Location loc,
                                 const TritonGPUTypeConverter *converter,
                                 Operation *op, ValueRange operands,
                                 SmallVectorImpl<Value> &newOperands) {
  if (operands.size() != op->getOperands().size())
    return failure();
  newOperands.reserve(operands.size());
  for (auto [operand, original] : llvm::zip(operands, op->getOperands())) {
    FailureOr<Value> coerced = coerceTleOperandToOriginalType(
        rewriter, loc, converter, operand, original);
    if (failed(coerced))
      return failure();
    newOperands.push_back(*coerced);
  }
  return success();
}
#endif

template <class Op> struct GenericOpPattern : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> retTypes;
#ifdef __TLE__
    auto typeConverter =
        this->template getTypeConverter<TritonGPUTypeConverter>();
    if (failed(
            convertTleResultTypes(typeConverter, op->getResults(), retTypes)))
      return failure();
    SmallVector<Value> operands;
    if (failed(coerceTleOperandsToOriginalTypes(
            rewriter, op.getLoc(), typeConverter, op.getOperation(),
            adaptor.getOperands(), operands)))
      return failure();
#else
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      retTypes)))
      return failure();
    ValueRange operands = adaptor.getOperands();
#endif
    rewriter.replaceOpWithNewOp<Op>(op, retTypes, operands, op->getAttrs());

    return success();
  }
};

class ArithConstantPattern : public OpConversionPattern<arith::ConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type retType =
#ifdef __TLE__
        convertTleResultType(getTypeConverter<TritonGPUTypeConverter>(),
                             op.getResult());
#else
        getTypeConverter()->convertType(op.getType());
#endif
    auto retShapedType = cast<ShapedType>(retType);
    auto value = dyn_cast<DenseElementsAttr>(adaptor.getValue());
    if (isa<RankedTensorType>(retShapedType)) {
      assert(value && "expected a dense elements attribute");
      // This is a hack. We just want to add encoding.
      value = value.reshape(retShapedType);
    }
    addNamedAttrs(rewriter.replaceOpWithNewOp<arith::ConstantOp>(
                      op, retShapedType, value),
                  adaptor.getAttributes());
    return success();
  }
};

void populateArithPatternsAndLegality(TritonGPUTypeConverter &typeConverter,
                                      RewritePatternSet &patterns,
                                      TritonGPUConversionTarget &target) {
  // --------------
  // Add legality and rewrite pattern rules for operations
  // from the Arith dialect. The basic premise is that
  // Arith operations require both inputs to have the same
  // non-null encoding
  // --------------
  MLIRContext *context = patterns.getContext();
  // TODO: there's probably a better way to avoid adding all ops one-by-one
  patterns.add<
      ArithConstantPattern, GenericOpPattern<arith::AddIOp>,
      GenericOpPattern<arith::SubIOp>, GenericOpPattern<arith::MulIOp>,
      GenericOpPattern<arith::DivUIOp>, GenericOpPattern<arith::DivSIOp>,
      GenericOpPattern<arith::CeilDivUIOp>,
      GenericOpPattern<arith::CeilDivSIOp>,
      GenericOpPattern<arith::FloorDivSIOp>, GenericOpPattern<arith::RemUIOp>,
      GenericOpPattern<arith::RemSIOp>, GenericOpPattern<arith::AndIOp>,
      GenericOpPattern<arith::OrIOp>, GenericOpPattern<arith::XOrIOp>,
      GenericOpPattern<arith::ShLIOp>, GenericOpPattern<arith::ShRUIOp>,
      GenericOpPattern<arith::ShRSIOp>, // NegFOp
      // Floating point
      GenericOpPattern<arith::AddFOp>, GenericOpPattern<arith::SubFOp>,
      // MaxMin
      GenericOpPattern<arith::MaximumFOp>, GenericOpPattern<arith::MaxNumFOp>,
      GenericOpPattern<arith::MaxSIOp>, GenericOpPattern<arith::MaxUIOp>,
      GenericOpPattern<arith::MinimumFOp>, GenericOpPattern<arith::MinNumFOp>,
      GenericOpPattern<arith::MinSIOp>, GenericOpPattern<arith::MinUIOp>,
      // Floating point
      GenericOpPattern<arith::MulFOp>, GenericOpPattern<arith::DivFOp>,
      GenericOpPattern<arith::RemFOp>,
      // Cmp
      GenericOpPattern<arith::CmpIOp>, GenericOpPattern<arith::CmpFOp>,
      // Select
      GenericOpPattern<arith::SelectOp>,
      // Cast Ops
      GenericOpPattern<arith::TruncIOp>, GenericOpPattern<arith::TruncFOp>,
      GenericOpPattern<arith::ExtUIOp>, GenericOpPattern<arith::ExtSIOp>,
      GenericOpPattern<arith::ExtFOp>, GenericOpPattern<arith::SIToFPOp>,
      GenericOpPattern<arith::FPToSIOp>, GenericOpPattern<arith::FPToUIOp>,
      GenericOpPattern<arith::UIToFPOp>>(typeConverter, context);
}

void populateMathPatternsAndLegality(TritonGPUTypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     TritonGPUConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  // Rewrite rule
  patterns.add<GenericOpPattern<math::ExpOp>, GenericOpPattern<math::Exp2Op>,
               GenericOpPattern<math::FloorOp>, GenericOpPattern<math::CeilOp>,
               GenericOpPattern<math::CosOp>, GenericOpPattern<math::SinOp>,
               GenericOpPattern<math::LogOp>, GenericOpPattern<math::Log2Op>,
               GenericOpPattern<math::ErfOp>, GenericOpPattern<math::AbsFOp>,
               GenericOpPattern<math::AbsIOp>, GenericOpPattern<math::SqrtOp>,
               GenericOpPattern<math::RsqrtOp>, GenericOpPattern<math::FmaOp>>(
      typeConverter, context);
}

//
// Triton patterns
//
struct TritonExpandDimsPattern
    : public OpConversionPattern<triton::ExpandDimsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = adaptor.getSrc();
#ifdef __TLE__
    FailureOr<Value> maybeSrc = coerceTleOperandToOriginalType(
        rewriter, op.getLoc(), getTypeConverter<TritonGPUTypeConverter>(), src,
        op.getSrc());
    if (failed(maybeSrc))
      return failure();
    src = *maybeSrc;
#endif
    // Type retType = op.getType());
    RankedTensorType argType = cast<RankedTensorType>(src.getType());
    Attribute _argEncoding = argType.getEncoding();
    if (!_argEncoding)
      return failure();
    auto argEncoding = cast<triton::gpu::BlockedEncodingAttr>(_argEncoding);
    // return shape
    auto retShape = argType.getShape().vec();
    retShape.insert(retShape.begin() + op.getAxis(), 1);
    auto newRank = retShape.size();
    // return encoding
    auto retSizePerThread = llvm::to_vector(argEncoding.getSizePerThread());
    retSizePerThread.insert(retSizePerThread.begin() + op.getAxis(), 1);
    auto retThreadsPerWarp = to_vector(argEncoding.getThreadsPerWarp());
    retThreadsPerWarp.insert(retThreadsPerWarp.begin() + op.getAxis(), 1);
    auto retWarpsPerCTA = to_vector(argEncoding.getWarpsPerCTA());
    retWarpsPerCTA.insert(retWarpsPerCTA.begin() + op.getAxis(), 1);
    SmallVector<unsigned, 4> retOrder(retShape.size());
    std::iota(retOrder.begin(), retOrder.end(), 0);

    auto ctaLl = argEncoding.getCTALayout().getLinearLayout();
    auto kBlock = *ctaLl.getInDimNames().begin();
    auto *ctx = kBlock.getContext();
    auto newDim = standardOutDimNames(ctx, newRank)[newRank - 1];
    ctaLl *= LinearLayout::identity1D(1, kBlock, newDim);
    // Move last dim to op.getAxis(). nb is this a std::rotate?
    auto newOrder = to_vector(llvm::seq<int32_t>(newRank));
    for (int i = newRank - 1; i >= op.getAxis() + 1; --i) {
      std::swap(newOrder[i], newOrder[i - 1]);
    }
    ctaLl = transposeLinearLayout(ctaLl, newOrder);
    auto retCTALayout = CTAEncodingAttr::get(ctx, std::move(ctaLl));
    triton::gpu::BlockedEncodingAttr retEncoding =
        triton::gpu::BlockedEncodingAttr::get(getContext(), retSizePerThread,
                                              retThreadsPerWarp, retWarpsPerCTA,
                                              retOrder, retCTALayout);
    // convert operand to slice of return type
    Attribute newArgEncoding = triton::gpu::SliceEncodingAttr::get(
        getContext(), op.getAxis(), retEncoding);
    RankedTensorType newArgType = argType.cloneWithEncoding(newArgEncoding);
    // construct new op
    auto newSrc = triton::gpu::ConvertLayoutOp::create(
        rewriter, op.getLoc(), newArgType, src);
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::ExpandDimsOp>(
                      op, newSrc, adaptor.getAxis()),
                  adaptor.getAttributes());
    return success();
  }

private:
  template <typename T>
  SmallVector<T> insertOne(ArrayRef<T> vec, unsigned axis) const {
    SmallVector<T> res(vec.begin(), vec.end());
    res.insert(res.begin() + axis, 1);
    return res;
  }

  // Example:    order = [   0, 2, 1, 3], dim = 2
  //          resOrder = [2, 0, 3, 1, 4]
  SmallVector<unsigned> insertOrder(ArrayRef<unsigned> order,
                                    unsigned axis) const {
    SmallVector<unsigned> resOrder(order.begin(), order.end());
    for (unsigned i = 0; i < resOrder.size(); ++i)
      if (resOrder[i] >= axis)
        ++resOrder[i];
    resOrder.insert(resOrder.begin(), axis);
    return resOrder;
  }
};

struct TritonDotPattern : public OpConversionPattern<triton::DotOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType origType = op.getType();
    auto origShape = origType.getShape();
    auto typeConverter = getTypeConverter<TritonGPUTypeConverter>();
#ifdef __TLE__
    int numWarps = typeConverter->getNumWarps(op.getResult());
#else
    int numWarps = typeConverter->getNumWarps();
#endif
    int threadsPerWarp = typeConverter->getThreadsPerWarp();
    int numCTAs = typeConverter->getNumCTAs();
    auto rank = origShape.size();
    SmallVector<unsigned> retSizePerThread(rank, 1);
    auto numElements = product<int64_t>(origShape);
    if (numElements / (numWarps * threadsPerWarp) >= 4) {
      retSizePerThread[rank - 1] = 2;
      retSizePerThread[rank - 2] = 2;
    }
    if (numElements / (numWarps * threadsPerWarp) >= 16) {
      retSizePerThread[rank - 1] = 4;
      retSizePerThread[rank - 2] = 4;
    }
    retSizePerThread[rank - 1] = std::min(
        retSizePerThread[rank - 1], static_cast<unsigned>(origShape[rank - 1]));
    retSizePerThread[rank - 2] = std::min(
        retSizePerThread[rank - 2], static_cast<unsigned>(origShape[rank - 2]));

    SmallVector<unsigned> retOrder(rank);
    for (unsigned i = 0; i < rank; ++i)
      retOrder[i] = rank - 1 - i;
    Attribute dEncoding = triton::gpu::BlockedEncodingAttr::get(
        getContext(), origShape, retSizePerThread, retOrder, numWarps,
        threadsPerWarp, numCTAs);
    RankedTensorType retType = origType.cloneWithEncoding(dEncoding);
    Value a = adaptor.getA();
    Value b = adaptor.getB();
    Value c = adaptor.getC();
#ifdef __TLE__
    FailureOr<Value> maybeA = coerceTleOperandToOriginalType(
        rewriter, op.getLoc(), typeConverter, a, op.getA());
    FailureOr<Value> maybeB = coerceTleOperandToOriginalType(
        rewriter, op.getLoc(), typeConverter, b, op.getB());
    FailureOr<Value> maybeC = coerceTleOperandToOriginalType(
        rewriter, op.getLoc(), typeConverter, c, op.getC());
    if (failed(maybeA) || failed(maybeB) || failed(maybeC))
      return failure();
    a = *maybeA;
    b = *maybeB;
    c = *maybeC;
#endif
    // a & b must be of smem layout
    auto aType = cast<RankedTensorType>(a.getType());
    auto bType = cast<RankedTensorType>(b.getType());
    Type aEltType = aType.getElementType();
    Type bEltType = bType.getElementType();
    Attribute aEncoding = aType.getEncoding();
    Attribute bEncoding = bType.getEncoding();
    if (!aEncoding || !bEncoding)
      return failure();
    if (!mlir::isa<triton::gpu::DotOperandEncodingAttr>(aEncoding)) {
      Attribute encoding = triton::gpu::DotOperandEncodingAttr::get(
          getContext(), 0, dEncoding, aEltType);
      auto dstType = aType.cloneWithEncoding(encoding);
      a = triton::gpu::ConvertLayoutOp::create(rewriter, a.getLoc(), dstType,
                                               a);
    }
    if (!mlir::isa<triton::gpu::DotOperandEncodingAttr>(bEncoding)) {
      Attribute encoding = triton::gpu::DotOperandEncodingAttr::get(
          getContext(), 1, dEncoding, bEltType);
      auto dstType = bType.cloneWithEncoding(encoding);
      b = triton::gpu::ConvertLayoutOp::create(rewriter, b.getLoc(), dstType,
                                               b);
    }
#ifdef __TLE__
    FailureOr<Value> maybeConvertedC =
        coerceTleTensorValueToType(rewriter, c.getLoc(), c, retType);
    if (failed(maybeConvertedC))
      return failure();
    c = *maybeConvertedC;
#else
    c = triton::gpu::ConvertLayoutOp::create(rewriter, c.getLoc(), retType, c);
#endif

    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::DotOp>(
                      op, retType, a, b, c, adaptor.getInputPrecision(),
                      adaptor.getMaxNumImpreciseAcc()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonCatPattern : public OpConversionPattern<triton::CatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::CatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // The cat op satisfy two conditions:
    // 1. output.numel = lhs.numel + rhs.numel
    // 2. output.total_elems_per_thread =
    // next_power_of_2(lhs.total_elems_per_thread + rhs.total_elems_per_thread)
    // For now, this behaves like generic, but this
    // will evolve when we add support for `can_reorder=False`.
    auto retType = cast<RankedTensorType>(
#ifdef __TLE__
        convertTleResultType(this->getTypeConverter<TritonGPUTypeConverter>(),
                             op.getResult()));
#else
        this->getTypeConverter()->convertType(op.getType()));
#endif
    auto retEncoding =
        cast<triton::gpu::BlockedEncodingAttr>(retType.getEncoding());
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
#ifdef __TLE__
    auto converter = getTypeConverter<TritonGPUTypeConverter>();
    FailureOr<Value> maybeLhs = coerceTleOperandToOriginalType(
        rewriter, op.getLoc(), converter, lhs, op.getLhs());
    FailureOr<Value> maybeRhs = coerceTleOperandToOriginalType(
        rewriter, op.getLoc(), converter, rhs, op.getRhs());
    if (failed(maybeLhs) || failed(maybeRhs))
      return failure();
    lhs = *maybeLhs;
    rhs = *maybeRhs;
#endif
    auto lhsType = lhs.getType();
    auto rhsType = rhs.getType();
    auto lhsTotalElemsPerThread = triton::gpu::getTotalElemsPerThread(lhsType);
    auto rhsTotalElemsPerThread = triton::gpu::getTotalElemsPerThread(rhsType);
    auto retTotalElemsPerThread = triton::gpu::getTotalElemsPerThread(retType);
    auto retShape = retType.getShape();
    auto retOrder = retEncoding.getOrder();
    auto retThreadsPerWarp = retEncoding.getThreadsPerWarp();
    auto retWarpsPerCTA = retEncoding.getWarpsPerCTA();
    // Get new retSizePerThread if ret elems per thread is not enough.
    // We have to round it up to the next power of 2 due to triton's tensor size
    // constraint.
    auto newRetTotalElemsPerThread =
        nextPowOf2(lhsTotalElemsPerThread + rhsTotalElemsPerThread);
    auto newRetSizePerThread = llvm::to_vector(retEncoding.getSizePerThread());
    newRetSizePerThread[retOrder[0]] *=
        newRetTotalElemsPerThread / retTotalElemsPerThread;
    triton::gpu::BlockedEncodingAttr newRetEncoding =
        triton::gpu::BlockedEncodingAttr::get(
            getContext(), newRetSizePerThread, retThreadsPerWarp,
            retWarpsPerCTA, retOrder, retEncoding.getCTALayout());
    auto newRetType = retType.cloneWithEncoding(newRetEncoding);
    SmallVector<Value> operands{lhs, rhs};
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::CatOp>(
                      op, newRetType, operands),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonJoinOpPattern : public OpConversionPattern<triton::JoinOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(JoinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
#ifdef __TLE__
    auto converter = getTypeConverter<TritonGPUTypeConverter>();
    FailureOr<Value> maybeLhs = coerceTleOperandToOriginalType(
        rewriter, op.getLoc(), converter, lhs, op.getLhs());
    FailureOr<Value> maybeRhs = coerceTleOperandToOriginalType(
        rewriter, op.getLoc(), converter, rhs, op.getRhs());
    if (failed(maybeLhs) || failed(maybeRhs))
      return failure();
    lhs = *maybeLhs;
    rhs = *maybeRhs;
#endif
    // Simply rely on type inference for this op.  (Notably, GenericOpPattern
    // does not do this, instead it assigns the default layout to the ins and
    // outs.)
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::JoinOp>(
                      op, lhs, rhs),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonSplitOpPattern : public OpConversionPattern<triton::SplitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(SplitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto src = adaptor.getSrc();
#ifdef __TLE__
    FailureOr<Value> maybeSrc = coerceTleOperandToOriginalType(
        rewriter, op.getLoc(), getTypeConverter<TritonGPUTypeConverter>(), src,
        op.getSrc());
    if (failed(maybeSrc))
      return failure();
    src = *maybeSrc;
#endif
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto srcEnc = dyn_cast<BlockedEncodingAttr>(srcTy.getEncoding());
    int rank = srcEnc.getOrder().size();
    auto typeConverter = getTypeConverter<TritonGPUTypeConverter>();

    // The operand to split must have:
    //  - a blocked layout, with
    //  - sizePerThread = 2 in the last dimension,
    //  - threadsPerWarp, warpsPerCTA, and CTAsPerCGA = 1 in the last dim, and
    //  - the last dimension minor.
    // If that's not the case, add a convert before the split.
    if (!srcEnc || srcEnc.getSizePerThread().back() != 2 ||
        srcEnc.getOrder().front() != rank - 1) {
      // If we take the default encoding for the op's result (i.e. post-split)
      // and add 1 to the end of each dim, that gives us what we want.  Other
      // than making a legal src encoding, our choice of layout doesn't matter;
      // it'll get fixed by RemoveLayoutConversions.
      auto defaultEnc = getDefaultBlockedEncoding(
          getContext(),
          cast<RankedTensorType>(op.getResult(0).getType()).getShape(),
          typeConverter->getNumWarps(), typeConverter->getThreadsPerWarp(),
          typeConverter->getNumCTAs());

      auto append = [&](ArrayRef<unsigned> vals, unsigned val) {
        SmallVector<unsigned> res(vals);
        res.push_back(val);
        return res;
      };
      auto prepend = [&](ArrayRef<unsigned> vals, unsigned val) {
        SmallVector<unsigned> res;
        res.push_back(val);
        res.append(vals.begin(), vals.end());
        return res;
      };

      auto layout = defaultEnc.getCTALayout().getLinearLayout();
      auto kBlock = StringAttr::get(getContext(), "block");
      auto newDim = standardOutDimNames(getContext(), rank)[rank - 1];
      layout *= LinearLayout::identity1D(1, kBlock, newDim);
      srcEnc = BlockedEncodingAttr::get(
          getContext(), append(defaultEnc.getSizePerThread(), 2),
          append(defaultEnc.getThreadsPerWarp(), 1),
          append(defaultEnc.getWarpsPerCTA(), 1),
          prepend(defaultEnc.getOrder(), rank - 1),
          CTAEncodingAttr::get(getContext(), layout));
      srcTy = srcTy.cloneWithEncoding(srcEnc);
      src = ConvertLayoutOp::create(rewriter, op.getLoc(), srcTy, src);
    }

    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::SplitOp>(op, src),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonTransPattern : public OpConversionPattern<TransOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = adaptor.getSrc();
#ifdef __TLE__
    FailureOr<Value> maybeSrc = coerceTleOperandToOriginalType(
        rewriter, op.getLoc(), getTypeConverter<TritonGPUTypeConverter>(), src,
        op.getSrc());
    if (failed(maybeSrc))
      return failure();
    src = *maybeSrc;
#endif
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto srcEnc = srcTy.getEncoding();
    if (!srcEnc)
      return failure();
    addNamedAttrs(rewriter.replaceOpWithNewOp<TransOp>(op, src, op.getOrder()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonBroadcastPattern
    : public OpConversionPattern<triton::BroadcastOp> {
  using OpConversionPattern::OpConversionPattern;

  // This creates a tensor with the new shape but the argument's layout
  LogicalResult
  matchAndRewrite(BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = adaptor.getSrc();
#ifdef __TLE__
    FailureOr<Value> maybeSrc = coerceTleOperandToOriginalType(
        rewriter, op.getLoc(), getTypeConverter<TritonGPUTypeConverter>(), src,
        op.getSrc());
    if (failed(maybeSrc))
      return failure();
    src = *maybeSrc;
#endif
    auto srcType = cast<RankedTensorType>(src.getType());
    auto srcEncoding = srcType.getEncoding();
    if (!srcEncoding)
      return failure();
    Type retType = op.getType().cloneWithEncoding(srcEncoding);
    // Type retType = this->getTypeConverter()->convertType(op.getType());
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::BroadcastOp>(
                      op, retType, src),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonReducePattern : public OpConversionPattern<triton::ReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange operands = adaptor.getOperands();
#ifdef __TLE__
    SmallVector<Value> coercedOperands;
    if (failed(coerceTleOperandsToOriginalTypes(
            rewriter, op.getLoc(), getTypeConverter<TritonGPUTypeConverter>(),
            op.getOperation(), adaptor.getOperands(), coercedOperands)))
      return failure();
    operands = coercedOperands;
#endif
    auto newReduce = triton::ReduceOp::create(
        rewriter, op.getLoc(), operands, adaptor.getAxis());
    addNamedAttrs(newReduce, adaptor.getAttributes());

    auto &newCombineOp = newReduce.getCombineOp();
    rewriter.cloneRegionBefore(op.getCombineOp(), newCombineOp,
                               newCombineOp.end());
    rewriter.replaceOp(op, newReduce.getResult());
    return success();
  }
};

struct TritonScanPattern : public OpConversionPattern<triton::ScanOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange operands = adaptor.getOperands();
#ifdef __TLE__
    SmallVector<Value> coercedOperands;
    if (failed(coerceTleOperandsToOriginalTypes(
            rewriter, op.getLoc(), getTypeConverter<TritonGPUTypeConverter>(),
            op.getOperation(), adaptor.getOperands(), coercedOperands)))
      return failure();
    operands = coercedOperands;
#endif
    auto newScan =
        triton::ScanOp::create(rewriter, op.getLoc(), operands,
                               adaptor.getAxis(), op.getReverse());
    addNamedAttrs(newScan, adaptor.getAttributes());

    auto &newCombineOp = newScan.getCombineOp();
    rewriter.cloneRegionBefore(op.getCombineOp(), newCombineOp,
                               newCombineOp.end());
    rewriter.replaceOp(op, newScan.getResult());
    return success();
  }
};

struct TritonMapElementwisePattern
    : public OpConversionPattern<triton::MapElementwiseOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::MapElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
#ifdef __TLE__
    auto converter = getTypeConverter<TritonGPUTypeConverter>();
#else
    auto converter = getTypeConverter();
#endif
    SmallVector<Type> resultTys;
#ifdef __TLE__
    auto err = convertTleResultTypes(converter, op.getResults(), resultTys);
#else
    auto err = converter->convertTypes(op.getResults().getType(), resultTys);
#endif
    if (failed(err)) {
      return err;
    }
    ValueRange operands = adaptor.getOperands();
#ifdef __TLE__
    SmallVector<Value> coercedOperands;
    if (failed(coerceTleOperandsToOriginalTypes(
            rewriter, op.getLoc(), converter, op.getOperation(),
            adaptor.getOperands(), coercedOperands)))
      return failure();
    operands = coercedOperands;
#endif

    auto newMapOp = triton::MapElementwiseOp::create(
        rewriter, op.getLoc(), resultTys, operands, op.getPack());
    addNamedAttrs(newMapOp, adaptor.getAttributes());

    auto &newScalarOp = newMapOp.getScalarOp();
    rewriter.cloneRegionBefore(op.getScalarOp(), newScalarOp,
                               newScalarOp.end());
    rewriter.replaceOp(op, newMapOp.getResult());
    return success();
  }
};

class TritonFuncOpPattern : public OpConversionPattern<triton::FuncOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
#ifdef __TLE__
    auto converter = getTypeConverter<TritonGPUTypeConverter>();
    TypeConverter::SignatureConversion result(op.getNumArguments());

    SmallVector<Type> newArgTypes;
    SmallVector<Type> newResultTypes;
    if (!op.getBody().empty()) {
      Block &entry = op.getBody().front();
      for (unsigned i = 0, e = entry.getNumArguments(); i < e; ++i) {
        Type newArgType = convertTleValueType(converter, entry.getArgument(i));
        if (!newArgType)
          return failure();
        result.addInputs(i, newArgType);
        newArgTypes.push_back(newArgType);
      }
    } else if (failed(converter->convertTypes(
                   op.getFunctionType().getInputs(), newArgTypes))) {
      return failure();
    }
    if (failed(converter->convertTypes(op.getFunctionType().getResults(),
                                       newResultTypes)))
      return failure();

    auto newOp = rewriter.replaceOpWithNewOp<triton::FuncOp>(
        op, op.getName(),
        FunctionType::get(op.getContext(), newArgTypes, newResultTypes));
#else
    auto converter = getTypeConverter();
    TypeConverter::SignatureConversion result(op.getNumArguments());
    auto newOp = rewriter.replaceOpWithNewOp<triton::FuncOp>(
        op, op.getName(), op.getFunctionType());
#endif
    addNamedAttrs(newOp, adaptor.getAttributes());
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(),
                                newOp.getBody().end());
    // Convert just the entry block. The remaining unstructured control flow is
    // converted by br patterns.
    if (!newOp.getBody().empty())
      rewriter.applySignatureConversion(&newOp.getBody().front(), result,
                                        converter);
    return success();
  }
};

class TritonCallOpPattern : public OpConversionPattern<triton::CallOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.replaceOpWithNewOp<triton::CallOp>(
        op, op.getCallee(), op.getResultTypes(), adaptor.getOperands());
    addNamedAttrs(newOp, adaptor.getAttributes());
    return success();
  }
};

class TritonReturnOpPattern : public OpConversionPattern<ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, ReturnOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

void populateTritonPatterns(TritonGPUTypeConverter &typeConverter,
                            RewritePatternSet &patterns, unsigned numCTAs) {
  MLIRContext *context = patterns.getContext();
  patterns.insert< // TODO: view should have custom pattern that views the
                   // layout
      // clang-format off
      GenericOpPattern<triton::AdvanceOp>,
      GenericOpPattern<triton::MakeTensorPtrOp>,
      GenericOpPattern<triton::ReshapeOp>,
      GenericOpPattern<triton::BitcastOp>,
      GenericOpPattern<triton::FpToFpOp>,
      GenericOpPattern<triton::IntToPtrOp>,
      GenericOpPattern<triton::PtrToIntOp>,
      GenericOpPattern<triton::SplatOp>,
      GenericOpPattern<triton::UnsplatOp>,
      GenericOpPattern<triton::AddPtrOp>,
      TritonBroadcastPattern,
      TritonCatPattern,
      TritonJoinOpPattern,
      TritonSplitOpPattern,
      GenericOpPattern<triton::ClampFOp>,
      GenericOpPattern<triton::PreciseSqrtOp>,
      GenericOpPattern<triton::PreciseDivFOp>,
      GenericOpPattern<triton::MulhiUIOp>,
      GenericOpPattern<triton::ElementwiseInlineAsmOp>,
      TritonReducePattern,
      GenericOpPattern<triton::ReduceReturnOp>,
      TritonScanPattern,
      GenericOpPattern<triton::ScanReturnOp>,
      GenericOpPattern<triton::MakeRangeOp>,
#ifdef __TLE__
      GenericOpPattern<triton::gpu::LocalAllocOp>,
      GenericOpPattern<triton::gpu::LocalStoreOp>,
      GenericOpPattern<triton::gpu::LocalLoadOp>,
#endif
      TritonExpandDimsPattern,
      TritonTransPattern,
      TritonDotPattern,
      TritonMapElementwisePattern,
      GatherScatterOpPattern<DescriptorGatherOp>,
      GatherScatterOpPattern<DescriptorScatterOp>,
      GenericOpPattern<triton::LoadOp>,
      GenericOpPattern<triton::StoreOp>,
      GenericOpPattern<triton::HistogramOp>,
      GenericOpPattern<triton::GatherOp>,
      GenericOpPattern<triton::ExternElementwiseOp>,
      GenericOpPattern<triton::PrintOp>,
      GenericOpPattern<triton::AssertOp>,
      GenericOpPattern<triton::AtomicCASOp>,
      GenericOpPattern<triton::AtomicRMWOp>,
      GenericOpPattern<triton::DescriptorLoadOp>,
      GenericOpPattern<triton::DescriptorStoreOp>,
      GenericOpPattern<triton::DescriptorReduceOp>,
      // this assumes the right layout will be set later for dot scaled.
      GenericOpPattern<triton::DotScaledOp>,
      GenericOpPattern<triton::CallOp>,
      GenericOpPattern<ReturnOp>,
      TritonFuncOpPattern
      // clang-format on
      >(typeConverter, context);
}
//
// SCF patterns
//
// This is borrowed from ConvertForOpTypes in
//    SCF/Transforms/StructuralTypeConversions.cpp
struct SCFForPattern : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern::OpConversionPattern;
  // Ref: ConvertForOpTypes
  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
#ifdef __TLE__
    auto typeConverter = getTypeConverter<TritonGPUTypeConverter>();
#else
    auto typeConverter = getTypeConverter();
#endif
    auto newOp =
        cast<scf::ForOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().end());

    // Now, update all the types.

    // Convert the types of block arguments within the given region. This
    // replaces each block with a new block containing the updated signature.
    // The entry block may have a special conversion if `entryConversion` is
    // provided. On success, the new entry block to the region is returned for
    // convenience. Otherwise, failure is returned.
#ifdef __TLE__
    if (failed(convertTleRegionTypes(&newOp.getRegion(), typeConverter,
                                     rewriter))) {
#else
    if (failed(
            rewriter.convertRegionTypes(&newOp.getRegion(), *typeConverter))) {
#endif
      return rewriter.notifyMatchFailure(op, "could not convert body types");
    }
    // Update the result types to the new converted types.
    SmallVector<Type> newResultTypes;
#ifdef __TLE__
    for (OpResult result : op.getResults()) {
      Type newType = convertTleResultType(typeConverter, result);
#else
    for (Type type : op.getResultTypes()) {
      Type newType = typeConverter->convertType(type);
#endif
      if (!newType)
        return rewriter.notifyMatchFailure(op, "not a 1:1 type conversion");
      newResultTypes.push_back(newType);
    }

    // Change the clone to use the updated operands. We could have cloned with
    // a IRMapping, but this seems a bit more direct.
#ifdef __TLE__
    SmallVector<Value> newOperands{adaptor.getLowerBound(),
                                   adaptor.getUpperBound(),
                                   adaptor.getStep()};
    auto initArgs = adaptor.getInitArgs();
    if (initArgs.size() != newResultTypes.size())
      return rewriter.notifyMatchFailure(op, "invalid loop init arity");
    for (auto [initArg, targetType] : llvm::zip(initArgs, newResultTypes)) {
      FailureOr<Value> coerced =
          coerceTleValueToType(rewriter, op.getLoc(), initArg, targetType);
      if (failed(coerced))
        return rewriter.notifyMatchFailure(op,
                                           "could not convert init operand");
      newOperands.push_back(*coerced);
    }
    newOp->setOperands(newOperands);
#else
    newOp->setOperands(adaptor.getOperands());
#endif

    for (auto t : llvm::zip(newOp.getResults(), newResultTypes))
      std::get<0>(t).setType(std::get<1>(t));

    rewriter.replaceOp(op, newOp.getResults());

    return success();
  }
};

// This is borrowed from ConvertFIfOpTypes in
//    SCF/Transforms/StructuralTypeConversions.cpp
class SCFIfPattern : public OpConversionPattern<scf::IfOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
#ifdef __TLE__
    auto typeConverter = getTypeConverter<TritonGPUTypeConverter>();
#else
    auto typeConverter = getTypeConverter();
#endif
    // TODO: Generalize this to any type conversion, not just 1:1.
    //
    // We need to implement something more sophisticated here that tracks which
    // types convert to which other types and does the appropriate
    // materialization logic.
    // For example, it's possible that one result type converts to 0 types and
    // another to 2 types, so newResultTypes would at least be the right size to
    // not crash in the llvm::zip call below, but then we would set the the
    // wrong type on the SSA values! These edge cases are also why we cannot
    // safely use the TypeConverter::convertTypes helper here.
    SmallVector<Type> newResultTypes;
#ifdef __TLE__
    for (OpResult result : op.getResults()) {
      Type newType = convertTleResultType(typeConverter, result);
#else
    for (auto type : op.getResultTypes()) {
      Type newType = typeConverter->convertType(type);
#endif
      if (!newType)
        return rewriter.notifyMatchFailure(op, "not a 1:1 type conversion");
      newResultTypes.push_back(newType);
    }

    // See comments in the ForOp pattern for why we clone without regions and
    // then inline.
    scf::IfOp newOp =
        cast<scf::IfOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
    rewriter.inlineRegionBefore(op.getThenRegion(), newOp.getThenRegion(),
                                newOp.getThenRegion().end());
    rewriter.inlineRegionBefore(op.getElseRegion(), newOp.getElseRegion(),
                                newOp.getElseRegion().end());

    // Update the operands and types.
    newOp->setOperands(adaptor.getOperands());
    for (auto t : llvm::zip(newOp.getResults(), newResultTypes))
      std::get<0>(t).setType(std::get<1>(t));
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

// This is borrowed from ConvertFIfOpTypes in
//    SCF/Transforms/StructuralTypeConversions.cpp
class SCFWhilePattern : public OpConversionPattern<scf::WhileOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::WhileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
#ifdef __TLE__
    auto *converter = getTypeConverter<TritonGPUTypeConverter>();
#else
    auto *converter = getTypeConverter();
#endif
    assert(converter);
    SmallVector<Type> newResultTypes;
#ifdef __TLE__
    if (failed(
            convertTleResultTypes(converter, op.getResults(), newResultTypes)))
#else
    if (failed(converter->convertTypes(op.getResultTypes(), newResultTypes)))
#endif
      return failure();

#ifdef __TLE__
    SmallVector<Value> newOperands;
    newOperands.reserve(adaptor.getOperands().size());
    if (adaptor.getOperands().size() != newResultTypes.size())
      return rewriter.notifyMatchFailure(op, "invalid while operand arity");
    for (auto [operand, targetType] :
         llvm::zip(adaptor.getOperands(), newResultTypes)) {
      FailureOr<Value> coerced =
          coerceTleValueToType(rewriter, op.getLoc(), operand, targetType);
      if (failed(coerced))
        return rewriter.notifyMatchFailure(op,
                                           "could not convert while operand");
      newOperands.push_back(*coerced);
    }
    auto newOp =
        scf::WhileOp::create(rewriter, op.getLoc(), newResultTypes, newOperands);
#else
    auto newOp = scf::WhileOp::create(rewriter, op.getLoc(), newResultTypes,
                                      adaptor.getOperands());
#endif
    for (auto i : {0u, 1u}) {
      auto &dstRegion = newOp.getRegion(i);
      rewriter.inlineRegionBefore(op.getRegion(i), dstRegion, dstRegion.end());
#ifdef __TLE__
      if (failed(convertTleRegionTypes(&dstRegion, converter, rewriter)))
#else
      if (failed(rewriter.convertRegionTypes(&dstRegion, *converter)))
#endif
        return rewriter.notifyMatchFailure(op, "could not convert body types");
    }
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

class SCFConditionPattern : public OpConversionPattern<scf::ConditionOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(scf::ConditionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
#ifdef __TLE__
    SmallVector<Value> newOperands;
    newOperands.reserve(adaptor.getOperands().size());
    auto converter = getTypeConverter<TritonGPUTypeConverter>();
    for (auto [operand, original] :
         llvm::zip(adaptor.getOperands(), op->getOperands())) {
      Type targetType = convertTleValueType(converter, original);
      FailureOr<Value> coerced =
          coerceTleValueToType(rewriter, op.getLoc(), operand, targetType);
      if (failed(coerced))
        return rewriter.notifyMatchFailure(op,
                                           "could not convert condition operand");
      newOperands.push_back(*coerced);
    }
    rewriter.modifyOpInPlace(op, [&]() { op->setOperands(newOperands); });
#else
    rewriter.modifyOpInPlace(op,
                             [&]() { op->setOperands(adaptor.getOperands()); });
#endif
    return success();
  }
};

class SCFYieldPattern : public OpConversionPattern<scf::YieldOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
#ifdef __TLE__
    SmallVector<Value> newOperands;
    newOperands.reserve(adaptor.getOperands().size());
    auto converter = getTypeConverter<TritonGPUTypeConverter>();
    for (auto [operand, original] :
         llvm::zip(adaptor.getOperands(), op->getOperands())) {
      Type targetType = convertTleValueType(converter, original);
      FailureOr<Value> coerced =
          coerceTleValueToType(rewriter, op.getLoc(), operand, targetType);
      if (failed(coerced))
        return rewriter.notifyMatchFailure(op,
                                           "could not convert yield operand");
      newOperands.push_back(*coerced);
    }
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, newOperands);
#else
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands());
#endif
    return success();
  }
};

void populateSCFPatterns(TritonGPUTypeConverter &typeConverter,
                         RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<SCFYieldPattern, SCFForPattern, SCFIfPattern, SCFWhilePattern,
               SCFConditionPattern>(typeConverter, context);
}

// CF

class CFBranchPattern : public OpConversionPattern<cf::BranchOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::BranchOp op, cf::BranchOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
#ifdef __TLE__
    auto converter = getTypeConverter<TritonGPUTypeConverter>();
    SmallVector<Value> operands;
    if (failed(coerceTleOperandsToOriginalTypes(
            rewriter, op.getLoc(), converter, op.getOperation(),
            adaptor.getOperands(), operands)))
      return failure();
    auto newOp = rewriter.replaceOpWithNewOp<cf::BranchOp>(
        op, op.getSuccessor(), operands);
    if (failed(convertTleRegionTypes(newOp.getSuccessor()->getParent(),
                                     converter, rewriter)))
      return failure();
#else
    auto converter = getTypeConverter();
    auto newOp = rewriter.replaceOpWithNewOp<cf::BranchOp>(
        op, op.getSuccessor(), adaptor.getOperands());
    if (failed(rewriter.convertRegionTypes(newOp.getSuccessor()->getParent(),
                                           *converter)))
      return failure();
#endif
    return success();
  }
};

class CFCondBranchPattern : public OpConversionPattern<cf::CondBranchOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::CondBranchOp op, cf::CondBranchOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
#ifdef __TLE__
    auto converter = getTypeConverter<TritonGPUTypeConverter>();
    FailureOr<Value> condition = coerceTleOperandToOriginalType(
        rewriter, op.getLoc(), converter, adaptor.getCondition(),
        op.getCondition());
    if (failed(condition))
      return failure();
    SmallVector<Value> trueOperands;
    SmallVector<Value> falseOperands;
    auto originalTrueOperands = op.getTrueDestOperands();
    if (adaptor.getTrueDestOperands().size() != originalTrueOperands.size())
      return failure();
    trueOperands.reserve(adaptor.getTrueDestOperands().size());
    for (auto [operand, original] :
         llvm::zip(adaptor.getTrueDestOperands(), originalTrueOperands)) {
      FailureOr<Value> coerced = coerceTleOperandToOriginalType(
          rewriter, op.getLoc(), converter, operand, original);
      if (failed(coerced))
        return failure();
      trueOperands.push_back(*coerced);
    }
    auto originalFalseOperands = op.getFalseDestOperands();
    if (adaptor.getFalseDestOperands().size() != originalFalseOperands.size())
      return failure();
    falseOperands.reserve(adaptor.getFalseDestOperands().size());
    for (auto [operand, original] :
         llvm::zip(adaptor.getFalseDestOperands(), originalFalseOperands)) {
      FailureOr<Value> coerced = coerceTleOperandToOriginalType(
          rewriter, op.getLoc(), converter, operand, original);
      if (failed(coerced))
        return failure();
      falseOperands.push_back(*coerced);
    }
    auto newOp = rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        op, *condition, op.getTrueDest(), trueOperands, op.getFalseDest(),
        falseOperands);
    addNamedAttrs(newOp, adaptor.getAttributes());

    if (failed(convertTleRegionTypes(newOp.getTrueDest()->getParent(),
                                     converter, rewriter)))
      return failure();
    if (failed(convertTleRegionTypes(newOp.getFalseDest()->getParent(),
                                     converter, rewriter)))
      return failure();
#else
    auto converter = getTypeConverter();
    auto newOp = rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        op, adaptor.getCondition(), op.getTrueDest(),
        adaptor.getTrueDestOperands(), op.getFalseDest(),
        adaptor.getFalseDestOperands());
    addNamedAttrs(newOp, adaptor.getAttributes());

    if (failed(rewriter.convertRegionTypes(newOp.getTrueDest()->getParent(),
                                           *converter)))
      return failure();
    if (failed(rewriter.convertRegionTypes(newOp.getFalseDest()->getParent(),
                                           *converter)))
      return failure();
#endif
    return success();
  }
};

void populateCFPatterns(TritonGPUTypeConverter &typeConverter,
                        RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<CFCondBranchPattern, CFBranchPattern>(typeConverter, context);
}

#ifdef __TLE__
class TleDSLRegionOpPattern : public OpConversionPattern<tle::DSLRegionOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tle::DSLRegionOp op, tle::DSLRegionOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.cloneWithoutRegions<tle::DSLRegionOp>(op);
    Region &body = op.getBody(), &newBody = newOp.getBody();
    rewriter.inlineRegionBefore(body, newBody, newBody.end());

    if (failed(convertTleRegionTypes(
            &newBody, getTypeConverter<TritonGPUTypeConverter>(), rewriter))) {
      return rewriter.notifyMatchFailure(op, "could not convert body types");
    }
    SmallVector<Value> operands;
    if (failed(coerceTleOperandsToOriginalTypes(
            rewriter, op.getLoc(), getTypeConverter<TritonGPUTypeConverter>(),
            op.getOperation(), adaptor.getOperands(), operands)))
      return failure();
    newOp->setOperands(operands);
    for (OpResult result : newOp->getResults()) {
#ifdef __TLE__
      result.setType(convertTleResultType(
          getTypeConverter<TritonGPUTypeConverter>(), result));
#else
      result.setType(getTypeConverter()->convertType(result.getType()));
#endif
    }

    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

class TleExtractTileOpPattern : public OpConversionPattern<tle::ExtractTileOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tle::ExtractTileOp op, tle::ExtractTileOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = adaptor.getSrc();
    FailureOr<Value> maybeSrc = coerceTleOperandToOriginalType(
        rewriter, op.getLoc(), getTypeConverter<TritonGPUTypeConverter>(), src,
        op.getSrc());
    if (failed(maybeSrc))
      return failure();
    src = *maybeSrc;

    auto srcType = dyn_cast<RankedTensorType>(src.getType());
    if (!srcType) {
      return op.emitError("source must be a ranked tensor");
    }
    auto srcEnc = srcType.getEncoding();
    if (!srcEnc) {
      return op.emitError("source tensor must have encoding attribute");
    }

    Type retType = op.getType().cloneWithEncoding(srcEnc);

    auto newOp = rewriter.replaceOpWithNewOp<tle::ExtractTileOp>(
        op, retType, src, adaptor.getIndex());

    if (auto tileShapeAttr = op->getAttr("tile_shape"))
      newOp->setAttr("tile_shape", tileShapeAttr);

    addNamedAttrs(newOp, adaptor.getAttributes());

    return success();
  }
};

// insert_tile op pattern
class TleInsertTileOpPattern : public OpConversionPattern<tle::InsertTileOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tle::InsertTileOp op, tle::InsertTileOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = adaptor.getSrc();
    Value tile = adaptor.getTile();
    FailureOr<Value> maybeSrc = coerceTleOperandToOriginalType(
        rewriter, op.getLoc(), getTypeConverter<TritonGPUTypeConverter>(), src,
        op.getSrc());
    FailureOr<Value> maybeTile = coerceTleOperandToOriginalType(
        rewriter, op.getLoc(), getTypeConverter<TritonGPUTypeConverter>(), tile,
        op.getTile());
    if (failed(maybeSrc) || failed(maybeTile))
      return failure();
    src = *maybeSrc;
    tile = *maybeTile;

    auto srcType = dyn_cast<RankedTensorType>(src.getType());
    if (!srcType) {
      return op.emitError("source must be a ranked tensor");
    }

    auto srcEnc = srcType.getEncoding();
    if (!srcEnc) {
      return op.emitError("source tensor must have encoding attribute");
    }

    auto tileType = dyn_cast<RankedTensorType>(tile.getType());
    if (!tileType) {
      return op.emitError("tile must be a ranked tensor");
    }

    auto tileEnc = tileType.getEncoding();
    if (!tileEnc) {
      return op.emitError("tile tensor must have encoding attribute");
    }

    Type retType = op.getType().cloneWithEncoding(srcEnc);

    auto newOp = rewriter.replaceOpWithNewOp<tle::InsertTileOp>(
        op, retType, src, tile, adaptor.getIndex());

    addNamedAttrs(newOp, adaptor.getAttributes());

    return success();
  }
};

class TleWGMMAOpPattern : public OpConversionPattern<tle::WGMMAOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tle::WGMMAOp op, tle::WGMMAOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter<TritonGPUTypeConverter>();
    FailureOr<RankedTensorType> maybeRetType =
        convertTleTensorResultTypeForOp(converter, op.getOperation(),
                                        op.getD());
    if (failed(maybeRetType))
      return failure();

    Value a = adaptor.getA();
    Value b = adaptor.getB();
    FailureOr<Value> maybeA = coerceTleOperandToOriginalType(
        rewriter, op.getLoc(), converter, a, op.getA());
    FailureOr<Value> maybeB = coerceTleOperandToOriginalType(
        rewriter, op.getLoc(), converter, b, op.getB());
    FailureOr<Value> maybeAcc = coerceTleTensorValueToType(
        rewriter, op.getLoc(), adaptor.getC(), *maybeRetType);
    if (failed(maybeA) || failed(maybeB) || failed(maybeAcc))
      return failure();
    a = *maybeA;
    b = *maybeB;

    SmallVector<Type> retTypes{*maybeRetType};
    SmallVector<Value> operands{a, b, *maybeAcc};
    rewriter.replaceOpWithNewOp<tle::WGMMAOp>(op, retTypes, operands,
                                              op->getAttrs());
    return success();
  }
};

class TleWGMMAWaitOpPattern
    : public OpConversionPattern<tle::WGMMAWaitOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tle::WGMMAWaitOp op, tle::WGMMAWaitOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter<TritonGPUTypeConverter>();
    FailureOr<RankedTensorType> maybeRetType =
        convertTleTensorResultTypeForOp(converter, op.getOperation(),
                                        op.getOutput());
    if (failed(maybeRetType))
      return failure();

    FailureOr<Value> maybeInput = coerceTleTensorValueToType(
        rewriter, op.getLoc(), adaptor.getInput(), *maybeRetType);
    if (failed(maybeInput))
      return failure();

    SmallVector<Type> retTypes{*maybeRetType};
    SmallVector<Value> operands{*maybeInput};
    rewriter.replaceOpWithNewOp<tle::WGMMAWaitOp>(op, retTypes, operands,
                                                  op->getAttrs());
    return success();
  }
};

// flagtree tle raw
void populateTleRawPatterns(TritonGPUTypeConverter &typeConverter,
                            RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns
      .add<TleDSLRegionOpPattern, TleExtractTileOpPattern,
           TleInsertTileOpPattern, GenericOpPattern<tle::LocalPointersOp>,
           GenericOpPattern<tle::RemotePointersOp>,
           GenericOpPattern<tle::ExclusiveCumsumOp>,
           TleWGMMAOpPattern, TleWGMMAWaitOpPattern,
           GenericOpPattern<tle::DistributedBarrierOp>,
           GenericOpPattern<tle::YieldOp>,
           GenericOpPattern<tle::ExtractAllocatedPtrOp>,
           GenericOpPattern<tle::ExtractAlignedPtrOp>,
           GenericOpPattern<tle::ExtractOffsetOp>,
           GenericOpPattern<tle::ExtractSizesOp>,
           GenericOpPattern<tle::ExtractStridesOp>,
           GenericOpPattern<tle::ExtractPtrOp>, GenericOpPattern<tle::PackOp>>(
          typeConverter, context);
}
#endif

class ConvertTritonToTritonGPU
    : public triton::impl::ConvertTritonToTritonGPUBase<
          ConvertTritonToTritonGPU> {
public:
  using ConvertTritonToTritonGPUBase::ConvertTritonToTritonGPUBase;

  void runOnOperation() override {
    if (target.getValue().empty()) {
      mlir::emitError(
          getOperation().getLoc(),
          "'convert-triton-to-tritongpu' requires 'target' option to be set");
      return signalPassFailure();
    }

    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    // type converter
    TritonGPUTypeConverter typeConverter(context, numWarps, threadsPerWarp,
                                         numCTAs, enableSourceRemat);
    TritonGPUConversionTarget target(*context, typeConverter);
    // rewrite patterns
    RewritePatternSet patterns(context);
    // add rules
    populateArithPatternsAndLegality(typeConverter, patterns, target);
    populateMathPatternsAndLegality(typeConverter, patterns, target);
    populateTritonPatterns(typeConverter, patterns, numCTAs);
    // TODO: can we use
    //    mlir::scf::populateSCFStructurealTypeConversionsAndLegality(...) here?
    populateSCFPatterns(typeConverter, patterns);
    populateCFPatterns(typeConverter, patterns);
#ifdef __TLE__
    populateTleRawPatterns(typeConverter, patterns);
#endif
    patterns.insert<GenericOpPattern<ub::PoisonOp>>(typeConverter, context);

    Builder b(&getContext());
    mod->setAttr(AttrNumWarpsName, b.getI32IntegerAttr(numWarps));
    mod->setAttr(AttrNumThreadsPerWarp, b.getI32IntegerAttr(threadsPerWarp));
    mod->setAttr(AttrNumCTAsName, b.getI32IntegerAttr(numCTAs));
    mod->setAttr(AttrTargetName, b.getStringAttr(this->target.getValue()));

    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
