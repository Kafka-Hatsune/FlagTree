#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::triton::tle {

#define GEN_PASS_DEF_TRITONTLESTAMPWARPCONTEXTENCODINGS
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

namespace tt = mlir::triton;

static constexpr llvm::StringLiteral kAsyncTaskHelperAttr =
    "tle_async_task_helper";
static constexpr llvm::StringLiteral kNumWarpsAttr = "ttg.num-warps";

static Type stampType(Type type, int numWarps) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (tensorType) {
    if (tensorType.getEncoding())
      return type;
    auto marker = WarpContextEncodingAttr::get(type.getContext(), numWarps);
    return tensorType.cloneWithEncoding(marker);
  }

  auto ptrType = dyn_cast<tt::PointerType>(type);
  if (!ptrType)
    return type;

  Type pointeeType = ptrType.getPointeeType();
  Type stampedPointeeType = stampType(pointeeType, numWarps);
  if (stampedPointeeType == pointeeType)
    return type;
  return tt::PointerType::get(stampedPointeeType, ptrType.getAddressSpace());
}

static bool wouldStampType(Type type, int numWarps) {
  return stampType(type, numWarps) != type;
}

static LogicalResult stampHelper(tt::FuncOp helper, int numWarps) {
  WalkResult unsupported = helper.walk([&](Operation *op) -> WalkResult {
    if (isa<tt::JoinOp, tt::SplitOp, tt::TransOp, tt::ReshapeOp>(op)) {
      op->emitOpError("is not supported inside async_task helpers that need "
                      "warp-context type stamping yet");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (unsupported.wasInterrupted())
    return failure();

  for (Type inputType : helper.getFunctionType().getInputs()) {
    if (wouldStampType(inputType, numWarps))
      return helper.emitOpError("async_task helper tensor arguments are not "
                                "supported by warp-context stamping yet");
  }
  for (Type resultType : helper.getFunctionType().getResults()) {
    if (wouldStampType(resultType, numWarps))
      return helper.emitOpError("async_task helper tensor results are not "
                                "supported by warp-context stamping yet");
  }

  for (Block &block : helper.getBody()) {
    for (BlockArgument arg : block.getArguments()) {
      Type stampedType = stampType(arg.getType(), numWarps);
      if (stampedType != arg.getType())
        arg.setType(stampedType);
    }
  }

  helper.walk([&](Operation *op) {
    for (OpResult result : op->getResults()) {
      Type stampedType = stampType(result.getType(), numWarps);
      if (stampedType == result.getType())
        continue;

      if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
        if (auto dense = dyn_cast<DenseElementsAttr>(constant.getValue()))
          constant->setAttr("value",
                            dense.reshape(cast<ShapedType>(stampedType)));
      }

        result.setType(stampedType);
    }
  });

  return success();
}

struct TritonTleStampWarpContextEncodingsPass
    : public impl::TritonTleStampWarpContextEncodingsBase<
          TritonTleStampWarpContextEncodingsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    WalkResult result = module.walk([&](tt::FuncOp funcOp) -> WalkResult {
      if (!funcOp->hasAttr(kAsyncTaskHelperAttr))
        return WalkResult::advance();

      auto numWarpsAttr = funcOp->getAttrOfType<IntegerAttr>(kNumWarpsAttr);
      if (!numWarpsAttr) {
        funcOp.emitOpError("async_task helper is missing `") << kNumWarpsAttr
                                                             << "`";
        return WalkResult::interrupt();
      }

      if (failed(stampHelper(funcOp, numWarpsAttr.getInt())))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });

    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::triton::tle
