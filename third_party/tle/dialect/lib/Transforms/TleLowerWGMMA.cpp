#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::triton::tle {

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

#define GEN_PASS_DEF_TRITONTLELOWERWGMMA
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

static LogicalResult verifyWGMMAUses(WGMMAOp op) {
  for (OpOperand &use : op.getD().getUses()) {
    Operation *user = use.getOwner();
    if (auto next = dyn_cast<WGMMAOp>(user)) {
      if (use.getOperandNumber() == 2)
        continue;
      return op.emitOpError("result may only feed the accumulator operand of "
                            "another tle.wgmma or tle.wgmma_wait");
    }
    if (isa<WGMMAWaitOp>(user))
      continue;
    return op.emitOpError("async result must be consumed by tle.wgmma_wait "
                          "before ordinary tensor use");
  }
  return success();
}

static RankedTensorType getMMAType(WGMMAOp op) {
  auto context = op.getContext();
  auto accType = cast<RankedTensorType>(op.getC().getType());
  auto aElemType =
      cast<ttg::TensorOrMemDesc>(op.getA().getType()).getElementType();
  std::optional<int> maybeNumWarps =
      ttg::maybeLookupNumWarps(op.getOperation());
  if (!maybeNumWarps) {
    op.emitOpError("requires a contextual `ttg.num-warps` to select the "
                   "WGMMA accumulator layout");
    return {};
  }

  unsigned numWarps = *maybeNumWarps;
  SmallVector<int64_t> retShapePerCTA =
      accType.getEncoding() ? ttg::getShapePerCTA(accType)
                            : SmallVector<int64_t>(accType.getShape());
  SmallVector<unsigned, 3> instrShape =
      mmaVersionToInstrShape(3, retShapePerCTA, aElemType, numWarps);
  SmallVector<unsigned, 2> warpsPerCTA = {numWarps, 1};
  SmallVector<unsigned, 2> CTAsPerCGA = {1, 1};
  SmallVector<unsigned, 2> CTASplitNum = {1, 1};
  SmallVector<unsigned, 2> CTAOrder = {1, 0};
  auto CTALayout = ttg::CTAEncodingAttr::fromSplitParams(
      context, CTAsPerCGA, CTASplitNum, CTAOrder);
  auto mmaEncoding = ttg::NvidiaMmaEncodingAttr::get(
      context, 3, 0, warpsPerCTA, CTALayout, instrShape);
  return RankedTensorType::get(accType.getShape(), accType.getElementType(),
                               mmaEncoding);
}

static Value materializeMMAAccumulator(OpBuilder &builder, WGMMAOp op,
                                       DenseMap<Value, Value> &encodedAccs) {
  Value acc = op.getC();
  if (Value mapped = encodedAccs.lookup(acc))
    return mapped;

  RankedTensorType mmaType = getMMAType(op);
  if (!mmaType)
    return {};
  auto accType = cast<RankedTensorType>(acc.getType());
  if (accType.getEncoding() == mmaType.getEncoding())
    return acc;
  return ttg::ConvertLayoutOp::create(builder, op.getLoc(), mmaType, acc);
}

struct TritonTleLowerWGMMAPass
    : public impl::TritonTleLowerWGMMABase<TritonTleLowerWGMMAPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool failed = false;

    module.walk([&](triton::FuncOp func) {
      if (failed)
        return;

      SmallVector<Operation *> worklist;
      SmallVector<WGMMAOp> wgmmas;
      func.walk([&](Operation *op) {
        if (isa<WGMMAOp, WGMMAWaitOp>(op))
          worklist.push_back(op);
        if (auto wgmma = dyn_cast<WGMMAOp>(op))
          wgmmas.push_back(wgmma);
      });

      for (WGMMAOp op : wgmmas) {
        if (mlir::failed(verifyWGMMAUses(op))) {
          failed = true;
          return;
        }
      }

      DenseMap<Value, Value> encodedAccs;
      for (Operation *op : worklist) {
        OpBuilder builder(op);
        if (auto wgmma = dyn_cast<WGMMAOp>(op)) {
          Value acc = materializeMMAAccumulator(builder, wgmma, encodedAccs);
          if (!acc) {
            failed = true;
            return;
          }
          auto nativeDot = ttng::WarpGroupDotOp::create(
              builder, wgmma.getLoc(), acc.getType(), wgmma.getA(),
              wgmma.getB(), acc, Value(), wgmma.getInputPrecision(),
              wgmma.getMaxNumImpreciseAcc(), wgmma.getIsAsync());
          encodedAccs[wgmma.getD()] = nativeDot.getD();
          continue;
        }

        auto wait = cast<WGMMAWaitOp>(op);
        Value encodedInput = encodedAccs.lookup(wait.getInput());
        if (!encodedInput) {
          wait.emitOpError("input must be the async result of tle.wgmma");
          failed = true;
          return;
        }

        SmallVector<Value, 1> waitInputs{encodedInput};
        auto nativeWait = ttng::WarpGroupDotWaitOp::create(
            builder, wait.getLoc(), waitInputs, wait.getPendings());
        Value waited = nativeWait.getResult(0);
        Value released = ttg::ConvertLayoutOp::create(
            builder, wait.getLoc(), wait.getOutput().getType(), waited);
        wait.getOutput().replaceAllUsesWith(released);
        wait.erase();
      }

      for (WGMMAOp op : llvm::reverse(wgmmas))
        op.erase();
    });

    if (failed)
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::triton::tle
