#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::triton::tle {

#define GEN_PASS_DEF_TRITONTLEINLINEASYNCTASKHELPERS
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

static constexpr llvm::StringLiteral kAsyncTaskHelperAttr =
    "tle.async_task.helper";

static LogicalResult verifyAsyncTaskHelper(tt::CallOp callOp,
                                           tt::FuncOp callee) {
  if (callee.getVisibility() != SymbolTable::Visibility::Private)
    return callOp.emitOpError("callee marked with `")
           << kAsyncTaskHelperAttr << "` must be private";
  if (callee.isExternal())
    return callOp.emitOpError("callee marked with `")
           << kAsyncTaskHelperAttr << "` must have a body";
  if (callee.getFunctionType().getNumResults() != 0)
    return callOp.emitOpError("callee marked with `")
           << kAsyncTaskHelperAttr << "` must not return values";
  if (callee.getNumArguments() != callOp.getNumOperands())
    return callOp.emitOpError("async_task helper call operand count must match "
                              "callee argument count");
  if (callOp.getNumResults() != 0)
    return callOp.emitOpError("async_task helper calls must be void");
  if (!llvm::hasSingleElement(callee.getBody()))
    return callOp.emitOpError("callee marked with `")
           << kAsyncTaskHelperAttr << "` must have a single block";

  Block &body = callee.getBody().front();
  auto returnOp = dyn_cast_or_null<tt::ReturnOp>(body.getTerminator());
  if (!returnOp || returnOp.getNumOperands() != 0)
    return callOp.emitOpError("callee marked with `")
           << kAsyncTaskHelperAttr << "` must end with an empty tt.return";
  return success();
}

static LogicalResult inlineAsyncTaskHelperCall(tt::CallOp callOp,
                                               tt::FuncOp callee) {
  auto *parentRegion = callOp->getParentRegion();
  auto partitions =
      dyn_cast_or_null<ttg::WarpSpecializePartitionsOp>(
          parentRegion ? parentRegion->getParentOp() : nullptr);
  if (!partitions)
    return callOp.emitOpError("async_task helper call must be directly inside "
                              "a ttg.warp_specialize partition region");

  if (failed(verifyAsyncTaskHelper(callOp, callee)))
    return failure();

  OpBuilder builder(callOp);
  IRMapping mapping;
  for (auto [arg, operand] : llvm::zip(callee.getArguments(),
                                       callOp.getOperands()))
    mapping.map(arg, operand);

  Block &body = callee.getBody().front();
  for (Operation &op : body.without_terminator())
    builder.clone(op, mapping);

  callOp.erase();
  return success();
}

struct TritonTleInlineAsyncTaskHelpersPass
    : public impl::TritonTleInlineAsyncTaskHelpersBase<
          TritonTleInlineAsyncTaskHelpersPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<tt::CallOp> calls;
    SmallPtrSet<Operation *, 8> helpers;

    module.walk([&](tt::CallOp callOp) {
      tt::FuncOp callee = module.lookupSymbol<tt::FuncOp>(callOp.getCallee());
      if (!callee || !callee->hasAttr(kAsyncTaskHelperAttr))
        return;
      calls.push_back(callOp);
      helpers.insert(callee.getOperation());
    });

    for (tt::CallOp callOp : calls) {
      tt::FuncOp callee = module.lookupSymbol<tt::FuncOp>(callOp.getCallee());
      if (!callee || !callee->hasAttr(kAsyncTaskHelperAttr)) {
        callOp.emitOpError("lost async_task helper callee before inlining");
        signalPassFailure();
        return;
      }
      if (failed(inlineAsyncTaskHelperCall(callOp, callee))) {
        signalPassFailure();
        return;
      }
    }

    for (Operation *helperOp : helpers) {
      auto helper = dyn_cast_or_null<tt::FuncOp>(helperOp);
      if (!helper)
        continue;
      if (SymbolTable::symbolKnownUseEmpty(helper, module)) {
        helper.erase();
        continue;
      }
      if (helper->hasAttr(kAsyncTaskHelperAttr)) {
        helper.emitOpError("still has call sites after async_task helper "
                           "inlining");
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace
} // namespace mlir::triton::tle
