#ifndef TLE_RAW_ATTR_DEFS
#define TLE_RAW_ATTR_DEFS

include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "tle/dialect/include/IR/TleDialect.td"

def WarpContextEncodingAttr : AttrDef<Tle_Dialect, "WarpContextEncoding"> {
  let mnemonic = "warp_context";
  let description = [{
    Temporary pre-TritonGPU-conversion marker used by TLE async_task helper
    bodies. The marker records which warp context should be used when assigning
    the real TTGPU layout for an otherwise unencoded tensor type.
  }];
  let parameters = (ins "int":$numWarps);
  let assemblyFormat = "`<` struct(params) `>`";
}

#endif
