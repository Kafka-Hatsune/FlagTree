import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, make_backend


_WGMMA_PIPELINE_MODE_ATTR = "tle.wgmma_pipeline_mode"
_USER_PROMISE_MODE = "user_promise"


@triton.jit
def _no_trigger_kernel(out):
    tl.store(out, 0)


@triton.jit
def _alloc_barriers_trigger_kernel():
    tle.gpu.alloc_barriers(2)


@triton.jit
def _alloc_barrier_no_trigger_kernel():
    tle.gpu.alloc_barrier()


@triton.jit
def _wgmma_wait_trigger_kernel():
    acc = tl.zeros((16, 16), tl.float32)
    tle.gpu.wgmma_wait(0, acc)


@triton.jit
def _async_dot_wait_no_trigger_kernel():
    acc = tl.zeros((16, 16), tl.float32)
    tle.gpu.async_dot_wait(0, acc)


def _make_cuda_backend(extra_options=None):
    target = GPUTarget("cuda", 90, 32)
    backend = make_backend(target)
    options = backend.parse_options({"num_warps": 4, **(extra_options or {})})
    return target, backend, options


def _make_ttir(kernel, signature=None, extra_options=None):
    target, backend, options = _make_cuda_backend(extra_options)
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    src = ASTSource(fn=kernel, signature=signature or {}, constexprs={})
    module = src.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )
    return str(module)


@pytest.mark.parametrize(
    ("kernel", "signature", "routes_user_promise"),
    [
        (_no_trigger_kernel, {"out": "*i32"}, False),
        (_alloc_barriers_trigger_kernel, {}, True),
        (_wgmma_wait_trigger_kernel, {}, True),
        (_alloc_barrier_no_trigger_kernel, {}, False),
        (_async_dot_wait_no_trigger_kernel, {}, False),
    ],
)
def test_tle_wgmma_pipeline_route_marker_exact_api_list(kernel, signature, routes_user_promise):
    ttir = _make_ttir(kernel, signature)

    assert (_WGMMA_PIPELINE_MODE_ATTR in ttir) is routes_user_promise
    assert (_USER_PROMISE_MODE in ttir) is routes_user_promise


def test_tle_wgmma_pipeline_env_var_does_not_force_user_promise(monkeypatch):
    monkeypatch.setenv("TLE_WGMMA_PIPELINE_MODE", "user_promise")

    ttir = _make_ttir(_no_trigger_kernel, {"out": "*i32"})

    assert _WGMMA_PIPELINE_MODE_ATTR not in ttir
    assert _USER_PROMISE_MODE not in ttir


def test_old_tle_wgmma_pipeline_mode_option_is_not_backend_option():
    _, _, options = _make_cuda_backend({"tle_wgmma_pipeline_mode": "user_promise"})

    assert not hasattr(options, "tle_wgmma_pipeline_mode")
