"""Compile the recorded operator IR on the reviewer's GPU and time both cache policies."""

import ctypes
import importlib.util
import json
import random
import statistics
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.runtime import driver

from fingerprints import sections

if importlib.util.find_spec("torch_musa") is not None:
    import torch_musa  # noqa: F401

POLICY = {
    "nv": " evictionPolicy = l2_evict_first",
    "mthreads": " cacheModifier = musa_i3_o3_l1",
    "metax": " cacheModifier = cg",
    "hcu": " cacheModifier = cs",
    "ppu": " cacheModifier = lu",
}
BINARY = {"nv": "cubin", "mthreads": "mubin", "metax": "mcfatbin", "hcu": "hsaco", "ppu": "hgbin"}
SAMPLES = 31
WARM_REPLAYS = 16


@triton.jit
def sweep_kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=0)
    tl.store(Y + i, x + 0.000001, i < N)


def platform_info():
    target = driver.active.get_current_target()
    api = torch.musa if target.backend == "musa" else torch.cuda
    device = torch.device("musa" if target.backend == "musa" else "cuda", api.current_device())
    name = api.get_device_name(device.index)
    platform = {
        "cuda": "ppu" if "ppu" in name.lower() else "nv", "ppu": "ppu", "musa": "mthreads", "maca": "metax", "hip":
        "hcu"
    }.get(target.backend)
    if platform is None:
        raise RuntimeError(f"No recorded workload for {target} ({name}).")
    return platform, api, device, name, target


def l2_reset(platform):
    if platform in ("nv", "ppu"):
        return ctypes.CDLL("libcuda.so.1").cuCtxResetPersistingL2Cache
    if platform == "mthreads":
        return ctypes.CDLL("libmusa.so.1").muCtxResetPersistingL2Cache
    return None


def make_inputs(case, op, device):
    rows, cols = case["shape"]
    step = 8 if case["layout"] == "stride8" else 1
    physical_rows = 1 if case["layout"] == "rowview" else rows
    inputs = []
    for _ in range(2 if "in1_ptr" in case["args"] else 1):
        base = torch.randn((physical_rows, cols * step), dtype=torch.float32, device=device)
        if op in ("sqrt", "rsqrt", "log1p"):
            base.abs_().add_(0.25)
        inputs.append(base[:, ::step].expand(rows, cols))
    return inputs, torch.empty((rows, cols), dtype=torch.float32, device=device)


def check_output(op, inputs, out):
    # Bound temporary reference tensors on the 512 MiB Hygon broadcasts.
    chunk = max(1, 8388608 // out.size(1))
    for start in range(0, out.size(0), chunk):
        x = inputs[0][start:start + chunk]
        if op.endswith("_and_mul"):
            activation = F.silu if op == "silu_and_mul" else F.gelu
            expected = activation(x) * inputs[1][start:start + chunk]
        else:
            fn = getattr(F, op) if op in ("relu", "silu", "gelu") else getattr(torch, op)
            expected = fn(x)
        torch.testing.assert_close(out[start:start + chunk], expected, rtol=2e-5, atol=2e-5)


def compile_pair(src, case, platform, inputs, out, tmp):
    hinted = src.read_text()
    modifier = POLICY[platform]
    if modifier not in hinted:
        raise ValueError(f"Recorded cache policy is missing from {src.name}.")
    baseline = tmp / src.name
    baseline.write_text(hinted.replace(modifier, ""))
    n = out.numel()
    # The generated rank-one kernels flatten the two contiguous dimensions.
    args = {
        "in0_ptr": inputs[0], "out0_ptr": out, "num_tasks": n, "tiles_per_cta": 1, "s0":
        out.size(0) if "s1" in case["args"] else n, "s1": out.size(1)
    }
    if len(inputs) == 2:
        args["in1_ptr"] = inputs[1]
    argv = [args[key] for key in case["args"]]
    kernels = {}
    images = {}
    for label, path in (("unhinted", baseline), ("hinted", src)):
        k = triton.compile(str(path), options=case["options"])
        kernels[label] = k
        images[label] = sections(k.asm[BINARY[platform]])
    if images["unhinted"] == images["hinted"]:
        raise RuntimeError(f"{src.stem}: the active compiler emitted identical code with and without the policy.")
    return kernels, argv, images


def sweep_buffers(buffers, options):
    x, y = buffers
    n = x.numel()
    grid = (triton.cdiv(n, 1024), )
    sweep_kernel[grid](x, y, n, 1024, **options)
    sweep_kernel[grid](y, x, n, 1024, **options)


def capture(api, platform, kernels, argv, grid, buffers, options):
    graphs = {}
    graph_type = api.MUSAGraph if platform == "mthreads" else api.CUDAGraph
    # Initialize JIT/driver handles before capture.
    launches = {k: kernel[tuple(grid)] for k, kernel in kernels.items()}
    for invoke in launches.values():
        invoke(*argv)
    sweep_options = dict(options, num_warps=4)
    if buffers is not None:
        sweep_buffers(buffers, sweep_options)
    api.synchronize()
    for label, invoke in launches.items():
        graph = graph_type()
        with api.graph(graph):
            if buffers is not None:
                sweep_buffers(buffers, sweep_options)
            invoke(*argv)
        graphs[label] = graph
    return graphs


def measure(graphs, api, kernel_name, warm, reset, seed, trace):
    rng = random.Random(seed)
    for _ in range(8):
        for label in rng.sample(list(graphs), len(graphs)):
            graphs[label].replay()
    api.synchronize()
    order = [label for _ in range(SAMPLES) for label in rng.sample(list(graphs), len(graphs))]
    calls = WARM_REPLAYS + 1 if warm else 1
    with torch.profiler.profile(activities=list(torch.profiler.supported_activities())) as prof:
        for label in order:
            if reset is not None and not warm:
                api.synchronize()
                if reset() != 0:
                    raise RuntimeError("The driver could not reset persisting L2 state.")
            for _ in range(calls):
                graphs[label].replay()
        api.synchronize()
    prof.export_chrome_trace(str(trace))
    events = sorted((e for e in json.loads(trace.read_text())["traceEvents"]
                     if e.get("cat") == "kernel" and e["name"] in (kernel_name, "sweep_kernel")), key=lambda e: e["ts"])
    pattern = [kernel_name] * calls if warm else ["sweep_kernel", "sweep_kernel", kernel_name]
    if [e["name"] for e in events] != pattern * len(order):
        raise RuntimeError("Profiler trace does not contain the expected sweep and target launches.")
    times = {label: [] for label in graphs}
    for label, event in zip(order, events[len(pattern) - 1::len(pattern)]):
        if event["dur"] <= 0:
            raise RuntimeError("Profiler returned a non-positive device duration.")
        times[label].append(event["dur"])
    medians = {label: statistics.median(vals) for label, vals in times.items()}
    return {"median_us": medians, "speedup": medians["unhinted"] / medians["hinted"], "samples_us": times}


def benchmark(op, case, platform, api, device, root, tmp, cold_round, buffers, reset):
    inputs, out = make_inputs(case, op, device)
    kernels, argv, images = compile_pair(root / "ir" / platform / (op + ".ttir"), case, platform, inputs, out, tmp)
    reference = case["executable"]
    matches = images["unhinted"] == reference["upstream"] and images["hinted"] == reference["hinted"]
    result = {
        "shape": case["shape"], "layout": case["layout"], "grid": case["grid"], "executable": images,
        "matches_recorded": matches
    }
    for warm in (False, True):
        graphs = capture(api, platform, kernels, argv, case["grid"], None if warm else buffers, case["options"])
        for graph in graphs.values():
            out.fill_(float("nan"))
            graph.replay()
            api.synchronize()
            check_output(op, inputs, out)
        if not warm:
            changed = inputs[0][:1] if inputs[0].stride(0) == 0 else inputs[0]
            changed.mul_(0.75).add_(0.15)
            for graph in graphs.values():
                out.fill_(float("nan"))
                graph.replay()
                api.synchronize()
                check_output(op, inputs, out)
        seed = 10007 if warm else (9623, 9677, 9749)[cold_round - 1]
        result["warm" if warm else "cold"] = measure(graphs, api, kernels["hinted"].name, warm, reset, seed,
                                                     tmp / "trace.json")
    return result


def run(args, root):
    platform, api, device, name, target = platform_info()
    cases = json.loads((root / "cases.json").read_text())[platform]
    if args.list:
        print(name, ", ".join(cases))
        return
    ops = list(cases) if args.all else [args.op]
    if any(op not in cases for op in ops):
        raise SystemExit(f"No row for {args.op!r} on {name}. Available: {', '.join(cases)}")
    mib = 512 if args.cold_round == 3 else 256
    print(f"{name} | {target} | torch {torch.__version__} | triton {triton.__version__}", flush=True)
    print(f"Cold round {args.cold_round}: two {mib} MiB buffers. Warm: replay 17. {SAMPLES} samples per variant.")
    print("Speedup = unhinted median / hinted median. 'match' covers both archived executable images.")
    print(f"{'Operator':<18} {'Cold base us':>12} {'Cold hint us':>12} {'Cold':>9} {'Warm':>9}   Code", flush=True)
    torch.manual_seed(9749)
    x = torch.randn(mib * 262144, device=device, dtype=torch.float32)
    y = torch.empty_like(x)
    results = {
        "device": name, "target": str(target), "torch": torch.__version__, "triton": triton.__version__, "cold_round":
        args.cold_round, "operators": {}
    }
    reset = l2_reset(platform)
    with tempfile.TemporaryDirectory(prefix="cache-hint-timing-") as tmp:
        for op in ops:
            result = benchmark(op, cases[op], platform, api, device, root, Path(tmp), args.cold_round, (x, y), reset)
            results["operators"][op] = result
            med = result["cold"]["median_us"]
            code = "match" if result["matches_recorded"] else "different build"
            print(
                f"{op:<18} {med['unhinted']:12.3f} {med['hinted']:12.3f} "
                f"{result['cold']['speedup']:8.4f}x {result['warm']['speedup']:8.4f}x   {code}", flush=True)
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")
