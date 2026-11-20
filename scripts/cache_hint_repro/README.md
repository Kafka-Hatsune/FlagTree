# Reproduce the cache hint measurements in FlagTree PR #1658

Use a Linux GPU environment with the vendor's PyTorch (including its device profiler and graph support) and FlagTree from [PR #1658](https://github.com/flagos-ai/FlagTree/pull/1658) built for that GPU. Run this in the Python environment that imports that FlagTree build.

```sh
curl -fL https://raw.githubusercontent.com/Kafka-Hatsune/FlagTree/refs/heads/codex/cache-hint-repro-1658/scripts/cache_hint_repro.pyz -o cache_hint_repro.pyz && python cache_hint_repro.pyz
```

The default run detects the GPU and measures **`abs`**. The first two times are its cold unhinted and hinted medians in microseconds. The next two columns give the cold and warm speedups. Both divide the unhinted median by the hinted median under the same cache condition.

```sh
python cache_hint_repro.pyz --all                 # All reported operators on this GPU
python cache_hint_repro.pyz --list                # Available operator names
python cache_hint_repro.pyz --op relu --json relu.json
```

| Measured GPU | FlagGems | FlagGems-vllm |
| --- | ---: | ---: |
| NVIDIA H800 | 10 | 0 |
| MThreads S5000 | 14 | 2 |
| MetaX C550 | 14 | 2 |
| Hygon gfx936 | 10 | 0 |
| PPU ZW810E | 14 | 2 |

## What is compiled

The archive contains readable Triton IR captured at the public FlagGems and FlagGems-vllm operator entry points. It uses one installed compiler to build each IR twice, with and without the cache attribute. The unhinted IR agrees with the IR from the original upstream run after removing debug locations. The runner uses the recorded FP32 shape, layout, launch grid and compiler options. It compares the results of each variant with PyTorch, including after changing a graph's input on the GPU.

`Code = match` means the executable sections from **both** builds match their separately recorded counterparts from upstream and the hinted PR build. A different compiler toolchain or GPU can produce different device code. The JSON output includes the section hashes and all latency samples. The benchmark covers device-kernel latency. All the inputs needed to run it are in the archive, including the downstream kernels.

The original measurements used [FlagTree upstream](https://github.com/flagos-ai/FlagTree/commit/20e0d0880518c9c77e445a579446337cdcacf4d7), [the hinted build for Hygon](https://github.com/Kafka-Hatsune/FlagTree/commit/a82119cd2b4a7ff87b5648acc2a0b01a8a0b7ad2) and [its parent for the other four platforms](https://github.com/Kafka-Hatsune/FlagTree/commit/be79b15bb398af7d17e85b6d685994528292007e). The operators came from [FlagGems](https://github.com/Kafka-Hatsune/FlagGems/commit/9dc7fc025785b6f7a7c05443dc8de8d5ef8c5350), [FlagGems with Hygon dispatch](https://github.com/Kafka-Hatsune/FlagGems/commit/a8b7963a5586438e66eb2f5f63636d0ec6558d7a) and [FlagGems-vllm](https://github.com/Kafka-Hatsune/FlagGems-vllm/commit/506f58239bd6a6fb5b300adbefddf70def1d4a76).

## How timing works

- **Cold** uses two 512 MiB buffers by default. Before each timed call, two device kernels copy the buffers in opposite directions. NVIDIA, MThreads and PPU also reset persisting L2 state. The profiler takes the duration of the target kernel. The default reproduces cold round 3, which supplied the per-operator latency and speedup columns.
- **Warm** replays the same kernel 16 times at the same input and output addresses and times the 17th replay. It uses the explicit warm protocol from the PR tables.
- Each median uses 31 device samples per variant in shuffled order. To repeat the three cold processes used for the platform overview, run the following on an idle device. Rounds 1 and 2 use two 256 MiB buffers.

```sh
for r in 1 2 3; do python cache_hint_repro.pyz --all --cold-round "$r" --json "round-$r.json"; done
```

To inspect or run the unpacked version, use `python scripts/cache_hint_repro` from a checkout of this branch. The same source can be repackaged with `python -m zipapp scripts/cache_hint_repro -c -o cache_hint_repro.pyz`.
