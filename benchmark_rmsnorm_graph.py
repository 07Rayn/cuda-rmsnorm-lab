"""Compare RMSNorm implementations under CUDA Graph replay, not eager latency."""
import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from rmsnorm_reference import rmsnorm_reference

PROJECT = Path(__file__).resolve().parent


def load_kernel(name, filename):
    # Separate modules: never link the two launch_rmsnorm definitions together.
    return load(
        name=name,
        sources=[str(PROJECT / "rmsnorm_binding.cpp"), str(PROJECT / filename)],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2"],
        verbose=False,
    )


def capture_batch(fn, iterations):
    # Warm up on a side stream before recording GPU operations.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(20):
            output = fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(iterations):
            output = fn()
    # Keep output alive: graph replay uses fixed memory addresses.
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    return graph, output


def measure_graphs(graphs, iterations, repeats, rng):
    samples = {name: [] for name in graphs}
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        order = list(graphs)
        rng.shuffle(order)  # Avoid always measuring one implementation first.
        for name in order:
            start.record()
            graphs[name][0].replay()
            end.record()
            end.synchronize()
            # One replay contains iterations RMSNorm calls.
            samples[name].append(start.elapsed_time(end) * 1000 / iterations)
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, default=PROJECT / "benchmark_graph_results.json")
    args = parser.parse_args()
    if min(args.iterations, args.repeats, args.rounds) < 1:
        parser.error("iterations, repeats and rounds must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable")

    print("Loading old and new CUDA modules...", flush=True)
    naive = load_kernel("rmsnorm_naive_graph_ext", "rmsnorm_kernel_native.cu")
    block = load_kernel("rmsnorm_block_graph_ext", "rmsnorm_kernel.cu")
    torch.manual_seed(42)
    rng = random.Random(42)
    eps = 1e-6
    shapes = [(1, 4096), (32, 4096), (256, 4096), (130, 257)]
    report = {
        "mode": "CUDA Graph batch replay; warm fixed inputs; not eager or pure kernel latency",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "dtype": "float32", "eps": eps,
        "iterations": args.iterations, "repeats": args.repeats, "rounds": args.rounds,
        "source_sha256": {
            name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest()
            for name in ["rmsnorm_binding.cpp", "rmsnorm_kernel.cu",
                         "rmsnorm_kernel_native.cu", "benchmark_rmsnorm_graph.py"]
        },
        "results": [],
    }
    print(report["gpu"], flush=True)
    print("CUDA Graph replay: microseconds per call (lower is better)", flush=True)
    with torch.no_grad():
        for rows, cols in shapes:
            x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
            weight = torch.randn(cols, device="cuda", dtype=torch.float32)
            functions = {
                "reference": lambda: rmsnorm_reference(x, weight, eps),
                "pytorch": lambda: F.rms_norm(x, (cols,), weight, eps),
                "naive": lambda: naive.rmsnorm(x, weight, eps),
                "block": lambda: block.rmsnorm(x, weight, eps),
            }
            expected = functions["reference"]()
            for fn in functions.values():
                torch.testing.assert_close(fn(), expected, rtol=1e-4, atol=1e-5)
            graphs = {name: capture_batch(fn, args.iterations) for name, fn in functions.items()}
            for graph, output in graphs.values():
                torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)

            # Verify replay reads the live input buffers, including on our custom stream.
            x.zero_()
            for graph, output in graphs.values():
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
            x.normal_()
            expected = functions["reference"]()
            for graph, output in graphs.values():
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)
            print(f"\nshape=({rows}, {cols}): eager and graph correctness PASS", flush=True)

            record = {"shape": [rows, cols], "rounds": []}
            for round_id in range(args.rounds):
                samples = measure_graphs(graphs, args.iterations, args.repeats, rng)
                medians = {name: statistics.median(values) for name, values in samples.items()}
                record["rounds"].append({"samples_us": samples, "median_us": medians})
                timings = "  ".join(f"{name}={value:.3f}" for name, value in medians.items())
                print(f"round {round_id + 1}: {timings}", flush=True)
                print(f"  naive/block={medians['naive']/medians['block']:.3f}x; "
                      f"pytorch/block={medians['pytorch']/medians['block']:.3f}x", flush=True)
            report["results"].append(record)
            torch.cuda.synchronize()
            del graphs, graph, output
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved raw samples and source hashes: {args.output}", flush=True)


if __name__ == "__main__":
    main()
