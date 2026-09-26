import argparse
import json
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import build_model, parameter_count
from Comparison.adapters import BASELINES, load_model
from utils import load_config, set_seed
from utils.engine import build_loaders, evaluate, resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description="Measure CM-S6 inference efficiency.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", choices=("full", "lite"))
    parser.add_argument("--baseline", choices=tuple(BASELINES))
    parser.add_argument("--factory", help="External model factory in package.module:function form")
    parser.add_argument("--external-root", type=Path, default=PROJECT_ROOT / "external")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=202601)
    parser.add_argument("--latency-warmup", type=int, default=50)
    parser.add_argument("--latency-repeats", type=int, default=200)
    parser.add_argument("--throughput-batch", type=int, default=16)
    parser.add_argument("--throughput-warmup", type=int, default=30)
    parser.add_argument("--throughput-repeats", type=int, default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-flops", action="store_true")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate checkpoint OA on the test split")
    args = parser.parse_args()
    if args.variant and args.baseline:
        parser.error("Choose either --variant or --baseline")
    if args.baseline and not args.factory:
        parser.error("--factory is required with --baseline")
    if args.evaluate and not args.checkpoint:
        parser.error("--evaluate requires --checkpoint")
    if not args.variant and not args.baseline:
        args.variant = "full"
    return args


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_inputs(config, batch_size, device):
    patch_size = int(config.get("patch_size", 11))
    return tuple(
        torch.randn(batch_size, channels, patch_size, patch_size, device=device)
        for channels in config["channels"]
    )


@torch.no_grad()
def timed_forward(model, inputs, warmup, repeats, device):
    for _ in range(warmup):
        model(*inputs)
    synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        model(*inputs)
    synchronize(device)
    return time.perf_counter() - start


def estimate_flops(model, inputs):
    try:
        from thop import profile
    except ImportError as error:
        raise RuntimeError("Install requirements-optional.txt to estimate FLOPs") from error
    macs, _ = profile(model, inputs=inputs, verbose=False)
    return 2.0 * float(macs)


def main():
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device)
    set_seed(args.seed)
    if args.baseline:
        model = load_model(args.factory, config, args.external_root)
        model_name = args.baseline
    else:
        model = build_model(config, args.variant)
        model_name = args.variant
    model = model.to(device).eval()
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)

    latency_inputs = make_inputs(config, 1, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    latency_seconds = timed_forward(
        model,
        latency_inputs,
        args.latency_warmup,
        args.latency_repeats,
        device,
    )

    throughput_inputs = make_inputs(config, args.throughput_batch, device)
    throughput_seconds = timed_forward(
        model,
        throughput_inputs,
        args.throughput_warmup,
        args.throughput_repeats,
        device,
    )
    peak_memory = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else None
    )
    flops = None if args.skip_flops else estimate_flops(model, latency_inputs)
    checkpoint_metrics = None
    if args.evaluate:
        _, _, _, test_loader = build_loaders(config, args.seed, device)
        checkpoint_metrics = evaluate(model, test_loader, device, config["num_classes"])

    results = {
        "dataset": config["dataset"],
        "model": model_name,
        "device": str(device),
        "parameters": parameter_count(model),
        "parameters_M": parameter_count(model) / 1e6,
        "FLOPs": flops,
        "latency_ms": 1000.0 * latency_seconds / args.latency_repeats,
        "throughput_samples_per_second": (
            args.throughput_batch * args.throughput_repeats / throughput_seconds
        ),
        "peak_memory_MiB": peak_memory,
        "checkpoint_metrics": checkpoint_metrics,
        "protocol": {
            "patch_size": int(config.get("patch_size", 11)),
            "latency_batch": 1,
            "latency_warmup": args.latency_warmup,
            "latency_repeats": args.latency_repeats,
            "throughput_batch": args.throughput_batch,
            "throughput_warmup": args.throughput_warmup,
            "throughput_repeats": args.throughput_repeats,
        },
    }
    output = args.output or (
        Path(config["output_root"]) / "analysis" / f"efficiency_{model_name}.json"
    )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2))
    print(f"results={output}")


if __name__ == "__main__":
    main()
