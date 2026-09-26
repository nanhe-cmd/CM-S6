import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Comparison.adapters import BASELINES, load_model
from utils import load_config, run_training, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train an external baseline with the CM-S6 protocol.")
    parser.add_argument("--config", required=True, help="Dataset YAML configuration")
    parser.add_argument("--baseline", choices=tuple(BASELINES), required=True)
    parser.add_argument(
        "--factory",
        required=True,
        help="External model factory in package.module:function form",
    )
    parser.add_argument("--external-root", type=Path, default=PROJECT_ROOT / "external")
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--seed", type=int, default=202601)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(args.seed)
    model = load_model(args.factory, config, args.external_root)
    run_training(
        config,
        model,
        run_name=f"comparison/{args.baseline}",
        run=args.run,
        seed=args.seed,
        device=args.device,
        output=args.output,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
