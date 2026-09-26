import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import ABLATIONS, build_ablation_model
from utils import load_config, run_training, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run a CM-S6 ablation experiment.")
    parser.add_argument("--config", required=True, help="Dataset YAML configuration")
    parser.add_argument("--experiment", choices=tuple(ABLATIONS), required=True)
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
    model = build_ablation_model(config, args.experiment)
    run_training(
        config,
        model,
        run_name=f"ablation/{args.experiment}",
        run=args.run,
        seed=args.seed,
        device=args.device,
        output=args.output,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
