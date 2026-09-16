"""Command line entry points, usable from an installed wheel."""

import argparse
import json
from pathlib import Path

from .artifacts import ARTIFACT_ROOT
from .evaluation import evaluate


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Dynamic Connectome: learned sparse routing in JAX"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "evaluate"):
        sub = commands.add_parser(
            name, help="Evaluate bundled or locally trained numeric checkpoints"
        )
        sub.add_argument("--artifacts", type=Path, default=ARTIFACT_ROOT)
        sub.add_argument("--seed", type=int, default=20260916)
        sub.add_argument(
            "--batch-size", type=positive, default=32 if name == "demo" else 256
        )
        sub.add_argument(
            "--batches", type=positive, default=2 if name == "demo" else 16
        )
        sub.add_argument("--models", nargs="+")
        sub.add_argument("--output", type=Path)
    sub = commands.add_parser(
        "train", help="Train Stage1, Stage2 and hybrid with independent data streams"
    )
    sub.add_argument("--config", type=Path, required=True)
    sub.add_argument("--output", type=Path, required=True)
    sub = commands.add_parser(
        "train-baselines", help="Train dense, random, heuristic and dense-soft controls"
    )
    sub.add_argument("--config", type=Path, required=True)
    sub.add_argument("--output", type=Path, required=True)
    sub.add_argument("--reference-artifacts", type=Path, default=ARTIFACT_ROOT)
    sub = commands.add_parser(
        "plot", help="Regenerate publication figures from an evaluation report"
    )
    sub.add_argument("--report", type=Path, required=True)
    sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            from .training import train

            report = train(args.config, args.output)
            print(
                f"Saved selected models to {args.output}; Stage2 test MSE {report['stage2']['test']['mse']:.4f}; hybrid {report['hybrid']['test']['mse']:.4f}"
            )
        elif args.command == "train-baselines":
            from .train_baselines import train_baselines

            train_baselines(args.config, args.output, args.reference_artifacts)
            print(f"Saved baseline models and held-out scores to {args.output}")
        elif args.command == "plot":
            from .plotting import plot_report

            plot_report(args.report, args.output)
        else:
            model_ids = args.models
            if (
                args.command == "demo"
                and model_ids is None
                and args.artifacts == ARTIFACT_ROOT
            ):
                model_ids = ["learned-0", "random-0", "dense_soft-0", "hybrid-0"]
            result = evaluate(
                args.artifacts,
                seed=args.seed,
                batch_size=args.batch_size,
                batches=args.batches,
                model_ids=model_ids,
            )
            if args.output:
                write_json(args.output, result)
            print(
                "Fixed-checkpoint evaluation on common synthetic draws (MSE; lower is better)"
            )
            for family, row in result["summary"].items():
                print(
                    f"{family:12s} {row['mean_mse']:.6f}  seed SD {row['std_across_seeds']:.6f}  n={row['checkpoints']}"
                )
            print(f"Unrestricted payload sum: {result['unrestricted_sum']['mse']:.6f}")
            if args.command == "demo":
                print(
                    "Small functionality demo; use dc evaluate for the documented 4,096-example table."
                )
    except (ValueError, FileExistsError) as error:
        parser.exit(2, f"Error: {error}\n")
