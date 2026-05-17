from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Task1 summaries into a report table.")
    parser.add_argument("--runs-dir", default="runs/task1")
    parser.add_argument("--out", default="reports/task1/task1_results.csv")
    parser.add_argument("--include-all", action="store_true", help="Include smoke and tracker-only runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for summary_path in sorted(Path(args.runs_dir).glob("*/summary.json")):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        experiment_name = summary["experiment_name"]
        if not args.include_all and (
            experiment_name.startswith("smoke_") or experiment_name.startswith("swanlab_local_")
        ):
            continue
        rows.append(
            {
                "experiment": experiment_name,
                "best_epoch": summary["best_epoch"],
                "best_val_acc": summary["best_val_top1"],
                "test_acc": summary["test_top1"],
                "test_top5": summary["test_top5"],
                "best_weights": summary["weights"]["best"],
                "curves": summary["artifacts"]["curves_png"],
            }
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    if df.empty:
        print("No summaries found.")
    else:
        try:
            print(df.to_markdown(index=False))
        except ImportError:
            print(df.to_string(index=False))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
