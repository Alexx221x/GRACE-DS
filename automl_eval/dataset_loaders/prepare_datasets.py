"""Prepare external datasets into GRACE task JSONs"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="prepare_datasets")
    p.add_argument(
        "--tml-root",
        default=None,
        help="Path to a cloned, prepared mykolapinchuk/tml-bench.",
    )
    p.add_argument(
        "--tabred-root",
        default=None,
        help="Path to a cloned, prepared yandex-research/tabred.",
    )
    p.add_argument(
        "--tabred-only",
        default=None,
        help="Comma-separated subset of TabReD dataset names (default: all 8).",
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Also build the two synthetic-DGP tasks.",
    )
    p.add_argument(
        "--include-toy",
        action="store_true",
        help="Include TML-bench toy_regression competition (excluded by default).",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=60000,
        help="Row cap for very large TabReD datasets (default 60000; 0 = no cap).",
    )
    p.add_argument(
        "--no-reference",
        action="store_true",
        help="Skip baseline/oracle reference-score computation (faster; normalization disabled).",
    )
    args = p.parse_args(argv)

    compute_reference = not args.no_reference
    max_rows = None if args.max_rows == 0 else args.max_rows
    written: list[Path] = []

    if args.tml_root:
        from automl_eval.dataset_loaders import tml_bench

        print(
            f"[tml-bench] discovering prepared competitions under {args.tml_root} ..."
        )
        paths = tml_bench.build_all(
            args.tml_root,
            include_toy=args.include_toy,
            compute_reference=compute_reference,
        )
        written.extend(paths)
        print(f"[tml-bench] wrote {len(paths)} task(s).")

    if args.tabred_root:
        from automl_eval.dataset_loaders import tabred

        names = tabred.available_datasets()
        if args.tabred_only:
            wanted = {n.strip() for n in args.tabred_only.split(",")}
            names = [n for n in names if n in wanted]
        print(f"[tabred] preparing {len(names)} dataset(s) from {args.tabred_root} ...")
        for name in names:
            try:
                path = tabred.build_task_json(
                    args.tabred_root,
                    name,
                    compute_reference=compute_reference,
                    max_rows=max_rows,
                )
                written.append(path)
                print(f"  [tabred] {name} -> {path.name}")
            except (FileNotFoundError, KeyError) as exc:
                print(f"  [tabred] SKIP {name}: {exc}")

    if args.synthetic:
        from automl_eval.dataset_loaders import synthetic

        print("[synthetic] building classification + regression tasks ...")
        written.append(
            synthetic.build_classification_task(compute_reference=compute_reference)
        )
        written.append(
            synthetic.build_regression_task(compute_reference=compute_reference)
        )
        print("[synthetic] done.")

    if not written:
        print(
            "Nothing prepared. Pass at least one of --tml-root / --tabred-root / --synthetic."
        )
        return 1

    print(f"\n[OK] prepared {len(written)} task JSON(s):")
    for pth in written:
        print(f"  {pth}")
    print("\nNext: list these task ids in your config and run:")
    print(
        "  bash scripts/run_experiments.sh --config configs/final.yaml --resume --rerun-transient"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
