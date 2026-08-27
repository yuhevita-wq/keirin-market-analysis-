from __future__ import annotations

import argparse

from .core import compute_d
from .io import read_market_csv, write_d_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="keirin-market-analysis")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="validate full markets and compute D for all 35 trios")
    analyze.add_argument("--trio", required=True, help="3連複CSV")
    analyze.add_argument("--trifecta", required=True, help="3連単CSV")
    analyze.add_argument("--out", required=True, help="DレポートCSV")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "analyze":
        trio_rows = read_market_csv(args.trio)
        trifecta_rows = read_market_csv(args.trifecta)
        rows = compute_d(trio_rows, trifecta_rows)
        write_d_csv(args.out, rows)
        print(f"OK: wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
