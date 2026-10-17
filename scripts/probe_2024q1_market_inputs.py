from __future__ import annotations

# Analysis branch trigger: v6.1-HM01 / 2024Q1. No scheme changes.
import csv
import json
from pathlib import Path

BASE = Path('data/2024/s_class_f1_all_parts/2024_q1')
FILES = [
    'races.csv',
    'entries.csv',
    'results.csv',
    'payouts.csv',
    'trio_final_odds.csv',
    'trifecta_final_odds.csv',
]


def inspect_csv(path: Path) -> dict:
    out = {'path': str(path), 'exists': path.exists()}
    if not path.exists():
        return out
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            out.update({'header': [], 'rows': 0, 'sample': []})
            return out
        sample = []
        rows = 0
        for row in reader:
            rows += 1
            if len(sample) < 3:
                sample.append(row)
        out.update({'header': header, 'rows': rows, 'sample': sample})
    return out


def main() -> None:
    result = {name: inspect_csv(BASE / name) for name in FILES}
    Path('artifacts').mkdir(exist_ok=True)
    out_path = Path('artifacts/2024q1_market_input_probe.json')
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(out_path.read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
