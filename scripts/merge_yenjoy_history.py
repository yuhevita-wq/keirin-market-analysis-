from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path


def iter_joined_files(root: Path):
    for path in sorted(root.rglob("joined_*.jsonl.gz")):
        yield path


def main() -> int:
    p = argparse.ArgumentParser(description="Merge monthly YenJoy normalized artifacts into yearly/history files")
    p.add_argument("--input-root", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    source_root = Path(args.input_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    sources = list(iter_joined_files(source_root))
    if not sources:
        raise SystemExit(f"no joined_*.jsonl.gz files under {source_root}")

    year_paths: dict[str, Path] = {}
    year_handles = {}
    year_counts: Counter[str] = Counter()
    quality_counts: dict[str, Counter[str]] = defaultdict(Counter)
    seen_race_ids: set[str] = set()
    duplicate_count = 0

    combined_path = output / "yenjoy_joined_2024_2026.jsonl.gz"
    try:
        with gzip.open(combined_path, "wt", encoding="utf-8") as combined:
            for src in sources:
                with gzip.open(src, "rt", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        race_id = rec.get("race_id")
                        if race_id in seen_race_ids:
                            duplicate_count += 1
                            continue
                        if race_id:
                            seen_race_ids.add(race_id)

                        year = str(rec.get("race_date", ""))[:4] or "unknown"
                        if year not in year_handles:
                            path = output / f"yenjoy_joined_{year}.jsonl.gz"
                            year_paths[year] = path
                            year_handles[year] = gzip.open(path, "wt", encoding="utf-8")

                        encoded = json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n"
                        year_handles[year].write(encoded)
                        combined.write(encoded)
                        year_counts[year] += 1

                        q = rec.get("quality") or {}
                        for key in ("has_entries", "has_result", "has_top3"):
                            if q.get(key):
                                quality_counts[year][key] += 1
    finally:
        for handle in year_handles.values():
            handle.close()

    manifest = {
        "schema_version": 1,
        "source": "yenjoy",
        "source_shards": [str(p.relative_to(source_root)) for p in sources],
        "source_shard_count": len(sources),
        "duplicate_race_ids_skipped": duplicate_count,
        "unique_race_count": sum(year_counts.values()),
        "years": {
            year: {
                "race_count": year_counts[year],
                "races_with_entries": quality_counts[year]["has_entries"],
                "races_with_results": quality_counts[year]["has_result"],
                "races_with_top3": quality_counts[year]["has_top3"],
                "file": year_paths[year].name,
            }
            for year in sorted(year_counts)
        },
        "combined_file": combined_path.name,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
