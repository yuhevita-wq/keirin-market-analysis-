from __future__ import annotations

import json
from pathlib import Path

from develop_v6_4_d02_q2q3 import collect, summarize

SCHEME_VERSION = "v6.4-D02-Q1-RETRO"
BASE_SCHEME_VERSION = "v6.1"
PARENT_SCHEME_VERSION = "v6.3-D01"
DATASET = "2024Q1"
STATUS = "RETROSPECTIVE_PARTLY_IN_SAMPLE"
D01_M_PRE_THRESHOLD = 0.35640013538348414
D02_PRUNE_DAMAGE_THRESHOLD = 0.3317657935777282

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "v6_4_d02_2024q1_retro"


def main():
    # collect() already applies the fixed D01 M_pre threshold.
    d01_rows = collect("2024_q1")
    d02_rows = [
        r for r in d01_rows
        if r.get("PruneDamage") is not None
        and float(r["PruneDamage"]) <= D02_PRUNE_DAMAGE_THRESHOLD
    ]
    removed = [r for r in d01_rows if r not in d02_rows]

    result = {
        "scheme_version": SCHEME_VERSION,
        "base_scheme_version": BASE_SCHEME_VERSION,
        "parent_scheme_version": PARENT_SCHEME_VERSION,
        "dataset": DATASET,
        "status": STATUS,
        "fixed_entry_filters": {
            "M_pre_min": D01_M_PRE_THRESHOLD,
            "PruneDamage_max": D02_PRUNE_DAMAGE_THRESHOLD,
        },
        "d01_before": summarize(d01_rows),
        "v6_4_d02": summarize(d02_rows),
        "removed_by_d02": summarize(removed),
        "warning": "2024Q1 was used to derive the D01 M_pre threshold, so this is not out-of-sample validation. The D02 PruneDamage threshold was fixed from 2024Q2+Q3 and is not re-estimated here.",
        "q1_tuning": False,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "v6_4_d02_2024q1_retro_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("V6_4_D02_Q1_RETRO_BEGIN")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("V6_4_D02_Q1_RETRO_END")


if __name__ == "__main__":
    main()
