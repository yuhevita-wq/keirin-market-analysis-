from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from . import search_three_year_conditions_v1 as base

YEARS = (2023, 2024, 2025)
OUT = Path("data/audits/v3_clean_2023_2025_proposal.json")

FORMS = {
    "前半": ["MIX2", "RIVAL4", "MAIN4_RIVAL"],
    "中盤": ["MIX2", "MAIN4_RIVAL", "MAIN6", "RIVAL4"],
    "後半": ["MAIN4_X", "MAIN6", "MAIN4_RIVAL", "RIVAL4"],
}


def assert_clean_scope() -> None:
    assert YEARS == (2023, 2024, 2025)
    # Hard research boundary: this module must not read any 2026 path or audit.
    forbidden = ("2026", "2026_h1", "2026_H1")
    text = Path(__file__).read_text(encoding="utf-8")
    # Allow the guard strings above, but no data/2026 reference.
    assert "data/2026" not in text
    assert "2026_h1/s_class_yosen" not in text
    assert "2026_H1/s_class_yosen" not in text
    for year in YEARS:
        assert Path(f"data/{year}/s_class_yosen").exists()


def build_rows():
    datasets = {}
    segment_counts = {}
    for year in YEARS:
        races, eb, tri, seg = base.load(year)
        ys = defaultdict(list)
        for race in races:
            rid = race["race_id"]
            s = seg.get(rid)
            if s not in ("前半", "中盤", "後半"):
                continue
            chosen = base.choose_main_line(eb[rid])
            if not chosen:
                continue
            main_id, main = chosen
            if len(main) < 3:
                continue
            rival = base.strongest_rival(eb[rid], main_id)
            if not rival or len(rival) < 2:
                continue
            row = base.feature_row(race, eb[rid], main_id, main, rival)
            row["year"] = year
            row["tri"] = tri[rid]
            row["forms"] = base.formations(row, eb[rid])
            ys[s].append(row)
        datasets[year] = ys
        segment_counts[str(year)] = {s: len(ys[s]) for s in ("前半", "中盤", "後半")}
    return datasets, segment_counts


def rule_defs():
    rules = [(a[0], [a]) for a in base.ATOMS]
    rules += [
        (a[0] + "__AND__" + b[0], [a, b])
        for i, a in enumerate(base.ATOMS)
        for b in base.ATOMS[i + 1 :]
        if base.compatible(a, b)
    ]
    return rules


def family_key(form: str, atoms) -> str:
    parts = sorted(f"{a[1]}:{a[2]}" for a in atoms)
    return form + "|" + "&".join(parts)


def summarize_candidate(rule_name, atoms, form, stats):
    rois = sorted(stats[y]["roi"] for y in YEARS)
    stake = sum(stats[y]["stake_yen"] for y in YEARS)
    payout = sum(stats[y]["payout_yen"] for y in YEARS)
    profitable_halves = sum(
        1 for y in YEARS for h in stats[y]["halves"] if h["roi"] > 1.0
    )
    worst_half_roi = min(h["roi"] for y in YEARS for h in stats[y]["halves"])
    return {
        "rule": rule_name,
        "complexity": len(atoms),
        "family": family_key(form, atoms),
        "formation": form,
        "periods": {str(y): stats[y] for y in YEARS},
        "worst_year_roi": rois[0],
        "median_year_roi": rois[1],
        "combined_roi": payout / stake if stake else 0.0,
        "combined_profit_yen": payout - stake,
        "min_races_per_year": min(stats[y]["races"] for y in YEARS),
        "min_hits_per_year": min(stats[y]["hits"] for y in YEARS),
        "max_top1_share": max(stats[y]["top1_payout_share"] for y in YEARS),
        "profitable_halves": profitable_halves,
        "worst_half_roi": worst_half_roi,
    }


def main() -> int:
    assert_clean_scope()
    datasets, segment_counts = build_rows()
    rules = rule_defs()

    out = {
        "status": "CLEAN_2023_2025_DEVELOPMENT_ONLY_AWAITING_USER_GO_FOR_2026",
        "years_read": list(YEARS),
        "forbidden_evaluation_year": 2026,
        "research_boundary": "No 2026 dataset, result, payout, or prior 2026-informed V3 audit is read by this script.",
        "guardrails": {
            "conditions": "1 or 2 coarse predeclared atoms only",
            "min_races_each_year": 20,
            "min_hits_each_year": 3,
            "roi_each_year": ">1.0",
            "max_top1_payout_share_each_year": 0.60,
            "each_half_year": "at least 1 hit",
            "profitable_half_years": ">=4 of 6",
            "family_support": ">=2 qualifying threshold variants in same formation+feature/direction family",
            "selection": "highest worst-year ROI; ties favor larger minimum annual sample, lower complexity, higher combined ROI",
        },
        "segment_counts": segment_counts,
        "segments": {},
    }

    for segment in ("前半", "中盤", "後半"):
        prelim = []
        for rule_name, atoms in rules:
            selected = {
                y: [r for r in datasets[y][segment] if all(base.passes(r, a) for a in atoms)]
                for y in YEARS
            }
            if min(len(selected[y]) for y in YEARS) < 20:
                continue
            for form in FORMS[segment]:
                if any(form not in r["forms"] for y in YEARS for r in selected[y]):
                    continue
                stats = {y: base.fin(selected[y], form) for y in YEARS}
                if min(stats[y]["hits"] for y in YEARS) < 3:
                    continue
                if min(stats[y]["roi"] for y in YEARS) <= 1.0:
                    continue
                if max(stats[y]["top1_payout_share"] for y in YEARS) > 0.60:
                    continue
                if any(min(h["hits"] for h in stats[y]["halves"]) < 1 for y in YEARS):
                    continue
                cand = summarize_candidate(rule_name, atoms, form, stats)
                if cand["profitable_halves"] < 4:
                    continue
                prelim.append(cand)

        family_counts = defaultdict(int)
        for c in prelim:
            family_counts[c["family"]] += 1
        robust = [c for c in prelim if family_counts[c["family"]] >= 2]
        robust.sort(
            key=lambda c: (
                c["worst_year_roi"],
                c["min_races_per_year"],
                -c["complexity"],
                c["combined_roi"],
            ),
            reverse=True,
        )
        for c in robust:
            c["family_qualifying_variants"] = family_counts[c["family"]]

        out["segments"][segment] = {
            "formation_candidates": FORMS[segment],
            "pre_family_filter_count": len(prelim),
            "robust_family_supported_count": len(robust),
            "recommended": robust[0] if robust else None,
            "shortlist": robust[:10],
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        s: {
            "count": out["segments"][s]["robust_family_supported_count"],
            "recommended": out["segments"][s]["recommended"],
        }
        for s in out["segments"]
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
