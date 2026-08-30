from __future__ import annotations

import csv, itertools, json, math, re, statistics
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data' / '2023' / 's_class_yosen'
OUT = ROOT / 'data' / 'audits' / 'temporally_robust_two_point_2023.json'

BETA = 0.02219612332210088
MUST_TV = 0.004982810992042711


def read_csv(p):
    with p.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def combo(s):
    s = str(s).strip().replace('=', '-').replace(',', '-')
    try:
        return tuple(sorted(int(x) for x in s.split('-') if x.strip()))
    except Exception:
        return ()


def parse_date(row):
    for k in ('race_date', 'date', 'held_date', 'event_date'):
        v = str(row.get(k, '') or '').strip()
        if v:
            m = re.search(r'(20\d{2})[-/]?(\d{2})[-/]?(\d{2})', v)
            if m:
                try:
                    return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                except ValueError:
                    pass
    rid = str(row.get('race_id', '') or '')
    m = re.search(r'(20\d{2})[-_/]?(\d{2})[-_/]?(\d{2})', rid)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    return None


def q_from_weights(combos, weights):
    raw = {c: math.prod(weights[x] for x in c) for c in combos}
    z = sum(raw.values())
    return {c: v / z for c, v in raw.items()}


def fit_maxent(riders, combos, target, tol=1e-12, max_iter=20000):
    w = {r: 1.0 for r in riders}
    for _ in range(max_iter):
        for r in riders:
            q = q_from_weights(combos, w)
            cur = sum(v for c, v in q.items() if r in c)
            t = target[r]
            f = (t * (1 - cur)) / (cur * (1 - t))
            w[r] *= f
        g = math.exp(sum(math.log(max(w[r], 1e-300)) for r in riders) / len(riders))
        for r in riders:
            w[r] /= g
        q = q_from_weights(combos, w)
        err = max(abs(sum(v for c, v in q.items() if r in c) - target[r]) for r in riders)
        if err < tol:
            return q, err
    return q, err


def p1_probs(m, R):
    vals = {c: m[c] * (R[c] ** BETA) for c in m}
    z = sum(vals.values())
    return {c: v / z for c, v in vals.items()}


def tv(a, b):
    return 0.5 * sum(abs(a[c] - b[c]) for c in a)


def payout_value(row):
    for key in ('payout_yen', 'payout', 'amount_yen', 'amount'):
        x = row.get(key)
        if x not in (None, ''):
            try:
                return int(float(str(x).replace(',', '')))
            except Exception:
                pass
    return None


def load_races():
    by = defaultdict(list)
    dates = {}
    for r in read_csv(DATA / 'trio_final_odds.csv'):
        if r.get('odds_status') != 'available':
            continue
        try:
            o = float(r['odds'])
        except Exception:
            continue
        c = combo(r.get('combination', ''))
        if len(c) == 3 and o > 0:
            rid = str(r['race_id'])
            by[rid].append((c, o))
            d = parse_date(r)
            if d:
                dates[rid] = d

    paid = defaultdict(list)
    for r in read_csv(DATA / 'payouts.csv'):
        if r.get('ticket_type') == '3連複' and r.get('status') == 'paid':
            c = combo(r.get('combination', ''))
            pv = payout_value(r)
            if len(c) == 3 and pv is not None:
                rid = str(r['race_id'])
                paid[rid].append((c, pv))
                if rid not in dates:
                    d = parse_date(r)
                    if d:
                        dates[rid] = d

    races = []
    accounting = {'market_races': len(by), 'excluded_non7_or_incomplete': 0, 'no_unique_paid_trio': 0, 'solver_failures': 0, 'non_must_enter': 0, 'date_parse_failures': 0}
    for rid, rows in sorted(by.items()):
        riders = sorted({x for c, _ in rows for x in c})
        if len(riders) != 7:
            accounting['excluded_non7_or_incomplete'] += 1
            continue
        combos = list(itertools.combinations(riders, 3))
        odds = {c: o for c, o in rows}
        if set(odds) != set(combos):
            accounting['excluded_non7_or_incomplete'] += 1
            continue
        ps = paid.get(rid, [])
        if len(ps) != 1:
            accounting['no_unique_paid_trio'] += 1
            continue
        d = dates.get(rid)
        if d is None:
            accounting['date_parse_failures'] += 1
            continue
        win, payout = ps[0]
        inv = {c: 1 / o for c, o in odds.items()}
        z = sum(inv.values())
        m = {c: v / z for c, v in inv.items()}
        support = {r: sum(v for c, v in m.items() if r in c) for r in riders}
        q, err = fit_maxent(riders, combos, support)
        if err >= 1e-9:
            accounting['solver_failures'] += 1
            continue
        R = {c: m[c] / q[c] for c in combos}
        p1 = p1_probs(m, R)
        if tv(p1, m) < MUST_TV:
            accounting['non_must_enter'] += 1
            continue
        rank_to_rider = {i + 1: r for i, r in enumerate(sorted(riders, key=lambda x: (-support[x], x)))}
        races.append({'race_id': rid, 'date': d, 'rank_to_rider': rank_to_rider, 'win': win, 'payout': payout})
    accounting['must_enter_evaluated'] = len(races)
    return races, accounting


def bucket(d):
    q = (d.month - 1) // 3 + 1
    return f'Q{q}'


def score_subset(rows, patterns):
    stake = 200 * len(rows)
    payout = hits = 0
    for r in rows:
        tickets = []
        for pat in patterns:
            riders = tuple(sorted(r['rank_to_rider'][int(ch)] for ch in pat))
            tickets.append(riders)
        if r['win'] in tickets:
            hits += 1
            payout += r['payout']
    return {
        'races': len(rows),
        'stake_yen': stake,
        'payout_yen': payout,
        'profit_yen': payout - stake,
        'roi_pct': (100 * payout / stake) if stake else None,
        'hit_races': hits,
        'hit_rate_pct': (100 * hits / len(rows)) if rows else None,
    }


def main():
    races, accounting = load_races()
    by_q = {f'Q{i}': [r for r in races if bucket(r['date']) == f'Q{i}'] for i in range(1,5)}
    halves = {
        'H1': [r for r in races if r['date'].month <= 6],
        'H2': [r for r in races if r['date'].month >= 7],
    }
    months = {f'{m:02d}': [r for r in races if r['date'].month == m] for m in range(1,13)}

    formations = []
    ranks = range(1, 8)
    for pair in itertools.combinations(ranks, 2):
        thirds = [x for x in ranks if x not in pair]
        for ts in itertools.combinations(thirds, 2):
            patterns = sorted(''.join(map(str, sorted((*pair, t)))) for t in ts)
            full = score_subset(races, patterns)
            quarters = {k: score_subset(v, patterns) for k, v in by_q.items()}
            half_scores = {k: score_subset(v, patterns) for k, v in halves.items()}
            month_scores = {k: score_subset(v, patterns) for k, v in months.items() if v}
            q_rois = [quarters[f'Q{i}']['roi_pct'] for i in range(1,5)]
            h_rois = [half_scores['H1']['roi_pct'], half_scores['H2']['roi_pct']]
            m_rois = [v['roi_pct'] for v in month_scores.values()]
            formations.append({
                'pair_support_ranks': ''.join(map(str,pair)),
                'third_support_ranks': ''.join(map(str,ts)),
                'formation': f"{pair[0]}-{pair[1]}-{''.join(map(str,ts))}",
                'tickets_as_rank_patterns': patterns,
                'full_year': full,
                'quarters': quarters,
                'halves': half_scores,
                'robustness': {
                    'worst_quarter_roi_pct': min(q_rois),
                    'median_quarter_roi_pct': statistics.median(q_rois),
                    'profitable_quarters': sum(x >= 100 for x in q_rois),
                    'quarters_ge_90': sum(x >= 90 for x in q_rois),
                    'worst_half_roi_pct': min(h_rois),
                    'profitable_halves': sum(x >= 100 for x in h_rois),
                    'median_month_roi_pct': statistics.median(m_rois),
                    'profitable_months': sum(x >= 100 for x in m_rois),
                }
            })

    by_worst_q = sorted(formations, key=lambda x: (-x['robustness']['worst_quarter_roi_pct'], -x['robustness']['median_quarter_roi_pct'], -x['full_year']['roi_pct']))
    by_consistency = sorted(formations, key=lambda x: (-x['robustness']['profitable_quarters'], -x['robustness']['quarters_ge_90'], -x['robustness']['worst_half_roi_pct'], -x['full_year']['roi_pct']))
    robust_positive = [x for x in formations if x['full_year']['roi_pct'] >= 100 and x['robustness']['profitable_halves'] == 2]
    robust_positive = sorted(robust_positive, key=lambda x: (-x['robustness']['profitable_quarters'], -x['robustness']['worst_quarter_roi_pct'], -x['full_year']['roi_pct']))

    out = {
        'status': 'TEMPORALLY_ROBUST_TWO_POINT_2023',
        'scope': '2023 only; no 2024 data used',
        'definition_of_universality_candidate': 'Fixed support-rank 2-point formation that does not rely only on full-year ROI; evaluate chronological stability across Q1-Q4 and H1/H2.',
        'accounting': accounting,
        'quarter_race_counts': {k: len(v) for k,v in by_q.items()},
        'half_race_counts': {k: len(v) for k,v in halves.items()},
        'candidate_count': len(formations),
        'ranked_by_worst_quarter_roi': by_worst_q,
        'ranked_by_consistency': by_consistency,
        'full_year_positive_and_both_halves_positive': robust_positive,
        'warning': 'This is still 2023 development. Temporal robustness reduces one-shot overfit risk but does not prove cross-year universality.'
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'accounting': accounting,
        'quarter_race_counts': out['quarter_race_counts'],
        'top_worst_quarter': by_worst_q[:10],
        'robust_positive_count': len(robust_positive),
        'top_robust_positive': robust_positive[:10],
    }, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
