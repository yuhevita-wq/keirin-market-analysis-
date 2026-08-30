from __future__ import annotations

import csv
import itertools
import json
import math
import statistics
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data' / '2023' / 's_class_yosen'
OUT = ROOT / 'data' / 'audits' / 'support_plus_entry_scheme_2023.json'

NUMERIC_FEATURES = [
    'score', 'win_rate', 'top2_rate', 'top3_rate', 'b_count',
    'nige_count', 'makuri_count', 'sashi_count', 'mark_count',
    'age', 'line_size', 'line_position', 'car_no',
]
CATEGORICAL_FEATURES = ['class', 'style', 'line_role']
RIDGE = 0.02
EPOCHS = 350
BOOTSTRAP_MONTHS = [1, 2, 3, 4]
CALIBRATION_MONTHS = [5, 6]
EVALUATION_MONTHS = [7, 8, 9, 10, 11, 12]


def read_csv(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def parse_combo(s: str):
    s = str(s or '').strip().replace('=', '-').replace(',', '-')
    try:
        xs = tuple(sorted(int(x) for x in s.split('-') if x.strip()))
    except Exception:
        return ()
    return xs


def safe_float(v, default=0.0):
    try:
        x = float(str(v).replace(',', '').strip())
        return x if math.isfinite(x) else default
    except Exception:
        return default


def clip(p, lo=1e-9, hi=1 - 1e-9):
    return min(hi, max(lo, p))


def sigmoid(z):
    if z >= 0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


def logit(p):
    p = clip(p)
    return math.log(p / (1.0 - p))


def q_nearest(values, q):
    xs = sorted(values)
    if not xs:
        return None
    idx = max(0, math.ceil(q * len(xs)) - 1)
    return xs[idx]


def load_races():
    entries = defaultdict(dict)
    for row in read_csv(DATA / 'entries.csv'):
        rid = str(row.get('race_id', '')).strip()
        try:
            car = int(row.get('car_no', ''))
        except Exception:
            continue
        if rid and car:
            entries[rid][car] = row

    odds_rows = defaultdict(list)
    dates = {}
    for row in read_csv(DATA / 'trio_final_odds.csv'):
        if row.get('odds_status') != 'available':
            continue
        rid = str(row.get('race_id', '')).strip()
        c = parse_combo(row.get('combination', ''))
        o = safe_float(row.get('odds'), -1.0)
        if rid and len(c) == 3 and o > 0:
            odds_rows[rid].append((c, o))
            ds = str(row.get('race_date', '')).strip()
            try:
                dates[rid] = date.fromisoformat(ds)
            except Exception:
                pass

    winners = defaultdict(list)
    for row in read_csv(DATA / 'payouts.csv'):
        if row.get('ticket_type') != '3連複' or row.get('status') != 'paid':
            continue
        rid = str(row.get('race_id', '')).strip()
        c = parse_combo(row.get('combination', ''))
        if rid and len(c) == 3:
            winners[rid].append(c)

    races = []
    accounting = {
        'odds_market_races': len(odds_rows),
        'excluded_non7_or_incomplete_market': 0,
        'excluded_incomplete_entries': 0,
        'excluded_no_unique_winner': 0,
        'eligible_complete_races': 0,
    }

    for rid, rows in sorted(odds_rows.items()):
        riders = sorted({x for c, _ in rows for x in c})
        if len(riders) != 7:
            accounting['excluded_non7_or_incomplete_market'] += 1
            continue
        combos = list(itertools.combinations(riders, 3))
        odds = {c: o for c, o in rows}
        if set(odds) != set(combos):
            accounting['excluded_non7_or_incomplete_market'] += 1
            continue
        ent = entries.get(rid, {})
        if set(ent) != set(riders):
            accounting['excluded_incomplete_entries'] += 1
            continue
        ws = winners.get(rid, [])
        if len(ws) != 1:
            accounting['excluded_no_unique_winner'] += 1
            continue
        d = dates.get(rid)
        if d is None:
            continue

        inv = {c: 1.0 / odds[c] for c in combos}
        z = sum(inv.values())
        m = {c: inv[c] / z for c in combos}
        support = {r: sum(v for c, v in m.items() if r in c) for r in riders}
        win = set(ws[0])

        persons = []
        for car in riders:
            row = ent[car]
            persons.append({
                'car': car,
                'row': row,
                's': support[car],
                'y': 1 if car in win else 0,
            })
        races.append({'race_id': rid, 'date': d, 'persons': persons, 'win': win})

    accounting['eligible_complete_races'] = len(races)
    return races, accounting


def build_encoder(train_races):
    vals = {k: [] for k in NUMERIC_FEATURES}
    levels = {k: set() for k in CATEGORICAL_FEATURES}
    for race in train_races:
        for p in race['persons']:
            row = p['row']
            for k in NUMERIC_FEATURES:
                vals[k].append(safe_float(row.get(k), 0.0))
            for k in CATEGORICAL_FEATURES:
                levels[k].add(str(row.get(k, '') or '').strip())

    means = {k: statistics.mean(v) if v else 0.0 for k, v in vals.items()}
    stds = {}
    for k, v in vals.items():
        if len(v) > 1:
            s = statistics.pstdev(v)
            stds[k] = s if s > 1e-12 else 1.0
        else:
            stds[k] = 1.0
    cat_levels = {k: sorted(v) for k, v in levels.items()}
    names = ['intercept'] + list(NUMERIC_FEATURES)
    for k in CATEGORICAL_FEATURES:
        names.extend(f'{k}={level}' for level in cat_levels[k])
    return {'means': means, 'stds': stds, 'levels': cat_levels, 'names': names}


def vector(row, enc):
    x = [1.0]
    for k in NUMERIC_FEATURES:
        x.append((safe_float(row.get(k), 0.0) - enc['means'][k]) / enc['stds'][k])
    for k in CATEGORICAL_FEATURES:
        val = str(row.get(k, '') or '').strip()
        for level in enc['levels'][k]:
            x.append(1.0 if val == level else 0.0)
    return x


def fit_offset_logistic(train_races):
    enc = build_encoder(train_races)
    samples = []
    for race in train_races:
        for p in race['persons']:
            samples.append((vector(p['row'], enc), logit(p['s']), p['y']))
    w = [0.0] * len(enc['names'])
    n = max(1, len(samples))
    for epoch in range(EPOCHS):
        g = [0.0] * len(w)
        for x, off, y in samples:
            z = off + sum(a * b for a, b in zip(w, x))
            pr = sigmoid(z)
            e = pr - y
            for j, xj in enumerate(x):
                g[j] += e * xj
        for j in range(len(w)):
            reg = 0.0 if j == 0 else RIDGE * w[j]
            g[j] = g[j] / n + reg
        lr = 0.12 / (1.0 + epoch / 120.0)
        for j in range(len(w)):
            w[j] -= lr * g[j]
    return enc, w


def calibrated_race_probs(race, enc, w):
    logits = {}
    for p in race['persons']:
        x = vector(p['row'], enc)
        logits[p['car']] = logit(p['s']) + sum(a * b for a, b in zip(w, x))
    lo, hi = -12.0, 12.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        total = sum(sigmoid(z + mid) for z in logits.values())
        if total > 3.0:
            hi = mid
        else:
            lo = mid
    shift = (lo + hi) / 2.0
    return {car: sigmoid(z + shift) for car, z in logits.items()}


def walk_forward_predictions(races):
    predictions = []
    month_models = {}
    for month in range(5, 13):
        train = [r for r in races if r['date'].month < month]
        test = [r for r in races if r['date'].month == month]
        if not train or not test:
            continue
        enc, w = fit_offset_logistic(train)
        month_models[str(month)] = {
            'train_races': len(train),
            'test_races': len(test),
            'coefficient_count': len(w),
        }
        for race in test:
            f = calibrated_race_probs(race, enc, w)
            s = {p['car']: p['s'] for p in race['persons']}
            e = {car: f[car] - s[car] for car in f}
            predictions.append({
                'race': race,
                's': s,
                'f': f,
                'e': e,
                'dmax': max(abs(v) for v in e.values()),
                'dtotal': sum(abs(v) for v in e.values()),
            })
    return predictions, month_models


def metrics(preds):
    if not preds:
        return {}
    brier_s = brier_f = ll_s = ll_f = 0.0
    n = 0
    exact_s = exact_f = 0
    overlap_s = overlap_f = 0
    for pr in preds:
        race = pr['race']
        yset = race['win']
        for p in race['persons']:
            car = p['car']
            y = p['y']
            ps = clip(pr['s'][car])
            pf = clip(pr['f'][car])
            brier_s += (ps - y) ** 2
            brier_f += (pf - y) ** 2
            ll_s += -(y * math.log(ps) + (1 - y) * math.log(1 - ps))
            ll_f += -(y * math.log(pf) + (1 - y) * math.log(1 - pf))
            n += 1
        top_s = set(sorted(pr['s'], key=lambda c: (-pr['s'][c], c))[:3])
        top_f = set(sorted(pr['f'], key=lambda c: (-pr['f'][c], c))[:3])
        exact_s += int(top_s == yset)
        exact_f += int(top_f == yset)
        overlap_s += len(top_s & yset)
        overlap_f += len(top_f & yset)
    races_n = len(preds)
    return {
        'races': races_n,
        'rider_rows': n,
        'support_brier': brier_s / n,
        'adjusted_brier': brier_f / n,
        'brier_improvement_pct': 100.0 * (brier_s - brier_f) / brier_s if brier_s else None,
        'support_logloss': ll_s / n,
        'adjusted_logloss': ll_f / n,
        'logloss_improvement_pct': 100.0 * (ll_s - ll_f) / ll_s if ll_s else None,
        'support_top3_exact_rate_pct': 100.0 * exact_s / races_n,
        'adjusted_top3_exact_rate_pct': 100.0 * exact_f / races_n,
        'support_avg_top3_overlap': overlap_s / races_n,
        'adjusted_avg_top3_overlap': overlap_f / races_n,
    }


def main():
    races, accounting = load_races()
    preds, month_models = walk_forward_predictions(races)
    calibration = [p for p in preds if p['race']['date'].month in CALIBRATION_MONTHS]
    evaluation = [p for p in preds if p['race']['date'].month in EVALUATION_MONTHS]

    dmax_q75 = q_nearest([p['dmax'] for p in calibration], 0.75)
    dtotal_q75 = q_nearest([p['dtotal'] for p in calibration], 0.75)
    positive_e = [v for p in calibration for v in p['e'].values() if v > 0]
    positive_e_median = q_nearest(positive_e, 0.50)

    def nplus(p):
        return sum(v > positive_e_median for v in p['e'].values())

    must_enter = [
        p for p in evaluation
        if p['dmax'] >= dmax_q75
        and p['dtotal'] >= dtotal_q75
        and nplus(p) >= 2
    ]
    non_enter = [p for p in evaluation if p not in must_enter]

    monthly = {}
    for month in EVALUATION_MONTHS:
        ps = [p for p in evaluation if p['race']['date'].month == month]
        ms = [p for p in must_enter if p['race']['date'].month == month]
        monthly[str(month)] = {
            'eligible_races': len(ps),
            'must_enter_races': len(ms),
            'must_enter_rate_pct': 100.0 * len(ms) / len(ps) if ps else None,
            'must_enter_metrics': metrics(ms),
        }

    out = {
        'status': 'SUPPORT_PLUS_ENTRY_SCHEME_2023',
        'scope': '2023 race-entry simulation only; no betting-ticket rule and therefore no ROI calculation',
        'data_note': 'Support S uses archived KDreams final trio odds. This is a historical diagnostic, not a T-10 executable backtest.',
        'design': {
            'market_baseline': 'Rider marginal support S from normalized inverse 3連複 odds; sum S=3 per race.',
            'race_card_model': 'Offset logistic model: logit(P_top3)=logit(S)+linear correction from pre-race entries features; per-race probabilities shifted to sum to 3.',
            'numeric_features': NUMERIC_FEATURES,
            'categorical_features': CATEGORICAL_FEATURES,
            'excluded_prediction_columns': ['prediction_mark', 'evaluation'],
            'walk_forward': 'Jan-Apr bootstrap; monthly refit/predict May-Dec using only prior-month outcomes.',
            'threshold_calibration': 'May-Jun model/market outputs only; no May-Jun outcomes used in threshold calculation.',
            'evaluation_window': 'Jul-Dec.',
            'must_enter_rule': 'Dmax>=calibration Q75 AND Dtotal>=calibration Q75 AND at least 2 riders with E above calibration median positive E.',
            'Dmax': 'max_i abs(F_i-S_i)',
            'Dtotal': 'sum_i abs(F_i-S_i)',
            'E': 'F_i-S_i',
        },
        'accounting': accounting,
        'walk_forward_month_models': month_models,
        'calibration': {
            'months': CALIBRATION_MONTHS,
            'races': len(calibration),
            'dmax_q75': dmax_q75,
            'dtotal_q75': dtotal_q75,
            'positive_e_median_threshold': positive_e_median,
        },
        'evaluation': {
            'months': EVALUATION_MONTHS,
            'eligible_races': len(evaluation),
            'must_enter_races': len(must_enter),
            'must_enter_rate_pct': 100.0 * len(must_enter) / len(evaluation) if evaluation else None,
            'all_eligible_metrics': metrics(evaluation),
            'must_enter_metrics': metrics(must_enter),
            'non_enter_metrics': metrics(non_enter),
            'must_enter_avg_dmax': statistics.mean(p['dmax'] for p in must_enter) if must_enter else None,
            'must_enter_avg_dtotal': statistics.mean(p['dtotal'] for p in must_enter) if must_enter else None,
            'must_enter_avg_nplus': statistics.mean(nplus(p) for p in must_enter) if must_enter else None,
            'monthly': monthly,
        },
        'decision_rule': {
            'entry_model_pass_condition': 'On MUST_ENTER subset, adjusted F should improve both Brier and logloss versus support S before proceeding to 2-point formation design.',
            'roi_not_reported_reason': 'No 2-point ticket-generation law has been frozen for this model yet.',
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
