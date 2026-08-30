from __future__ import annotations

import csv, itertools, json, math, statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data' / '2023' / 's_class_yosen'
OUT = ROOT / 'data' / 'audits' / 'support_pair_formation_simulation_2023.json'

BETA = 0.02219612332210088
MUST_TV = 0.004982810992042711
GROSS_THRESHOLDS = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)


def read_csv(p):
    with p.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def combo(s):
    s = str(s).strip().replace('=', '-').replace(',', '-')
    try:
        return tuple(sorted(int(x) for x in s.split('-') if x.strip()))
    except Exception:
        return ()


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
    for r in read_csv(DATA / 'trio_final_odds.csv'):
        if r.get('odds_status') != 'available':
            continue
        try:
            o = float(r['odds'])
        except Exception:
            continue
        c = combo(r.get('combination', ''))
        if len(c) == 3 and o > 0:
            by[str(r['race_id'])].append((c, o))

    paid = defaultdict(list)
    for r in read_csv(DATA / 'payouts.csv'):
        if r.get('ticket_type') == '3連複' and r.get('status') == 'paid':
            c = combo(r.get('combination', ''))
            pv = payout_value(r)
            if len(c) == 3 and pv is not None:
                paid[str(r['race_id'])].append((c, pv))

    races = []
    accounting = {'market_races': len(by), 'excluded_non7_or_incomplete': 0, 'no_unique_paid_trio': 0, 'solver_failures': 0, 'non_must_enter': 0}
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
        distance = tv(p1, m)
        if distance < MUST_TV:
            accounting['non_must_enter'] += 1
            continue
        races.append({'race_id': rid, 'riders': riders, 'combos': combos, 'odds': odds, 'p1': p1, 'win': win, 'payout': payout, 'tv': distance})
    accounting['must_enter_evaluated'] = len(races)
    return races, accounting


def enumerate_pair_formations(r):
    out = []
    for pair in itertools.combinations(r['riders'], 2):
        thirds = [x for x in r['riders'] if x not in pair]
        for k in range(2, 6):
            for subset in itertools.combinations(thirds, k):
                tickets = tuple(sorted(tuple(sorted((*pair, x))) for x in subset))
                coverage = sum(r['p1'][c] for c in tickets)
                invsum = sum(1.0 / r['odds'][c] for c in tickets)
                synthetic_odds = 1.0 / invsum
                model_ev_dutch = coverage * synthetic_odds
                out.append({
                    'pair': pair,
                    'tickets': tickets,
                    'k': len(tickets),
                    'coverage': coverage,
                    'synthetic_odds': synthetic_odds,
                    'model_ev_dutch': model_ev_dutch,
                })
    return out


def choose_formation(forms, objective, gross_min):
    cand = [f for f in forms if f['synthetic_odds'] >= gross_min]
    if not cand:
        return None
    if objective == 'coverage':
        cand.sort(key=lambda f: (-f['coverage'], -f['model_ev_dutch'], f['k'], f['pair'], f['tickets']))
    elif objective == 'model_ev':
        cand.sort(key=lambda f: (-f['model_ev_dutch'], -f['coverage'], f['k'], f['pair'], f['tickets']))
    else:
        raise ValueError(objective)
    return cand[0]


def score_rule(races, objective, gross_min):
    stake = payout = hit_races = bet_races = tickets = 0
    ks = []
    syns = []
    covs = []
    model_evs = []
    cur_loss = max_loss = 0
    pair_counts = defaultdict(int)
    for r in races:
        f = choose_formation(enumerate_pair_formations(r), objective, gross_min)
        if f is None:
            continue
        bet_races += 1
        n = f['k']
        tickets += n
        ks.append(n)
        syns.append(f['synthetic_odds'])
        covs.append(f['coverage'])
        model_evs.append(f['model_ev_dutch'])
        pair_counts['-'.join(map(str, f['pair']))] += 1
        stake += 100 * n
        if r['win'] in f['tickets']:
            hit_races += 1
            payout += r['payout']
            cur_loss = 0
        else:
            cur_loss += 1
            max_loss = max(max_loss, cur_loss)
    if cur_loss > max_loss:
        max_loss = cur_loss
    return {
        'objective': objective,
        'synthetic_odds_min': gross_min,
        'must_enter_races': len(races),
        'bet_races': bet_races,
        'purchase_rate_pct': 100 * bet_races / len(races) if races else None,
        'tickets': tickets,
        'avg_tickets_per_bet_race': statistics.mean(ks) if ks else None,
        'median_tickets_per_bet_race': statistics.median(ks) if ks else None,
        'hit_races': hit_races,
        'hit_rate_pct': 100 * hit_races / bet_races if bet_races else None,
        'stake_yen': stake,
        'payout_yen': payout,
        'profit_yen': payout - stake,
        'roi_pct': 100 * payout / stake if stake else None,
        'avg_selected_synthetic_odds': statistics.mean(syns) if syns else None,
        'avg_selected_model_coverage': statistics.mean(covs) if covs else None,
        'avg_selected_model_ev_dutch': statistics.mean(model_evs) if model_evs else None,
        'max_losing_streak_bet_races': max_loss,
        'top_pair_cores': sorted(pair_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10],
    }


def main():
    races, accounting = load_races()
    rules = []
    for objective in ('coverage', 'model_ev'):
        for g in GROSS_THRESHOLDS:
            rules.append(score_rule(races, objective, g))
    ranked = sorted(rules, key=lambda x: (-(x['roi_pct'] if x['roi_pct'] is not None else -1), -(x['hit_rate_pct'] or 0), x['synthetic_odds_min'], x['objective']))
    out = {
        'status': 'SUPPORT_PAIR_FORMATION_SIMULATION_2023',
        'scope': '2023 only, in-sample exploratory simulation',
        'frozen_inputs': {
            'beta': BETA,
            'must_enter_tv_threshold': MUST_TV,
            'market_only_ticket_generation': True,
            'outcome_used_only_for_scoring': True,
            'odds_phase': 'final',
            'stake_rule': '100 yen equal stake per selected trio',
        },
        'formation_universe': {
            'description': 'For every 2-rider core, enumerate every 2-5 third-rider subset. 21 pair cores * 26 subsets = 546 formations per race.',
            'synthetic_odds': '1 / sum(1 / trio_odds) over selected tickets',
            'coverage': 'sum of support-augmented probabilities P1 over selected tickets',
            'model_ev_dutch': 'coverage * synthetic_odds; descriptive only because all probabilities are market-derived and final odds are not executable snapshots',
            'objectives': ['coverage', 'model_ev'],
            'synthetic_odds_min_grid': list(GROSS_THRESHOLDS),
        },
        'accounting': accounting,
        'ranked_rules_by_realized_2023_roi': ranked,
        'warning': 'The best 2023 rule is selected in-sample from a predefined 16-rule grid. It is not validated edge. Final odds are historical final prices, not T-10 executable odds.',
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'accounting': accounting, 'top5': ranked[:5], 'bottom3': ranked[-3:]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
