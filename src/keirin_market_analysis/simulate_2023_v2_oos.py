from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from .simulate_mainline_v1 import choose_main_line, num, segment_for
from .simulate_early_candidate_v0 import strongest_rival as early_strongest_rival
from .simulate_middle_candidate_v0 import strongest_rival as middle_strongest_rival

EARLY_LEADER_GAP_MAX = -2.0
EARLY_PAIR_GAP_MAX = 6.0
MIDDLE_PAIR_TOP2_MIN = 70.0
LATE_PAIR_TOP3_MIN = 106.1
LATE_PAIR_WIN_MIN = 54.7


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def fnum(v: object) -> float:
    x = num(v)
    return 0.0 if x == float('-inf') else float(x)


def max_losing_streak(rows: list[dict[str, object]]) -> int:
    streak = best = 0
    for row in sorted(rows, key=lambda r: (str(r['race_date']), str(r['track']), int(r['race_no']))):
        if not int(row['purchased']):
            continue
        if int(row['hit']):
            streak = 0
        else:
            streak += 1
            best = max(best, streak)
    return best


def monthly(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for month in range(1, 13):
        key = f'2023-{month:02d}'
        rs = [r for r in rows if int(r['purchased']) and str(r['race_date']).startswith(key)]
        stake = sum(int(r['stake_yen']) for r in rs)
        payout = sum(int(r['payout_yen']) for r in rs)
        hits = sum(int(r['hit']) for r in rs)
        out[key] = {
            'purchased_races': len(rs),
            'hits': hits,
            'stake_yen': stake,
            'payout_yen': payout,
            'profit_yen': payout - stake,
            'roi': payout / stake if stake else 0.0,
        }
    return out


def summarize(rows: list[dict[str, object]], eligible_base: int) -> dict[str, object]:
    bought = [r for r in rows if int(r['purchased'])]
    hits = sum(int(r['hit']) for r in bought)
    stake = sum(int(r['stake_yen']) for r in bought)
    payout = sum(int(r['payout_yen']) for r in bought)
    return {
        'eligible_base': eligible_base,
        'purchased_races': len(bought),
        'purchase_rate': len(bought) / eligible_base if eligible_base else 0.0,
        'hits': hits,
        'hit_rate': hits / len(bought) if bought else 0.0,
        'stake_yen': stake,
        'payout_yen': payout,
        'profit_yen': payout - stake,
        'roi': payout / stake if stake else 0.0,
        'max_losing_streak': max_losing_streak(bought),
        'monthly': monthly(bought),
    }


def load_and_assert_specs(repo_root: Path) -> dict[str, object]:
    def load(name: str) -> dict[str, object]:
        p = repo_root / 'data' / 'strategy_specs' / name
        return json.loads(p.read_text(encoding='utf-8'))

    early = load('early_v2_pre2023.json')
    middle = load('middle_v2_pre2023.json')
    late = load('late_v2_pre2023.json')

    assert early['status'] == 'FROZEN_BEFORE_2023_OUTCOME_EVALUATION'
    assert early['entrance_rule']['leader_score_gap'] == 'A score - R1L score <= -2.0'
    assert early['entrance_rule']['pair_score_gap'] == '(A+B score) - (R1L+R1B score) <= 6.0'
    assert len(early['formation_100_yen_each']) == 4

    assert middle['status'] == 'FROZEN_BEFORE_2023_OUTCOME_EVALUATION'
    assert middle['entrance_rule'] == 'A+B top2-rate sum > 70.0'
    assert len(middle['formation_100_yen_each']) == 6

    assert late['status'] == 'FROZEN_BEFORE_2023_OUTCOME_EVALUATION'
    assert late['entrance_rule'] == 'A+B top3-rate sum > 106.1 AND A+B win-rate sum > 54.7'
    assert len(late['formation_100_yen_each']) == 4

    return {'early_v2': early, 'middle_v2': middle, 'late_v2': late}


def main() -> None:
    parser = argparse.ArgumentParser(description='Apply frozen V2 strategies to untouched 2023 exact-label S-class preliminary races')
    parser.add_argument('--data-dir', default='data/2023/s_class_yosen')
    parser.add_argument('--out-dir', default='data/2023/s_class_yosen/oos_v2')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    repo_root = Path('.')
    if '2023' not in str(data_dir).replace('\\', '/'):
        raise ValueError('This OOS simulator is intentionally restricted to the 2023 dataset')

    specs = load_and_assert_specs(repo_root)
    races = read_csv(data_dir / 'races.csv')
    entries = read_csv(data_dir / 'entries.csv')
    payouts = read_csv(data_dir / 'payouts.csv')

    entries_by: dict[str, list[dict[str, str]]] = defaultdict(list)
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    trifecta: dict[str, dict[str, int]] = defaultdict(dict)
    for e in entries:
        entries_by[e['race_id']].append(e)
    for r in races:
        groups[(r['race_date'], r['track'])].append(r)
    for p in payouts:
        if p.get('ticket_type') == '3連単' and p.get('status') == 'paid' and p.get('combination'):
            try:
                trifecta[p['race_id']][p['combination']] = int(p['payout_yen'])
            except (TypeError, ValueError):
                pass

    segment: dict[str, str] = {}
    for group in groups.values():
        group.sort(key=lambda r: int(r['race_no']))
        total = len(group)
        for pos, r in enumerate(group, 1):
            segment[r['race_id']] = segment_for(pos, total)

    decision_rows: list[dict[str, object]] = []
    bet_rows: list[dict[str, object]] = []
    eligible = {'early_v2': 0, 'middle_v2': 0, 'late_v2': 0}

    def record(race: dict[str, str], strategy: str, decision: str, purchased: bool, bets: list[str], extra: dict[str, object]) -> None:
        rid = race['race_id']
        stake = 100 * len(bets) if purchased else 0
        payout = sum(trifecta[rid].get(combo, 0) for combo in bets) if purchased else 0
        hit_combos = [combo for combo in bets if trifecta[rid].get(combo, 0)] if purchased else []
        row = {
            'race_id': rid,
            'race_date': race['race_date'],
            'track': race['track'],
            'race_no': int(race['race_no']),
            'segment': segment[rid],
            'strategy': strategy,
            'decision': decision,
            'purchased': int(purchased),
            'bet_count': len(bets) if purchased else 0,
            'stake_yen': stake,
            'payout_yen': payout,
            'profit_yen': payout - stake,
            'hit': int(payout > 0),
            'hit_combinations': '/'.join(hit_combos),
            **extra,
        }
        decision_rows.append(row)
        if purchased:
            for combo in bets:
                bp = trifecta[rid].get(combo, 0)
                bet_rows.append({
                    'race_id': rid,
                    'race_date': race['race_date'],
                    'track': race['track'],
                    'race_no': int(race['race_no']),
                    'segment': segment[rid],
                    'strategy': strategy,
                    'combination': combo,
                    'stake_yen': 100,
                    'payout_yen': bp,
                    'profit_yen': bp - 100 if bp else -100,
                    'hit': int(bp > 0),
                })

    for race in sorted(races, key=lambda r: (r['race_date'], r['track'], int(r['race_no']))):
        rid = race['race_id']
        es = entries_by[rid]
        chosen = choose_main_line(es)
        if not chosen:
            continue
        main_id, main = chosen
        seg = segment[rid]

        if seg == '前半':
            if len(main) < 3:
                continue
            rival = early_strongest_rival(es, main_id)
            if not rival or len(rival) < 2:
                continue
            eligible['early_v2'] += 1
            A, B = main[0], main[1]
            R1L, R1B = rival[0], rival[1]
            a, b = int(A['car_no']), int(B['car_no'])
            r1l, r1b = int(R1L['car_no']), int(R1B['car_no'])
            leader_gap = fnum(A.get('score')) - fnum(R1L.get('score'))
            pair_gap = (fnum(A.get('score')) + fnum(B.get('score'))) - (fnum(R1L.get('score')) + fnum(R1B.get('score')))
            buy = leader_gap <= EARLY_LEADER_GAP_MAX and pair_gap <= EARLY_PAIR_GAP_MAX
            bets = [f'{r1l}-{r1b}-{a}', f'{r1l}-{r1b}-{b}', f'{r1b}-{r1l}-{a}', f'{r1b}-{r1l}-{b}'] if buy else []
            assert not buy or len(bets) == 4
            record(race, 'early_v2', 'buy' if buy else 'skip_threshold', buy, bets, {
                'a_car': a, 'b_car': b, 'm3_car': int(main[2]['car_no']), 'r1l_car': r1l, 'r1b_car': r1b,
                'leader_score_gap': leader_gap, 'pair_score_gap': pair_gap,
                'pair_top2_sum': '', 'pair_top3_sum': '', 'pair_win_sum': '', 'x_car': '',
            })
            continue

        if seg == '中盤':
            if len(main) < 3:
                continue
            rival_info = middle_strongest_rival(es, main_id)
            if not rival_info:
                continue
            _, rival = rival_info
            if len(rival) < 2:
                continue
            eligible['middle_v2'] += 1
            A, B, M3 = main[0], main[1], main[2]
            R1L, R1B = rival[0], rival[1]
            a, b, m3 = int(A['car_no']), int(B['car_no']), int(M3['car_no'])
            r1l, r1b = int(R1L['car_no']), int(R1B['car_no'])
            pair_top2 = fnum(A.get('top2_rate')) + fnum(B.get('top2_rate'))
            buy = pair_top2 > MIDDLE_PAIR_TOP2_MIN
            bets = [
                f'{a}-{b}-{m3}', f'{b}-{a}-{m3}',
                f'{a}-{b}-{r1l}', f'{b}-{a}-{r1l}',
                f'{a}-{b}-{r1b}', f'{b}-{a}-{r1b}',
            ] if buy else []
            assert not buy or len(bets) == 6
            record(race, 'middle_v2', 'buy' if buy else 'skip_threshold', buy, bets, {
                'a_car': a, 'b_car': b, 'm3_car': m3, 'r1l_car': r1l, 'r1b_car': r1b,
                'leader_score_gap': '', 'pair_score_gap': '', 'pair_top2_sum': pair_top2,
                'pair_top3_sum': '', 'pair_win_sum': '', 'x_car': '',
            })
            continue

        if seg == '後半':
            if len(main) < 3:
                continue
            eligible['late_v2'] += 1
            A, B, M3 = main[0], main[1], main[2]
            a, b, m3 = int(A['car_no']), int(B['car_no']), int(M3['car_no'])
            pair_top3 = fnum(A.get('top3_rate')) + fnum(B.get('top3_rate'))
            pair_win = fnum(A.get('win_rate')) + fnum(B.get('win_rate'))
            buy = pair_top3 > LATE_PAIR_TOP3_MIN and pair_win > LATE_PAIR_WIN_MIN
            excluded = {a, b, m3}
            remain = [e for e in es if int(e['car_no']) not in excluded]
            X = max(remain, key=lambda e: (fnum(e.get('score')), -int(e['car_no']))) if remain else None
            x = int(X['car_no']) if X else None
            bets = [f'{a}-{b}-{m3}', f'{b}-{a}-{m3}', f'{a}-{b}-{x}', f'{b}-{a}-{x}'] if buy and x is not None else []
            purchased = buy and x is not None
            assert not purchased or len(bets) == 4
            record(race, 'late_v2', 'buy' if purchased else ('skip_no_x' if buy else 'skip_threshold'), purchased, bets, {
                'a_car': a, 'b_car': b, 'm3_car': m3, 'r1l_car': '', 'r1b_car': '',
                'leader_score_gap': '', 'pair_score_gap': '', 'pair_top2_sum': '',
                'pair_top3_sum': pair_top3, 'pair_win_sum': pair_win, 'x_car': x if x is not None else '',
            })

    by_strategy = {s: [r for r in decision_rows if r['strategy'] == s] for s in eligible}
    purchased_all = [r for r in decision_rows if int(r['purchased'])]

    summary = {
        'scope': '2023 exact-label Ｓ級予選 complete out-of-sample evaluation of frozen V2 strategies',
        'dataset_races': len(races),
        'oos_integrity': {
            'strategy_specs_asserted_before_evaluation': True,
            'development_years': [2024, 2025],
            'oos_year': 2023,
            'no_threshold_or_formation_changes_after_2023_outcomes': True,
        },
        'frozen_rule_manifest': {
            'segmentation': 'within each race_date x track, exact Ｓ級予選 sorted by race_no; floor((position-1)*3/total)',
            'main_line': 'existing choose_main_line implementation',
            'early_v2': specs['early_v2'],
            'middle_v2': specs['middle_v2'],
            'late_v2': specs['late_v2'],
            'stake': '100 yen per trifecta combination',
        },
        'early_v2': summarize(by_strategy['early_v2'], eligible['early_v2']),
        'middle_v2': summarize(by_strategy['middle_v2'], eligible['middle_v2']),
        'late_v2': summarize(by_strategy['late_v2'], eligible['late_v2']),
        'combined': summarize(purchased_all, sum(eligible.values())),
    }

    # Accounting invariants.
    assert summary['combined']['stake_yen'] == sum(int(r['stake_yen']) for r in purchased_all)
    assert summary['combined']['payout_yen'] == sum(int(r['payout_yen']) for r in purchased_all)
    assert len(bet_rows) * 100 == summary['combined']['stake_yen']

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    decision_fields = [
        'race_id','race_date','track','race_no','segment','strategy','decision','purchased','bet_count','stake_yen','payout_yen','profit_yen','hit','hit_combinations',
        'a_car','b_car','m3_car','r1l_car','r1b_car','leader_score_gap','pair_score_gap','pair_top2_sum','pair_top3_sum','pair_win_sum','x_car'
    ]
    bet_fields = ['race_id','race_date','track','race_no','segment','strategy','combination','stake_yen','payout_yen','profit_yen','hit']
    write_csv(out_dir / 'race_decisions.csv', decision_rows, decision_fields)
    write_csv(out_dir / 'bets.csv', bet_rows, bet_fields)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
