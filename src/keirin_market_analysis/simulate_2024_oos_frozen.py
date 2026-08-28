from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from .simulate_mainline_v1 import choose_main_line, num, segment_for
from .simulate_early_candidate_v0 import strongest_rival as early_strongest_rival, formation as early_formation
from .simulate_middle_candidate_v0 import strongest_rival as middle_strongest_rival, fnum as middle_fnum
from .simulate_branching_v1 import classify, mainline_bets, strongest_rival as late_strongest_rival
from .simulate_branching_v3_confidence import rough_bets as late_rough_bets, f as late_f

EARLY_THRESHOLD = 22.35
MIDDLE_B_SCORE_THRESHOLD = 105.0
LATE_SCORE_GAP_SKIP = 10.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def pct(n: int, d: int) -> float:
    return n / d if d else 0.0


def max_losing_streak(rows: list[dict[str, object]]) -> int:
    streak = 0
    best = 0
    for row in sorted(rows, key=lambda r: (str(r['race_date']), str(r['track']), int(r['race_no']))):
        if not int(row['purchased']):
            continue
        if int(row['hit']):
            streak = 0
        else:
            streak += 1
            best = max(best, streak)
    return best


def monthly(rows: list[dict[str, object]], year: int) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for month in range(1, 13):
        key = f'{year:04d}-{month:02d}'
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


def summarize(rows: list[dict[str, object]], eligible_base: int, year: int) -> dict[str, object]:
    bought = [r for r in rows if int(r['purchased'])]
    hits = sum(int(r['hit']) for r in bought)
    stake = sum(int(r['stake_yen']) for r in bought)
    payout = sum(int(r['payout_yen']) for r in bought)
    return {
        'eligible_base': eligible_base,
        'purchased_races': len(bought),
        'purchase_rate': pct(len(bought), eligible_base),
        'hits': hits,
        'hit_rate': pct(hits, len(bought)),
        'stake_yen': stake,
        'payout_yen': payout,
        'profit_yen': payout - stake,
        'roi': payout / stake if stake else 0.0,
        'max_losing_streak': max_losing_streak(bought),
        'monthly': monthly(bought, year),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Apply frozen 2025 S級予選 strategies without modification')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--year', type=int, required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    year = args.year

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
    eligible = {'early_v0': 0, 'middle_v0': 0, 'late_v1': 0}

    def record(
        race: dict[str, str], strategy: str, decision: str, purchased: bool,
        bets: list[str], extra: dict[str, object],
    ) -> None:
        rid = race['race_id']
        returned = sum(trifecta[rid].get(combo, 0) for combo in bets) if purchased else 0
        stake = 100 * len(bets) if purchased else 0
        hits = [combo for combo in bets if trifecta[rid].get(combo, 0)] if purchased else []
        row = {
            'race_id': rid,
            'race_date': race['race_date'],
            'track': race['track'],
            'race_no': int(race['race_no']),
            'segment': segment[rid],
            'strategy': strategy,
            'decision': decision,
            'purchased': 1 if purchased else 0,
            'bet_count': len(bets) if purchased else 0,
            'stake_yen': stake,
            'payout_yen': returned,
            'profit_yen': returned - stake,
            'hit': 1 if returned else 0,
            'hit_combinations': '/'.join(hits),
            **extra,
        }
        decision_rows.append(row)
        if purchased:
            for combo in bets:
                payout = trifecta[rid].get(combo, 0)
                bet_rows.append({
                    'race_id': rid,
                    'race_date': race['race_date'],
                    'track': race['track'],
                    'race_no': int(race['race_no']),
                    'segment': segment[rid],
                    'strategy': strategy,
                    'decision': decision,
                    'combination': combo,
                    'stake_yen': 100,
                    'payout_yen': payout,
                    'profit_yen': payout - 100 if payout else -100,
                    'hit': 1 if payout else 0,
                })

    for race in sorted(races, key=lambda r: (r['race_date'], r['track'], int(r['race_no']))):
        rid = race['race_id']
        es = entries_by[rid]
        chosen = choose_main_line(es)
        if not chosen:
            continue
        main_id, main = chosen

        if segment[rid] == '前半':
            if len(main) < 3:
                continue
            rival = early_strongest_rival(es, main_id)
            if not rival or len(rival) < 2:
                continue
            eligible['early_v0'] += 1
            a = int(main[0]['car_no']); b = int(main[1]['car_no'])
            r1l = int(rival[0]['car_no']); r1b = int(rival[1]['car_no'])
            aw = num(main[0].get('win_rate', '')); bw = num(main[1].get('win_rate', ''))
            pair_win = (0.0 if aw == float('-inf') else aw) + (0.0 if bw == float('-inf') else bw)
            buy = pair_win <= EARLY_THRESHOLD
            bets = early_formation(a, b, r1l, r1b) if buy else []
            record(race, 'early_v0', 'buy' if buy else 'skip_threshold', buy, bets, {
                'main_line_id': main_id,
                'main_line_size': len(main),
                'a_car': a, 'b_car': b, 'm3_car': int(main[2]['car_no']),
                'r1l_car': r1l, 'r1b_car': r1b,
                'pair_win_sum': pair_win,
                'b_score': main[1].get('score', ''),
                'pair_top3_sum': '', 'b_top3': '', 'score_gap_A_vs_R1L': '',
            })
            continue

        if segment[rid] == '中盤':
            if len(main) < 3:
                continue
            rival_info = middle_strongest_rival(es, main_id)
            if not rival_info:
                continue
            _, rival = rival_info
            if len(rival) < 2:
                continue
            eligible['middle_v0'] += 1
            a = int(main[0]['car_no']); b = int(main[1]['car_no'])
            r1l = int(rival[0]['car_no']); r1b = int(rival[1]['car_no'])
            b_score = middle_fnum(main[1].get('score'))
            buy = b_score > MIDDLE_B_SCORE_THRESHOLD
            bets = [f'{a}-{b}-{r1b}', f'{r1l}-{r1b}-{b}'] if buy else []
            record(race, 'middle_v0', 'buy' if buy else 'skip_threshold', buy, bets, {
                'main_line_id': main_id,
                'main_line_size': len(main),
                'a_car': a, 'b_car': b, 'm3_car': int(main[2]['car_no']),
                'r1l_car': r1l, 'r1b_car': r1b,
                'pair_win_sum': '', 'b_score': b_score,
                'pair_top3_sum': '', 'b_top3': '', 'score_gap_A_vs_R1L': '',
            })
            continue

        if segment[rid] == '後半':
            if len(main) < 3:
                continue
            eligible['late_v1'] += 1
            a = int(main[0]['car_no']); b = int(main[1]['car_no']); m3 = int(main[2]['car_no'])
            branch, pair_top3, pair_win, b_top3 = classify(main)
            rival = late_strongest_rival(es, main_id)
            r1l = int(rival[0]['car_no']) if rival and len(rival) >= 2 else ''
            r1b = int(rival[1]['car_no']) if rival and len(rival) >= 2 else ''
            score_gap: float | str = ''

            if branch == 'A_mainline':
                bets = mainline_bets(es, main)
                decision = 'mainline_4pt'
                buy = True
            elif branch == 'B_rough':
                if not rival or len(rival) < 2:
                    bets = []
                    decision = 'skip_no_rival'
                    buy = False
                else:
                    score_gap = late_f(main[0], 'score') - late_f(rival[0], 'score')
                    if score_gap >= LATE_SCORE_GAP_SKIP:
                        bets = []
                        decision = 'skip_rough_score_gap_ge_10'
                        buy = False
                    else:
                        bets = late_rough_bets(es, main_id, main)
                        decision = 'rough_18pt'
                        buy = True
            else:
                bets = []
                decision = 'skip_middle'
                buy = False

            record(race, 'late_v1', decision, buy, bets, {
                'main_line_id': main_id,
                'main_line_size': len(main),
                'a_car': a, 'b_car': b, 'm3_car': m3,
                'r1l_car': r1l, 'r1b_car': r1b,
                'pair_win_sum': pair_win, 'b_score': main[1].get('score', ''),
                'pair_top3_sum': pair_top3, 'b_top3': b_top3,
                'score_gap_A_vs_R1L': score_gap,
            })

    early_rows = [r for r in decision_rows if r['strategy'] == 'early_v0']
    middle_rows = [r for r in decision_rows if r['strategy'] == 'middle_v0']
    late_rows = [r for r in decision_rows if r['strategy'] == 'late_v1']
    combined_rows = [r for r in decision_rows if int(r['purchased'])]

    summary = {
        'scope': f'{year} exact-label Ｓ級予選 out-of-sample application' if year != 2025 else '2025 frozen-rule reproduction check',
        'dataset_races': len(races),
        'frozen_rule_manifest': {
            'segmentation': 'within each race_date x track, exact Ｓ級予選 sorted by race_no; floor((position-1)*3/total)',
            'main_line': 'existing 2025 choose_main_line implementation unchanged',
            'early_v0': {
                'eligible': '前半, main line size >=3, strongest rival line 2+ exists',
                'condition': 'A+B win-rate sum <= 22.35',
                'bets': ['R1L-R1B-A','R1L-R1B-B','R1B-R1L-A','R1B-R1L-B'],
                'stake_per_race_yen': 400,
            },
            'middle_v0': {
                'eligible': '中盤, main line size >=3, strongest rival line 2+ exists',
                'condition': 'B score > 105.0',
                'bets': ['A-B-R1B','R1L-R1B-B'],
                'stake_per_race_yen': 200,
            },
            'late_v1': {
                'eligible': '後半, main line size >=3',
                'mainline_condition': 'pair_top3 > 106.1 AND pair_win > 54.7',
                'mainline_bets': ['A-B-M3','B-A-M3','A-B-X','B-A-X'],
                'rough_condition': 'pair_top3 <= 106.1 AND B top3 <= 35.7',
                'rough_skip': 'A score - R1L score >= 10.0 => no bet',
                'rough_bets': '18-point formation: 1st A/R1L; 2nd A/B/R1L/R1B; 3rd A/B/M3/R1L/R1B, duplicates removed',
                'middle_branch': 'skip',
            },
            'stake': '100 yen per trifecta combination',
            'no_post_result_filters': True,
        },
        'early_v0': summarize(early_rows, eligible['early_v0'], year),
        'middle_v0': summarize(middle_rows, eligible['middle_v0'], year),
        'late_v1': summarize(late_rows, eligible['late_v1'], year),
        'combined': summarize(combined_rows, eligible['early_v0'] + eligible['middle_v0'] + eligible['late_v1'], year),
        'late_decision_counts': {
            key: sum(1 for r in late_rows if r['decision'] == key)
            for key in ('mainline_4pt','rough_18pt','skip_rough_score_gap_ge_10','skip_middle','skip_no_rival')
        },
        'warning': 'For OOS validation, thresholds, branch rules, bet formations and skip rules are frozen from 2025. Results/payouts are evaluation labels only.',
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    decision_fields = [
        'race_id','race_date','track','race_no','segment','strategy','decision','purchased','bet_count','stake_yen','payout_yen','profit_yen','hit','hit_combinations',
        'main_line_id','main_line_size','a_car','b_car','m3_car','r1l_car','r1b_car','pair_win_sum','b_score','pair_top3_sum','b_top3','score_gap_A_vs_R1L'
    ]
    bet_fields = ['race_id','race_date','track','race_no','segment','strategy','decision','combination','stake_yen','payout_yen','profit_yen','hit']
    write_csv(out_dir / 'race_decisions.csv', decision_rows, decision_fields)
    write_csv(out_dir / 'bets.csv', bet_rows, bet_fields)
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    # Mechanical integrity checks only. These do not tune the strategy.
    assert all(int(r['bet_count']) == 4 for r in late_rows if r['decision'] == 'mainline_4pt')
    assert all(int(r['bet_count']) == 18 for r in late_rows if r['decision'] == 'rough_18pt')
    assert all(int(r['bet_count']) == 0 for r in late_rows if str(r['decision']).startswith('skip_'))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
