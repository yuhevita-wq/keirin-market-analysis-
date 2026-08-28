from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from .analyze_middle_structure_v1 import FEATURES as MIDDLE_FEATURES, thresholds as middle_thresholds
from .analyze_mainline_branch_v1 import FEATURES as LATE_FEATURES, candidate_thresholds as late_thresholds, make_feature_row
from .analyze_rough_pruning_v1 import CANDIDATES, combos as pruning_combos, role_map as pruning_role_map
from .analyze_rough_confidence_narrowing_v1 import ZONES
from .analyze_rough_winner_split_v1 import actual_winner, metrics as leader_metrics
from .simulate_branching_v1 import classify, mainline_bets, strongest_rival as late_strongest_rival
from .simulate_early_candidate_v0 import formation as early_formation, strongest_rival as early_strongest_rival
from .simulate_mainline_v1 import choose_main_line, num, segment_for
from .simulate_middle_candidate_v0 import strongest_rival as middle_strongest_rival

DATA_DIR = Path('data/2025/s_class_yosen')
OUT = Path('data/audits/threshold_overfit_2025.json')


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def f(v: object) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def metric(e: dict[str, str], name: str) -> float:
    v = num(e.get(name, ''))
    return 0.0 if v == float('-inf') else v


def line_map(entries: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    by: dict[int, list[dict[str, str]]] = defaultdict(list)
    for e in entries:
        if e.get('line_id', '').isdigit() and e.get('line_position', '').isdigit():
            by[int(e['line_id'])].append(e)
    for lid in by:
        by[lid].sort(key=lambda e: int(e['line_position']))
    return by


def line_pair_strength(members: list[dict[str, str]]) -> float:
    if len(members) < 2:
        return float('-inf')
    return metric(members[0], 'score') + metric(members[1], 'score')


def gini(rows: list[dict[str, object]], label: str) -> float:
    if not rows:
        return 0.0
    p = sum(int(r[label]) for r in rows) / len(rows)
    return 2 * p * (1 - p)


def split_gain(rows: list[dict[str, object]], feature: str, threshold: float, label: str, min_leaf: int = 20) -> dict[str, object] | None:
    left = [r for r in rows if float(r[feature]) <= threshold]
    right = [r for r in rows if float(r[feature]) > threshold]
    if len(left) < min_leaf or len(right) < min_leaf:
        return None
    base = gini(rows, label)
    weighted = (len(left) * gini(left, label) + len(right) * gini(right, label)) / len(rows)
    return {
        'feature': feature,
        'threshold': threshold,
        'gain': base - weighted,
        'left_n': len(left),
        'left_rate': sum(int(r[label]) for r in left) / len(left),
        'right_n': len(right),
        'right_rate': sum(int(r[label]) for r in right) / len(right),
    }


def rank_of(item: dict[str, object] | None, candidates: list[dict[str, object]]) -> int | None:
    if item is None:
        return None
    gain = float(item['gain'])
    return 1 + sum(float(c['gain']) > gain + 1e-15 for c in candidates)


def financial(rows: list[dict[str, object]]) -> dict[str, object]:
    n = len(rows)
    stake = sum(int(r['stake_yen']) for r in rows)
    payout = sum(int(r['payout_yen']) for r in rows)
    hits = sum(int(r['payout_yen']) > 0 for r in rows)
    return {
        'races': n,
        'hits': hits,
        'hit_rate': hits / n if n else 0.0,
        'stake_yen': stake,
        'payout_yen': payout,
        'profit_yen': payout - stake,
        'roi': payout / stake if stake else 0.0,
    }


def subset_stats(rows: list[dict[str, object]], label: str) -> dict[str, object]:
    return {
        'races': len(rows),
        'positives': sum(int(r[label]) for r in rows),
        'rate': sum(int(r[label]) for r in rows) / len(rows) if rows else 0.0,
    }


def main() -> None:
    # Hard guard: this audit intentionally never loads 2024 data.
    assert '2025' in str(DATA_DIR) and '2024' not in str(DATA_DIR)

    races = read_csv(DATA_DIR / 'races.csv')
    entries = read_csv(DATA_DIR / 'entries.csv')
    results = read_csv(DATA_DIR / 'results.csv')
    payouts = read_csv(DATA_DIR / 'payouts.csv')

    entries_by: dict[str, list[dict[str, str]]] = defaultdict(list)
    results_by: dict[str, list[dict[str, str]]] = defaultdict(list)
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    trifecta: dict[str, dict[str, int]] = defaultdict(dict)
    for e in entries:
        entries_by[e['race_id']].append(e)
    for r in results:
        results_by[r['race_id']].append(r)
    for r in races:
        groups[(r['race_date'], r['track'])].append(r)
    for p in payouts:
        if p.get('ticket_type') == '3連単' and p.get('status') == 'paid' and p.get('combination'):
            try:
                trifecta[p['race_id']][p['combination']] = int(p['payout_yen'])
            except (TypeError, ValueError):
                pass

    segment: dict[str, str] = {}
    ordinal: dict[str, int] = {}
    cardn: dict[str, int] = {}
    for group in groups.values():
        group.sort(key=lambda r: int(r['race_no']))
        n = len(group)
        for pos, r in enumerate(group, 1):
            segment[r['race_id']] = segment_for(pos, n)
            ordinal[r['race_id']] = pos
            cardn[r['race_id']] = n

    # ---------- Early: reconstruct the documented H1 structural threshold search ----------
    early_rows: list[dict[str, object]] = []
    for race in sorted(races, key=lambda r: (r['race_date'], r['track'], int(r['race_no']))):
        rid = race['race_id']
        if segment[rid] != '前半':
            continue
        chosen = choose_main_line(entries_by[rid])
        if not chosen:
            continue
        main_id, main_members = chosen
        if len(main_members) < 3:
            continue
        rival = early_strongest_rival(entries_by[rid], main_id)
        if not rival or len(rival) < 2:
            continue
        a, b = main_members[0], main_members[1]
        r1l, r1b = rival[0], rival[1]
        pair_win = metric(a, 'win_rate') + metric(b, 'win_rate')
        result_map = {int(x['car_no']): str(x.get('finish_position', '')) for x in results_by[rid]}
        rival_pair_top2 = int({result_map.get(int(r1l['car_no']), ''), result_map.get(int(r1b['car_no']), '')} == {'1', '2'})
        bets = early_formation(int(a['car_no']), int(b['car_no']), int(r1l['car_no']), int(r1b['car_no']))
        ret = sum(trifecta[rid].get(c, 0) for c in bets)
        early_rows.append({
            'race_id': rid,
            'half': 'H1' if race['race_date'] <= '2025-06-30' else 'H2',
            'pair_win': pair_win,
            'rival_pair_top2': rival_pair_top2,
            'stake_yen': 400,
            'payout_yen': ret,
        })

    early_h1 = [r for r in early_rows if r['half'] == 'H1']
    early_h2 = [r for r in early_rows if r['half'] == 'H2']
    early_vals = sorted({float(r['pair_win']) for r in early_h1})
    early_thresholds = [(a + b) / 2 for a, b in zip(early_vals, early_vals[1:])]
    early_candidates = [x for t in early_thresholds if (x := split_gain(early_h1, 'pair_win', t, 'rival_pair_top2')) is not None]
    early_candidates.sort(key=lambda x: float(x['gain']), reverse=True)
    early_frozen = split_gain(early_h1, 'pair_win', 22.35, 'rival_pair_top2')

    early_sensitivity = {}
    for t in (20.0, 21.0, 22.0, 22.35, 23.0, 24.0):
        h1 = [r for r in early_h1 if float(r['pair_win']) <= t]
        h2 = [r for r in early_h2 if float(r['pair_win']) <= t]
        allr = h1 + h2
        early_sensitivity[str(t)] = {
            'H1_structure': subset_stats(h1, 'rival_pair_top2'),
            'H2_structure': subset_stats(h2, 'rival_pair_top2'),
            'H1_betting': financial(h1),
            'H2_betting': financial(h2),
            'ALL_betting': financial(allr),
        }

    early_audit = {
        'documented_origin': 'H1 structural classification of R1L+R1B occupying 1st/2nd; payout not used for threshold selection.',
        'selection_code_preserved': False,
        'selection_provenance_gap': 'The final report documents the origin and sensitivity, but the exact standalone H1 threshold-selector that produced 22.35 is not preserved in the current tree.',
        'H1_rows': len(early_h1),
        'H2_rows': len(early_h2),
        'one_feature_midpoint_candidates': len(early_thresholds),
        'valid_candidates_min_leaf20': len(early_candidates),
        'frozen_22_35_split': early_frozen,
        'frozen_rank_by_H1_gini_gain': rank_of(early_frozen, early_candidates),
        'best_H1_split': early_candidates[0] if early_candidates else None,
        'nearby_threshold_sensitivity': early_sensitivity,
    }

    # ---------- Middle: reproduce the broad H1 feature/threshold search ----------
    middle_structure_rows: list[dict[str, object]] = []
    middle_action_rows: list[dict[str, object]] = []
    for race in sorted(races, key=lambda r: (r['race_date'], r['track'], int(r['race_no']))):
        rid = race['race_id']
        if segment[rid] != '中盤':
            continue
        es = entries_by[rid]
        chosen = choose_main_line(es)
        if not chosen:
            continue
        main_id, main = chosen
        if len(main) < 3:
            continue
        by = line_map(es)
        rivals = [(line_pair_strength(m), lid, m) for lid, m in by.items() if lid != main_id and len(m) >= 2]
        rivals.sort(key=lambda x: (-x[0], x[1]))
        rival = rivals[0][2] if rivals else []
        a, b, c = main[0], main[1], main[2]
        r1 = rival[0] if len(rival) >= 1 else {}
        r2 = rival[1] if len(rival) >= 2 else {}
        result_map = {int(x['car_no']): x for x in results_by[rid]}
        fa = result_map.get(int(a['car_no']), {}).get('finish_position', '')
        fb = result_map.get(int(b['car_no']), {}).get('finish_position', '')
        y = int({str(fa), str(fb)} == {'1', '2'})
        row: dict[str, object] = {
            'race_id': rid,
            'half': 'H1' if race['race_date'] <= '2025-06-30' else 'H2',
            'mainline_top2': y,
            'main_pair_score_sum': metric(a, 'score') + metric(b, 'score'),
            'pair_score_gap': metric(a, 'score') + metric(b, 'score') - (line_pair_strength(rival) if rival else 0.0),
            'main_leader_score': metric(a, 'score'),
            'main_second_score': metric(b, 'score'),
            'main_third_score': metric(c, 'score'),
            'leader_score_gap': metric(a, 'score') - (metric(r1, 'score') if r1 else 0.0),
            'second_score_gap': metric(b, 'score') - (metric(r2, 'score') if r2 else 0.0),
            'main_pair_top2_sum': metric(a, 'top2_rate') + metric(b, 'top2_rate'),
            'main_pair_top3_sum': metric(a, 'top3_rate') + metric(b, 'top3_rate'),
            'main_pair_win_sum': metric(a, 'win_rate') + metric(b, 'win_rate'),
            'main_leader_top2': metric(a, 'top2_rate'),
            'main_second_top2': metric(b, 'top2_rate'),
            'main_leader_top3': metric(a, 'top3_rate'),
            'main_second_top3': metric(b, 'top3_rate'),
            'main_third_top3': metric(c, 'top3_rate'),
            'main_leader_b': metric(a, 'b_count'),
            'main_second_mark': metric(b, 'mark_count'),
            'line_count': len(by),
            'multi_line_count': sum(1 for m in by.values() if len(m) >= 2),
            'main_line_size': len(main),
            'race_no': int(race['race_no']),
            'segment_ordinal': ordinal[rid],
            's_yosen_count': cardn[rid],
        }
        middle_structure_rows.append(row)
        if rival:
            bets = [f"{int(a['car_no'])}-{int(b['car_no'])}-{int(rival[1]['car_no'])}", f"{int(rival[0]['car_no'])}-{int(rival[1]['car_no'])}-{int(b['car_no'])}"]
            ret = sum(trifecta[rid].get(cmb, 0) for cmb in bets)
            middle_action_rows.append({
                'race_id': rid,
                'half': row['half'],
                'b_score': metric(b, 'score'),
                'stake_yen': 200,
                'payout_yen': ret,
            })

    middle_h1 = [r for r in middle_structure_rows if r['half'] == 'H1']
    middle_feature_candidates: list[dict[str, object]] = []
    feature_best: list[dict[str, object]] = []
    for feature in MIDDLE_FEATURES:
        fcands = []
        for t in middle_thresholds(middle_h1, feature):
            cand = split_gain(middle_h1, feature, t, 'mainline_top2')
            if cand:
                fcands.append(cand)
                middle_feature_candidates.append(cand)
        if fcands:
            feature_best.append(max(fcands, key=lambda x: float(x['gain'])))
    middle_feature_candidates.sort(key=lambda x: float(x['gain']), reverse=True)
    feature_best.sort(key=lambda x: float(x['gain']), reverse=True)
    middle_frozen_split = split_gain(middle_h1, 'main_second_score', 105.0, 'mainline_top2')

    middle_action_h1 = [r for r in middle_action_rows if r['half'] == 'H1']
    bvals = sorted({float(r['b_score']) for r in middle_action_h1})
    bthresholds = [(a + b) / 2 for a, b in zip(bvals, bvals[1:])]
    roi_scan = []
    for t in bthresholds:
        h1 = [r for r in middle_action_rows if r['half'] == 'H1' and float(r['b_score']) > t]
        h2 = [r for r in middle_action_rows if r['half'] == 'H2' and float(r['b_score']) > t]
        if len(h1) < 20:
            continue
        allr = h1 + h2
        fh1, fh2, fall = financial(h1), financial(h2), financial(allr)
        roi_scan.append({'threshold': t, 'H1': fh1, 'H2': fh2, 'ALL': fall})

    frozen_h1 = [r for r in middle_action_rows if r['half'] == 'H1' and float(r['b_score']) > 105.0]
    frozen_h2 = [r for r in middle_action_rows if r['half'] == 'H2' and float(r['b_score']) > 105.0]
    frozen_middle_fin = {'threshold': 105.0, 'H1': financial(frozen_h1), 'H2': financial(frozen_h2), 'ALL': financial(frozen_h1 + frozen_h2)}
    h1_roi_rank = 1 + sum(float(x['H1']['roi']) > float(frozen_middle_fin['H1']['roi']) + 1e-15 for x in roi_scan)
    all_roi_rank = 1 + sum(float(x['ALL']['roi']) > float(frozen_middle_fin['ALL']['roi']) + 1e-15 for x in roi_scan)

    middle_audit = {
        'structure_audit_to_candidate_commit_gap_minutes': 11.18,
        'structure_rows_all_2025': len(middle_structure_rows),
        'actionable_rows_with_rival': len(middle_action_rows),
        'H1_structure_rows': len(middle_h1),
        'features_scanned': len(MIDDLE_FEATURES),
        'valid_H1_feature_threshold_candidates_min_leaf20': len(middle_feature_candidates),
        'frozen_105_structural_split': middle_frozen_split,
        'frozen_105_global_gini_rank': rank_of(middle_frozen_split, middle_feature_candidates),
        'frozen_105_rank_among_feature_best': rank_of(middle_frozen_split, feature_best),
        'best_H1_structural_split': middle_feature_candidates[0] if middle_feature_candidates else None,
        'final_rule_provenance_gap': 'No preserved deterministic selection rule explains why exactly B score >105.0 and exactly the two final trifecta orders were chosen after the broad 2025 structure audit.',
        'counterfactual_B_score_threshold_scan': {
            'candidate_thresholds_with_H1_n_ge20': len(roi_scan),
            'H1_profitable_thresholds': sum(float(x['H1']['roi']) > 1 for x in roi_scan),
            'H2_profitable_thresholds': sum(float(x['H2']['roi']) > 1 for x in roi_scan),
            'both_halves_profitable_thresholds': sum(float(x['H1']['roi']) > 1 and float(x['H2']['roi']) > 1 for x in roi_scan),
            'frozen_105': frozen_middle_fin,
            'frozen_105_H1_roi_rank': h1_roi_rank,
            'frozen_105_full2025_roi_rank': all_roi_rank,
            'best_H1_roi_candidate': max(roi_scan, key=lambda x: float(x['H1']['roi'])) if roi_scan else None,
            'best_full2025_roi_candidate': max(roi_scan, key=lambda x: float(x['ALL']['roi'])) if roi_scan else None,
            'note': 'This threshold scan is a counterfactual measure of available researcher degrees of freedom, not a claim that every threshold was actually tried historically.'
        },
    }

    # ---------- Late routing: reproduce H1 feature scan and frozen manual intersections ----------
    late_rows: list[dict[str, object]] = []
    for race in sorted(races, key=lambda r: (r['race_date'], r['track'], int(r['race_no']))):
        rid = race['race_id']
        if segment[rid] != '後半':
            continue
        row = make_feature_row(race, entries_by[rid], results_by[rid], ordinal[rid], cardn[rid])
        if row is not None:
            late_rows.append(row)
    late_h1 = [r for r in late_rows if r['half'] == 'H1']
    late_h2 = [r for r in late_rows if r['half'] == 'H2']
    late_candidates: list[dict[str, object]] = []
    for feature in LATE_FEATURES:
        for t in late_thresholds(late_h1, feature):
            cand = split_gain(late_h1, feature, t, 'mainline_top2')
            if cand:
                late_candidates.append(cand)
    late_candidates.sort(key=lambda x: float(x['gain']), reverse=True)
    top3_frozen = split_gain(late_h1, 'main_pair_top3_sum', 106.1, 'mainline_top2')
    win_frozen = split_gain(late_h1, 'main_pair_win_sum', 54.7, 'mainline_top2')

    def late_A(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        return [r for r in rows if float(r['main_pair_top3_sum']) > 106.1 and float(r['main_pair_win_sum']) > 54.7]

    def late_B(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        return [r for r in rows if not (float(r['main_pair_top3_sum']) > 106.1 and float(r['main_pair_win_sum']) > 54.7) and float(r['main_pair_top3_sum']) <= 106.1 and float(r['main_second_top3']) <= 35.7]

    late_A_h1, late_A_h2 = late_A(late_h1), late_A(late_h2)
    late_B_h1, late_B_h2 = late_B(late_h1), late_B(late_h2)
    late_audit = {
        'H1_rows': len(late_h1),
        'H2_rows': len(late_h2),
        'features_scanned': len(LATE_FEATURES),
        'valid_H1_feature_threshold_candidates_min_leaf20': len(late_candidates),
        'pair_top3_106_1_split': top3_frozen,
        'pair_top3_106_1_global_gini_rank': rank_of(top3_frozen, late_candidates),
        'pair_win_54_7_split': win_frozen,
        'pair_win_54_7_global_gini_rank': rank_of(win_frozen, late_candidates),
        'branch_A_manual_intersection': {
            'H1': subset_stats(late_A_h1, 'mainline_top2'),
            'H2': subset_stats(late_A_h2, 'mainline_top2'),
        },
        'branch_B_manual_intersection': {
            'H1_mainline': subset_stats(late_B_h1, 'mainline_top2'),
            'H2_mainline': subset_stats(late_B_h2, 'mainline_top2'),
            'H1_other_rate': 1 - subset_stats(late_B_h1, 'mainline_top2')['rate'] if late_B_h1 else 0.0,
            'H2_other_rate': 1 - subset_stats(late_B_h2, 'mainline_top2')['rate'] if late_B_h2 else 0.0,
        },
        'method_note': 'Single-feature thresholds were selected from H1; the final 3-way routing was manually composed from those signals and then checked on H2. The H2 direction did persist, so routing overfit is less severe than the later ticket optimization.'
    }

    # ---------- Rough branch: enumerate the sequential search breadth and payout-driven pruning ----------
    rough_rows = []
    mainline_fin_rows = []
    for race in sorted(races, key=lambda r: (r['race_date'], r['track'], int(r['race_no']))):
        rid = race['race_id']
        if segment[rid] != '後半':
            continue
        es = entries_by[rid]
        chosen = choose_main_line(es)
        if not chosen:
            continue
        main_id, members = chosen
        if len(members) < 3:
            continue
        branch, *_ = classify(members)
        half = 'H1' if race['race_date'] <= '2025-06-30' else 'H2'
        if branch == 'A_mainline':
            cs = mainline_bets(es, members)
            ret = sum(trifecta[rid].get(c, 0) for c in cs)
            mainline_fin_rows.append({'half': half, 'stake_yen': len(cs) * 100, 'payout_yen': ret})
        elif branch == 'B_rough':
            roles = pruning_role_map(es, main_id, members)
            rival = late_strongest_rival(es, main_id)
            if not roles or not rival or len(rival) < 2:
                continue
            winner = actual_winner(results_by[rid])
            am = leader_metrics(members[0]); rm = leader_metrics(rival[0])
            rec = {
                'race_id': rid,
                'half': half,
                'roles': roles,
                'payouts': trifecta[rid],
                'score_gap': am['score'] - rm['score'],
                'winner_role': 'A' if winner == int(members[0]['car_no']) else ('R1L' if winner == int(rival[0]['car_no']) else 'OTHER'),
            }
            for k in am:
                rec[f'd_{k}'] = am[k] - rm[k]
            rough_rows.append(rec)

    def candidate_fin(rows: list[dict[str, object]], name: str) -> dict[str, object]:
        p1, p2, p3 = CANDIDATES[name]
        x = []
        for r in rows:
            cs = pruning_combos(r['roles'], p1, p2, p3)
            ret = sum(r['payouts'].get(c, 0) for c in cs)
            x.append({'stake_yen': len(cs) * 100, 'payout_yen': ret})
        return financial(x)

    pruning_results = {}
    eligible_selection = []
    for name in CANDIDATES:
        h1m = candidate_fin([r for r in rough_rows if r['half'] == 'H1'], name)
        h2m = candidate_fin([r for r in rough_rows if r['half'] == 'H2'], name)
        allm = candidate_fin(rough_rows, name)
        pruning_results[name] = {'H1': h1m, 'H2': h2m, 'ALL': allm}
        # Exact historical selection rule in analyze_rough_pruning_v1.
        if h1m['hit_rate'] >= 0.50:
            avg_points = h1m['stake_yen'] / 100 / h1m['races'] if h1m['races'] else 999.0
            eligible_selection.append((avg_points, -h1m['roi'], -h1m['hit_rate'], name))
    selected_pruning = sorted(eligible_selection)[0][-1] if eligible_selection else 'P27_base'

    zone_results = {}
    for name, fn in ZONES.items():
        z = [r for r in rough_rows if fn(r)]
        def zsum(sub):
            a = sum(r['winner_role'] == 'A' for r in sub)
            q = sum(r['winner_role'] == 'R1L' for r in sub)
            o = sum(r['winner_role'] == 'OTHER' for r in sub)
            return {'races': len(sub), 'A': a, 'R1L': q, 'OTHER': o, 'A_share_vs_R1L': a / (a + q) if a + q else 0.0}
        zone_results[name] = {
            'ALL': zsum(z),
            'H1': zsum([r for r in z if r['half'] == 'H1']),
            'H2': zsum([r for r in z if r['half'] == 'H2']),
        }

    # Reconstruct v3 before the final same-2025 decision to turn >=10 from 9-point into SKIP.
    p18 = 'P18_first_A_R1L'
    p09 = 'P09_first_A'
    pre_skip_rough_rows = []
    final_skip_rough_rows = []
    for r in rough_rows:
        name = p09 if float(r['score_gap']) >= 10.0 else p18
        p1, p2, p3 = CANDIDATES[name]
        cs = pruning_combos(r['roles'], p1, p2, p3)
        ret = sum(r['payouts'].get(c, 0) for c in cs)
        pre_skip_rough_rows.append({'stake_yen': len(cs) * 100, 'payout_yen': ret})
        if float(r['score_gap']) < 10.0:
            final_skip_rough_rows.append({'stake_yen': len(cs) * 100, 'payout_yen': ret})

    main_fin = financial(mainline_fin_rows)
    pre_rough = financial(pre_skip_rough_rows)
    final_rough = financial(final_skip_rough_rows)
    pre_combined = {
        'stake_yen': main_fin['stake_yen'] + pre_rough['stake_yen'],
        'payout_yen': main_fin['payout_yen'] + pre_rough['payout_yen'],
    }
    pre_combined['profit_yen'] = pre_combined['payout_yen'] - pre_combined['stake_yen']
    pre_combined['roi'] = pre_combined['payout_yen'] / pre_combined['stake_yen']
    final_combined = {
        'stake_yen': main_fin['stake_yen'] + final_rough['stake_yen'],
        'payout_yen': main_fin['payout_yen'] + final_rough['payout_yen'],
    }
    final_combined['profit_yen'] = final_combined['payout_yen'] - final_combined['stake_yen']
    final_combined['roi'] = final_combined['payout_yen'] / final_combined['stake_yen']

    rough_audit = {
        'explicit_search_breadth_minimum': {
            'role_candidate_sets_in_rough_branch_candidates_v1': 8,
            'formation_definitions_in_rough_branch_candidates_v1': 3,
            'pruning_ticket_candidates': len(CANDIDATES),
            'confidence_zones': len(ZONES),
            'simple_total_not_independent': 8 + 3 + len(CANDIDATES) + len(ZONES),
            'warning': 'These are sequential and correlated alternatives, so the simple total is not a statistical multiple-testing count; it is only a lower bound on explicit researcher degrees of freedom.'
        },
        'pruning_candidates_H1_selection': {
            'historical_selection_rule': 'H1 only: require hit rate >=50%; choose fewest points; tie-break higher H1 ROI then hit rate.',
            'selected': selected_pruning,
            'all_candidates': pruning_results,
        },
        'confidence_zone_search': {
            'tested_zones': list(ZONES.keys()),
            'zone_results': zone_results,
            'score_ge_10': zone_results.get('score_ge_10'),
        },
        'same_2025_payout_driven_final_pruning': {
            'before_final_skip_ge10': pre_combined,
            'after_final_skip_ge10': final_combined,
            'in_sample_profit_uplift_yen': final_combined['profit_yen'] - pre_combined['profit_yen'],
            'in_sample_roi_uplift_percentage_points': (final_combined['roi'] - pre_combined['roi']) * 100,
            'interpretation': 'The >=10 zone was first introduced structurally as a 9-point narrowing. After those same eight 2025 races produced very poor payout economics, the final V1 changed them to SKIP. This is direct same-sample outcome-driven pruning and the clearest overfit pathway in the final strategy.'
        },
    }

    report = {
        'scope': '2025 only. Quantifies discovery/search freedom and internal temporal checks; 2024 is deliberately excluded.',
        'dataset_races': len(races),
        'early_v0': early_audit,
        'middle_v0': middle_audit,
        'late_routing': late_audit,
        'late_rough_ticket_path': rough_audit,
        'methodological_readout': {
            'early_v0': 'LOW_TO_MODERATE overfit risk: one documented structural H1 feature with similar H2 structural rate and nearby-threshold stability, but exact selector code is not preserved and payout tail inflated the headline.',
            'middle_v0': 'HIGH overfit risk: broad multi-feature H1 search preceded the final rule, while the exact 105.0 threshold and exact two ticket orders lack a preserved deterministic selection rule.',
            'late_mainline_routing': 'MODERATE overfit risk: large H1 split search and manual threshold intersection, but directional separation persisted in untouched H2.',
            'late_rough_ticket_path': 'VERY_HIGH overfit risk: repeated sequential role-set, formation, pruning and confidence-zone searches on the same 2025 data, followed by direct payout-driven removal of the losing >=10 branch.',
            'combined_2025_headline': 'HIGHLY_OPTIMISTIC_IN_SAMPLE: the final combined ROI is assembled after repeated adaptive reuse of 2025, even though the later 2024 frozen test itself remains a valid OOS test.'
        }
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
