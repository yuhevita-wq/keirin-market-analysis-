from __future__ import annotations

"""v9.0-F27: universal race-card fundamental first-branch filter.

This layer does not replace the v8.25-F26 market candidate generator.
It applies the same deterministic race-card transformation to every race type
and may only remove complete FIRST-position rider branches from a market-built
candidate.

Race-card inputs are historical race/meeting snapshots from entries.csv:
- score
- top3_rate
- b_count
- nige_count
- makuri_count
- sashi_count
- mark_count
- line_position
- line_size

No fitted coefficient, race-type-specific threshold, payout, result, prediction
mark, or evaluation mark is used.
"""

from math import isfinite
from typing import Iterable

from v8_25_f26_final_set_lock_only import build_v8_25_f26

SCHEME_VERSION = 'v9.0-F27'
BASE_SCHEME = 'v8.25-F26'
STATUS = 'DEVELOPMENT_Q1Q2Q3_UNIVERSAL_FUNDAMENTAL_LAYER'


def _pf(x):
    try:
        v = float(x)
        return v if isfinite(v) else None
    except Exception:
        return None


def _pi(x):
    try:
        return int(float(x))
    except Exception:
        return None


def _rank01(values: dict[int, float]) -> dict[int, float]:
    """Within-race ordinal normalization. Best=1, worst=0; ties stay tied."""
    n = len(values)
    if n <= 1:
        return {k: 1.0 for k in values}
    out = {}
    raw = list(values.values())
    for k, v in values.items():
        out[k] = sum(1 for z in raw if z < v) / (n - 1)
    return out


def fundamental_scores(entry_rows: Iterable[dict]) -> dict:
    rows = list(entry_rows)
    by_car = {}
    required = (
        'score', 'top3_rate', 'b_count', 'nige_count', 'makuri_count',
        'sashi_count', 'mark_count', 'line_position', 'line_size',
    )
    for r in rows:
        car = _pi(r.get('car_no'))
        if car is None or car in by_car:
            return {'ok': False, 'reason': 'FUNDAMENTAL_INVALID_CAR_ROWS'}
        vals = {k: _pf(r.get(k)) for k in required[:-2]}
        pos, size = _pi(r.get('line_position')), _pi(r.get('line_size'))
        if any(vals[k] is None for k in vals) or pos is None or size is None or pos < 1 or size < 1:
            return {'ok': False, 'reason': 'FUNDAMENTAL_INPUT_INCOMPLETE'}
        by_car[car] = {**vals, 'line_position': pos, 'line_size': size}

    if len(by_car) != 7:
        return {'ok': False, 'reason': 'FUNDAMENTAL_NOT_SEVEN_COMPLETE_ROWS'}

    def ranks(field):
        return _rank01({c: v[field] for c, v in by_car.items()})

    score_r = ranks('score')
    top3_r = ranks('top3_rate')
    b_r = ranks('b_count')
    nige_r = ranks('nige_count')
    makuri_r = ranks('makuri_count')
    sashi_r = ranks('sashi_count')
    mark_r = ranks('mark_count')

    f = {}
    components = {}
    for car, v in by_car.items():
        attack = (b_r[car] + nige_r[car] + makuri_r[car]) / 3.0
        follow = (sashi_r[car] + mark_r[car]) / 2.0
        if v['line_size'] == 1:
            role = (attack + follow) / 2.0
        elif v['line_position'] == 1:
            role = attack
        else:
            role = follow
        fi = (score_r[car] + top3_r[car] + role) / 3.0
        f[car] = fi
        components[car] = {
            'score_rank01': score_r[car],
            'top3_rank01': top3_r[car],
            'attack_rank01': attack,
            'follow_rank01': follow,
            'role_fit_rank01': role,
            'F': fi,
            'line_position': v['line_position'],
            'line_size': v['line_size'],
        }

    order = sorted(f, key=lambda c: (-f[c], c))
    gaps = [f[order[i]] - f[order[i + 1]] for i in range(len(order) - 1)]
    max_gap = max(gaps) if gaps else 0.0
    if max_gap <= 0:
        cut = len(order)
    else:
        # Conservative deterministic tie handling: use the LAST equal-largest
        # gap, which preserves the larger head block.
        eps = 1e-12
        cut = max(i + 1 for i, g in enumerate(gaps) if abs(g - max_gap) <= eps)
    head_block = tuple(order[:cut])

    return {
        'ok': True,
        'F': f,
        'components': components,
        'order': tuple(order),
        'adjacent_gaps': tuple(gaps),
        'max_gap': max_gap,
        'head_block': head_block,
    }


def build_v9_0_f27(trio_odds, trifecta_odds, predicted_line_formation: str,
                    race_type: str, entry_rows: Iterable[dict]):
    base = dict(build_v8_25_f26(
        trio_odds, trifecta_odds, predicted_line_formation, race_type
    ))
    base['base_scheme_version'] = base.get('scheme_version', BASE_SCHEME)
    base['scheme_version'] = SCHEME_VERSION
    base['fundamental_policy'] = (
        'UNIVERSAL FIRST-BRANCH FILTER: F=equal-weight mean of within-race '
        'score rank, top3-rate rank, and line-position role-fit rank; '
        'head block cut at the largest adjacent F gap; only complete first-rider '
        'branches may be removed.'
    )
    base['development_status'] = STATUS

    if not base.get('buy'):
        base['fundamental_filter_action'] = 'BASE_NO_BET_PASSTHROUGH'
        return base

    fd = fundamental_scores(entry_rows)
    if not fd.get('ok'):
        return {
            **base,
            'buy': False,
            'tickets': [],
            'ticket_count': 0,
            'reason': fd.get('reason', 'FUNDAMENTAL_INPUT_INVALID'),
            'fundamental_filter_action': 'NO_BET_INVALID_FUNDAMENTAL_INPUT',
        }

    tickets = [tuple(t) for t in base.get('tickets', [])]
    if not tickets:
        return {
            **base,
            'buy': False,
            'ticket_count': 0,
            'reason': 'FUNDAMENTAL_BASE_TICKETS_EMPTY',
            'fundamental_filter_action': 'NO_BET_INVALID_BASE_TICKETS',
        }

    base_first = tuple(sorted({int(t[0]) for t in tickets}))
    head = tuple(int(x) for x in fd['head_block'])
    surviving = tuple(c for c in base_first if c in set(head))

    fundamental_meta = {
        'fundamental_F': {str(k): v for k, v in fd['F'].items()},
        'fundamental_components': {str(k): v for k, v in fd['components'].items()},
        'fundamental_order': list(fd['order']),
        'fundamental_head_block': list(head),
        'fundamental_max_adjacent_gap': fd['max_gap'],
        'base_first_riders': list(base_first),
        'surviving_first_riders': list(surviving),
    }

    if not surviving:
        return {
            **base,
            **fundamental_meta,
            'buy': False,
            'tickets': [],
            'ticket_count': 0,
            'reason': 'FUNDAMENTAL_HEAD_CONFLICT',
            'fundamental_filter_action': 'DROP_ALL_FIRST_BRANCHES',
        }

    kept = [t for t in tickets if int(t[0]) in set(surviving)]
    action = 'PASS_ALL_FIRST_BRANCHES' if len(kept) == len(tickets) else 'DROP_UNSUPPORTED_FIRST_BRANCHES'
    return {
        **base,
        **fundamental_meta,
        'buy': True,
        'tickets': kept,
        'ticket_count': len(kept),
        'fundamental_filter_action': action,
        'branch_policy': f"{base.get('branch_policy', '')} | v9 universal fundamental first-branch filter",
    }
