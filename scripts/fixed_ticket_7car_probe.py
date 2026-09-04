import csv, glob, itertools, json, re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETS = {
    '2024': glob.glob(str(ROOT / 'data/2024/s_class_f1_all_parts/2024_q*')),
    '2025': glob.glob(str(ROOT / 'data/2025/s_class_f1_all_parts/2025_q*')),
    '2026H1': [str(ROOT / 'data/2026_h1/s_class_f1_all')],
}
TYPES = ['2車複', '2車単', '3連複', '3連単', 'ワイド', '2枠複', '2枠単']
UNORDERED = {'2車複', '3連複', 'ワイド', '2枠複'}


def rows(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        yield from csv.DictReader(f)


def norm(ticket_type, combination):
    nums = [int(x) for x in re.findall(r'\d+', combination or '')]
    need = 3 if ticket_type.startswith('3連') else 2
    if len(nums) != need:
        return None
    if ticket_type in UNORDERED:
        nums.sort()
    return tuple(nums)


def universe(ticket_type):
    n = 6 if ticket_type.startswith('2枠') else 7
    k = 3 if ticket_type.startswith('3連') else 2
    if ticket_type in UNORDERED:
        return list(itertools.combinations(range(1, n + 1), k))
    return list(itertools.permutations(range(1, n + 1), k))


D = {}
for year, dirs in SETS.items():
    seven = set()
    active = defaultdict(set)
    paid = defaultdict(lambda: defaultdict(int))
    blanket_refund = defaultdict(set)
    for d in dirs:
        rp = Path(d) / 'races.csv'
        if rp.exists():
            for r in rows(rp):
                if int(r.get('entry_count') or 0) == 7:
                    seven.add(r['race_id'])
        pp = Path(d) / 'payouts.csv'
        if pp.exists():
            for r in rows(pp):
                tt = r.get('ticket_type')
                if tt not in TYPES:
                    continue
                rid = r.get('race_id')
                status = r.get('status', '')
                if status != 'not_offered':
                    active[tt].add(rid)
                if status == 'refund' and not r.get('combination'):
                    blanket_refund[tt].add(rid)
                if status == 'paid':
                    combo = norm(tt, r.get('combination', ''))
                    if combo:
                        payout = int(float(r.get('payout_yen') or 0))
                        paid[(tt, combo)][rid] = max(paid[(tt, combo)][rid], payout)
    D[year] = {'seven': seven, 'active': active, 'paid': paid, 'refund': blanket_refund}


def evaluate(year, tt, combo):
    d = D[year]
    races = d['active'][tt] & d['seven']
    n = len(races)
    returned = sum(v for rid, v in d['paid'][(tt, combo)].items() if rid in races)
    returned += 100 * len(d['refund'][tt] & races)
    hits = sum(1 for rid in d['paid'][(tt, combo)] if rid in races)
    stake = 100 * n
    return {'races': n, 'hits': hits, 'hit_rate': hits / n if n else None,
            'stake_yen': stake, 'return_yen': returned, 'profit_yen': returned - stake,
            'roi': returned / stake if stake else None}


def pooled(combo, tt, years):
    by_year = {y: evaluate(y, tt, combo) for y in years}
    stake = sum(v['stake_yen'] for v in by_year.values())
    returned = sum(v['return_yen'] for v in by_year.values())
    return {'combo': list(combo), 'by_year': by_year,
            'races': sum(v['races'] for v in by_year.values()),
            'hits': sum(v['hits'] for v in by_year.values()),
            'stake_yen': stake, 'return_yen': returned,
            'profit_yen': returned - stake, 'roi': returned / stake if stake else None}


def best_on(tt, years):
    xs = [pooled(c, tt, years) for c in universe(tt)]
    xs = [x for x in xs if x['stake_yen']]
    xs.sort(key=lambda z: (z['roi'], z['profit_yen']), reverse=True)
    return xs[0] if xs else None


out = {'scope': 'F1 S級 7車立て限定・固定100円買い',
       'periods': ['2024', '2025', '2026H1'],
       'seven_car_race_counts': {y: len(D[y]['seven']) for y in D},
       'ticket_types': {}}

for tt in TYPES:
    available = {y: len(D[y]['active'][tt] & D[y]['seven']) for y in D}
    pick24 = best_on(tt, ['2024'])
    frozen24 = pooled(tuple(pick24['combo']), tt, ['2024', '2025', '2026H1']) if pick24 else None
    pick2425 = best_on(tt, ['2024', '2025'])
    frozen2425 = None
    if pick2425:
        c = tuple(pick2425['combo'])
        frozen2425 = {'combo': list(c),
                      'development_2024_2025': pooled(c, tt, ['2024', '2025']),
                      'oos_2026H1': evaluate('2026H1', tt, c)}
    cross = []
    for c in universe(tt):
        by = {y: evaluate(y, tt, c) for y in D}
        if all(by[y]['roi'] is not None and by[y]['roi'] > 1.0 for y in D):
            stake = sum(by[y]['stake_yen'] for y in D)
            ret = sum(by[y]['return_yen'] for y in D)
            cross.append({'combo': list(c), 'min_period_roi': min(by[y]['roi'] for y in D),
                          'pooled_roi': ret / stake, 'pooled_profit_yen': ret - stake, 'by_year': by})
    cross.sort(key=lambda z: (z['min_period_roi'], z['pooled_roi']), reverse=True)
    out['ticket_types'][tt] = {
        'available_races': available,
        'best_selected_on_2024': pick24,
        'frozen_2024_pick_all_periods': frozen24,
        'best_selected_on_2024_2025_then_2026H1_oos': frozen2425,
        'cross_period_positive_count': len(cross),
        'cross_period_positive_top20': cross[:20],
        'hindsight_best_all_periods': best_on(tt, ['2024', '2025', '2026H1']),
    }

p = ROOT / 'results/fixed_ticket_7car_f1_20260904.json'
p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print('seven_car_race_counts', out['seven_car_race_counts'])
for tt, x in out['ticket_types'].items():
    print(tt, 'available', x['available_races'], 'cross_positive', x['cross_period_positive_count'])
