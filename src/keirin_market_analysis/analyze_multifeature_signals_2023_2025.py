from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .research_high_payout_structures_v1 import parse_combination
from .research_weakest_line_2023_2025 import weakest_line
from .search_three_year_conditions_v1 import choose_main_line, strongest_rival
from .simulate_early_candidate_v0 import line_map
from .simulate_mainline_v1 import num, segment_for

YEARS = (2023, 2024, 2025)
SEGMENTS = ('前半', '中盤', '後半')
OUT = Path('data/audits/multifeature_signals_2023_2025.json')


def read_csv(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def f(v):
    x = num(v)
    return 0.0 if x == float('-inf') else float(x)


def load_year(year: int):
    d = Path(f'data/{year}/s_class_yosen')
    races = read_csv(d / 'races.csv')
    entries = read_csv(d / 'entries.csv')
    payouts = read_csv(d / 'payouts.csv')
    eb = defaultdict(list)
    for e in entries:
        eb[e['race_id']].append(e)
    tri = defaultdict(list)
    for p in payouts:
        if p.get('ticket_type') == '3連単' and p.get('status') == 'paid' and p.get('combination'):
            try:
                tri[p['race_id']].append((p['combination'], int(p['payout_yen'])))
            except (TypeError, ValueError):
                pass
    groups = defaultdict(list)
    for r in races:
        groups[(r['race_date'], r['track'])].append(r)
    seg = {}
    for g in groups.values():
        g.sort(key=lambda r: int(r['race_no']))
        for i, r in enumerate(g, 1):
            seg[r['race_id']] = segment_for(i, len(g))
    return races, eb, tri, seg


def sm(a, b, key):
    return f(a.get(key)) + f(b.get(key))


def role_numeric(prefix, e):
    keys = ('score','win_rate','top2_rate','top3_rate','s_count','b_count','nige_count','makuri_count','sashi_count','mark_count')
    out = {f'{prefix}_{k}': f(e.get(k)) for k in keys}
    out[f'{prefix}_attack'] = out[f'{prefix}_nige_count'] + out[f'{prefix}_makuri_count']
    out[f'{prefix}_finish'] = out[f'{prefix}_sashi_count'] + out[f'{prefix}_mark_count']
    return out


def pair_features(name, lead, second):
    out = {}
    for k in ('score','win_rate','top2_rate','top3_rate','s_count','b_count','nige_count','makuri_count','sashi_count','mark_count'):
        out[f'{name}_{k}'] = sm(lead, second, k)
    out[f'{name}_attack'] = f(lead.get('nige_count')) + f(lead.get('makuri_count'))
    out[f'{name}_finish'] = f(second.get('sashi_count')) + f(second.get('mark_count'))
    return out


def line_structure(es):
    by = line_map(es)
    sizes = [len(v) for v in by.values()]
    return {
        'line_count_all': len(sizes),
        'line_count_2plus': sum(x >= 2 for x in sizes),
        'solo_count': sum(x == 1 for x in sizes),
        'max_line_len': max(sizes) if sizes else 0,
    }


def bool_signal_features(x):
    # Score-vs-performance disagreement. True means the score ordering and historical-rate ordering disagree.
    return {
        'main_vs_rival_score_top2_discord': (x['main_score'] - x['rival_score']) * (x['main_top2_rate'] - x['rival_top2_rate']) < 0,
        'main_vs_rival_score_top3_discord': (x['main_score'] - x['rival_score']) * (x['main_top3_rate'] - x['rival_top3_rate']) < 0,
        'main_score_ahead_top3_behind': x['main_score'] > x['rival_score'] and x['main_top3_rate'] < x['rival_top3_rate'],
        'rival_score_ahead_main_top3_behind': x['rival_score'] > x['main_score'] and x['rival_top3_rate'] < x['main_top3_rate'],
    }


def add_gap_features(x, left, right, prefix):
    for k in ('score','win_rate','top2_rate','top3_rate','attack','finish','b_count','s_count'):
        lk = f'{left}_{k}'; rk = f'{right}_{k}'
        if lk in x and rk in x:
            x[f'{prefix}_{k}_gap'] = x[lk] - x[rk]


def build_rows():
    general = []
    weakrows = []
    exclusions = defaultdict(Counter)
    style_counts = defaultdict(Counter)
    for year in YEARS:
        races, eb, tri, seg = load_year(year)
        for race in races:
            rid = race['race_id']; s = seg.get(rid)
            if s not in SEGMENTS or not tri.get(rid):
                continue
            chosen = choose_main_line(eb[rid])
            if not chosen:
                exclusions[str(year)]['no_main'] += 1; continue
            main_id, main = chosen
            if len(main) < 3:
                exclusions[str(year)]['main_under_3'] += 1; continue
            rival = strongest_rival(eb[rid], main_id)
            if not rival or len(rival) < 2:
                exclusions[str(year)]['no_rival'] += 1; continue

            A,B,M3 = main[0], main[1], main[2]
            R1L,R1B = rival[0], rival[1]
            # Use the highest published winning 3連単 payout if dead-heat rows exist.
            wins = tri[rid]
            max_payout = max(p for _, p in wins)
            winning_combos = [c for c, p in wins if p == max_payout]
            x = {
                'year': year, 'segment': s, 'race_id': rid, 'payout_yen': max_payout,
                'high10000': max_payout >= 10000, 'high20000': max_payout >= 20000,
                'main_len': len(main), 'rival_len': len(rival),
            }
            x.update(line_structure(eb[rid]))
            for name,e in (('a',A),('b',B),('r1l',R1L),('r1b',R1B)):
                x.update(role_numeric(name,e))
                style_counts[name][str(e.get('style',''))] += 1
            x.update(pair_features('main', A, B))
            x.update(pair_features('rival', R1L, R1B))
            add_gap_features(x, 'main', 'rival', 'main_rival')
            x.update(bool_signal_features(x))
            general.append(x)

            weak, reason = weakest_line(eb[rid], main_id)
            if not weak:
                exclusions[str(year)][reason] += 1; continue
            weak_id, weak_mem, _ = weak
            rival_id = int(rival[0]['line_id'])
            if weak_id == rival_id:
                exclusions[str(year)]['weak_equals_rival'] += 1; continue
            W1L,W1B = weak_mem[0], weak_mem[1]
            w = dict(x)
            w['weak_len'] = len(weak_mem)
            for name,e in (('w1l',W1L),('w1b',W1B)):
                w.update(role_numeric(name,e))
                style_counts[name][str(e.get('style',''))] += 1
            w.update(pair_features('weak', W1L, W1B))
            add_gap_features(w, 'rival', 'weak', 'rival_weak')
            add_gap_features(w, 'main', 'weak', 'main_weak')
            w.update({
                'weak_hidden_top2_vs_rival': w['weak_score'] < w['rival_score'] and w['weak_top2_rate'] >= w['rival_top2_rate'],
                'weak_hidden_top3_vs_rival': w['weak_score'] < w['rival_score'] and w['weak_top3_rate'] >= w['rival_top3_rate'],
                'weak_hidden_attack_vs_rival': w['weak_score'] < w['rival_score'] and w['weak_attack'] >= w['rival_attack'],
                'weak_lead_score_top3_discord_vs_r1l': (w['w1l_score']-w['r1l_score'])*(w['w1l_top3_rate']-w['r1l_top3_rate']) < 0,
                'weak_second_score_top3_discord_vs_r1b': (w['w1b_score']-w['r1b_score'])*(w['w1b_top3_rate']-w['r1b_top3_rate']) < 0,
            })
            roles = {int(A['car_no']):'A', int(B['car_no']):'B', int(M3['car_no']):'M3', int(R1L['car_no']):'R1L', int(R1B['car_no']):'R1B', int(W1L['car_no']):'W1L', int(W1B['car_no']):'W1B'}
            # Outcome roles across all published winning combinations; any occurrence counts as true.
            orders = []
            for combo,_p in wins:
                cars = parse_combination(combo)
                orders.append(tuple(roles.get(c,'OTHER') for c in cars))
            w['weak_included'] = any(any(z in {'W1L','W1B'} for z in o) for o in orders)
            w['weak_head'] = any(o[0] in {'W1L','W1B'} for o in orders)
            w['weak_main2'] = any(sum(z in {'W1L','W1B'} for z in o)==1 and sum(z in {'A','B','M3'} for z in o)==2 for o in orders)
            weakrows.append(w)
    return general, weakrows, exclusions, style_counts


def quantile_cut(vals, q):
    xs = sorted(vals)
    if not xs: return 0.0
    pos = (len(xs)-1)*q
    lo = math.floor(pos); hi = math.ceil(pos)
    if lo == hi: return xs[lo]
    return xs[lo] + (xs[hi]-xs[lo])*(pos-lo)


def rate(rows, target):
    return sum(bool(r[target]) for r in rows)/len(rows) if rows else 0.0


def numeric_signals(rows, target, segment):
    sr = [r for r in rows if r['segment']==segment]
    if not sr: return []
    excluded = {'year','segment','race_id','payout_yen'}
    features = []
    for k,v in sr[0].items():
        if k in excluded or k == target or isinstance(v,bool): continue
        if isinstance(v,(int,float)):
            features.append(k)
    out=[]
    for feat in features:
        vals=[float(r[feat]) for r in sr if isinstance(r.get(feat),(int,float))]
        if len(vals)<100: continue
        q25=quantile_cut(vals,.25); q75=quantile_cut(vals,.75)
        yearly={}; signs=[]; effects=[]
        for y in YEARS:
            yr=[r for r in sr if r['year']==y]
            low=[r for r in yr if float(r[feat])<=q25]
            high=[r for r in yr if float(r[feat])>=q75]
            if min(len(low),len(high))<10: break
            lr=rate(low,target); hr=rate(high,target); diff=hr-lr
            yearly[str(y)]={'low_n':len(low),'high_n':len(high),'low_rate':lr,'high_rate':hr,'high_minus_low_pp':diff*100}
            signs.append(1 if diff>0 else -1 if diff<0 else 0); effects.append(abs(diff)*100)
        if len(yearly)!=3: continue
        nonzero=[s for s in signs if s]
        stable=bool(nonzero) and len(set(nonzero))==1
        out.append({'feature':feat,'q25':q25,'q75':q75,'stable_direction':stable,
                    'direction':'HIGHER=>TARGET' if stable and nonzero[0]>0 else 'LOWER=>TARGET' if stable else 'MIXED',
                    'min_abs_effect_pp':min(effects),'mean_abs_effect_pp':sum(effects)/3,'by_year':yearly})
    out.sort(key=lambda z:(z['stable_direction'],z['min_abs_effect_pp'],z['mean_abs_effect_pp']),reverse=True)
    return out


def boolean_signals(rows, target, segment):
    sr=[r for r in rows if r['segment']==segment]
    bools=[k for k,v in (sr[0].items() if sr else []) if isinstance(v,bool) and k!=target]
    out=[]
    for feat in bools:
        yearly={}; signs=[]; effects=[]
        for y in YEARS:
            yr=[r for r in sr if r['year']==y]
            t=[r for r in yr if r[feat]]; frows=[r for r in yr if not r[feat]]
            if min(len(t),len(frows))<10: break
            tr=rate(t,target); fr=rate(frows,target); diff=tr-fr
            yearly[str(y)]={'true_n':len(t),'false_n':len(frows),'true_rate':tr,'false_rate':fr,'true_minus_false_pp':diff*100}
            signs.append(1 if diff>0 else -1 if diff<0 else 0); effects.append(abs(diff)*100)
        if len(yearly)!=3: continue
        nonzero=[s for s in signs if s]
        stable=bool(nonzero) and len(set(nonzero))==1
        out.append({'feature':feat,'stable_direction':stable,
                    'direction':'TRUE=>TARGET' if stable and nonzero[0]>0 else 'FALSE=>TARGET' if stable else 'MIXED',
                    'min_abs_effect_pp':min(effects),'mean_abs_effect_pp':sum(effects)/3,'by_year':yearly})
    out.sort(key=lambda z:(z['stable_direction'],z['min_abs_effect_pp'],z['mean_abs_effect_pp']),reverse=True)
    return out


def family(feat):
    if any(s in feat for s in ('nige','makuri','sashi','mark','attack','finish','b_count','s_count')): return 'TACTICAL_COUNTS'
    if any(s in feat for s in ('win_rate','top2_rate','top3_rate')): return 'HISTORICAL_RATES'
    if 'score' in feat: return 'SCORE'
    if any(s in feat for s in ('line_count','line_len','solo_count','main_len','rival_len','weak_len','max_line_len')): return 'LINE_STRUCTURE'
    if 'discord' in feat or 'hidden' in feat: return 'DISCORDANCE'
    return 'OTHER'


def compact(signals, n=12):
    stable=[x for x in signals if x['stable_direction']]
    return stable[:n]


def family_champions(signals):
    best={}
    for x in signals:
        if not x['stable_direction']: continue
        fam=family(x['feature'])
        if fam not in best or x['min_abs_effect_pp']>best[fam]['min_abs_effect_pp']:
            best[fam]=x
    return best


def main():
    general, weakrows, exclusions, style_counts = build_rows()
    out={
        'status':'MULTIFEATURE_SIGNAL_ANALYSIS_2023_2025_ONLY',
        'years_read':list(YEARS),
        'evaluation_year_2026_used':False,
        'purpose':'Descriptive stability audit of pre-race score, historical-rate, tactical-count, line-structure, and score-vs-rate discordance features. Not a strategy search.',
        'entry_fields_available':['style','score','s_count','b_count','nige_count','makuri_count','sashi_count','mark_count','win_rate','top2_rate','top3_rate'],
        'method':'Pooled 25th/75th percentile cuts per segment for numeric features; compare target rates in low vs high quartiles separately in each year. Stable means the direction agrees in 2023, 2024, 2025. Boolean signals compare true vs false in each year.',
        'exclusions':{y:dict(c) for y,c in exclusions.items()},
        'style_value_counts':{k:dict(v) for k,v in style_counts.items()},
        'populations':{
            'general_main3_rival2':{str(y):{s:sum(r['year']==y and r['segment']==s for r in general) for s in SEGMENTS} for y in YEARS},
            'weakest_line_3plus_lines':{str(y):{s:sum(r['year']==y and r['segment']==s for r in weakrows) for s in SEGMENTS} for y in YEARS},
        },
        'targets':{}
    }
    targets=[('high10000',general),('high20000',general),('weak_included',weakrows),('weak_main2',weakrows),('weak_head',weakrows)]
    for target,rows in targets:
        out['targets'][target]={}
        for s in SEGMENTS:
            nr=numeric_signals(rows,target,s); br=boolean_signals(rows,target,s)
            baseline={str(y):{'n':sum(r['year']==y and r['segment']==s for r in rows),
                              'rate':rate([r for r in rows if r['year']==y and r['segment']==s],target)} for y in YEARS}
            out['targets'][target][s]={
                'baseline':baseline,
                'top_stable_numeric':compact(nr,15),
                'top_stable_boolean':compact(br,10),
                'numeric_family_champions':family_champions(nr),
                'boolean_family_champions':family_champions(br),
                'stable_numeric_count':sum(x['stable_direction'] for x in nr),
                'stable_boolean_count':sum(x['stable_direction'] for x in br),
            }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({t:{s:{'baseline':out['targets'][t][s]['baseline'],'top_numeric':[x['feature'] for x in out['targets'][t][s]['top_stable_numeric'][:5]],'top_boolean':[x['feature'] for x in out['targets'][t][s]['top_stable_boolean'][:5]],'family_champions':{k:v['feature'] for k,v in out['targets'][t][s]['numeric_family_champions'].items()}} for s in SEGMENTS} for t,_ in targets},ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
