from __future__ import annotations

import csv, itertools, json, math, statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'audits' / 'must_enter_formation_development_2023_forward.json'
DATASETS = {
    '2023': ROOT/'data'/'2023'/'s_class_yosen',
    '2024': ROOT/'data'/'2024'/'s_class_yosen',
    '2025': ROOT/'data'/'2025'/'s_class_yosen',
    '2026_h1': ROOT/'data'/'2026_h1'/'s_class_yosen',
}

# Frozen by prior incremental-information test. Do not refit here.
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


def q_from_weights(combos, weights):
    raw = {c: math.prod(weights[x] for x in c) for c in combos}
    z = sum(raw.values())
    return {c: v/z for c, v in raw.items()}


def fit_maxent(riders, combos, target, tol=1e-12, max_iter=20000):
    w = {r: 1.0 for r in riders}
    for _ in range(max_iter):
        for r in riders:
            q = q_from_weights(combos, w)
            cur = sum(v for c, v in q.items() if r in c)
            t = target[r]
            f = (t*(1-cur))/(cur*(1-t))
            w[r] *= f
        g = math.exp(sum(math.log(max(w[r], 1e-300)) for r in riders)/len(riders))
        for r in riders:
            w[r] /= g
        q = q_from_weights(combos, w)
        err = max(abs(sum(v for c,v in q.items() if r in c)-target[r]) for r in riders)
        if err < tol:
            return q, err
    return q, err


def p1_probs(m, R):
    vals = {c: m[c]*(R[c]**BETA) for c in m}
    z = sum(vals.values())
    return {c:v/z for c,v in vals.items()}


def tv(a,b):
    return .5*sum(abs(a[c]-b[c]) for c in a)


def payout_value(row):
    for key in ('payout_yen','payout','amount_yen','amount'):
        x = row.get(key)
        if x not in (None,''):
            try:
                return int(float(str(x).replace(',','')))
            except Exception:
                pass
    return None


def load_year(name):
    data = DATASETS[name]
    by = defaultdict(list)
    for r in read_csv(data/'trio_final_odds.csv'):
        if r.get('odds_status') != 'available':
            continue
        try:
            o = float(r['odds'])
        except Exception:
            continue
        c = combo(r.get('combination',''))
        if len(c)==3 and o>0:
            by[str(r['race_id'])].append((c,o))

    paid = defaultdict(list)
    for r in read_csv(data/'payouts.csv'):
        if r.get('ticket_type') == '3連複' and r.get('status') == 'paid':
            c = combo(r.get('combination',''))
            pv = payout_value(r)
            if len(c)==3 and pv is not None:
                paid[str(r['race_id'])].append((c,pv))

    races=[]; excluded=0; solver_fail=0; no_unique_payout=0; non_must=0
    for rid, rows in sorted(by.items()):
        riders = sorted({x for c,_ in rows for x in c})
        if len(riders)!=7:
            excluded += 1; continue
        combos = list(itertools.combinations(riders,3))
        odds = {c:o for c,o in rows}
        if set(odds)!=set(combos):
            excluded += 1; continue
        # Require one paid trio row for clean scoring.
        ps = paid.get(rid,[])
        if len(ps)!=1:
            no_unique_payout += 1; continue
        win,payout = ps[0]
        if win not in odds:
            excluded += 1; continue
        inv={c:1/o for c,o in odds.items()}; z=sum(inv.values()); m={c:v/z for c,v in inv.items()}
        support={r:sum(v for c,v in m.items() if r in c) for r in riders}
        q,err=fit_maxent(riders,combos,support)
        if err>=1e-9:
            solver_fail += 1; continue
        R={c:m[c]/q[c] for c in combos}
        p1=p1_probs(m,R)
        distance=tv(p1,m)
        if distance < MUST_TV:
            non_must += 1; continue
        races.append({'race_id':rid,'riders':riders,'combos':combos,'odds':odds,'m':m,'R':R,'p1':p1,'win':win,'payout':payout,'tv':distance})
    return races, {
        'market_races':len(by),
        'excluded_non7_or_incomplete':excluded,
        'no_unique_paid_trio':no_unique_payout,
        'solver_failures':solver_fail,
        'non_must_enter':non_must,
        'must_enter_evaluated':len(races),
    }


def pair_mass(r, pair):
    return sum(r['p1'][c] for c in r['combos'] if pair[0] in c and pair[1] in c)


def third_candidates(r, pair):
    out=[]
    for x in r['riders']:
        if x in pair:
            continue
        c=tuple(sorted((pair[0],pair[1],x)))
        uplift=r['p1'][c]/r['m'][c]
        mixed=r['p1'][c]*uplift
        out.append({'third':x,'combo':c,'prob':r['p1'][c],'uplift':uplift,'mixed':mixed})
    return out


def tickets_for(r, mode, pairs_n, k):
    pairs=list(itertools.combinations(r['riders'],2))
    pairs.sort(key=lambda p:(-pair_mass(r,p),p))
    chosen=pairs[:pairs_n]
    tickets=set()
    for p in chosen:
        cs=third_candidates(r,p)
        key = {'prob':'prob','uplift':'uplift','mixed':'mixed'}[mode]
        cs.sort(key=lambda x:(-x[key],x['combo']))
        tickets.update(x['combo'] for x in cs[:k])
    return sorted(tickets)


def candidate_specs():
    out=[]
    for mode in ('prob','uplift','mixed'):
        # Deliberately small predefined family: all are formations, never 1-point bets.
        for pairs_n,k in ((1,4),(1,5),(2,3),(2,4)):
            out.append({'mode':mode,'pairs_n':pairs_n,'thirds_per_pair':k,'name':f'{mode.upper()}_P{pairs_n}_K{k}'})
    return out


def score(races,spec):
    stake=0; ret=0; hits=0; tickets_n=0; max_loss=0; cur_loss=0; counts=[]
    for r in races:
        t=tickets_for(r,spec['mode'],spec['pairs_n'],spec['thirds_per_pair'])
        n=len(t); counts.append(n); tickets_n+=n; stake += 100*n
        if r['win'] in t:
            hits += 1; ret += r['payout']; cur_loss=0
        else:
            cur_loss += 1; max_loss=max(max_loss,cur_loss)
    return {
        'races':len(races),
        'tickets':tickets_n,
        'avg_tickets_per_race':statistics.mean(counts) if counts else None,
        'min_tickets_per_race':min(counts) if counts else None,
        'max_tickets_per_race':max(counts) if counts else None,
        'hit_races':hits,
        'hit_rate_pct':100*hits/len(races) if races else None,
        'stake_yen':stake,
        'payout_yen':ret,
        'profit_yen':ret-stake,
        'roi_pct':100*ret/stake if stake else None,
        'max_losing_streak_races':max(max_loss,cur_loss),
    }


def main():
    datasets={}; accounting={}
    for y in DATASETS:
        datasets[y],accounting[y]=load_year(y)

    specs=candidate_specs()
    dev_rows=[]
    for spec in specs:
        m=score(datasets['2023'],spec)
        dev_rows.append({**spec,'metrics':m})
    # 2023 is development only. Highest ROI wins; ties prefer hit rate, then shorter drought.
    dev_rows.sort(key=lambda x:(-x['metrics']['roi_pct'],-x['metrics']['hit_rate_pct'],x['metrics']['max_losing_streak_races'],x['name']))
    winner={k:v for k,v in dev_rows[0].items() if k!='metrics'}

    forward={}
    for y in DATASETS:
        forward[y]=score(datasets[y],winner)

    out={
        'status':'MUST_ENTER_FORMATION_DEVELOPMENT_2023_FORWARD',
        'frozen_inputs':{
            'beta':BETA,
            'must_enter_tv_threshold':MUST_TV,
            'must_enter_definition':'TV(P1,M) >= frozen 2023 75th percentile threshold from prior incremental-information test.',
            'market_and_support_only_for_ticket_selection':True,
        },
        'formation_family':{
            'pair_core':'Rank all 21 rider pairs by the sum of support-augmented probability P1 over the five trios containing that pair.',
            'prob':'For chosen pair core(s), choose third riders by P1.',
            'uplift':'For chosen pair core(s), choose third riders by P1/M, i.e. combinations upgraded most by the support model versus raw market.',
            'mixed':'For chosen pair core(s), choose third riders by P1*(P1/M), balancing probability and model uplift.',
            'candidate_grid':specs,
            'development_selection_rule':'Choose the highest 2023 ROI candidate; tie-break by higher hit rate then shorter losing streak. 2024+ is not consulted.',
            'stake_per_ticket_yen':100,
        },
        'accounting':accounting,
        'development_2023_ranked':dev_rows,
        'frozen_winner':winner,
        'forward_evaluation_same_rule':forward,
        'warning':'2023 is in-sample development for the formation rule. Only 2024, 2025 and 2026_h1 are forward checks. A profitable 2023 winner is not evidence by itself.',
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'winner':winner,'forward':forward,'top5_2023':dev_rows[:5],'accounting':accounting},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
