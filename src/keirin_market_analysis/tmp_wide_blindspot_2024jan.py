import csv, json, math, statistics
from collections import defaultdict
from pathlib import Path

TRAIN_DIR = Path('data/2023/s_class_yosen')
TEST_DIR = Path('data/2024/s_class_f1_all_parts/2024_q1')
OUT = Path('data/audits/wide_blindspot_2024jan.json')
RACE_TYPE = 'Ｓ級予選'
MIN_BUCKET_N = 30
MAIN_PERCENTILE = 0.30
SENSITIVITY = [0.20, 0.30, 0.40, 0.50, 1.00]


def read_csv(path):
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def parse_combo(s):
    if not s:
        return None
    xs = s.replace('-', '=').split('=')
    try:
        return tuple(sorted(int(x) for x in xs if x))
    except ValueError:
        return None


def parse_line(s):
    if not s:
        return None
    rider_info = {}
    groups = []
    for gi, part in enumerate(str(s).split('/')):
        vals = []
        for tok in part.strip().split('-'):
            tok = tok.strip()
            if tok.isdigit():
                vals.append(int(tok))
        if not vals:
            continue
        groups.append(vals)
        for pos, rider in enumerate(vals):
            if len(vals) == 1:
                role = 'single'
            elif pos == 0:
                role = 'head'
            elif pos == 1:
                role = 'second'
            else:
                role = 'third_plus'
            rider_info[rider] = {'group': gi, 'pos': pos, 'role': role, 'line_len': len(vals)}
    return rider_info if rider_info else None


def percentile(vals, p):
    vals = sorted(vals)
    if not vals:
        return None
    if p <= 0:
        return vals[0]
    if p >= 1:
        return vals[-1]
    x = (len(vals) - 1) * p
    lo, hi = math.floor(x), math.ceil(x)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - x) + vals[hi] * (x - lo)


def race_map(path):
    out = {}
    for r in read_csv(path):
        try:
            n = int(r.get('entry_count') or 0)
        except ValueError:
            n = 0
        out[r['race_id']] = {
            'race_id': r['race_id'],
            'race_date': r.get('race_date',''),
            'track': r.get('track',''),
            'race_no': r.get('race_no',''),
            'race_type': r.get('race_type',''),
            'entry_count': n,
            'start_time': r.get('start_time',''),
            'line': r.get('predicted_line_formation',''),
        }
    return out


def load_trio(path):
    by = defaultdict(list)
    for r in read_csv(path):
        if r.get('ticket_type') != '3連複' or r.get('odds_status') != 'available':
            continue
        c = parse_combo(r.get('combination'))
        try:
            o = float(r.get('odds') or 0)
        except ValueError:
            continue
        if c and len(c) == 3 and o > 0:
            by[r['race_id']].append((c, o))
    return by


def market_features(rows):
    # Exactly 35 unique combinations for a complete 7-car trio market.
    d = {}
    for c, o in rows:
        d[c] = o
    if len(d) != 35:
        return None
    weights = {c: 1.0/o for c,o in d.items()}
    z = sum(weights.values())
    if z <= 0:
        return None
    q = {c:w/z for c,w in weights.items()}
    riders = sorted({x for c in q for x in c})
    if len(riders) != 7:
        return None
    M = {i: sum(v for c,v in q.items() if i in c) for i in riders}
    order = sorted(riders, key=lambda i: (-M[i], i))
    rank = {i:k+1 for k,i in enumerate(order)}
    C = {}
    for a in riders:
        for b in riders:
            if a < b:
                C[(a,b)] = sum(v for c,v in q.items() if a in c and b in c)
    return {'M':M, 'rank':rank, 'C':C}


def wide_payouts(path):
    by = defaultdict(dict)
    for r in read_csv(path):
        if r.get('ticket_type') != 'ワイド' or r.get('status') != 'paid':
            continue
        c = parse_combo(r.get('combination'))
        if not c or len(c) != 2:
            continue
        try:
            p = int(float(r.get('payout_yen') or 0))
        except ValueError:
            p = 0
        if p > 0:
            by[r['race_id']][tuple(c)] = max(p, by[r['race_id']].get(tuple(c), 0))
    return by


def candidate_records(races, trio, payouts=None, jan_only=False):
    recs = []
    valid_races = 0
    for rid, race in races.items():
        if race['race_type'] != RACE_TYPE or race['entry_count'] != 7:
            continue
        if jan_only and not race['race_date'].startswith('2024-01-'):
            continue
        line = parse_line(race['line'])
        mf = market_features(trio.get(rid, []))
        if not line or not mf or any(i not in line for i in mf['rank']):
            continue
        valid_races += 1
        inv_rank = {rk:i for i,rk in mf['rank'].items()}
        for ar in (1,2):
            a = inv_rank[ar]
            for pr in (4,5,6,7):
                b = inv_rank[pr]
                pair = tuple(sorted((a,b)))
                ai, bi = line[a], line[b]
                relation = 'same_line' if ai['group'] == bi['group'] else 'different_lines'
                bucket = (ar, pr, relation, bi['role'])
                hit = None
                payout = None
                if payouts is not None:
                    payout = payouts.get(rid, {}).get(pair, 0)
                    hit = 1 if payout > 0 else 0
                recs.append({
                    'race_id':rid, 'race_date':race['race_date'], 'track':race['track'],
                    'race_no':race['race_no'], 'start_time':race['start_time'],
                    'line':race['line'], 'anchor':a, 'partner':b, 'pair':pair,
                    'anchor_rank':ar, 'partner_rank':pr, 'relation':relation,
                    'partner_role':bi['role'], 'Cij':mf['C'][pair], 'bucket':bucket,
                    'hit':hit, 'payout':payout,
                })
    return recs, valid_races


def summarize_bets(bets):
    stake = 100 * len(bets)
    ret = sum(b['payout'] for b in bets)
    hits = sum(1 for b in bets if b['payout'] > 0)
    hit_pays = [b['payout'] for b in bets if b['payout'] > 0]
    streak = max_streak = 0
    for b in bets:
        if b['payout'] > 0:
            streak = 0
        else:
            streak += 1
            max_streak = max(max_streak, streak)
    return {
        'bets':len(bets), 'hits':hits,
        'hit_rate': hits/len(bets) if bets else None,
        'stake_yen':stake, 'return_yen':ret, 'profit_yen':ret-stake,
        'roi': ret/stake if stake else None,
        'avg_hit_payout_yen': statistics.mean(hit_pays) if hit_pays else None,
        'median_hit_payout_yen': statistics.median(hit_pays) if hit_pays else None,
        'max_hit_payout_yen': max(hit_pays) if hit_pays else None,
        'max_losing_streak':max_streak,
    }


def main():
    train_races = race_map(TRAIN_DIR/'races.csv')
    test_races = race_map(TEST_DIR/'races.csv')
    train_trio = load_trio(TRAIN_DIR/'trio_final_odds.csv')
    test_trio = load_trio(TEST_DIR/'trio_final_odds.csv')
    train_pay = wide_payouts(TRAIN_DIR/'payouts.csv')
    test_pay = wide_payouts(TEST_DIR/'payouts.csv')

    train, train_valid = candidate_records(train_races, train_trio, train_pay, jan_only=False)
    test_unscored, test_valid = candidate_records(test_races, test_trio, None, jan_only=True)

    # Learn bucket calibration ONLY from 2023 outcomes.
    agg = defaultdict(lambda: {'n':0,'hits':0,'c_sum':0.0})
    for r in train:
        a = agg[r['bucket']]
        a['n'] += 1; a['hits'] += r['hit']; a['c_sum'] += r['Cij']
    calib = {}
    for k,a in agg.items():
        n=a['n']; actual=a['hits']/n; avgc=a['c_sum']/n
        calib[k] = {'n':n,'hits':a['hits'],'actual_hit_rate':actual,
                    'avg_market_pair_prob':avgc,
                    'edge_ratio':actual/avgc if avgc>0 else None}

    train_cijs = [r['Cij'] for r in train]
    cutoffs = {str(int(p*100)):percentile(train_cijs,p) for p in SENSITIVITY}

    # Generate all 2024-Jan picks WITHOUT seeing 2024 payouts.
    by_race = defaultdict(list)
    for r in test_unscored:
        by_race[r['race_id']].append(r)

    all_tiers = {}
    pick_lists = {}
    for p in SENSITIVITY:
        cutoff = cutoffs[str(int(p*100))]
        picks=[]
        for rid, candidates in by_race.items():
            eligible=[]
            for r in candidates:
                c = calib.get(r['bucket'])
                if not c or c['n'] < MIN_BUCKET_N or not c['edge_ratio'] or c['edge_ratio'] <= 1.0:
                    continue
                if r['Cij'] > cutoff:
                    continue
                x=dict(r)
                x['bucket_n']=c['n']; x['edge_ratio']=c['edge_ratio']; x['historical_hit_rate']=c['actual_hit_rate']
                x['estimated_pair_prob']=min(0.999, r['Cij']*c['edge_ratio'])
                eligible.append(x)
            if not eligible:
                continue
            eligible.sort(key=lambda x:(-x['edge_ratio'], x['Cij'], -x['partner_rank'], x['pair']))
            picks.append(eligible[0])
        picks.sort(key=lambda x:(x['race_date'], x['start_time'], x['race_id']))

        # Only now attach 2024 outcome/payout labels.
        scored=[]
        for x in picks:
            y=dict(x)
            y['payout']=test_pay.get(x['race_id'],{}).get(tuple(x['pair']),0)
            y['hit']=1 if y['payout']>0 else 0
            y['pair']='='.join(map(str,x['pair']))
            y['bucket']=list(x['bucket'])
            scored.append(y)
        pick_lists[str(int(p*100))]=scored
        all_tiers[str(int(p*100))] = {
            'risk_percentile':p,
            'Cij_cutoff':cutoff,
            **summarize_bets(scored)
        }

    main_key=str(int(MAIN_PERCENTILE*100))
    main_picks=pick_lists[main_key]
    role_breakdown={}
    for keyfunc_name, keyfunc in [
        ('partner_role', lambda x:x['partner_role']),
        ('relation', lambda x:x['relation']),
        ('anchor_partner_rank', lambda x:f"{x['anchor_rank']}-{x['partner_rank']}"),
    ]:
        groups=defaultdict(list)
        for b in main_picks: groups[keyfunc(b)].append(b)
        role_breakdown[keyfunc_name]={k:summarize_bets(v) for k,v in sorted(groups.items())}

    top_calib=[]
    for k,c in calib.items():
        if c['n']>=MIN_BUCKET_N and c['edge_ratio'] is not None:
            top_calib.append({'bucket':list(k),**c})
    top_calib.sort(key=lambda x:-x['edge_ratio'])

    out={
        'strategy':'Wide Blindspot v0.1 frozen before 2024-Jan scoring',
        'training_period':'2023 full year, exact S級予選 only',
        'test_period':'2024-01-01 through 2024-01-31, exact S級予選 only',
        'stake_per_bet_yen':100,
        'rules':{
            'entry_count':7,
            'complete_trio_market_required':True,
            'line_required':True,
            'anchor_market_rider_rank':[1,2],
            'partner_market_rider_rank':[4,5,6,7],
            'bucket_features':['anchor_rank','partner_rank','same/different line','partner line role'],
            'bucket_min_n':MIN_BUCKET_N,
            'bet_only_if_2023_edge_ratio_gt':1.0,
            'main_high_risk_Cij_percentile':MAIN_PERCENTILE,
            'selection':'max 2023 edge_ratio, tie -> lower Cij, then higher partner rank',
            'one_bet_max_per_race':True,
            '2024_outcomes_used_only_after_pick_generation':True,
        },
        'data_counts':{
            'train_valid_races':train_valid,
            'train_candidate_pairs':len(train),
            'test_jan_valid_races':test_valid,
            'test_jan_candidate_pairs':len(test_unscored),
        },
        'sensitivity':all_tiers,
        'main_30pct':all_tiers[main_key],
        'main_breakdown':role_breakdown,
        'top_2023_calibration_buckets':top_calib[:20],
        'main_picks':main_picks,
        'warning':'Uses historical final trio odds as the market feature; this is not a T-10-minute executable-odds backtest.'
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({
        'train_valid_races':train_valid,
        'test_jan_valid_races':test_valid,
        'main':all_tiers[main_key],
        'sensitivity':all_tiers,
    }, ensure_ascii=False, indent=2))

if __name__=='__main__':
    main()
