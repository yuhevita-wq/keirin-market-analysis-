from __future__ import annotations
import csv, io, json, math, statistics, zipfile
from collections import defaultdict, Counter
from itertools import product
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ARCH=ROOT/'data'/'grade_races'/'g3'
OUT=ROOT/'results'/'g3_day3_reality'

def dec(b):
    for e in ('utf-8-sig','utf-8','cp932','shift_jis'):
        try:return b.decode(e)
        except UnicodeDecodeError:pass
    return b.decode('utf-8',errors='replace')
def read(z,n):
    xs=[x for x in z.namelist() if Path(x).name==n]
    return list(csv.DictReader(io.StringIO(dec(z.read(xs[0]))))) if xs else []
def f(v,d=0.0):
    try:
        x=float(v or ''); return x if math.isfinite(x) else d
    except:return d
def i(v,d=999):
    try:return int(v)
    except:return d
def day3(r):return len(r)==16 and r.isdigit() and r[10:12]=='03'
def parse_combo(s):
    if not s:return None
    for sep in ('=','-'):
        p=s.split(sep)
        if len(p)==3 and all(x.strip().isdigit() for x in p):return tuple(sorted(int(x) for x in p))
    return None
def paid(ps):
    for p in ps:
        if p.get('ticket_type')=='3連複' and p.get('status')=='paid':
            c=parse_combo(p.get('combination',''))
            if c:
                try:pay=int(float(p.get('payout_yen') or 0))
                except:pay=0
                return c,pay
    return None,0
def market(rows):
    vals=[]
    for r in rows:
        c=parse_combo(r.get('combination','')); od=f(r.get('odds'))
        if c and od>0: vals.append((c,od,1/od))
    den=sum(x[2] for x in vals)
    if not den:return {},{}
    return {c:w/den for c,od,w in vals},{c:od for c,od,w in vals}
def rank_role(es,keyfn):
    return [int(e['car_no']) for e in sorted(es,key=lambda e:(-keyfn(e),-f(e.get('score')),i(e.get('car_no'))))]
def combos_for(W,Q2,Q3,nw,n2,n3):
    out=set()
    for a,b,c in product(W[:nw],Q2[:n2],Q3[:n3]):
        if len({a,b,c})==3:out.add(tuple(sorted((a,b,c))))
    return out
def role_signature(win, role_ranks):
    # Best rank among the 3 winning riders for each role axis.
    sig=[]
    for name in ('W','Q2','Q3'):
        rr=role_ranks[name]
        sig.append(min(rr[c] for c in win if c in rr))
    return tuple(sig)

def main():
    races={};E=defaultdict(list);P=defaultdict(list);O=defaultdict(list)
    for pth in sorted(ARCH.rglob('*.zip')):
        with zipfile.ZipFile(pth) as z:
            for x in read(z,'races.csv'):races.setdefault(x['race_id'],x)
            for x in read(z,'entries.csv'):E[x['race_id']].append(x)
            for x in read(z,'payouts.csv'):P[x['race_id']].append(x)
            for x in read(z,'trio_final_odds.csv'):O[x['race_id']].append(x)

    configs=[(nw,n2,n3) for nw in range(1,4) for n2 in range(1,4) for n3 in range(1,5)]
    stats={cfg:{sp:{'n':0,'points':0,'hit':0,'market':0.0,'ret':0,'eligible':0} for sp in ('train','test')} for cfg in configs}
    sigs={sp:Counter() for sp in ('train','test')}
    bucket={sp:{'W':Counter(),'Q2':Counter(),'Q3':Counter()} for sp in ('train','test')}
    styles={sp:{'W1':Counter(),'Q21':Counter(),'Q31':Counter()} for sp in ('train','test')}
    score_baseline={k:{sp:{'n':0,'points':0,'hit':0,'market':0.0,'ret':0} for sp in ('train','test')} for k in range(3,9)}

    for race in races.values():
        rid=race['race_id']
        if not day3(rid):continue
        es=[e for e in E[rid] if e.get('car_no','').isdigit()]
        if len(es)<7:continue
        win,pay=paid(P[rid]); q,odds=market(O[rid])
        if not win or not q or win not in q:continue
        try:y=int(race.get('race_date','9999')[:4])
        except:continue
        sp='train' if y<=2024 else 'test'

        W=rank_role(es,lambda e:f(e.get('win_rate')))
        Q2=rank_role(es,lambda e:max(0.0,f(e.get('top2_rate'))-f(e.get('win_rate'))))
        Q3=rank_role(es,lambda e:max(0.0,f(e.get('top3_rate'))-f(e.get('top2_rate'))))
        rr={name:{c:j+1 for j,c in enumerate(lst)} for name,lst in [('W',W),('Q2',Q2),('Q3',Q3)]}
        sigs[sp][role_signature(win,rr)]+=1
        for name in ('W','Q2','Q3'):
            ranks=sorted(rr[name][c] for c in win)
            bucket[sp][name][f'best{ranks[0]}']+=1
            bucket[sp][name][f'contains_top1']+=int(1 in ranks)
            bucket[sp][name][f'contains_top2']+=int(any(x<=2 for x in ranks))
            bucket[sp][name][f'contains_top3']+=int(any(x<=3 for x in ranks))
        bycar={int(e['car_no']):e for e in es}
        styles[sp]['W1'][bycar[W[0]].get('style','')]+=1
        styles[sp]['Q21'][bycar[Q2[0]].get('style','')]+=1
        styles[sp]['Q31'][bycar[Q3[0]].get('style','')]+=1

        for cfg in configs:
            tickets=combos_for(W,Q2,Q3,*cfg)
            tickets={c for c in tickets if c in q}
            s=stats[cfg][sp]; s['n']+=1; s['points']+=len(tickets); s['eligible']+=int(bool(tickets))
            h=int(win in tickets); s['hit']+=h; s['ret']+=pay if h else 0
            s['market']+=sum(q[c] for c in tickets)

        # line-free benchmark: every trio among top-k score riders.
        sr=rank_role(es,lambda e:f(e.get('score')))
        from itertools import combinations
        for k in range(3,9):
            tickets={tuple(sorted(c)) for c in combinations(sr[:k],3)}
            tickets={c for c in tickets if c in q}
            s=score_baseline[k][sp]; s['n']+=1;s['points']+=len(tickets)
            h=int(win in tickets);s['hit']+=h;s['ret']+=pay if h else 0;s['market']+=sum(q[c] for c in tickets)

    def summarize(v):
        n=v['n']; pts=v['points']/n if n else 0; hit=v['hit']/n if n else 0; m=v['market']/n if n else 0
        stake=v['points']*100
        return {'n':n,'avg_points':pts,'hit':hit,'market':m,'ratio':hit/m if m else 0,'diff':hit-m,'roi':v['ret']/stake if stake else 0}

    results={}
    for cfg in configs:
        results[cfg]={sp:summarize(stats[cfg][sp]) for sp in ('train','test')}
    # Candidates selected using TRAIN ONLY: 2-8 average points, strongest market-diff then ratio.
    eligible=[]
    for cfg,r in results.items():
        tr=r['train']
        if 2.0<=tr['avg_points']<=8.0:
            eligible.append((tr['diff'],tr['ratio'],tr['hit'],cfg))
    eligible.sort(reverse=True)
    selected=[x[3] for x in eligible[:12]]

    md=['# G3三日目 ライン非依存・役割軸分析','',
        '役割は事前個体成績のみ。ライン情報・予想印は候補選定に不使用。',
        '- W（勝ち切り役） = win_rate',
        '- Q2（2着残存役） = top2_rate - win_rate',
        '- Q3（3着滑り込み役） = top3_rate - top2_rate',
        '- 各レース内でW/Q2/Q3を別々に順位付け。',
        '- formation Wn × Q2m × Q3k。重複選手は除外し3人が異なる組だけ購入。','',
        'train=2022-2024 / test=2025-2026H1','']
    md.append('## 勝ち組3人に各役割上位が含まれる率')
    for sp in ('train','test'):
        n=sum(sigs[sp].values());md.append(f'### {sp}: {n}R')
        for role in ('W','Q2','Q3'):
            c=bucket[sp][role]
            md.append(f"- {role}: top1 {c['contains_top1']/n:.1%} / top2以内 {c['contains_top2']/n:.1%} / top3以内 {c['contains_top3']/n:.1%}")
        md.append(f"- 役割1位の脚質 W1={dict(styles[sp]['W1'])} / Q2-1={dict(styles[sp]['Q21'])} / Q3-1={dict(styles[sp]['Q31'])}")
    md.append('')
    md.append('## trainだけで選んだ2〜8点帯 上位フォーメーション')
    for cfg in selected:
        md.append(f"### W{cfg[0]} × Q2{cfg[1]} × Q3{cfg[2]}")
        for sp in ('train','test'):
            s=results[cfg][sp]
            md.append(f"- {sp}: {s['avg_points']:.2f}点 hit {s['hit']:.2%} / market {s['market']:.2%} / ratio {s['ratio']:.3f} / diff {s['diff']:+.2%} / ROI {s['roi']:.1%}")
    md.append('')
    md.append('## 得点上位BOX benchmark')
    for k in range(3,9):
        md.append(f'### score top{k} BOX')
        for sp in ('train','test'):
            s=summarize(score_baseline[k][sp])
            md.append(f"- {sp}: {s['avg_points']:.2f}点 hit {s['hit']:.2%} / market {s['market']:.2%} / ratio {s['ratio']:.3f} / ROI {s['roi']:.1%}")
    md.append('')
    md.append('## 勝ち組の役割順位シグネチャ上位')
    for sp in ('train','test'):
        n=sum(sigs[sp].values());md.append(f'### {sp}')
        for sig,c in sigs[sp].most_common(15):
            md.append(f'- best W/Q2/Q3 ranks={sig}: {c} ({c/n:.2%})')

    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'role_axis.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    jout={f'{a}-{b}-{c}':v for (a,b,c),v in results.items()}
    (OUT/'role_axis.json').write_text(json.dumps({'formations':jout,'selected':['-'.join(map(str,x)) for x in selected]},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('\n'.join(md))
if __name__=='__main__':main()
