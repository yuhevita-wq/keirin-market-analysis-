from __future__ import annotations

import csv, io, json, math, statistics, zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCH = ROOT / 'data' / 'grade_races' / 'g3'
OUT = ROOT / 'results' / 'g3_day3_reality'


def dec(b):
    for e in ('utf-8-sig','utf-8','cp932','shift_jis'):
        try: return b.decode(e)
        except UnicodeDecodeError: pass
    return b.decode('utf-8', errors='replace')

def read(z,n):
    xs=[x for x in z.namelist() if Path(x).name==n]
    return list(csv.DictReader(io.StringIO(dec(z.read(xs[0]))))) if xs else []

def num(v,d=float('-inf')):
    try:
        x=float(v or '')
        return x if math.isfinite(x) else d
    except: return d

def ival(v):
    try: return int(v or '')
    except: return None

def day3(rid): return len(rid)==16 and rid.isdigit() and rid[10:12]=='03'
def combo_key(xs): return '='.join(map(str,sorted(xs)))
def parse_combo(s):
    if not s: return None
    for sep in ('=','-'):
        p=s.split(sep)
        if len(p)==3 and all(x.strip().isdigit() for x in p): return tuple(sorted(int(x) for x in p))
    return None

def line_groups(es):
    by=defaultdict(list); assigned=set()
    for e in es:
        lid=ival(e.get('line_id')); pos=ival(e.get('line_position')); car=ival(e.get('car_no'))
        if lid is not None and pos is not None and car is not None:
            by[lid].append(e); assigned.add(car)
    for lid in list(by): by[lid].sort(key=lambda e: ival(e.get('line_position')) or 99)
    singles=[]
    for lid,ms in by.items():
        if len(ms)==1: singles.append(ms[0])
    for e in es:
        car=ival(e.get('car_no'))
        if car is not None and car not in assigned: singles.append(e)
    seen=set(); singles2=[]
    for e in singles:
        c=ival(e.get('car_no'))
        if c is not None and c not in seen: seen.add(c); singles2.append(e)
    multi=[]
    for lid,ms in by.items():
        if len(ms)<2: continue
        strength=num(ms[0].get('score'))+num(ms[1].get('score'))
        key=(strength,num(ms[0].get('score')),1 if len(ms)>=3 else 0,-lid)
        multi.append((key,lid,ms))
    multi.sort(reverse=True,key=lambda x:x[0])
    return multi,singles2

def paid_trio(ps):
    for p in ps:
        if p.get('ticket_type')=='3連複' and p.get('status')=='paid':
            c=parse_combo(p.get('combination',''))
            if c:
                try: pay=int(float(p.get('payout_yen') or 0))
                except: pay=0
                return c,pay
    return None,0

def market_probs(rows):
    raw=[]
    for o in rows:
        c=parse_combo(o.get('combination','')); od=num(o.get('odds'))
        if c and od>0 and math.isfinite(od): raw.append((c,od,1.0/od))
    den=sum(x[2] for x in raw)
    if den<=0: return {},{}
    q={c:w/den for c,od,w in raw}
    odds={c:od for c,od,w in raw}
    return q,odds

def cat_flags(c, ctx):
    s=set(c); main=ctx['main']; second=ctx['second']; weak=ctx['weak']; singles=ctx['singles']
    flags={}
    if main:
        ab=set(main[:2]); flags['AB_pair']=ab<=s
        flags['AB_plus_outside']=ab<=s and any(x not in main for x in s)
        flags['main_tail_any']=any(x in s for x in main[2:])
    else:
        flags['AB_pair']=flags['AB_plus_outside']=flags['main_tail_any']=False
    if second:
        cd=set(second[:2]); flags['CD_pair']=cd<=s
        flags['second_tail_any']=any(x in s for x in second[2:])
    else:
        flags['CD_pair']=flags['second_tail_any']=False
    if weak:
        flags['weak_pair']=set(weak[:2])<=s
        flags['weak_any']=any(x in s for x in weak)
        flags['weak_leader_any']=weak[0] in s
        flags['weak_second_any']=weak[1] in s if len(weak)>1 else False
        flags['weak_tail_any']=any(x in s for x in weak[2:])
    else:
        flags['weak_pair']=flags['weak_any']=flags['weak_leader_any']=flags['weak_second_any']=flags['weak_tail_any']=False
    nsg=sum(x in singles for x in s)
    flags['singleton_any']=nsg>=1
    flags['singleton_2plus']=nsg>=2
    units=[]
    for x in s:
        unit=('S',x)
        for idx,line in enumerate(ctx['all_lines']):
            if x in line: unit=('L',idx); break
        units.append(unit)
    flags['three_units']=len(set(units))==3
    flags['any_non_top2_line']=any((x in ln) for ln in ctx['all_lines'][2:] for x in s) if len(ctx['all_lines'])>=3 else False
    return flags

def exact_ticket_stats(records, name):
    out={}
    for split in ('train','test'):
        rs=[r for r in records[name] if r['split']==split]
        n=len(rs); hits=sum(r['hit'] for r in rs); mq=sum(r['q'] for r in rs)
        st=n*100; ret=sum(r['pay'] if r['hit'] else 0 for r in rs)
        ods=[r['odds'] for r in rs if r['odds'] is not None]
        out[split]={
            'eligible':n,'hit_rate':hits/n if n else 0,'market_prob':mq/n if n else 0,
            'edge_ratio':(hits/mq if mq else 0),'roi':ret/st if st else 0,
            'median_odds':statistics.median(ods) if ods else None
        }
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    races={}; E=defaultdict(list); P=defaultdict(list); O=defaultdict(list)
    for pth in sorted(ARCH.rglob('*.zip')):
        with zipfile.ZipFile(pth) as z:
            for x in read(z,'races.csv'): races.setdefault(x['race_id'],x)
            for x in read(z,'entries.csv'): E[x['race_id']].append(x)
            for x in read(z,'payouts.csv'): P[x['race_id']].append(x)
            for x in read(z,'trio_final_odds.csv'): O[x['race_id']].append(x)

    cats=defaultdict(lambda: defaultdict(lambda:{'n':0,'actual':0,'market':0.0}))
    exact=defaultdict(list)
    struct=defaultdict(Counter)
    singleton_rider=defaultdict(lambda:{'n':0,'actual_top3':0,'actual_win':0,'market_top3':0.0})
    valid=0

    for race in races.values():
        rid=race['race_id']
        if not day3(rid): continue
        lines,single_entries=line_groups(E[rid])
        if not lines: continue
        win,pay=paid_trio(P[rid]); q,odds=market_probs(O[rid])
        if not win or not q or win not in q: continue
        split='train' if int(race.get('race_date','9999')[:4])<=2024 else 'test'
        all_lines=[[ival(e.get('car_no')) for e in ms if ival(e.get('car_no')) is not None] for _,_,ms in lines]
        singles=[ival(e.get('car_no')) for e in single_entries if ival(e.get('car_no')) is not None]
        main=all_lines[0] if all_lines else []
        second=all_lines[1] if len(all_lines)>=2 else []
        weak=all_lines[-1] if all_lines else []
        ctx={'main':main,'second':second,'weak':weak,'singles':singles,'all_lines':all_lines}
        valid+=1
        struct[split]['races']+=1
        struct[split][f'main_size_{len(main)}']+=1
        struct[split][f'multi_lines_{len(all_lines)}']+=1
        struct[split][f'singletons_{len(singles)}']+=1
        if singles: struct[split]['races_with_singleton']+=1
        if len(all_lines)>=2:
            struct[split]['races_2plus_lines']+=1
            if set(main[:2])<=set(win): struct[split]['AB_pair_actual']+=1
        if weak:
            if any(x in win for x in weak): struct[split]['weak_any_actual']+=1
            if set(weak[:2])<=set(win): struct[split]['weak_pair_actual']+=1
            if weak[0] in win: struct[split]['weak_leader_top3']+=1
            if weak[1] in win: struct[split]['weak_second_top3']+=1
            if win[0] in weak: struct[split]['weak_winner']+=1
        if singles:
            if any(x in win for x in singles): struct[split]['singleton_any_actual']+=1
            if win[0] in singles: struct[split]['singleton_winner']+=1

        # Category calibration: actual winning-category rate versus market-implied category probability.
        win_flags=cat_flags(win,ctx)
        market_cat=defaultdict(float)
        for c,prob in q.items():
            for k,v in cat_flags(c,ctx).items():
                if v: market_cat[k]+=prob
        for k in win_flags:
            rec=cats[k][split]; rec['n']+=1; rec['actual']+=int(win_flags[k]); rec['market']+=market_cat.get(k,0.0)

        # Rider-level singleton calibration for top-3 inclusion.
        for s in singles:
            key=split
            rr=singleton_rider[key]; rr['n']+=1; rr['actual_top3']+=int(s in win); rr['actual_win']+=int(win[0]==s)
            rr['market_top3']+=sum(prob for c,prob in q.items() if s in c)

        # Exact structural tickets. Missing roles are simply not eligible.
        candidates={}
        if len(main)>=2:
            A,B=main[:2]
            if len(main)>=3: candidates['AB_main3']=(A,B,main[2])
            if len(second)>=1: candidates['AB_rank2_leader']=(A,B,second[0])
            if len(second)>=2: candidates['AB_rank2_second']=(A,B,second[1])
            if weak and weak!=second:
                candidates['AB_weak_leader']=(A,B,weak[0])
                candidates['AB_weak_second']=(A,B,weak[1])
            if singles:
                best=max(single_entries,key=lambda e:(num(e.get('score')),-(ival(e.get('car_no')) or 99)))
                candidates['AB_best_singleton']=(A,B,ival(best.get('car_no')))
        if len(second)>=2:
            C,D=second[:2]
            if main: candidates['CD_A']=(C,D,main[0])
            if len(main)>=2: candidates['CD_B']=(C,D,main[1])
            if len(second)>=3: candidates['CD_rank2_3rd']=(C,D,second[2])
            if weak and weak!=second:
                candidates['CD_weak_leader']=(C,D,weak[0])
                candidates['CD_weak_second']=(C,D,weak[1])
            if singles:
                best=max(single_entries,key=lambda e:(num(e.get('score')),-(ival(e.get('car_no')) or 99)))
                candidates['CD_best_singleton']=(C,D,ival(best.get('car_no')))
        if weak and len(weak)>=2:
            W1,W2=weak[:2]
            if main: candidates['WEAKPAIR_A']=(W1,W2,main[0])
            if len(main)>=2: candidates['WEAKPAIR_B']=(W1,W2,main[1])
            if singles:
                best=max(single_entries,key=lambda e:(num(e.get('score')),-(ival(e.get('car_no')) or 99)))
                candidates['WEAKPAIR_best_singleton']=(W1,W2,ival(best.get('car_no')))

        for name,t in candidates.items():
            if None in t or len(set(t))<3: continue
            c=tuple(sorted(t))
            if c not in q: continue
            exact[name].append({'split':split,'hit':int(c==win),'q':q[c],'odds':odds.get(c),'pay':pay})

    out={'valid_races':valid,'structure':{},'categories':{},'singleton_rider':{},'exact_tickets':{}}
    md=['# G3三日目 市場残差分析','', '頻度ではなく、実現率と正規化市場期待確率の差を見る。train=2022-2024 / test=2025-2026H1','']

    md.append('## ライン長・最弱ライン・単騎の実現率')
    for split in ('train','test'):
        c=struct[split]; n=c['races']; out['structure'][split]=dict(c)
        md.append(f'### {split}: {n}R')
        sizes=', '.join(f'{k.replace("main_size_","")}車={v}R' for k,v in sorted(c.items()) if k.startswith('main_size_'))
        md.append(f'- 本命ライン長: {sizes}')
        if n:
            md.append(f"- 最弱ライン1人以上TOP3: {c['weak_any_actual']/n:.1%} / 最弱ライン先頭TOP3 {c['weak_leader_top3']/n:.1%} / 番手TOP3 {c['weak_second_top3']/n:.1%} / 最弱ライン勝者 {c['weak_winner']/n:.1%} / 最弱ペア同時TOP3 {c['weak_pair_actual']/n:.1%}")
        sr=c['races_with_singleton']
        if sr:
            md.append(f"- 単騎あり {sr}R: 単騎1人以上TOP3 {c['singleton_any_actual']/sr:.1%} / 単騎勝者 {c['singleton_winner']/sr:.1%}")
        md.append('')

    md.append('## 構造カテゴリ 市場較正')
    order=['AB_pair','AB_plus_outside','main_tail_any','CD_pair','second_tail_any','weak_pair','weak_any','weak_leader_any','weak_second_any','weak_tail_any','singleton_any','singleton_2plus','three_units','any_non_top2_line']
    for k in order:
        out['categories'][k]={}; md.append(f'### {k}')
        for split in ('train','test'):
            r=cats[k][split]; n=r['n']; a=r['actual']/n if n else 0; m=r['market']/n if n else 0; ratio=a/m if m else 0
            out['categories'][k][split]={'races':n,'actual_rate':a,'market_rate':m,'edge_ratio':ratio,'diff':a-m}
            md.append(f'- {split}: actual {a:.1%} / market {m:.1%} / actual-market {(a-m)*100:+.2f}pt / ratio {ratio:.3f}')
        md.append('')

    md.append('## 単騎 rider-level 市場較正')
    for split in ('train','test'):
        r=singleton_rider[split]; n=r['n']; a=r['actual_top3']/n if n else 0; w=r['actual_win']/n if n else 0; m=r['market_top3']/n if n else 0
        out['singleton_rider'][split]={'starts':n,'top3_rate':a,'win_rate':w,'market_top3_rate':m,'edge_ratio':a/m if m else 0}
        md.append(f'- {split}: 単騎延べ{n}人 top3 {a:.1%} / win {w:.1%} / 市場top3期待 {m:.1%} / ratio {(a/m if m else 0):.3f}')
    md.append('')

    md.append('## 代表的な1点構造 市場期待との差と実ROI')
    for name in sorted(exact):
        s=exact_ticket_stats(exact,name); out['exact_tickets'][name]=s; md.append(f'### {name}')
        for split in ('train','test'):
            r=s[split]; med='-' if r['median_odds'] is None else f"{r['median_odds']:.1f}"
            md.append(f"- {split}: n={r['eligible']} hit {r['hit_rate']:.2%} / market {r['market_prob']:.2%} / ratio {r['edge_ratio']:.3f} / ROI {r['roi']:.1%} / median odds {med}")
        md.append('')

    (OUT/'market_residuals.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'market_residuals.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('\n'.join(md))

if __name__=='__main__': main()
