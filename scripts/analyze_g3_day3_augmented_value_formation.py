from __future__ import annotations
import csv, io, json, math, zipfile
from collections import defaultdict
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
def f(v,d=float('-inf')):
    try:
        x=float(v or '');return x if math.isfinite(x) else d
    except:return d
def day3(r):return len(r)==16 and r.isdigit() and r[10:12]=='03'
def tk(*xs):return '='.join(map(str,sorted(xs)))
def lns(es):
    by=defaultdict(list)
    for e in es:
        if e.get('line_id','').isdigit() and e.get('line_position','').isdigit():by[int(e['line_id'])].append(e)
    out=[]
    for lid,ms in by.items():
        ms=sorted(ms,key=lambda x:int(x['line_position']))
        if len(ms)>=2:out.append((f(ms[0].get('score'))+f(ms[1].get('score')),f(ms[0].get('score')),lid,ms))
    out.sort(key=lambda x:(-x[0],-x[1],x[2]));return by,out
def probs(od):
    inv={}
    for c,r in od.items():
        try:o=float(r.get('odds') or 0)
        except:o=0
        if o>0:inv[c]=1/o
    s=sum(inv.values());return {c:v/s for c,v in inv.items()} if s else {}
def paid(ps):
    for p in ps:
        if p.get('ticket_type')=='3連複' and p.get('status')=='paid':
            try:return p.get('combination'),int(float(p.get('payout_yen') or 0))
            except:return p.get('combination'),0
    return None,0

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    races={};E=defaultdict(list);P=defaultdict(list);O=defaultdict(dict)
    for pth in sorted(ARCH.rglob('*.zip')):
        with zipfile.ZipFile(pth) as z:
            for x in read(z,'races.csv'):races.setdefault(x['race_id'],x)
            for x in read(z,'entries.csv'):E[x['race_id']].append(x)
            for x in read(z,'payouts.csv'):P[x['race_id']].append(x)
            for x in read(z,'trio_final_odds.csv'):O[x['race_id']][x.get('combination','')]=x
    st=defaultdict(lambda:defaultdict(lambda:{'n':0,'pts':0,'hit':0,'m':0.0,'ret':0,'abfail_n':0,'abfail_hit':0,'offwin_n':0,'offwin_hit':0,'deep_n':0,'deep_hit':0}))
    for race in races.values():
        rid=race['race_id']
        if not day3(rid):continue
        win,pay=paid(P[rid]);od=O[rid]
        if not win or not od:continue
        by,rl=lns(E[rid])
        if len(rl)<2:continue
        mp=probs(od);split='train' if int(race['race_date'][:4])<=2024 else 'test'
        m1,m2=rl[0][3],rl[1][3]
        A,B=int(m1[0]['car_no']),int(m1[1]['car_no']);C,D=int(m2[0]['car_no']),int(m2[1]['car_no'])
        T=int(m1[2]['car_no']) if len(m1)>=3 else None;U=int(m2[2]['car_no']) if len(m2)>=3 else None
        top4={A,B,C,D};winset=set(map(int,win.split('=')))
        singles=[]
        for e in E[rid]:
            car=int(e['car_no']);lid=int(e['line_id']) if e.get('line_id','').isdigit() else None
            if lid is None or len(by.get(lid,[]))<2:singles.append(car)
        lower=rl[2:]
        lower_all=[int(e['car_no']) for _,_,_,ms in lower for e in ms]
        lower_leaders=[int(ms[0]['car_no']) for _,_,_,ms in lower if ms]
        lower_seconds=[int(ms[1]['car_no']) for _,_,_,ms in lower if len(ms)>=2]
        lower_tails=[int(e['car_no']) for _,_,_,ms in lower for e in ms[2:]]
        def base():
            xs=set()
            if T is not None:
                xs.add(tk(A,B,T));xs.add(tk(B,D,T))
            if U is not None:
                xs.add(tk(C,D,U));xs.add(tk(B,C,U))
            return xs
        def cd_with(cars):return {tk(C,D,x) for x in cars if x not in (C,D)}
        def same_with(cars):
            out=set()
            for x in cars:
                if x not in (A,B):out.add(tk(A,B,x))
                if x not in (C,D):out.add(tk(C,D,x))
            return out
        B0=base()
        forms={
          'BASE4':B0,
          'BASE+CD_SINGLE':B0|cd_with(singles),
          'BASE+CD_LOWER':B0|cd_with(lower_all),
          'BASE+CD_LOWER_LEAD':B0|cd_with(lower_leaders),
          'BASE+CD_SINGLE+LOWER_LEAD':B0|cd_with(singles)|cd_with(lower_leaders),
          'BASE+SAME_LOWER_LEAD':B0|same_with(lower_leaders),
          'BASE+SAME_SINGLE':B0|same_with(singles),
          'BASE+CD_RESID':B0|cd_with(singles+lower_all),
          'BASE+CD_BACK':B0|cd_with(singles+lower_tails),
          'BASE+CD_LEADSECOND':B0|cd_with(lower_leaders+lower_seconds),
        }
        # diagnostics
        abpair={A,B}.issubset(winset);cdpair={C,D}.issubset(winset)
        winner_car=None
        # result combo unordered, use payout/result winner unavailable here; approximate off-main as no A/B? leave structural diagnostics only
        deep=not abpair and not cdpair
        for name,combos in forms.items():
            combos={x for x in combos if x in mp}
            if not combos:continue
            s=st[name][split];s['n']+=1;s['pts']+=len(combos);h=win in combos;s['hit']+=int(h);s['m']+=sum(mp[x] for x in combos);s['ret']+=pay if h else 0
            if not abpair:s['abfail_n']+=1;s['abfail_hit']+=int(h)
            if deep:s['deep_n']+=1;s['deep_hit']+=int(h)
    md=['# G3三日目 価値核＋単騎・下位ライン拡張','', 'BASE4=ABT, BDT, CDU, BCU（T/U存在時のみ）。ここへ残差枝を追加。','']
    out={}
    for name in forms.keys():
        md.append(f'## {name}');out[name]={}
        for split in ('train','test'):
            s=st[name][split];n=s['n'];pts=s['pts'];hit=s['hit'];actual=hit/n if n else 0;market=s['m']/n if n else 0;ratio=actual/market if market else 0;roi=s['ret']/(pts*100) if pts else 0;avg=pts/n if n else 0
            abf=s['abfail_hit']/s['abfail_n'] if s['abfail_n'] else 0;deep=s['deep_hit']/s['deep_n'] if s['deep_n'] else 0
            out[name][split]={'n':n,'avg_points':avg,'hit_rate':actual,'market_rate':market,'ratio':ratio,'roi':roi,'ab_fail_hit':abf,'deep_hit':deep}
            md.append(f"- {split}: n={n} pts={avg:.2f} hit={actual:.2%} market={market:.2%} ratio={ratio:.3f} ROI={roi:.1%} / AB崩れ={abf:.2%} / AB・CD両崩れ={deep:.2%}")
        md.append('')
    (OUT/'augmented_value_formation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'augmented_value_formation.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('\n'.join(md))
if __name__=='__main__':main()
