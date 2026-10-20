"""Freeze a pilot-based, day-clustered sample-size plan; never stop on significance.
This is a Keirin adaptation of the paper's conditional-return relationship,
not a proof of profitable prospective trading or a full JRA replication.
"""
import argparse,gzip,hashlib,json,math,os,re,subprocess,sys
from collections import Counter,defaultdict
from datetime import datetime,timedelta,timezone
from pathlib import Path
from statistics import NormalDist

POOLS=('umaren','umatan','trio')
POWER_TAG='path-odds-power-plan-v1'
INDEX='path-odds-archive-index-v1'
MDE=0.034  # Reference: paper Table 2, ~3.4 percentage points per 10% odds movement.
ALPHA_FAMILY=0.05
POWER=0.90
MIN_DAYS=30  # Practical floor for day-cluster variance planning, not sufficiency by itself.

def canonical(pool,key):
    p=key.replace('=','-').split('-')
    if pool in ('umaren','trio'):p=sorted(p,key=int)
    return '-'.join(p)

def identity(r):return f"{r['race_date']}|{r['track']}|{r['race_no']}"

def pairs(r,pool):
    official=r.get('official',{})
    if pool not in official.get('complete_pools',[]) or not official.get('deadline'):return [],'official_missing'
    if official.get('status')!='complete':return [],'official_pending'
    end=datetime.fromisoformat(official['deadline']);target=end-timedelta(minutes=5)
    candidates=[];finals=[]
    for f in r.get('pools',{}).get(pool,{}).get('frames',[]):
        if f.get('final'):finals.append(f);continue
        try:
            hh,mm=map(int,f['label'].split(':'));t=end.replace(hour=hh,minute=mm,second=0,microsecond=0)
        except (ValueError,KeyError):continue
        if t<=target and (target-t).total_seconds()<=120:candidates.append((t,f))
    if not candidates:return [],'minus5_missing_or_stale'
    if len(finals)!=1:return [],'final_missing_or_ambiguous'
    pre=max(candidates,key=lambda x:x[0])[1]
    a={canonical(pool,k):float(v) for k,v in pre['odds'].items()}
    b={canonical(pool,k):float(v) for k,v in finals[0]['odds'].items()}
    pays={canonical(pool,k):v for k,v in official['payouts_per_100_yen'][pool].items()}
    out=[]
    for ticket in sorted(a.keys()&b.keys()):
        if a[ticket]>=1 and b[ticket]>=1 and math.isfinite(a[ticket]+b[ticket]):
            out.append((b[ticket],b[ticket]/a[ticket]-1,float(pays.get(ticket,0))/100))
    return out,'eligible' if out else 'no_paired_tickets'

def counts(records,plan):
    pilot=set(plan.get('pilot_race_ids',[]));n=Counter();days={p:Counter() for p in POOLS}
    for r in records:
        if identity(r) in pilot:continue
        for p in POOLS:
            rows,_=pairs(r,p)
            if rows:n[p]+=1;days[p][r['race_date']]+=1
    return {p:{'races':n[p],'days':len(days[p]),'date_counts':dict(days[p])} for p in POOLS}

def gate(plan,available):
    return bool(plan.get('frozen') and all(available.get(p,{}).get('races',0)>=plan['targets'][p]['races'] and available.get(p,{}).get('days',0)>=plan['targets'][p]['days'] for p in POOLS))

def estimate(records,pool):
    import numpy as np
    rows=[];ids=set();reasons=Counter()
    for r in records:
        observations,reason=pairs(r,pool);reasons[reason]+=1
        if not observations:continue
        ids.add(identity(r))
        for price,change,y in observations:rows.append((r,price,change,y))
    days=sorted({r[0]['race_date'] for r in rows});n=len(ids)
    out={'pilot_races':n,'pilot_days':len(days),'paired_ticket_observations':len(rows),'eligibility':dict(reasons)}
    if len(days)<MIN_DAYS or n<100:
        out['status']='need_more_pilot_clusters';return out
    prices=np.array([r[1] for r in rows]);median=float(np.median(prices));scaled=prices/median
    change=np.array([r[2] for r in rows]);y=np.array([r[3] for r in rows])
    # Paper-style conditional-return equation with available nuisance controls.
    categories=[]
    for func in [lambda r:r['track'],lambda r:r.get('grade','unknown'),lambda r:len(r.get('cars',[])),lambda r:r['race_date'][:7]]:
        values=[str(func(row[0])) for row in rows];levels=sorted(set(values))
        categories.extend([np.array([v==level for v in values],dtype=float) for level in levels[1:]])
    Z=np.column_stack([np.ones(len(rows)),scaled,*categories]);D=np.column_stack([change,change*scaled])
    X=D-Z@np.linalg.lstsq(Z,D,rcond=None)[0];yr=y-Z@np.linalg.lstsq(Z,y,rcond=None)[0]
    if np.linalg.matrix_rank(X)<2:out['status']='insufficient_price_variation';return out
    beta=np.linalg.lstsq(X,yr,rcond=None)[0];resid=yr-X@beta
    bread=np.linalg.pinv(X.T@X);contrast=np.array([-.1,-.1])
    scores=[]
    day_array=np.array([r[0]['race_date'] for r in rows])
    for day in days:
        mask=day_array==day;scores.append(float(contrast@bread@(X[mask].T@resid[mask])))
    squares=np.square(scores);G=len(days);df=len(rows)-np.linalg.matrix_rank(Z)-2
    if df<=0:out['status']='insufficient_residual_df';return out
    correction=G/(G-1)*(len(rows)-1)/df
    var=float(sum(squares)*correction)
    # Upper pilot variance bound from day-resampling, used only for planning.
    rng=np.random.default_rng(250914645)
    boot=squares[rng.integers(0,G,size=(2000,G))].sum(axis=1)*correction
    upper=float(np.quantile(boot,.95));upper=max(var,upper)
    if not math.isfinite(upper) or upper<=0:out['status']='variance_unidentified';return out
    z=NormalDist().inv_cdf(1-ALPHA_FAMILY/(2*len(POOLS)))+NormalDist().inv_cdf(POWER)
    targets={str(effect):max(1,math.ceil(n*z*z*upper/(effect*effect))) for effect in [.02,MDE,.05,.10]}
    required=targets[str(MDE)]
    # Preserve at least the pilot's day-cluster support in the independent holdout.
    required_days=max(MIN_DAYS,math.ceil(G*required/n))
    out.update(status='estimated',planning_races=required,planning_days=required_days,median_final_odds=median,
      contrast_se=math.sqrt(var),upper_planning_se=math.sqrt(upper),sensitivity_required_races=targets,
      model='return ~ final_odds + change + final_odds*change + track/grade/car_count/year_month fixed effects',
      warning='Planning approximation assumes comparable covariance, clustering, and price support in the independent period. No observed significance or effect sign is used for stopping.')
    return out

def load_archives(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    p=subprocess.run(['gh','api','--paginate',f"repos/{os.environ['GH_REPO']}/releases?per_page=100"],capture_output=True,check=True)
    raw=p.stdout.decode();dec=json.JSONDecoder();releases=[]
    while raw.strip():
        value,end=dec.raw_decode(raw.lstrip());releases.extend(value);raw=raw.lstrip()[end:]
    records=[]
    for release in releases:
        if not re.fullmatch(r'path-odds-\d{4}-\d{2}',release.get('tag_name','')):continue
        directory=root/release['tag_name'];directory.mkdir(exist_ok=True)
        for a in release.get('assets',[]):
            if not re.fullmatch(r'\d{4}-\d{2}-\d{2}\.json\.gz',a['name']):continue
            subprocess.run(['gh','release','download',release['tag_name'],'--pattern',a['name'],'--dir',str(directory),'--clobber'],capture_output=True,check=True)
            data=json.loads(gzip.decompress((directory/a['name']).read_bytes()));records.extend(data.values())
    unique={identity(r):r for r in records};return list(unique.values())

def audit(args):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'odds_path_archive'))
    from collect import Store
    store=Store(args.output,True);existing=store.load(POWER_TAG,'sample_plan.json.gz',{})
    records=load_archives(Path(args.output)/'downloaded')
    if existing.get('frozen'):
        plan=existing
    else:
        estimates={p:estimate(records,p) for p in POOLS}
        ready=all(v.get('status')=='estimated' for v in estimates.values())
        plan=dict(schema=1,frozen=ready,created_at=datetime.now(timezone.utc).isoformat(),alpha_family=ALPHA_FAMILY,power=POWER,
          minimum_detectable_return_difference=MDE,reference='Paper Table 2: approximately 3.4 percentage points per 10% relative odds decline; this is a reference-sized effect, not a measured Keirin effect.',
          timestamp_rule='Latest observed quote at or before betting cutoff minus 5 minutes, at most 120 seconds stale; final quote for retrospective controls only.',
          scope='Three accepted pools; tickets observed at both time points. Does not generalize to missing tails or demonstrate prospective profits.',
          estimates=estimates,targets={p:{'races':v['planning_races'],'days':v['planning_days']} for p,v in estimates.items()} if ready else {},
          pilot_race_ids=sorted(identity(r) for r in records) if ready else [],
          stopping_rule='Once frozen, collect independent races until all three race/day targets are met, irrespective of effect sign or p-values. Do not refit or shrink the target based on significance.')
        if ready:plan['plan_id']=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
        store.save(POWER_TAG,'sample_plan.json.gz',plan)
    available=counts(records,plan) if plan.get('frozen') else {p:{'races':0,'days':0} for p in POOLS}
    status=dict(frozen=plan.get('frozen',False),plan_id=plan.get('plan_id'),coverage=available,targets=plan.get('targets',{}),sufficient=gate(plan,available),
      recorded_races=len(records),pilot_estimates=plan.get('estimates',{}),updated_at=datetime.now(timezone.utc).isoformat())
    store.save(POWER_TAG,'sample_status.json.gz',status)
    # Machine-readable progress summary in authenticated draft release metadata.
    readable=Path(args.output)/'status_notes.md';readable.write_text('Sample-size planning and independent collection status\n\n```json\n'+json.dumps(status,ensure_ascii=False,indent=2)+'\n```')
    subprocess.run(['gh','release','edit',POWER_TAG,'--notes-file',str(readable)],check=True)
    print(json.dumps(status,ensure_ascii=False,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='work/path_odds_power');a=p.parse_args();audit(a)

if __name__=='__main__':main()
