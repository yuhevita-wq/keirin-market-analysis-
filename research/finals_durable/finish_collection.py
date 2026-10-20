"""Run to a verified end in a live session; commit each race, bound each batch.

Exit 0 only when the core audit has no gaps. Exit 2 preserves unresolved gaps.
No detached watchdog is used. Re-running resumes the SQLite ledger.
"""
import argparse, fcntl, itertools, json, os, sqlite3, subprocess, sys, time
from pathlib import Path
import pipeline as p

def gaps(payload):
    d=json.loads(payload or '{}')
    cars=[e['car_no'] for e in d.get('entries',[])]
    errors=[]
    if len(cars)<3 or len(set(cars))!=len(cars):return ['entries']
    if not d.get('results'):errors.append('results')
    if d.get('line',{}).get('status')!='published':errors.append('line')
    for key,expected in [('trifecta',set(itertools.permutations(cars,3))),('trio',set(itertools.combinations(sorted(cars),3)))]:
        selections=d.get(key,[])
        if len(selections)!=len(expected) or {tuple(x['cars']) for x in selections}!=expected or any(x.get('status')!='available' or not x.get('odds') for x in selections):errors.append(key)
    return errors

def report(c):
    counts={k:0 for k in ['entries','results','trifecta','trio']};missing=[]
    for rid,payload in c.execute("select race_id,payload from races where status!='excluded'"):
        d=json.loads(payload or '{}')
        for key in counts:counts[key]+=bool(d.get(key))
        errors=gaps(payload)
        if errors:missing.append({'race_id':rid,'missing':errors})
    return {'at':p.now(),'field_presence':counts,'core_gaps':missing,'all_core_records_complete':not missing,'all_finals_complete':False,'note':'Rider registration identity and historical bank applicability require separate verification.'}

def save_report(c,data):
    r=report(c);tmp=data/'finish_status.tmp';tmp.write_text(json.dumps(r,ensure_ascii=False,indent=2));tmp.replace(data/'finish_status.json')
    print(json.dumps({k:v for k,v in r.items() if k!='core_gaps'}),flush=True)
    return r

def worker(data,workers,limit):
    # Retry at the outer batch level. urllib3's five internal retries previously
    # multiplied waits and could honor an unbounded Retry-After header.
    original=p.base.make_session
    def session():
        s=original()
        from requests.adapters import HTTPAdapter
        s.mount('https://',HTTPAdapter(max_retries=0))
        get=s.get
        s.get=lambda url,**kw:get(url,**{**kw,'timeout':(8,20)})
        return s
    p.base.make_session=session
    c=p.db_open(data)
    rows=[]
    for row in c.execute("select race_id,day,url,label,grade,category,payload from races where status!='excluded' order by coalesce(updated,''),day,race_id"):
        if gaps(row[6]):
            row=list(row);row[2]=row[2].split('?')[0];rows.append(row)
            if len(rows)>=limit:break
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(p.collect_one,row,data):row[0] for row in rows}
        for future in cf.as_completed(futures):
            try:rid,status,payload,errors=future.result()
            except Exception as exc:print('worker_error',futures[future],repr(exc),flush=True);continue
            c.execute('update races set status=?,payload=?,error=?,updated=?,category=? where race_id=?',(status,p.dump(payload),p.dump(errors),p.now(),payload.get('category','unresolved'),rid));c.commit()
            print('saved',rid,flush=True)
    save_report(c,data)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=p.ROOT/'data');parser.add_argument('--worker',action='store_true');parser.add_argument('--workers',type=int,default=12);parser.add_argument('--batch',type=int,default=96);parser.add_argument('--passes',type=int,default=3);a=parser.parse_args()
    if a.worker:return worker(a.data,a.workers,a.batch)
    lock=open(a.data/'finish.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    c=p.db_open(a.data);initial=save_report(c,a.data);total=len(initial['core_gaps'])
    for batch in range(a.passes*((total+a.batch-1)//a.batch)):
        if not report(c)['core_gaps']:break
        cmd=[sys.executable,__file__,'--worker','--data',str(a.data),'--workers',str(a.workers),'--batch',str(a.batch)]
        try:
            child=subprocess.run(cmd,timeout=180)
            event={'at':p.now(),'batch':batch,'returncode':child.returncode}
        except subprocess.TimeoutExpired:event={'at':p.now(),'batch':batch,'reason':'batch_timeout_180s'}
        with open(a.data/'finish_events.jsonl','a') as f:f.write(json.dumps(event)+'\n')
        # Independent backup; no remote save credentials in the collection path.
        backup=a.data/'finish_checkpoint.tmp';dest=sqlite3.connect(backup);c.backup(dest);dest.close();backup.replace(a.data/'finish_checkpoint.sqlite')
        r=save_report(c,a.data)
    r=save_report(c,a.data)
    return 0 if r['all_core_records_complete'] else 2

if __name__=='__main__':sys.exit(main())
