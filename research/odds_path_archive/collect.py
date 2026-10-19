"""Restartable historical 3-pool acquisition. No betting, fitting, or imputation.
Persistence uses authenticated GitHub draft Release assets, never expiring artifacts.
"""
import argparse,gzip,hashlib,json,math,os,re,subprocess,sys,tempfile,time
from datetime import date,datetime,timedelta,timezone
from pathlib import Path
from urllib.parse import urlencode

POOLS={'umaren':'2車複','umatan':'2車単','trio':'3連複'}
JST=timezone(timedelta(hours=9))
INDEX='path-odds-archive-index-v1'

def run(args,check=True):
    p=subprocess.run(args,capture_output=True)
    if check and p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[-600:])
    return p

class Store:
    def __init__(self,root,github=False):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.github=github;self.tags=set()
    def ensure(self,tag):
        if not self.github or tag in self.tags:return
        p=run(['gh','release','view',tag],False)
        if p.returncode:
            run(['gh','release','create',tag,'--draft','--title',tag,'--notes','Private research acquisition checkpoints. Historical odds and official settlement provenance; not a betting recommendation.','--target','main'])
        self.tags.add(tag)
    def load(self,tag,name,default):
        dest=self.root/name
        if self.github:
            self.ensure(tag)
            listing=json.loads(run(['gh','release','view',tag,'--json','assets']).stdout)
            if not any(a['name']==name for a in listing['assets']):return default
            run(['gh','release','download',tag,'--pattern',name,'--dir',str(self.root),'--clobber'])
        if not dest.exists():return default
        return json.loads(gzip.decompress(dest.read_bytes()))
    def save(self,tag,name,payload):
        dest=self.root/name;tmp=dest.with_suffix('.tmp')
        tmp.write_bytes(gzip.compress(json.dumps(payload,ensure_ascii=False,separators=(',',':')).encode(),mtime=0));tmp.replace(dest)
        if self.github:
            self.ensure(tag);run(['gh','release','upload',tag,str(dest),'--clobber'])

class Source:
    def __init__(self,pause=1.5):self.last=0.;self.pause=pause
    def get(self,url):
        for attempt in range(3):
            time.sleep(max(0,self.pause-(time.monotonic()-self.last)));self.last=time.monotonic()
            p=run(['curl','-L','--fail','--max-time','30','--retry','0','-sS',url],False)
            if p.returncode==0:return p.stdout,datetime.now(JST).isoformat()
            if attempt<2:time.sleep(2**attempt)
        raise RuntimeError('HTTP retrieval failed: '+url+' '+p.stderr.decode(errors='replace')[-160:])
    def api(self,day,track=None,race=None):
        q=dict(race_kind='kr',race_date=day,display_times=24,page=1)
        if track is not None:q['race_track']=track
        if race is not None:q['race_no']=race
        url='https://keirinodds.com/odds?'+urlencode(q);raw,received=self.get(url);x=json.loads(raw)
        if 'kr_info' not in x:raise ValueError('Missing kr_info: '+str(x)[:200])
        return x,dict(url=url,received_at=received,sha256=hashlib.sha256(raw).hexdigest())

def matched(x,day,track,race):
    for r in x.get('kr_info',{}).get('race_info') or []:
        i=r['race_info']
        if i['race_date']==day and i['track']==track and int(i['race'])==race:return r
    raise ValueError('Requested race not present; never accept fallback dates/races')

def normalize(r,source):
    i=r['race_info'];od=r['odds_info'];cars=sorted(int(k) for k in r.get('horse_info',{}))
    if len(cars)<3:raise ValueError('Incomplete rider roster')
    n=len(cars);expected={'umaren':math.comb(n,2),'umatan':n*(n-1),'trio':math.comb(n,3)}
    out=dict(source_race_id=i['id'],race_date=i['race_date'],track=i['track'],race_no=i['race'],race_type=i.get('syumoku'),grade=i.get('grade'),cars=cars,
      start_time=i['race_date_time'],official_url=r.get('link_info',{}).get('kdream'),quote_source=source,pools={},
      admission_rule='Use only combinations actually observed before the decision. Up to 50; no final-rank selection or missing-as-loss.')
    for pool,label in POOLS.items():
        frames=od.get('time_pops',{}).get(pool,[]);times=od.get('time_odds_times',{}).get(pool,[])
        if len(frames)!=len(times):raise ValueError('Mismatched quote times and arrays')
        records=[]
        for stamp,values in zip(times,frames):
            if len(values)>50:raise ValueError('Source exceeded accepted 50-combination scope; audit before changing rules')
            invalid={};good={}
            for ticket,v in values.items():
                parts=ticket.split('-');want=3 if pool=='trio' else 2
                if len(parts)!=want or len(set(parts))!=want or any(not p.isdigit() or int(p) not in cars for p in parts):raise ValueError('Invalid combination')
                if not isinstance(v,(int,float)) or not math.isfinite(v) or v<1:invalid[ticket]=v
                else:good[ticket]=v
            records.append(dict(label=stamp,final=stamp=='最終',odds=good,invalid_quotes=invalid,stored_count=len(values),expected_count=expected[pool],complete=len(good)==expected[pool]))
        out['pools'][pool]=dict(name=label,frames=records)
    if not any(v['frames'] for v in out['pools'].values()):raise ValueError('No target-pool observations')
    return out

def official(raw,source,day):
    h=raw.decode('utf-8');m=re.search(r'<dt class="bet">.*?</dt>(.{0,220})',h,re.S)
    cutoff=re.search(r'(\d{2}:\d{2})',re.sub('<[^>]+>',' ',m[1])) if m else None
    deadline=day+'T'+cutoff[1]+':00+09:00' if cutoff else None
    table=re.search(r'<table class="refund_table">(.*?)</table>',h,re.S);payouts={};complete=[]
    if table:
        trs=re.findall(r'<tr>(.*?)</tr>',table[1],re.S);cells=[re.findall(r'<(?:td|th)\b[^>]*>(.*?)</(?:td|th)>',t,re.S) for t in trs]
        if list(map(len,cells))==[11,6]:
            for pool,cell in [('umaren',cells[0][5]),('umatan',cells[1][3]),('trio',cells[0][8])]:
                pairs=re.findall(r'<dt>([1-9](?:[-=][1-9]){1,2})</dt>\s*<dd>([\d,]+)円',cell)
                if pairs:payouts[pool]={k.replace('=','-'):int(v.replace(',','')) for k,v in pairs};complete.append(pool)
    return dict(source=source,deadline=deadline,payouts_per_100_yen=payouts,complete_pools=complete,
      status='complete' if deadline and len(complete)==3 else 'pending',
      evidence_refund_html=table[0] if table else None,evidence_cutoff_html=m[0] if m else None)

def metrics(record):
    pools=record.get('pools',{});frames=[f for p in pools.values() for f in p['frames']]
    return dict(races=1,interim_races=int(any(not f['final'] for f in frames)),frames=len(frames),observations=sum(len(f['odds']) for f in frames),official_complete=int(record.get('official',{}).get('status')=='complete'))

def collect(args):
    store=Store(args.output,args.github);source=Source(args.pause)
    state=store.load(INDEX,'state.json.gz',dict(schema=1,next_date=args.end,start_date=args.start,end_date=args.end,pending_dates=[],days={},totals={k:0 for k in ['races','interim_races','frames','observations','official_complete']},warnings=[]))
    if state['start_date']!=args.start:raise ValueError('Existing archive start differs; do not silently reset')
    if args.end>state['end_date']:
        # Queue newly requested days without losing the historical cursor.
        d=date.fromisoformat(state['end_date'])+timedelta(days=1)
        while d<=date.fromisoformat(args.end):state['pending_dates'].append(d.isoformat());d+=timedelta(days=1)
        state['end_date']=args.end
    until=time.monotonic()+args.minutes*60;new_count=0
    pending=list(dict.fromkeys(state.get('pending_dates',[])))[:1]
    cursor=date.fromisoformat(state['next_date']);days=pending[:]
    while cursor>=date.fromisoformat(args.start):
        if cursor.isoformat() not in days:days.append(cursor.isoformat())
        cursor-=timedelta(days=1)
    print(json.dumps({'event':'started','cursor':state['next_date'],'scope':list(POOLS.values()),'accepted_top':50,'stop_after_minutes':args.minutes,'totals':state['totals']},ensure_ascii=False),flush=True)
    for day in days:
        if time.monotonic()>=until:break
        tag='path-odds-'+day[:7];name=day+'.json.gz';records=store.load(tag,name,{})
        dirty=False;errors=[];interrupted=False
        def checkpoint():
            nonlocal dirty
            if dirty:store.save(tag,name,records);dirty=False
            # Rebuild this day's counters from its durable records, including a
            # possible asset upload completed before an interrupted index upload.
            totals={k:0 for k in state['totals']}
            for saved in records.values():
                for k,v in metrics(saved).items():totals[k]+=v
            state.setdefault('day_totals',{})[day]=totals
            state['totals']={k:sum(v.get(k,0) for v in state['day_totals'].values()) for k in totals}
            state['updated_at']=datetime.now(JST).isoformat();store.save(INDEX,'state.json.gz',state)
            local=store.root/'progress.json';local.write_text(json.dumps(state,ensure_ascii=False,indent=2))
        try:
            catalog,_=source.api(day);tracks=catalog['kr_info'].get('track_info') or []
            if not tracks:raise ValueError('No explicit meeting list returned for requested date')
            for track in tracks:
                if time.monotonic()>=until:interrupted=True;break
                try:
                    first,firstmeta=source.api(day,track,1);r1=matched(first,day,track,1)
                    numbers=first['kr_info'].get('races') or list(range(1,int(r1['race_info']['race_count'])+1))
                except Exception as e:errors.append({'day':day,'track':track,'error':str(e)});continue
                for number in numbers:
                    if time.monotonic()>=until or (args.max_races and new_count>=args.max_races):interrupted=True;break
                    try:
                        number=int(number);key=f'{track}:{number}';old=records.get(key)
                        if old and old.get('official',{}).get('status')=='complete':continue
                        if number==1:x,meta=first,firstmeta;r=r1
                        else:x,meta=source.api(day,track,number);r=matched(x,day,track,number)
                        if 'ガール' in r['race_info'].get('syumoku',''):continue
                        rec=normalize(r,meta)
                        if rec['official_url']:
                            try:
                                raw,received=source.get(rec['official_url']);rec['official']=official(raw,dict(url=rec['official_url'],received_at=received,sha256=hashlib.sha256(raw).hexdigest()),day)
                            except Exception as e:rec['official']={'status':'pending','error':str(e)}
                        else:rec['official']={'status':'pending','error':'No official race link'}
                        if old:
                            for k,v in metrics(old).items():state['totals'][k]-=v
                        records[key]=rec;dirty=True
                        for k,v in metrics(rec).items():state['totals'][k]+=v
                        new_count+=1
                        if rec['official']['status']!='complete':errors.append({'day':day,'track':track,'race':number,'error':'official_pending'})
                        print(json.dumps({'event':'race_saved','date':day,'track':track,'race':number,'interim':metrics(rec)['interim_races'],'observations':metrics(rec)['observations'],'official':rec['official']['status'],'totals':state['totals']},ensure_ascii=False),flush=True)
                        if new_count==1 or new_count%args.checkpoint_every==0:checkpoint()
                    except Exception as e:errors.append({'day':day,'track':track,'race':number,'error':str(e)});print(json.dumps(errors[-1],ensure_ascii=False),flush=True)
                if interrupted:break
        except Exception as e:errors.append({'day':day,'error':str(e)})
        finally:
            if not interrupted:
                state['days'][day]={'races':len(records),'errors':len(errors),'status':'complete' if not errors else 'partial'}
                if day==state['next_date']:state['next_date']=(date.fromisoformat(day)-timedelta(days=1)).isoformat()
                state['pending_dates']=[d for d in state['pending_dates'] if d!=day]
                if errors:state['pending_dates'].append(day)
            state['warnings']=(state['warnings']+errors)[-500:]
            checkpoint()
        if interrupted:break
    print(json.dumps({'event':'checkpointed_stop','next_date':state['next_date'],'totals':state['totals'],'pending_dates':len(state['pending_dates'])},ensure_ascii=False),flush=True)

def self_test():
    # Actual Kdreams payout table shape, with only the three accepted pools extracted.
    cells=['x','複','x','x','複','<dt>1=5</dt><dd>550円</dd>','x','複','<dt>1=4=5</dt><dd>520円</dd>','x','x']
    row2=['単','x','単','<dt>1-5</dt><dd>790円</dd>','単','<dt>1-5-4</dt><dd>2,070円</dd>']
    h='<dt class="bet">投票締切</dt><dd>11:05</dd><table class="refund_table">'+''.join('<tr>'+''.join('<td>'+c+'</td>' for c in row)+'</tr>' for row in [cells,row2])+'</table>'
    o=official(h.encode(),{},'2026-05-02');assert o['deadline']=='2026-05-02T11:05:00+09:00' and o['payouts_per_100_yen']['umatan']=={'1-5':790} and 'tierce' not in o['payouts_per_100_yen']
    r={'race_info':{'id':1,'race_date':'2026-05-02','track':'平塚','race':2,'race_count':11,'race_date_time':'2026-05-02T11:10:00+09:00'},'horse_info':{str(i):{} for i in range(1,10)},'odds_info':{'time_pops':{'umatan':[{'1-2':3.0}]},'time_odds_times':{'umatan':['11:00']}}}
    n=normalize(r,{});assert n['pools']['umatan']['frames'][0]['expected_count']==72 and set(n['pools'])==set(POOLS)
    try:matched({'kr_info':{'race_info':[r]}},'2026-05-01','平塚',2)
    except ValueError:pass
    else:raise AssertionError('Fallback date incorrectly accepted')
    print('PASS: official payout axes, cutoff, pool scope, actual roster cardinality, fallback-date rejection')

def main():
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--github',action='store_true');p.add_argument('--start',default='2021-01-01');p.add_argument('--end',default=(datetime.now(JST).date()-timedelta(days=1)).isoformat());p.add_argument('--minutes',type=float,default=45);p.add_argument('--pause',type=float,default=1.5);p.add_argument('--checkpoint-every',type=int,default=5);p.add_argument('--max-races',type=int,default=0);p.add_argument('--output',default='work/path_odds_archive');a=p.parse_args()
    if a.self_test:self_test()
    else:collect(a)

if __name__=='__main__':main()
