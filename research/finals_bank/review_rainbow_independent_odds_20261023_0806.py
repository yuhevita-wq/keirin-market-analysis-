"""Bounded recovery of unsaved independent published final odds; no DB changes."""
import pathlib,json,urllib.request,urllib.parse,time,hashlib,concurrent.futures,itertools,re,decimal,datetime
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=ROOT/'data/scope_review_0457';OUT=ROOT/'data/scope_review_0806';OUT.mkdir(exist_ok=True)
EVENTS=json.loads((BASE/'rainbow_event_candidates.json').read_text())
def fetch(ev):
 key=ev['day'].replace('-','')+ev['venue_code']+str(ev['race_no']).zfill(2);p=OUT/(key+'_odds.json');meta=OUT/(key+'_meta.json')
 if p.exists() and meta.exists():return ev,key,p,json.loads(meta.read_text())
 fields={'input':'UTF-8','output':'json','class':'AplRaceOdds','method':'get','compress':'0','race_id':key}
 req=urllib.request.Request('https://keirin.netkeiba.com/api/race/',data=urllib.parse.urlencode(fields).encode(),headers={'User-Agent':'Mozilla/5.0','Accept-Encoding':'identity','Content-Type':'application/x-www-form-urlencoded'})
 t=time.monotonic();chunks=[]
 try:
  with urllib.request.urlopen(req,timeout=25) as r:
   headers=dict(r.headers);status=r.status
   while True:
    b=r.read(16384)
    if not b:break
    chunks.append(b)
    if time.monotonic()-t>55:raise TimeoutError('whole transfer exceeded 55 seconds')
  raw=b''.join(chunks);data=json.loads(raw);assert data['status']=='OK'
  p.write_bytes(raw);m={'url':req.full_url,'method':'POST','form':fields,'status':status,'headers':headers,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'retrieved_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'recovery_note':'Previous unsaved retrieval lost after workspace pruning; recovered once, not new race completion.'};meta.write_text(json.dumps(m,ensure_ascii=False,indent=2));return ev,key,p,m
 except Exception as e:
  m={'key':key,'error':type(e).__name__+': '+str(e),'retry_this_run':False};meta.write_text(json.dumps(m));return ev,key,None,m
def audit(ev,key,p,m):
 assert p is not None,m
 data=json.loads(p.read_bytes());d=data['data'];k='nkrace_odds::'+key;assert k in d
 record=d[k];assert record['official_dt'].startswith(ev['day']),record['official_dt']
 prefix=ev['day'].replace('-','')+'_'+ev['venue_code']
 if ev['race_no']!=9:prefix+='_r'+str(ev['race_no'])
 old=BASE/'chariloto_api_odds.html' if ev['day']=='2023-12-05' else BASE/'other_rainbows'/(prefix+'_odds.html')
 c=json.loads(old.read_text());assert c['has_error'] is False
 markets={}
 for field,bet,ordered in [('list_9','sanrentan',True),('list_8','sanrenfuku',False)]:
  rows=record[field];actual={}
  for row in rows:
   assert re.fullmatch(r'\d{6}',row[0]),row
   cars=tuple(int(row[0][i:i+2]) for i in (0,2,4));assert cars not in actual
   rate=decimal.Decimal(row[1]);assert rate.is_finite() and rate>0
   actual[cars]=rate
  expected=set(itertools.permutations(range(1,10),3) if ordered else itertools.combinations(range(1,10),3));assert set(actual)==expected
  comparator={}
  for row in c['odds']:
   if row['bet_type']!=bet:continue
   cars=tuple(map(int,re.findall(r'\d+',row['deme'])));assert cars not in comparator
   assert row['is_valid'] is True and row['on_final'] is True
   rate=decimal.Decimal(str(row['rate']));assert rate==decimal.Decimal(str(row['normalize_rate']))
   comparator[cars]=rate
  assert actual==comparator
  markets[bet]={'count':len(actual),'exact_expected_set':True,'duplicates':0,'positive_finite':True,'exact_decimal_odds_equal':True,'independent_provider':'netkeiba','comparator':'saved Chariloto final odds','raw_flags_preserved':True}
 return {**ev,'netkeiba_race_id':key,'official_dt':record['official_dt'],'source':m,'comparator_path':str(old.relative_to(ROOT)),'comparator_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'markets':markets,'canonical_db_inserted':False}
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:retrieved=list(ex.map(fetch,EVENTS))
 events=[];errors=[]
 for ev,key,p,m in retrieved:
  try:events.append(audit(ev,key,p,m))
  except Exception as e:errors.append({'race_id':key,'error':type(e).__name__+': '+str(e)})
 report={'all_finals_complete':False,'canonical_db_changed':False,'independent_market_sets_verified':sum(len(x['markets']) for x in events),'independent_odds_combinations_verified':sum(v['count'] for x in events for v in x['markets'].values()),'events':events,'errors':errors,'unresolved':['historical entry statistics and malformed prior-history dates require independent verification','canonical key policy and complete race insertion audit','independent actual results and lines audit','historical bank applicability','other final-equivalent labels and full scope audit'],'recovery_note':'Only independent full-market validation persisted this run; prior unsaved card/history/result fetches are not claimed as verified.'}
 (OUT/'independent_odds_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['events','unresolved']},ensure_ascii=False));assert len(events)==14 and not errors
