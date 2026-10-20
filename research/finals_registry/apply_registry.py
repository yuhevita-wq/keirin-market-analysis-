import sqlite3,json,pathlib,re,collections,datetime
DATA=pathlib.Path(__file__).resolve().parents[1]/'data'
def norm(x):return re.sub(r'\s+','',str(x))
def key(e):
 term=re.search(r'\d+',str(e.get('term','')))
 return (norm(e.get('player_name',e.get('name',''))),term[0] if term else '',norm(e.get('prefecture','')))
def apply(data):
 records=json.loads((data/'registry_official_sources.json').read_text());matches=collections.defaultdict(list)
 for r in records:matches[(r['name'],r['term'],r['prefecture'])].append(r)
 conflicting={k for k,v in matches.items() if len({x['registration_number'] for x in v})!=1}
 c=sqlite3.connect(data/'finals.sqlite');new=0;verified=0;unresolved=0;persons=set();clashes=[];unmatched={}
 for rid,day,payload in c.execute("select race_id,day,payload from races where status!='excluded'").fetchall():
  d=json.loads(payload or '{}');changed=False
  for e in d.get('entries',[]):
   k=key(e);v=[r for r in matches.get(k,[]) if r.get('valid_from','0000-01-01')<=day<r.get('valid_until_exclusive','9999-12-31')]
   if v and k not in conflicting:
    number=v[0]['registration_number']
    if e.get('rider_registration_number') not in [None,number]:clashes.append({'race_id':rid,'key':k,'old':e.get('rider_registration_number'),'new':number});continue
    new+=not bool(e.get('rider_source_id'));e.update(rider_source_id='https://www.keirin.jp/pc/racerprofile?snum='+number,rider_registration_number=number,rider_id_status='verified_official_name_term_prefecture',rider_identity_evidence={'matching_fields':['name','term','prefecture'],'official_sources':v,'use':'registration identity only; historical stats unchanged'});changed=True;persons.add(number)
   if e.get('rider_source_id'):verified+=1
   else:unresolved+=1;unmatched[k]=unmatched.get(k,0)+1
  if changed:c.execute('update races set payload=? where race_id=?',(json.dumps(d,ensure_ascii=False),rid))
 c.commit();check=c.execute('pragma integrity_check').fetchone()[0];assert check=='ok';c.close()
 report={'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_rows':len(records),'official_unique_registration_ids':len({r['registration_number'] for r in records}),'matched_registration_ids':len(persons),'newly_verified_entry_records':new,'verified_entry_records':verified,'unresolved_entry_records':unresolved,'conflicting_source_keys':[list(x) for x in conflicting],'conflicts':clashes,'integrity_check':check,'all_finals_complete':False,'unmatched_targets':[{'name':k[0],'term':k[1],'prefecture':k[2],'appearances':v} for k,v in sorted(unmatched.items(),key=lambda kv:-kv[1])]}
 (data/'registry_integrity.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print({k:v for k,v in report.items() if k!='unmatched_targets'});return report
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--data',type=pathlib.Path,default=DATA);apply(a.parse_args().data)
