"""Offline identity adapter prototype audit. No collector integration or canonical SQL write."""
import pathlib,json,re,datetime,hashlib,sqlite3,copy,importlib.util,urllib.parse
ROOT=pathlib.Path('current_data/unpacked/finals_ai')
KEYS=ROOT/'data/scope_review_1746/source_key_mapping_review.json'
KEYSHA='755a2c57cbfb5b4c4e66a326d089ba81005b7af780e36d457072d06e3865673a'
DBSHA='04c4de11b6937c3b3a499f042410786cfdde4c2892ca0af46999ca059c5e2d91'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def actual(day,venue,rno):
 assert datetime.date.fromisoformat(day).isoformat()==day and '2023-01-01'<=day<='2026-10-19'
 assert re.fullmatch(r'[0-9]{2}',venue) and 1<=int(venue)<=99
 assert type(rno)==int and 1<=rno<=12
 return {'day':day,'venue_code':venue,'race_no':rno}
def local_identity(event):
 t=event['tuple'];a=actual(t['day'],t['venue_code'],t['race_no']);key=f"keirin:{a['day']}:{a['venue_code']}:{a['race_no']:02d}"
 assert event['local_key_namespace']=='locally_defined_actual_race_date_venue_race_number_v1' and event['local_composite_key']==key
 s=event['observed_source_keys'];nk=a['day'].replace('-','')+a['venue_code']+f"{a['race_no']:02d}"
 assert s['netkeiba_race_id']==nk and s['netkeiba_actual_result_links']==['https://keirin.netkeiba.com/race/result/?race_id='+nk]
 assert s['oddspark_explicit_hidden_tuple']=={'joCode':a['venue_code'],'kaisaiBi':a['day'].replace('-',''),'raceNo':a['race_no']}
 assert type(s['chariloto_program_status_id'])==int and s['chariloto_program_status_id']>0
 assert event['confirmed_kdreams_race_id'] is None and event['source_key_mapping_status']=='independently_verified_observed_keys_and_actual_race_tuple'
 return dict(a,canonical_key=key,provider='local_composite',confirmed_kdreams_race_id=None)
def verified_mapping():
 assert digest(KEYS)==KEYSHA
 d=json.loads(KEYS.read_text());assert d['SQL_rows_inserted']==0 and d['confirmed_KDreams_target_keys']==0
 assert len(d['events'])==14
 return d

def legacy_identity(rid,day,url,payload):
 assert re.fullmatch(r'[0-9]{16}',rid)
 r=payload['race'];raw=r['race_no'];assert type(raw)==int or (type(raw)==str and re.fullmatch(r'[0-9]{1,3}',raw));a=actual(day,rid[:2],int(raw))
 assert r['race_date']==day and r['race_id']==rid and payload['race_id']==rid
 assert '/racedetail/'+rid+'/' in url and urllib.parse.urlparse(r['source_url']).path==urllib.parse.urlparse(url).path
 assert int(rid[-4:])==a['race_no']
 return dict(a,canonical_key=rid,provider='kdreams_observed_legacy',confirmed_kdreams_race_id=rid)
def select_explicit(records,identity,track):
 # Does not interpret local string as provider ID or infer date from meeting prefix.
 a=actual(identity['day'],identity['venue_code'],identity['race_no'])
 hits=[r for r in records if r['venue_code']==a['venue_code'] and r['track']==track and r['valid_from']<=a['day']<r['valid_until_exclusive']]
 assert len(hits)<=1
 return hits[0] if hits else None

def main():
 d=verified_mapping();events=d['events'];ids=[local_identity(e) for e in events];assert len({i['canonical_key'] for i in ids})==14
 spec=importlib.util.spec_from_file_location('oldbank',ROOT/'recovery/apply_historical_bank.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 cat=json.loads((ROOT/'data/historical_bank_catalog.json').read_text())['records'];old.validate(cat)
 db=ROOT/'data/finals.sqlite';assert db.stat().st_size==281206784 and digest(db)==DBSHA
 assert not any(pathlib.Path(str(db)+s).exists() for s in ['-wal','-shm','-journal'])
 n=verified=unresolved=excluded=0;anomalies=[]
 with sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True) as c:
  assert c.execute('pragma integrity_check').fetchall()==[('ok',)]
  for rid,day,url,status,raw in c.execute('select race_id,day,url,status,payload from races'):
   p=json.loads(raw);n+=1
   if status=='excluded':
    assert rid=='1120240912030011' and p['scope_review']['status']=='excluded_women_source_verified';excluded+=1;continue
   try:identity=legacy_identity(rid,day,url,p)
   except (AssertionError,KeyError,ValueError) as err:
    anomalies.append({'race_id':rid,'day':day,'url':url,'raw_race_no':p.get('race',{}).get('race_no'),'status':'unresolved_legacy_identity_adapter','exception_type':type(err).__name__});continue
   track=p['race']['track'];a=select_explicit(cat,identity,track);assert a==old.select(cat,rid,day,track)
   if a:verified+=1
   else:unresolved+=1
 assert n==5354 and excluded==1 and verified+unresolved+len(anomalies)==5353
 # Reproduce concrete incompatibilities without calling network/collect/apply.
 failures=[]
 for i in ids:
  try:int(i['canonical_key'][-4:])
  except ValueError:failures.append(i['canonical_key'])
 assert len(failures)==14
 ito=next(i for i in ids if i['day']=='2023-12-05' and i['venue_code']=='37');bank=select_explicit(cat,ito,'伊東競輪');assert bank['bank_version_id']=='37-ito20231202-reviewed'
 assert old.select(cat,ito['canonical_key'],ito['day'],'伊東競輪') is None
 muts=[]
 for field,value in [('local_composite_key','keirin:2023-06-06:44:10'),('local_key_namespace','kdreams'),('confirmed_kdreams_race_id',events[0]['candidate_kdreams_race_id_retained_unconfirmed'])]:
  e=copy.deepcopy(events[0]);e[field]=value
  try:local_identity(e)
  except AssertionError:muts.append(field)
  else:raise AssertionError('bad identity accepted')
 for field,value in [('day','2023-06-07'),('venue_code','45'),('race_no',10)]:
  e=copy.deepcopy(events[0]);e['tuple'][field]=value
  try:local_identity(e)
  except AssertionError:muts.append('tuple.'+field)
  else:raise AssertionError('bad tuple accepted')
 assert len(muts)==6
 assert digest(db)==DBSHA
 report={'status':'offline_identity_adapter_prototype_verified_not_integrated','all_finals_complete':False,'canonical_DB_changed':False,'SQL_rows_inserted':0,'network_requests':0,'legacy_rows_scanned':n,'legacy_male_identity_adapter_accepted':verified+unresolved,'legacy_identity_anomalies':anomalies,'excluded_women_preserved':excluded,'legacy_bank_selection_exact_parity_on_accepted_identities':{'verified':verified,'unresolved':unresolved},'local_identity_mappings_checked':14,'confirmed_KDreams_targets':0,'legacy_collector_suffix_parse_rejected_local_keys':failures,'legacy_bank_prefix_misses_verified_Ito_candidate':True,'explicit_tuple_selector_finds_exact_Ito_candidate':bank['bank_version_id'],'candidate_bank_not_added_existing_count':True,'mutations_rejected':muts,'source_key_review_sha256':KEYSHA,'code_dependencies':{p:digest(ROOT/p) for p in ['pipeline.py','recovery/apply_historical_bank.py','recovery/audit_historical_bank.py']},'integration_blockers':['pipeline.collect_one uses int(rid[-4:]) and provider-specific HTML parsers','apply_historical_bank.select and audit_historical_bank use rid[:2]','audit_historical_bank and older builders hardcode 5354/5353; do not relax without independent scope manifest','Complete race payload adapter, raw provenance, readiness gate and isolated insertion audit still pending','Legacy identity anomalies must be resolved from actual source labels; do not strip 100 from raw race_no by assumption','13 candidate bank sources,17 empty planned histories, broader final-equivalent scope unresolved'],'events':ids}
 out=ROOT/'data/scope_review_1918';out.mkdir(exist_ok=True);(out/'key_adapter_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True));print(json.dumps({k:v for k,v in report.items() if k not in ['events','legacy_collector_suffix_parse_rejected_local_keys','code_dependencies']},ensure_ascii=False))
if __name__=='__main__':main()
