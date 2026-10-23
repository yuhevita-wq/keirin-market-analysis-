"""Establish namespaced, observed source-key mappings for pending Rainbow finals.
The local composite key is explicitly local, never a fabricated KDreams ID.
No SQL mutation; insertion adapters and all unresolved evidence remain pending.
"""
import pathlib,json,hashlib,re,sqlite3,urllib.parse,datetime,copy,importlib.util
from lxml import html
ROOT=pathlib.Path('current_data/unpacked/finals_ai');DATA=ROOT/'data';OUT=DATA/'scope_review_1746'
DBSHA='04c4de11b6937c3b3a499f042410786cfdde4c2892ca0af46999ca059c5e2d91'
def sha(b):return hashlib.sha256(b).hexdigest()
def module():
 s=importlib.util.spec_from_file_location('cards0831',ROOT/'recovery/review_rainbow_independent_card_results_0831.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def verify(e,c,op,nk,m):
 day=datetime.date.fromisoformat(e['day']);assert datetime.date(2023,1,1)<=day<=datetime.date(2026,10,19)
 venue=e['venue_code'];rr=e['race_no'];assert re.fullmatch(r'\d{2}',venue) and type(rr)==int and 1<=rr<=12
 k=e['day'].replace('-','')+venue+str(rr).zfill(2);b=c['box'];program=c['race_program'];status=c['race_status']
 for record in [b,program,status]:assert (record['open_day'],record['vel_code'],record['race_num'])==(e['day'],venue,rr)
 assert b['is_end_day'] and not b['is_girls'] and not b['on_abort'] and not b['is_invalid']
 pid=program['id'];assert type(pid)==int and pid>0 and pid==status['id']==e['source_race_key']['race_info_id']
 assert (e['source_race_key']['day'],e['source_race_key']['venue_code'],e['source_race_key']['race_no'])==(e['day'],venue,rr)
 assert 'Ｆ' in program['event_name'] and program['name']=='Ａ級' and e['category'] in ['A12','A3']
 for f,v in [('joCode',venue),('kaisaiBi',e['day'].replace('-','')),('raceNo',str(rr))]:assert op.xpath('//input[@id="'+f+'"]/@value')==[v]
 assert nk.xpath('//input[@name="plp-raceid"]/@value')==[k]
 # Observe this exact key in actual DOM links as well as hidden field.
 links=[a.get('href') for a in nk.xpath('//a[@href]') if urllib.parse.urlparse(a.get('href')).path in ['/race/result/','/race/result//'] and urllib.parse.parse_qs(urllib.parse.urlparse(a.get('href')).query).get('race_id')==[k]]
 assert links and all(urllib.parse.urlparse(u).hostname=='keirin.netkeiba.com' for u in links)
 ids=[];cars=[]
 for row in nk.xpath('//table[@id="All_Result_Table"]//tr[td]'):
  cs=row.xpath('./td');assert len(cs)==8;car=int(m.norm(cs[2].text_content()));a=c['race_athletes'][str(car)];rid=int(a['un_id']);cars.append(car);ids.append(rid)
  ns=cs[3].xpath('.//a[@href]');assert len(ns)==1 and int(urllib.parse.parse_qs(urllib.parse.urlparse(ns[0].get('href')).query)['id'][0])==rid
  assert [m.norm(x) for x in cs[3].xpath('.//img/@alt')]==[m.norm(a['name'])]
  assert re.search(r'(\d+)期',m.norm(cs[3].text_content()))[1]==a['graduate_period']
  ors=op.xpath('//table[@summary="出走表"]/tr[td[@class="no'+str(car)+'"]]');assert len(ors)==1;als=ors[0].xpath('./td')[-8].xpath('.//a[@href]');assert len(als)==1
  assert int(urllib.parse.parse_qs(urllib.parse.urlparse(als[0].get('href')).query)['playerCd'][0])==rid and m.norm(als[0].text_content())==m.norm(a['name'])
 assert sorted(cars)==list(range(1,10)) and len(ids)==len(set(ids))==9
 return {'local_composite_key':'keirin:'+e['day']+':'+venue+':'+str(rr).zfill(2),'local_key_namespace':'locally_defined_actual_race_date_venue_race_number_v1','tuple':{'day':e['day'],'venue_code':venue,'race_no':rr},'observed_source_keys':{'chariloto_program_status_id':pid,'chariloto_box_id':b['id'],'netkeiba_race_id':k,'netkeiba_actual_result_links':sorted(set(links)),'oddspark_explicit_hidden_tuple':{'joCode':venue,'kaisaiBi':e['day'].replace('-',''),'raceNo':rr}},'registration_numbers_by_result_order':[str(x).zfill(6) for x in ids],'candidate_kdreams_race_id_retained_unconfirmed':e['candidate_kdreams_race_id'],'confirmed_kdreams_race_id':None,'source_key_mapping_status':'independently_verified_observed_keys_and_actual_race_tuple','SQL_insert_adapter_audit_status':'pending_not_inserted','all_finals_complete':False}
def audit_absence(conn,mapping):
 wanted=(mapping['tuple']['day'],mapping['tuple']['venue_code'],mapping['tuple']['race_no']);collisions=[]
 for table in ['races','excluded_races']:
  for rid,day,payload in conn.execute('select race_id,day,payload from '+table+' where day=?',(wanted[0],)):
   assert re.fullmatch(r'\d{16}',rid),('unexpected legacy key',rid)
   d=json.loads(payload);race=d['race'];assert race['race_id']==rid and race['race_date']==day
   assert rid in race['source_url'];actual=(day,rid[:2],int(race['race_no']))
   if actual==wanted:collisions.append({'table':table,'race_id':rid})
 for day,url,rno in conn.execute('select day,event_card_url,race_no from cancellations where day=?',(wanted[0],)):
  match=re.search(r'/racecard/(\d{14})/',url);assert match
  if (day,match[1][:2],int(rno))==wanted:collisions.append({'table':'cancellations','event_card_url':url})
 assert not collisions,collisions
 return {'existing_races_excluded_cancellations_same_actual_tuple':0,'existing_legacy_KDreams_ids_unchanged':True}
def main():
 OUT.mkdir(exist_ok=True);m=module();manifest=json.loads((DATA/'scope_review_0831/raw_manifest.json').read_text())
 for n,v in manifest.items():
  b=(pathlib.Path('raw_independent_0831')/n).read_bytes();assert len(b)==v['bytes'] and sha(b)==v['sha256']
 db=(DATA/'finals.sqlite').resolve();before=db.read_bytes();assert len(before)==281206784 and sha(before)==DBSHA;assert not any(pathlib.Path(str(db)+s).exists() for s in ['-journal','-wal','-shm'])
 conn=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True);assert conn.execute('pragma integrity_check').fetchall()==[('ok',)]
 scope=json.loads((DATA/'scope_review_0457/rainbow_scope_independent_review.json').read_text());out=[];inputs=[]
 for e in scope['events']:
  for v in e['source_files']:
   b=(DATA/'scope_review_0457'/v['path']).read_bytes();assert len(b)==v['bytes'] and sha(b)==v['sha256']
  k=e['day'].replace('-','')+e['venue_code']+str(e['race_no']).zfill(2);c=json.loads(m.cardfile(e).read_text());op,om=m.doc(k,'card');nk,nm=m.doc(k,'result');v=verify(e,c,op,nk,m);v['collision_audit']=audit_absence(conn,v);out.append(v);inputs.append({'local_composite_key':v['local_composite_key'],'Chariloto_source':e['source_files'][0],'OP_meta':om,'NK_meta':nm})
 assert len(out)==14 and len({v['local_composite_key'] for v in out})==14 and len({v['observed_source_keys']['chariloto_program_status_id'] for v in out})==14 and len({v['observed_source_keys']['netkeiba_race_id'] for v in out})==14
 e=next(e for e in scope['events'] if e['day']=='2025-12-23');k='202512236211';c=json.loads(m.cardfile(e).read_text());op,_=m.doc(k,'card');nk,_=m.doc(k,'result');tests=[]
 for mode in ['wrong_day','wrong_venue','wrong_R','wrong_program_id','wrong_NK_hidden','wrong_explicit_rider_ID']:
  ev=copy.deepcopy(e);card=copy.deepcopy(c);doc=copy.deepcopy(nk)
  if mode=='wrong_day':ev['day']='2025-12-22'
  elif mode=='wrong_venue':ev['venue_code']='61'
  elif mode=='wrong_R':ev['race_no']=9
  elif mode=='wrong_program_id':card['race_program']['id']+=1
  elif mode=='wrong_NK_hidden':doc.xpath('//input[@name="plp-raceid"]')[0].set('value','202512236209')
  else:
   a=doc.xpath('//table[@id="All_Result_Table"]//tr[td]')[0].xpath('./td')[3].xpath('.//a[@href]')[0];a.set('href','https://keirin.netkeiba.com/db/profile/?id=999999')
  raw=html.tostring(doc);mutated_sha=sha(raw)
  try:verify(ev,card,op,html.fromstring(raw),m)
  except (AssertionError,KeyError,ValueError,TypeError):tests.append({'mutation':mode,'rejected':True,'mutated_NK_raw_sha256':mutated_sha})
  else:raise AssertionError('accepted mutation '+mode)
 assert conn.execute('select count(*) from races').fetchone()==(5354,);conn.close();assert db.read_bytes()==before
 report={'all_finals_complete':False,'canonical_DB_changed':False,'SQL_rows_inserted':0,'new_raw_fetches':0,'observed_provider_key_mappings_verified':14,'explicit_entry_IDs_linked':126,'unique_local_composite_keys':14,'legacy_existing_key_rewrites':0,'confirmed_KDreams_target_keys':0,'canonical_key_policy':'Use the explicit local namespace plus verified actual date/venue/R tuple with observed provider ID mappings. Local composite is not any providers ID. Preserve unconfirmed KDreams candidates separately, never elevate them by construction. SQL insert compatibility and unresolved race requirements remain pending.','events':out,'sources':inputs,'mutations':tests,'unresolved':['SQL insertion adapter and independent insertion audit','13 candidate historical banks','17 empty planned history date mappings','other final-equivalent full-period scope','5123 existing male historical banks']}
 (OUT/'source_key_mapping_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print('14 observed source-key mappings verified/126 explicit IDs; composite keys14 unique; collisions0; mutations6 rejected; SQL0')
if __name__=='__main__':main()
