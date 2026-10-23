"""Independent reconstruction of blocked legacy-field drafts and SQL roundtrip."""
import copy,hashlib,json,sqlite3,tarfile,unicodedata,itertools
from collections import Counter,defaultdict
from decimal import Decimal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/scope_review_0835'
FIXED=[('data/scope_review_0552/integrated_candidates.tar.xz','integrated_candidates.json','c1a030bc66b78fec4bda307046402f9db15fa67dd386845b8d19862cd3f29e3a','b9ed0533f9c8a654236a4c4c8d9d275161acdd6c154ad6c3efbaa444cb307e3f'),('data/scope_review_0636/payout_fragments.tar.xz','payout_fragments.json','f0729f417e0410bc0b26e1ad2d772b83a5bab3ee790c6a24703909744e6d28eb','b468330e89e778acc6cd481dcc3fab654d3a1bc56fcb8a5e202491439c4ccd33'),('data/scope_review_0730/entry_extension_overlay.tar.xz','entry_extension_overlay.json','10cc8901c0d7674dad51ba1fa3353d4e266b6741552e6e7c4b4bcc646b6433e6','3597429ce71d38b3a66ac777c083afaea4543bd3b86f33cfbd94c082d57c1503')]
def digest(b):return hashlib.sha256(b).hexdigest()
def serialize(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def load_fixed():
 result=[]
 for path,member,ah,mh in FIXED:
  body=(ROOT/path).read_bytes();assert digest(body)==ah
  with tarfile.open(ROOT/path) as a:assert a.getnames()==[member];raw=a.extractfile(member).read()
  assert digest(raw)==mh;result.append(json.loads(raw))
 return result
def check_contract(contract):
 db=sqlite3.connect((ROOT/'data/finals.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
 assert db.execute("select sql from sqlite_master where name='races'").fetchone()[0]==contract['races_DDL']
 counts=defaultdict(Counter);presence=Counter();whole=hashlib.sha256();males=nentry=nresult=0;excluded=[]
 for rid,status,text in db.execute('select race_id,status,payload from races order by race_id'):
  if rid=='1120240912030011':excluded.append({'race_id':rid,'status':status});continue
  payload=json.loads(text);males+=1;nentry+=len(payload['entries']);nresult+=len(payload['results']);presence.update(payload.keys())
  b=serialize([rid,status,text]);whole.update(len(b).to_bytes(8,'big'));whole.update(b)
  stack=[('/'+part,payload[part]) for part in ['race','entries','results','line','trifecta','trio','payouts','odds_header']]
  while stack:
   p,v=stack.pop();counts[p][type(v).__name__]+=1
   if isinstance(v,dict):stack.extend((p+'/'+k,x) for k,x in v.items())
   elif isinstance(v,list):stack.extend((p+'/*',x) for x in v)
 db.close()
 assert (males,nentry,nresult)==(5353,37883,37883)
 assert (contract['male_rows'],contract['entry_rows'],contract['result_rows'])==(males,nentry,nresult)
 assert contract['excluded']==excluded and contract['top_field_presence']==dict(presence)
 assert contract['ordered_original_payload_text_digest']==whole.hexdigest()
 assert contract['observed_types']=={p:dict(c) for p,c in counts.items()}
def validate(d,source,pay,ext):
 assert d['schema']=='rainbow_legacy_payload_adapter_draft_v1' and len(d['events'])==14
 assert d['all_finals_complete'] is d['full_payload_SQL_adapter_verified'] is d['complete_payload_insertion_audit'] is False
 assert d['canonical_SQL_inserts']==d['new_completed_races']==d['network_requests']==0
 for path,h in d['input_pins'].items():assert digest((ROOT/path).read_bytes())==h
 for draft,event,payout,extension in zip(d['events'],source['events'],pay['events'],ext['events']):
  assert draft['identity']==event['identity']==payout['identity']==extension['identity']
  assert draft['sql_status']=='partial_adapter_draft' and all(draft[f] is False for f in ['complete_payload','canonical_SQL_inserted','ready_for_completed_export','full_payload_SQL_adapter_verified'])
  expectedblock=event['evidence_integration']['readiness_inventory']['blockers']+['legacy_payload_unmapped_fields_remain','line_roles_not_published_in_selected_sources','race_times_and_vote_totals_single_source_only','odds_timestamp_semantics_unverified','entry_extension_unresolved_fields_retained','complete_payload_independent_insertion_audit_pending']
  assert draft['blockers']==expectedblock
  ref=event['component_raw_provenance']['card'];b=(ROOT/ref['path']).read_bytes();assert len(b)==ref['bytes'] and digest(b)==ref['sha256'];card=json.loads(b);box=card['box'];ident=event['identity'];p=draft['payload']
  url=event['observed_provider_keys']['netkeiba_actual_result_links'][0]
  race=dict(race_id=ident['canonical_key'],race_date=ident['day'],race_no=ident['race_no'],source_url=url,track=box['vel_name'],entry_count=9,race_type=event['program_raw']['event_name'],meeting_grade=event['meeting_grade_raw'],event_name=box['race_name'],start_time=box['race_start_at_string'],deadline=box['sale_to_at_string'])
  assert p['race']==race
  assert set(p)=={'race_id','discovered_day','expected_label','grade','category','race','entries','results','line','payouts','trifecta','trio','odds_header'}
  assert p['race_id']==ident['canonical_key'] and p['discovered_day']==ident['day'] and p['expected_label']==event['program_raw']['event_name'] and p['grade']==event['meeting_grade_raw'] and p['category']==event['category']
  assert len(p['entries'])==len(draft['evidence']['entries'])==9
  missingentries={}
  for saved,entry,extra,evidence in zip(p['entries'],event['entries'],extension['entries'],draft['evidence']['entries']):
   a=card['race_athletes'][str(entry['car_no'])]
   assert int(entry['registration_number'])==a['un_id'] and saved['car_no']==a['entry_num']
   expected={'car_no':a['entry_num'],'player_name':a['name'],'prefecture':a['pref_name'],'age':str(a['age']),'term':a['graduate_period'],'class':unicodedata.normalize('NFKC',a['class_name']+a['group_name']),'score':str(Decimal(a['average_point'])/Decimal(1000)),'gear':a['gear_ratio'],'style':a['leg_quality'],'rider_registration_number':entry['registration_number']}
   for name,raw in [('first_count','first_count'),('second_count','second_count'),('third_count','third_count'),('outside_count','out_rank_count'),('nige_count','nige_count'),('makuri_count','makuri_count'),('sashi_count','sashi_count'),('mark_count','mark_count')]:expected[name]=str(a[raw])
   expected.update(extra['legacy_entry_fields_verified']);expected.update({k:race[k] for k in ['race_id','race_date','race_no','source_url','track','race_type','start_time','deadline']})
   assert saved==expected
   assert evidence=={'car_no':entry['car_no'],'registration_number':entry['registration_number'],'core_source_pointer':entry['source_pointer'],'core_identity':entry['evidence_integration']['core_identity'],'history_adapter':entry['history_adapter'],'history_DOM':entry['evidence_integration']['history_DOM'],'entry_extension':extra}
   missing=['captured_at_utc','evaluation','player_profile','prediction_mark','raw_row_json','rider_source_id','rider_id_status','rider_identity_evidence']
   if 'top3_rate' not in expected:missing.append('top3_rate')
   missingentries[str(entry['car_no'])]=missing
  groups=event['published_line_groups'];line=p['line']
  assert line['status']=='published' and line['formation']=='/'.join('-'.join(str(x) for x in g) for g in groups)
  assert line['groups']==[[{'car_no':car} for car in group] for group in groups]
  assert line['riders']=={str(car):{'line_id':i+1,'line_size':len(group),'line_position':j+1} for i,group in enumerate(groups) for j,car in enumerate(group)}
  assert set(line)=={'status','formation','groups','riders'}
  for saved,original in zip(p['results'],event['results']):
   nk=original['netkeiba_raw'];ch=original['chariloto_raw'];f=original['finish_text']
   assert saved=={'car_no':original['car_no'],'player_name':ch['name'],'finish_text':f,'finish_position':int(f) if f.isdecimal() else f,'last200':nk['last_lap_raw'],'margin':nk['margin_raw'],'winning_method':nk['move_raw'],'sb':nk['SB_raw'],'raw_row_json':serialize({'chariloto_original':ch,'netkeiba_original':nk}).decode()}
  assert len(p['results'])==9
  for src,dest in [('sanrentan','trifecta'),('sanrenfuku','trio')]:
   assert len(p[dest])==len(event['markets'][src])==(504 if dest=='trifecta' else 84)
   for row,original in zip(p[dest],event['markets'][src]):
    assert set(row)=={'cars','odds','status'} and row['cars']==original['cars'] and row['status']=='available' and type(row['odds']) is float and Decimal(str(row['odds']))==Decimal(original['odds_decimal'])
   assert draft['evidence']['market_decimal_sources'][dest]==event['markets'][src]
  assert p['payouts']==payout['payload_payouts_fragment'] and draft['evidence']['payout_evidence']==payout['evidence']
  assert p['odds_header']=={'3連単':{'total_votes':card['all_votes']['sanrentan']},'3連複':{'total_votes':card['all_votes']['sanrenfuku']}}
  assert draft['evidence']['total_votes_verification']=='historical_Chariloto_explicit_single_source_only'
  ref=event['component_raw_provenance']['independent_odds'];b=(ROOT/ref['path']).read_bytes();assert len(b)==ref['bytes'] and digest(b)==ref['sha256']
  actual=json.loads(b)['data']['nkrace_odds::'+event['observed_provider_keys']['netkeiba_race_id']]['official_dt']
  assert draft['evidence']['NK_official_dt_raw_not_relabelled']==actual
  assert draft['evidence']['source_refs']==event['component_raw_provenance'] and draft['evidence']['line_tokens']==event['published_line_tokens_raw'] and draft['evidence']['original_results']==event['results']
  assert draft['evidence']['single_source_race_metadata']=={'box':box,'program':event['program_raw']}
  assert draft['unmapped_fields']=={'top_level':['captured','racecard_tables'],'race':['captured_at_utc'],'entries':missingentries,'line':['groups/*/role','riders/*/line_role','provider'],'results':['result_comment'],'odds_header':['3連単/odds_as_of','3連複/odds_as_of']}
def main():
 source,pay,ext=load_fixed();report=json.loads((OUT/'legacy_payload_adapter_review.json').read_bytes());arc=report['archive'];raw=(ROOT/arc['path']).read_bytes();assert digest(raw)==arc['sha256'] and len(raw)==arc['bytes']
 with tarfile.open(ROOT/arc['path']) as t:assert t.getnames()==[arc['member']];body=t.extractfile(arc['member']).read()
 assert digest(body)==arc['member_sha256'] and len(body)==arc['member_bytes'];d=json.loads(body);check_contract(d['observed_legacy_contract']);validate(d,source,pay,ext)
 # Exact live DDL, isolated in memory. The payload is deliberately PARTIAL.
 mem=sqlite3.connect(':memory:');mem.execute(d['observed_legacy_contract']['races_DDL'])
 for e in d['events']:
  p=e['payload'];mem.execute('insert into races values (?,?,?,?,?,?,?,?,?,?)',(p['race_id'],e['identity']['day'],p['race']['source_url'],p['expected_label'],p['grade'],p['category'],'partial_adapter_draft',serialize(p).decode(),serialize(e['blockers']).decode(),None))
 for rid,status,p in mem.execute('select race_id,status,payload from races'):
  original=next(e['payload'] for e in d['events'] if e['payload']['race_id']==rid);assert status=='partial_adapter_draft' and json.loads(p)==original
 assert mem.execute("select count(*) from races where status in ('completed','collected')").fetchone()[0]==0;mem.close()
 mutations=[]
 cases=[('historical_score',lambda e:e['payload']['entries'][0].__setitem__('score','999.999')),('explicit_ID',lambda e:e['payload']['entries'][0].__setitem__('rider_registration_number','999999')),('local_key_as_KD',lambda e:e['payload']['race'].__setitem__('race_id','4420230603040009')),('line_role_invention',lambda e:e['payload']['line']['groups'][0][0].__setitem__('role','先行')),('line_boundary',lambda e:e['payload']['line']['riders']['3'].__setitem__('line_position',9)),('market_price',lambda e:e['payload']['trifecta'][0].__setitem__('odds',1.1)),('market_deleted',lambda e:e['payload']['trio'].pop()),('result_finish',lambda e:e['payload']['results'][0].__setitem__('finish_position',9)),('odds_time_false_mapping',lambda e:e['payload']['odds_header']['3連単'].__setitem__('odds_as_of',e['evidence']['NK_official_dt_raw_not_relabelled'])),('gap_erased',lambda e:e['unmapped_fields'].__setitem__('line',[])),('single_source_false_promotion',lambda e:e['evidence'].__setitem__('total_votes_verification','independently_verified')),('SQL_completion_promotion',lambda e:e.__setitem__('sql_status','collected')),('complete_flag',lambda e:e.__setitem__('complete_payload',True)),('history_raw_year',lambda e:e['evidence']['entries'][0]['history_adapter']['adapted_history_slots'][0]['original'].__setitem__('open_from','20230603'))]
 for name,fn in cases:
  bad=copy.deepcopy(d);fn(bad['events'][0])
  try:validate(bad,source,pay,ext)
  except (AssertionError,KeyError,ValueError):mutations.append(name)
  else:raise AssertionError('accepted mutation '+name)
 # Explicit DQ and tied places must survive roundtrip with original tokens.
 dq=[r for e in d['events'] for r in e['payload']['results'] if not r['finish_text'].isdecimal()];assert len(dq)==2 and all(r['finish_position']==r['finish_text']=='失' for r in dq)
 first=d['events'][0]['payload']['results'];assert sum(r['finish_text']=='7' and r['finish_position']==7 for r in first)==2
 out={'status':'independent_partial_legacy_adapter_audit_passed','events':14,'entries':126,'results':126,'market_selections':8232,'exact_live_races_DDL_in_memory_roundtrip':14,'in_memory_SQL_status':'partial_adapter_draft','in_memory_completed_or_collected_rows':0,'legacy_contract_male_rows':5353,'legacy_contract_entry_rows':37883,'original_DQ_tokens_retained':2,'Ogaki_tied_7th_rows_retained':2,'line_roles_inferred':0,'unresolved_top3_fields_absent':2,'mutations_rejected':mutations,'canonical_SQL_inserts':0,'new_completed_races':0,'all_finals_complete':False,'full_payload_SQL_adapter_verified':False,'complete_payload_insertion_audit':False,'second_SQLite_artifact_saved':False}
 (OUT/'independent_legacy_payload_adapter_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True)+'\n');print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
