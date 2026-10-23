"""Map saved verified components to legacy field names; incomplete drafts only.

No canonical writer is provided. SQLite shape acceptance is not completion.
"""
import copy,hashlib,io,itertools,json,math,sqlite3,tarfile,unicodedata
from collections import Counter,defaultdict
from decimal import Decimal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/scope_review_0835'
PINS={
'pipeline.py':'122cd6a9b06821618e539e84b2fc98a031f6df065f2332c9a88e8b6b969b4d81',
'data/scope_review_0552/integrated_candidates.tar.xz':'c1a030bc66b78fec4bda307046402f9db15fa67dd386845b8d19862cd3f29e3a',
'data/scope_review_0552/independent_integrated_candidate_audit.json':'c7656fc4864e85692afac82910348f132f6b62264dd00a3877975d158f008537',
'data/scope_review_0636/payout_fragments.tar.xz':'f0729f417e0410bc0b26e1ad2d772b83a5bab3ee790c6a24703909744e6d28eb',
'data/scope_review_0636/independent_payout_fragment_audit.json':'1086167dc621eac0e44d0574407876abb5ddfc597edf61adf48aba106261386f',
'data/scope_review_0730/entry_extension_overlay.tar.xz':'10cc8901c0d7674dad51ba1fa3353d4e266b6741552e6e7c4b4bcc646b6433e6',
'data/scope_review_0730/independent_entry_extension_audit.json':'f00ce1019229d6b28cf37b14411c80063d3e3f206b7e62300035e657298f1ecd'}
MEMBERS=[('data/scope_review_0552/integrated_candidates.tar.xz','integrated_candidates.json','b9ed0533f9c8a654236a4c4c8d9d275161acdd6c154ad6c3efbaa444cb307e3f'),('data/scope_review_0636/payout_fragments.tar.xz','payout_fragments.json','b468330e89e778acc6cd481dcc3fab654d3a1bc56fcb8a5e202491439c4ccd33'),('data/scope_review_0730/entry_extension_overlay.tar.xz','entry_extension_overlay.json','3597429ce71d38b3a66ac777c083afaea4543bd3b86f33cfbd94c082d57c1503')]
def sha(b):return hashlib.sha256(b).hexdigest()
def enc(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def norm(v):return unicodedata.normalize('NFKC',v)
def inputs():
 for p,h in PINS.items():assert sha((ROOT/p).read_bytes())==h
 out=[]
 for p,name,h in MEMBERS:
  with tarfile.open(ROOT/p) as t:assert t.getnames()==[name];b=t.extractfile(name).read()
  assert sha(b)==h;out.append(json.loads(b))
 assert all(x['all_finals_complete'] is False and len(x['events'])==14 for x in out)
 return out
def observe_contract():
 c=sqlite3.connect((ROOT/'data/finals.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
 ddl=c.execute("select sql from sqlite_master where name='races'").fetchone()[0]
 types=defaultdict(Counter);present=Counter();digest=hashlib.sha256();males=entries=results=0;excluded=[]
 def visit(v,path):
  types[path][type(v).__name__]+=1
  if isinstance(v,dict):
   for k,x in v.items():visit(x,path+'/'+k)
  elif isinstance(v,list):
   for x in v:visit(x,path+'/*')
 for rid,status,p in c.execute('select race_id,status,payload from races order by race_id'):
  if rid=='1120240912030011':excluded.append({'race_id':rid,'status':status});continue
  d=json.loads(p);males+=1;entries+=len(d['entries']);results+=len(d['results'])
  b=enc([rid,status,p]);digest.update(len(b).to_bytes(8,'big'));digest.update(b)
  for k in d:present[k]+=1
  # Raw evidence JSON and rider provenance are opaque strings/dicts, not features.
  for k in ['race','entries','results','line','trifecta','trio','payouts','odds_header']:visit(d[k],'/'+k)
 assert males==5353 and entries==results==37883 and len(excluded)==1
 c.close()
 return {'races_DDL':ddl,'male_rows':males,'entry_rows':entries,'result_rows':results,'excluded':excluded,'ordered_original_payload_text_digest':digest.hexdigest(),'top_field_presence':dict(present),'observed_types':{p:dict(v) for p,v in sorted(types.items())},'scope':'observed persisted contract; not a universal schema or readiness proof'}
def readref(ref):
 b=(ROOT/ref['path']).read_bytes();assert sha(b)==ref['sha256'] and len(b)==ref['bytes'];return b
def map_event(e,p,x,contract):
 ident=e['identity'];key=ident['canonical_key'];assert ident==p['identity']==x['identity'];assert p['complete_payload'] is x['complete_payload'] is False
 refs=e['component_raw_provenance'];card=json.loads(readref(refs['card']));box=card['box']
 assert (box['open_day'],box['vel_code'],box['race_num'])==(ident['day'],ident['venue_code'],ident['race_no'])
 # These exact URLs were observed in the saved result links, not synthesized KD URLs.
 url=e['observed_provider_keys']['netkeiba_actual_result_links'][0]
 assert url=='https://keirin.netkeiba.com/race/result/?race_id='+e['observed_provider_keys']['netkeiba_race_id']
 race={'race_id':key,'race_date':ident['day'],'race_no':ident['race_no'],'source_url':url,'track':box['vel_name'],'entry_count':len(e['entries']),'race_type':e['program_raw']['event_name'],'meeting_grade':e['meeting_grade_raw'],'event_name':box['race_name'],'start_time':box['race_start_at_string'],'deadline':box['sale_to_at_string']}
 gap={'top_level':['captured','racecard_tables'],'race':['captured_at_utc'],'entries':{},'line':['groups/*/role','riders/*/line_role','provider'],'results':['result_comment'],'odds_header':['3連単/odds_as_of','3連複/odds_as_of']}
 entries=[];proofs=[]
 for a,extra in zip(e['entries'],x['entries']):
  assert a['car_no']==extra['car_no'] and a['registration_number']==extra['registration_number'] and extra['complete_payload'] is False
  old=card['race_athletes'][str(a['car_no'])];assert int(a['registration_number'])==old['un_id']
  assert a['score_decimal']==str(Decimal(old['average_point'])/1000)
  ent={'car_no':a['car_no'],'player_name':a['name_raw'],'prefecture':a['prefecture_raw'],'age':str(a['age_at_race']),'term':a['term'],'class':norm(a['class_raw']),'score':a['score_decimal'],'gear':a['gear_raw'],'style':a['style_raw'],'rider_registration_number':a['registration_number']}
  for k,v in zip(['first_count','second_count','third_count','outside_count'],a['finish_counts']):ent[k]=str(v)
  for k,v in zip(['nige_count','makuri_count','sashi_count','mark_count'],a['winning_move_counts']):ent[k]=str(v)
  ent.update(copy.deepcopy(extra['legacy_entry_fields_verified']))
  for k in ['race_id','race_date','race_no','source_url','track','race_type','start_time','deadline']:ent[k]=race[k]
  missing=['captured_at_utc','evaluation','player_profile','prediction_mark','raw_row_json','rider_source_id','rider_id_status','rider_identity_evidence']
  if 'top3_rate' not in ent:missing.append('top3_rate')
  gap['entries'][str(a['car_no'])]=missing
  entries.append(ent);proofs.append({'car_no':a['car_no'],'registration_number':a['registration_number'],'core_source_pointer':a['source_pointer'],'core_identity':copy.deepcopy(a['evidence_integration']['core_identity']),'history_adapter':copy.deepcopy(a['history_adapter']),'history_DOM':copy.deepcopy(a['evidence_integration']['history_DOM']),'entry_extension':copy.deepcopy(extra)})
 groups=[];riders={}
 for i,g in enumerate(e['published_line_groups'],1):
  groups.append([{'car_no':car} for car in g])
  for j,car in enumerate(g,1):assert str(car) not in riders;riders[str(car)]={'line_id':i,'line_size':len(g),'line_position':j}
 assert set(map(int,riders))=={a['car_no'] for a in entries}
 line={'status':'published','formation':'/'.join('-'.join(map(str,g)) for g in e['published_line_groups']),'groups':groups,'riders':riders}
 results=[]
 for a in e['results']:
  n=a['netkeiba_raw'];ch=a['chariloto_raw'];assert a['car_no']==n['car_no']==ch['car_no'] and int(a['registration_number'])==int(n['registration_number'])
  finish=a['finish_text'];assert finish==n['finish_text']==ch['finish_text']
  results.append({'car_no':a['car_no'],'player_name':ch['name'],'finish_text':finish,'finish_position':int(finish) if finish.isdecimal() else finish,'last200':n['last_lap_raw'],'margin':n['margin_raw'],'winning_method':n['move_raw'],'sb':n['SB_raw'],'raw_row_json':enc({'chariloto_original':ch,'netkeiba_original':n}).decode()})
 markets={};decimal_evidence={}
 for source,target in [('sanrentan','trifecta'),('sanrenfuku','trio')]:
  rows=e['markets'][source];assert len(rows)==(504 if target=='trifecta' else 84)
  got=set();dest=[]
  for a in rows:
   cars=tuple(a['cars']);assert cars not in got and len(set(cars))==3;got.add(cars)
   assert a['is_valid_raw'] is a['on_final_raw'] is True;d=Decimal(a['odds_decimal']);f=float(d);assert math.isfinite(f) and f>0 and Decimal(str(f))==d
   dest.append({'cars':list(cars),'odds':f,'status':'available'})
  expected=set(itertools.permutations(range(1,10),3) if target=='trifecta' else itertools.combinations(range(1,10),3));assert got==expected
  markets[target]=dest;decimal_evidence[target]=copy.deepcopy(rows)
 # Total vote counts are explicitly published in the historical card, ONE source.
 # NK official_dt is preserved separately; its meaning is not silently re-labelled odds_as_of.
 headers={label:{'total_votes':card['all_votes'][source]} for label,source in [('3連単','sanrentan'),('3連複','sanrenfuku')]}
 for v in headers.values():assert type(v['total_votes']) is int and v['total_votes']>0
 nkodds=json.loads(readref(refs['independent_odds']));nkmeta=nkodds['data']['nkrace_odds::'+e['observed_provider_keys']['netkeiba_race_id']]['official_dt']
 payload={'race_id':key,'discovered_day':ident['day'],'expected_label':e['program_raw']['event_name'],'grade':e['meeting_grade_raw'],'category':e['category'],'race':race,'entries':entries,'results':results,'line':line,'payouts':copy.deepcopy(p['payload_payouts_fragment']),'trifecta':markets['trifecta'],'trio':markets['trio'],'odds_header':headers}
 for section in ['race','entries','results']:
  for row in payload[section] if isinstance(payload[section],list) else [payload[section]]:
   for k,v in row.items():
    path='/'+section+('/*' if isinstance(payload[section],list) else '')+'/'+k
    assert type(v).__name__ in contract['observed_types'][path],(path,type(v))
 blockers=copy.deepcopy(e['evidence_integration']['readiness_inventory']['blockers'])
 blockers+=['legacy_payload_unmapped_fields_remain','line_roles_not_published_in_selected_sources','race_times_and_vote_totals_single_source_only','odds_timestamp_semantics_unverified','entry_extension_unresolved_fields_retained','complete_payload_independent_insertion_audit_pending']
 return {'identity':copy.deepcopy(ident),'payload':payload,'unmapped_fields':gap,'evidence':{'source_refs':copy.deepcopy(refs),'entries':proofs,'original_results':copy.deepcopy(e['results']),'market_decimal_sources':decimal_evidence,'line_tokens':copy.deepcopy(e['published_line_tokens_raw']),'payout_evidence':copy.deepcopy(p['evidence']),'single_source_race_metadata':{'box':copy.deepcopy(box),'program':copy.deepcopy(e['program_raw'])},'total_votes_verification':'historical_Chariloto_explicit_single_source_only','NK_official_dt_raw_not_relabelled':nkmeta,'odds_precision':'float legacy wire type, Decimal(str(float)) equals original price for every selection'},'blockers':blockers,'sql_status':'partial_adapter_draft','full_payload_SQL_adapter_verified':False,'complete_payload':False,'canonical_SQL_inserted':False,'ready_for_completed_export':False}
def reject_completed_export(draft):
 raise ValueError('incomplete adapter draft: no completed-row export capability')
def main():
 base,payout,extension=inputs();contract=observe_contract();events=[map_event(e,p,x,contract) for e,p,x in zip(base['events'],payout['events'],extension['events'])]
 doc={'schema':'rainbow_legacy_payload_adapter_draft_v1','input_pins':PINS,'observed_legacy_contract':contract,'events':events,'all_finals_complete':False,'canonical_SQL_inserts':0,'new_completed_races':0,'network_requests':0,'full_payload_SQL_adapter_verified':False,'complete_payload_insertion_audit':False}
 OUT.mkdir(parents=True,exist_ok=True);body=enc(doc);p=OUT/'legacy_payload_drafts.tar.xz'
 with tarfile.open(p,'w:xz',preset=9) as t:m=tarfile.TarInfo('legacy_payload_drafts.json');m.size=len(body);m.mtime=0;t.addfile(m,io.BytesIO(body))
 with tarfile.open(p) as t:assert t.getnames()==['legacy_payload_drafts.json'] and t.extractfile(t.getmembers()[0]).read()==body
 report={'status':'legacy_field_mapping_drafts_only','events':14,'entries':126,'results':126,'odds_selections':8232,'mapped_line_roles':0,'unmapped_top3_rates':2,'original_DB_unchanged':True,'observed_contract':contract,'field_mappings':{'entries':'saved historical core + 0730 verified fields; no current profile import','results':'NK raw display; numeric finish only when explicit decimal; Char raw retained','line':'published groups -> line id/size/position only; role/provider absent','markets':'sanrentan -> trifecta / sanrenfuku -> trio; exact Decimal roundtrip float wire type','odds_header':'Char explicit total_votes single source; odds_as_of absent; NK official_dt raw separately retained','identity':'explicit local key + actual observed NK URL; no KD ID assumptions'},'archive':{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p.read_bytes()),'member':'legacy_payload_drafts.json','member_bytes':len(body),'member_sha256':sha(body)},'full_payload_SQL_adapter_verified':False,'complete_payload_insertion_audit':False,'canonical_SQL_inserts':0,'new_completed_races':0,'all_finals_complete':False}
 (OUT/'legacy_payload_adapter_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='observed_contract'},ensure_ascii=False))
if __name__=='__main__':main()
