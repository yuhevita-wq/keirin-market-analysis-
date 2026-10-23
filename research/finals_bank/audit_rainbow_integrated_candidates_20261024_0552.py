"""Independent audit of selected evidence integration, never completed SQL rows."""
import copy,gzip,hashlib,io,json,sqlite3,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PINS={'data/scope_review_2308/component_drafts.tar.xz': 'aed23e460c08eccba068b28f45b405f02168f908192e19ba9020d3d8bdf7f2ea', 'data/scope_review_0106/history_adapter_overlay.tar.xz': 'c81b9260ef8e3fb8f5d2186fc6477e45fdd953cec5c1a4118d69513628290248', 'data/scope_review_0253/history_raw_graph.tar.xz': '02dc8fc3185494745e2d58841a26de895d68f2c981d67ec1557f4c5a304d3146', 'data/scope_review_0414/dependency_subgraphs.tar.xz': '725d1d766f21428949babfbcf0ea8ea3f735c47c951aa9a12b7e1295920d1e80', 'data/scope_review_0115/identity_scope_provenance_review.json': '4b4819ab2b8da379f0835490bbde6e59defdfbad3bd1d1eb075767d5bf71083f', 'data/scope_review_0323/alternative_identity_review.json': 'de3e3ff5a50f87d3b43fb5d8e15e3b656a9978088c95eb4d031c6166d29143f8', 'data/scope_review_2308/independent_component_draft_audit.json': '78df35f2b7436dab1802545d33fc13ca392bfe7af5676c7407b8d4221aba01a7', 'data/scope_review_0106/independent_history_adapter_audit.json': '7f42ec57395e9abc25ef6f99391b34185e27a17bd1e0a5dc2ca03becefc030e4', 'data/scope_review_0253/independent_history_raw_graph_audit.json': 'ddf1bfbbe5ae7446ff7edd2b6eb0fce793cbbf080b4f32e1d5e0c60a86bad239', 'data/scope_review_0414/independent_dependency_subgraphs_audit.json': 'e0ef2acc3408ca8f94e58672908bbfa89b7b12302f3ffce39c2d078782c4bc3d', 'data/scope_review_0115/independent_identity_scope_provenance_audit.json': '4b2052438d10f1fe2382da3acc4b73f8a90d4c2bb844bb129ac0793c686bc2c0', 'data/scope_review_0323/independent_alternative_identity_audit.json': 'dcfb9a7acb285febdc204b83700811469bb524798cec5bfd80039f763658667f'}
def digest(b):return hashlib.sha256(b).hexdigest()
def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def text(x):return ''.join(x.split())
def read_archive(p,member=None):
 with tarfile.open(ROOT/p) as f:
  ns=f.getnames();assert len(ns)==len(set(ns));data={n:f.extractfile(n).read() for n in ns}
 if member is not None:assert list(data)==[member];return json.loads(data[member])
 return data
class RawCheck:
 def __init__(self):self.cache={};self.expected={}
 def check(self,ref):
  if 'archive' in ref:
   a=ref['archive'];path=a if isinstance(a,str) else a['path'];b=(ROOT/path).read_bytes()
   assert digest(b)==(ref['archive_sha256'] if isinstance(a,str) else a['sha256'])
   if isinstance(a,dict):assert len(b)==a['bytes']
   if path not in self.cache:self.cache[path]=read_archive(path)
   raw=self.cache[path][ref['member']];assert digest(raw)==ref['sha256'] and len(raw)==ref['bytes']
  elif 'stored_sha256' in ref:
   b=(ROOT/ref['path']).read_bytes();assert digest(b)==ref['stored_sha256'] and len(b)==ref['stored_bytes'];raw=gzip.decompress(b);assert digest(raw)==ref['raw_sha256'] and len(raw)==ref['raw_bytes']
  else:
   raw=(ROOT/ref['path']).read_bytes();assert digest(raw)==ref['sha256'] and len(raw)==ref['bytes']
  self.expected[digest(canonical(ref))]=ref
def originals():
 for p,h in PINS.items():assert digest((ROOT/p).read_bytes())==h
 return (read_archive('data/scope_review_2308/component_drafts.tar.xz','component_drafts.json'),read_archive('data/scope_review_0106/history_adapter_overlay.tar.xz','history_adapter_overlay.json'),read_archive('data/scope_review_0253/history_raw_graph.tar.xz','history_raw_graph.json'),read_archive('data/scope_review_0414/dependency_subgraphs.tar.xz','dependency_subgraphs.json'),json.loads((ROOT/'data/scope_review_0115/identity_scope_provenance_review.json').read_bytes()),json.loads((ROOT/'data/scope_review_0323/alternative_identity_review.json').read_bytes()))
BASE,HISTORY,DOM,DEPS,IDENTITY,ALT=originals()
def check(doc):
 assert doc['schema']=='rainbow_integrated_candidate_v1' and doc['status']=='selected_evidence_integrated_candidate_only'
 assert doc['all_finals_complete'] is False and doc['production_payload_insertion_audit'] is False and doc['canonical_SQL_inserts']==doc['new_completed_races']==doc['network_requests']==0
 assert doc['input_pins']==PINS and doc['original_component_recoverable'] is True
 assert len(doc['events'])==len(BASE['events'])==14
 raw=RawCheck();counts={'entries':0,'core_identity':0,'scope_events':0,'history_slots':0,'planned_unresolved':0,'prior':0,'meeting_context':0,'dependency_edges':0,'bank_unresolved':0}
 for pool in [DOM['source_pool'],DEPS['source_pool']]:
  for ref in pool.values():raw.check(ref)
 for i,(event,base) in enumerate(zip(doc['events'],BASE['events'])):
  h=HISTORY['events'][i];d=DOM['events'][i];official=IDENTITY['events'][i];key=base['identity']['canonical_key']
  assert event['identity']==base['identity'] and h['local_key']==d['local_key']==key
  assert event['history_adapter_status']=='attached_candidate_only_unresolved_retained'
  stripped=copy.deepcopy(event);stripped.pop('evidence_integration');stripped.pop('history_adapter_status')
  for a in stripped['entries']:a.pop('history_adapter');a.pop('evidence_integration')
  assert stripped==base
  planned=[];event_edges=[]
  for j,(a,b) in enumerate(zip(event['entries'],base['entries'])):
   ha=h['entries'][j];dn=d['entries'][j];ip=official['entries'][j]
   assert a['history_adapter']==ha and a['evidence_integration']['history_DOM']==dn
   reg=a['registration_number'];car=a['car_no'];assert reg==ha['registration_number']==dn['registration_number']==ip['registration_number'];assert car==ha['car_no']==dn['car_no']==ip['car_no']
   assert text(ip['name'])==text(a['name_raw']) and ip['term']==a['term']
   assert ip['prefecture']=={'神奈':'神奈川','和歌':'和歌山','北海':'北海道'}.get(a['prefecture_raw'],a['prefecture_raw'])
   core=a['evidence_integration']['core_identity']
   if ip['locator'] is None:
    assert key==ALT['target']['local_key'] and reg==ALT['target']['registration_number'] and car==ALT['target']['car_no']
    refs=[ALT['official_allocation']['locator']]+list(ALT['event_time_sources'].values())
    expected={'kind':'alternative_official_identity_plus_event_time_cards','original_0115_failed_proof_retained':ip,'alternative_proof':ALT,'source_ids':[digest(canonical(x)) for x in refs],'affiliation_interval_inferred':False,'transfer_body_recovered':False}
   else:
    refs=[ip['locator']];expected={'kind':'selected_official_identity','original_proof':ip,'source_ids':[digest(canonical(ip['locator']))]}
   assert core==expected
   for ref in refs:raw.check(ref)
   edges=[e for e in DEPS['edges'] if e['local_key']==key and e['car_no']==car];assert a['evidence_integration']['history_dependency_edges']==edges;event_edges+=edges
   for k,(slot,node) in enumerate(zip(ha['adapted_history_slots'],dn['slots'])):
    assert slot['original']==b['history_slots'][k]['original'];assert node['overlay_pointer']==['events',i,'entries',j,'adapted_history_slots',k]
    assert node['original_history_sha256']==digest(canonical(slot['original']))
    if slot['kind']=='planned_slot_date_unresolved':
     assert slot['actual_start_date'] is None and slot['independent_group'] is None;planned.append({'car_no':car,'registration_number':reg,'slot':slot['slot']});counts['planned_unresolved']+=1
    elif slot['kind']=='actual_prior_meeting':counts['prior']+=1
    else:assert slot['kind']=='current_meeting_context_not_participation' and not slot['rider_firstday_participation_inferred'];counts['meeting_context']+=1
    counts['history_slots']+=1
   counts['entries']+=1;counts['core_identity']+=1
  evi=event['evidence_integration'];assert evi['status']=='selected_evidence_integrated_candidate_only';assert evi['official_scope_raw']==official['official_scope_raw']
  for ref in evi['official_scope_raw']:raw.check({k:v for k,v in ref.items() if k!='original_scope_proof'})
  for ref in base['component_raw_provenance'].values():raw.check(ref)
  assert evi['dependency_roots_used']=={k:DEPS['roots'][k] for k in {edge['root'] for edge in event_edges}}
  bank=base['historical_bank']['status']=='unresolved'
  expected_inventory={'selected_core_identity_raw_bound':9,'official_scope_raw_bound':True,'history_adapter_and_DOM_bound':True,'supporting_dependencies_bound':True,'planned_slots_unresolved':planned,'historical_bank_unresolved':bank,'production_payload_schema_verified':False,'independent_complete_payload_SQL_insertion_audit':False,'all_registry_source_history_graph_verified':False,'broader_target_scope_verified':False,'ready_for_canonical_completed_row':False,'blockers':['production_payload_schema_and_adapter_not_verified','independent_complete_payload_SQL_insertion_audit_pending','all_registry_source_history_graph_pending','broader_target_scope_pending']+(['planned_slot_dates_explicit_IDs_unresolved'] if planned else [])+(['historical_bank_unresolved'] if bank else [])}
  assert evi['readiness_inventory']==expected_inventory
  counts['scope_events']+=1;counts['dependency_edges']+=len(event_edges);counts['bank_unresolved']+=bank
 assert counts==doc['counts']=={'entries':126,'core_identity':126,'scope_events':14,'history_slots':378,'planned_unresolved':17,'prior':235,'meeting_context':126,'dependency_edges':21,'bank_unresolved':13}
 assert doc['source_pool']==raw.expected
 return counts
def main():
 report=json.loads((ROOT/'data/scope_review_0552/integrated_candidate_review.json').read_bytes());a=report['archive'];b=(ROOT/a['path']).read_bytes();assert digest(b)==a['sha256'] and len(b)==a['bytes'];members=read_archive(a['path']);assert list(members)==[a['member']];payload=members[a['member']];assert digest(payload)==a['member_sha256'] and len(payload)==a['member_bytes'];doc=json.loads(payload);counts=check(doc)
 db=sqlite3.connect(':memory:');db.execute("create table candidate_drafts(key text primary key,status text check(status='blocked'),payload text)")
 for e in doc['events']:db.execute('insert into candidate_drafts values(?,?,?)',(e['identity']['canonical_key'],'blocked',canonical(e).decode()))
 assert [json.loads(row[0]) for row in db.execute('select payload from candidate_drafts order by rowid')]==doc['events'];assert db.execute("select name from sqlite_master where type='table'").fetchall()==[('candidate_drafts',)];db.close()
 import integrate_rainbow_candidate_evidence_0552 as producer
 assert producer.recover_original(doc)==BASE
 for e in doc['events']:
  try:producer.reject_completed_export(e)
  except ValueError:pass
  else:raise AssertionError('accepted unfinished candidate')
 tests={
 'historical_score':lambda x:x['events'][0]['entries'][0].__setitem__('score_decimal','999.999'),
 'explicit_registration':lambda x:x['events'][0]['entries'][0].__setitem__('registration_number','099999'),
 'official_identity_binding':lambda x:x['events'][0]['entries'][0]['evidence_integration']['core_identity']['original_proof'].__setitem__('term','1'),
 'scope_binding':lambda x:x['events'][0]['evidence_integration']['official_scope_raw'][0].__setitem__('sha256','0'*64),
 'raw_history_year':lambda x:x['events'][0]['entries'][0]['history_adapter']['adapted_history_slots'][0]['original'].__setitem__('open_from','20230603'),
 'history_DOM_pointer':lambda x:x['events'][0]['entries'][0]['evidence_integration']['history_DOM']['slots'][0].__setitem__('overlay_pointer',[]),
 'remove_dependency_edge':lambda x:next(a for e in x['events'] for a in e['entries'] if a['evidence_integration']['history_dependency_edges'])['evidence_integration']['history_dependency_edges'].pop(),
 'false_transfer_recovery':lambda x:next(a for e in x['events'] for a in e['entries'] if a['evidence_integration']['core_identity']['kind']=='alternative_official_identity_plus_event_time_cards')['evidence_integration']['core_identity'].__setitem__('transfer_body_recovered',True),
 'planned_success':lambda x:next(s for e in x['events'] for a in e['entries'] for s in a['history_adapter']['adapted_history_slots'] if s['kind']=='planned_slot_date_unresolved').__setitem__('actual_start_date','2023-01-01'),
 'bank_success':lambda x:x['events'][0]['historical_bank'].__setitem__('status','verified'),
 'readiness_promotion':lambda x:x['events'][0]['evidence_integration']['readiness_inventory'].__setitem__('ready_for_canonical_completed_row',True),
 'completion_flag':lambda x:x.__setitem__('all_finals_complete',True),
 'source_bytes':lambda x:next(iter(x['source_pool'].values())).__setitem__('bytes',1),
 }
 rejected=[]
 for name,mutate in tests.items():
  changed=copy.deepcopy(doc);mutate(changed)
  try:check(changed)
  except (AssertionError,KeyError,TypeError):rejected.append(name)
  else:raise AssertionError('mutation accepted '+name)
 output={'status':'selected_candidate_evidence_integration_audit_passed','counts':counts,'source_locators':len(doc['source_pool']),'lossless_original_component_recovery':True,'isolated_blocked_candidate_roundtrip_rows':14,'independent_complete_payload_SQL_insertion_audit':False,'completed_export_rejections':14,'mutations_rejected':rejected,'canonical_SQL_inserts':0,'second_SQLite_artifact_saved':False,'new_completed_races':0,'all_finals_complete':False}
 (ROOT/'data/scope_review_0552/independent_integrated_candidate_audit.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,sort_keys=True)+'\n');print(json.dumps(output,ensure_ascii=False))
if __name__=='__main__':main()
