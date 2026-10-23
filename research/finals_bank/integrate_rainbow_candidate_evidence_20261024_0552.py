"""Lossless integration of saved candidate components and selected evidence.

Offline candidate format only. No production race rows or guessed missing values.
"""
import copy,gzip,hashlib,io,json,tarfile
from pathlib import Path
from adapt_rainbow_history_0106 import attach_verified_overlay

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/scope_review_0552'
PINS={'data/scope_review_2308/component_drafts.tar.xz': 'aed23e460c08eccba068b28f45b405f02168f908192e19ba9020d3d8bdf7f2ea', 'data/scope_review_0106/history_adapter_overlay.tar.xz': 'c81b9260ef8e3fb8f5d2186fc6477e45fdd953cec5c1a4118d69513628290248', 'data/scope_review_0253/history_raw_graph.tar.xz': '02dc8fc3185494745e2d58841a26de895d68f2c981d67ec1557f4c5a304d3146', 'data/scope_review_0414/dependency_subgraphs.tar.xz': '725d1d766f21428949babfbcf0ea8ea3f735c47c951aa9a12b7e1295920d1e80', 'data/scope_review_0115/identity_scope_provenance_review.json': '4b4819ab2b8da379f0835490bbde6e59defdfbad3bd1d1eb075767d5bf71083f', 'data/scope_review_0323/alternative_identity_review.json': 'de3e3ff5a50f87d3b43fb5d8e15e3b656a9978088c95eb4d031c6166d29143f8', 'data/scope_review_2308/independent_component_draft_audit.json': '78df35f2b7436dab1802545d33fc13ca392bfe7af5676c7407b8d4221aba01a7', 'data/scope_review_0106/independent_history_adapter_audit.json': '7f42ec57395e9abc25ef6f99391b34185e27a17bd1e0a5dc2ca03becefc030e4', 'data/scope_review_0253/independent_history_raw_graph_audit.json': 'ddf1bfbbe5ae7446ff7edd2b6eb0fce793cbbf080b4f32e1d5e0c60a86bad239', 'data/scope_review_0414/independent_dependency_subgraphs_audit.json': 'e0ef2acc3408ca8f94e58672908bbfa89b7b12302f3ffce39c2d078782c4bc3d', 'data/scope_review_0115/independent_identity_scope_provenance_audit.json': '4b2052438d10f1fe2382da3acc4b73f8a90d4c2bb844bb129ac0793c686bc2c0', 'data/scope_review_0323/independent_alternative_identity_audit.json': 'dcfb9a7acb285febdc204b83700811469bb524798cec5bfd80039f763658667f'}
ARCHIVES={
 'component':('data/scope_review_2308/component_drafts.tar.xz','component_drafts.json','eec2b7f45058a277ce6e0ab17b898735d416d17783cef807fe4a0193fc4b833c'),
 'history':('data/scope_review_0106/history_adapter_overlay.tar.xz','history_adapter_overlay.json','53a19c888f8063062d7541ee8879e95796d2999c6c57d5e597260a0c8d765522'),
 'DOM':('data/scope_review_0253/history_raw_graph.tar.xz','history_raw_graph.json','5d5529ee8b7eb6be9b730bc44f0b75ded9ebfb062d90f486a5e0305ea9b466e0'),
 'dependencies':('data/scope_review_0414/dependency_subgraphs.tar.xz','dependency_subgraphs.json','16af1719302c9ade5a559f4077a399f45f073f47ae806668f0ab2b96a68aa152'),
}
def sha(b):return hashlib.sha256(b).hexdigest()
def enc(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def norm(x):return ''.join(x.split())
class Saved:
 def __init__(self):self.arcs={};self.pool={}
 def arc(self,p,h=None,size=None):
  b=(ROOT/p).read_bytes()
  if h is not None:assert sha(b)==h
  if size is not None:assert len(b)==size
  if p not in self.arcs:
   with tarfile.open(fileobj=io.BytesIO(b)) as t:
    assert len(t.getnames())==len(set(t.getnames()));assert all(m.isfile() for m in t.getmembers())
    self.arcs[p]={m.name:t.extractfile(m).read() for m in t.getmembers()}
  return self.arcs[p]
 def source(self,ref):
  if 'stored_sha256' in ref:
   b=(ROOT/ref['path']).read_bytes();assert sha(b)==ref['stored_sha256'] and len(b)==ref['stored_bytes']
   raw=gzip.decompress(b);assert sha(raw)==ref['raw_sha256'] and len(raw)==ref['raw_bytes']
  elif isinstance(ref.get('archive'),str):
   b=self.arc(ref['archive'],ref['archive_sha256'])[ref['member']];assert sha(b)==ref['sha256'] and len(b)==ref['bytes']
  elif 'archive' in ref:
   a=ref['archive'];b=self.arc(a['path'],a['sha256'],a['bytes'])[ref['member']];assert sha(b)==ref['sha256'] and len(b)==ref['bytes']
  else:
   b=(ROOT/ref['path']).read_bytes();assert sha(b)==ref['sha256'] and len(b)==ref['bytes']
  k=sha(enc(ref));self.pool[k]=copy.deepcopy(ref);return k
 def document(self,tag):
  p,m,h=ARCHIVES[tag];raw=(ROOT/p).read_bytes();assert sha(raw)==PINS[p]
  arc=self.arc(p);assert list(arc)==[m];b=arc[m];assert sha(b)==h
  return b,json.loads(b)
def inputs():
 assert len(PINS)>=12
 for p,h in PINS.items():assert sha((ROOT/p).read_bytes())==h,p
 s=Saved();docs={tag:s.document(tag) for tag in ARCHIVES}
 identity=json.loads((ROOT/'data/scope_review_0115/identity_scope_provenance_review.json').read_bytes())
 alt=json.loads((ROOT/'data/scope_review_0323/alternative_identity_review.json').read_bytes())
 return s,docs,identity,alt
def integration():
 s,docs,identity,alt=inputs();component=docs['component'][0];hist=docs['history'][1];dom=docs['DOM'][1];deps=docs['dependencies'][1]
 assert deps['base_overlay_member_sha256']==ARCHIVES['history'][2] and deps['base_history_DOM_graph_member_sha256']==ARCHIVES['DOM'][2]
 drafts=attach_verified_overlay(component,docs['history'][0]);events=[];counts={'entries':0,'core_identity':0,'scope_events':0,'history_slots':0,'planned_unresolved':0,'prior':0,'meeting_context':0,'dependency_edges':0,'bank_unresolved':0}
 for pool in [dom['source_pool'],deps['source_pool']]:
  for ref in pool.values():s.source(ref)
 for ei,draft in enumerate(drafts):
  ident=draft['identity'];key=ident['canonical_key'];old=identity['events'][ei];he=hist['events'][ei];de=dom['events'][ei]
  assert (ident['day'],ident['venue_code'],ident['race_no'])==(old['day'],old['venue_code'],old['race_no'])
  assert he['local_key']==de['local_key']==key
  assert not draft['canonical_DB_inserted'] and not draft['all_finals_complete'] and ident['confirmed_kdreams_race_id'] is None
  official_scope=copy.deepcopy(old['official_scope_raw']);assert official_scope
  for ref in official_scope:s.source({k:v for k,v in ref.items() if k!='original_scope_proof'})
  for ref in draft['component_raw_provenance'].values():s.source(ref)
  selected=[]
  for ai,entry in enumerate(draft['entries']):
   reg=entry['registration_number'];car=entry['car_no'];proof=old['entries'][ai];node=de['entries'][ai];ha=entry['history_adapter']
   assert proof['car_no']==node['car_no']==ha['car_no']==car and proof['registration_number']==node['registration_number']==ha['registration_number']==reg
   assert norm(proof['name'])==norm(entry['name_raw']) and proof['term']==entry['term'] and proof['prefecture']=={'神奈':'神奈川','和歌':'和歌山','北海':'北海道'}.get(entry['prefecture_raw'],entry['prefecture_raw'])
   if proof['locator'] is not None:
    source_ids=[s.source(proof['locator'])];core={'kind':'selected_official_identity','original_proof':copy.deepcopy(proof),'source_ids':source_ids}
   else:
    target=alt['target'];assert target['local_key']==key and target['car_no']==car and target['registration_number']==reg and target['term']==entry['term'] and target['prefecture_at_event']==entry['prefecture_raw']
    assert norm(target['name'])==norm(entry['name_raw']);assert alt['original_transfer_page_body_status']=='still_unavailable_HTTP403_not_recovered_not_retried'
    source_ids=[s.source(alt['official_allocation']['locator'])]+[s.source(ref) for ref in alt['event_time_sources'].values()]
    core={'kind':'alternative_official_identity_plus_event_time_cards','original_0115_failed_proof_retained':copy.deepcopy(proof),'alternative_proof':copy.deepcopy(alt),'source_ids':source_ids,'affiliation_interval_inferred':False,'transfer_body_recovered':False}
   assert len(ha['adapted_history_slots'])==len(node['slots'])==3
   for si,(slot,loc) in enumerate(zip(ha['adapted_history_slots'],node['slots'])):
    assert loc['overlay_pointer']==['events',ei,'entries',ai,'adapted_history_slots',si]
    assert loc['original_history_sha256']==sha(enc(slot['original'])) and loc['kind']==slot['kind'] and loc['actual_start_date']==slot['actual_start_date']
    assert slot['original']==entry['history_slots'][si]['original'];counts['history_slots']+=1
    if slot['kind']=='planned_slot_date_unresolved':assert slot['actual_start_date'] is None and slot['independent_group'] is None;counts['planned_unresolved']+=1
    elif slot['kind']=='actual_prior_meeting':counts['prior']+=1
    else:assert slot['kind']=='current_meeting_context_not_participation' and slot['rider_firstday_participation_inferred'] is False;counts['meeting_context']+=1
   edges=[]
   for edge in deps['edges']:
    if edge['local_key']!=key or edge['car_no']!=car:continue
    assert edge['registration_number']==reg and edge['overlay_pointer'][:4]==['events',ei,'entries',ai]
    p=edge['overlay_pointer'];v=hist
    for part in p:v=v[part]
    group=v['independent_group'] if p[-2]=='adapted_history_slots' else v['NK_group'];assert group['group_id']==edge['group_id'] and sha(enc(group))==edge['group_sha256']
    edges.append(copy.deepcopy(edge))
   entry['evidence_integration']={'core_identity':core,'history_DOM':copy.deepcopy(node),'history_dependency_edges':edges}
   counts['dependency_edges']+=len(edges);counts['entries']+=1;counts['core_identity']+=1
   selected.extend(edges)
  planned=[{'car_no':a['car_no'],'registration_number':a['registration_number'],'slot':v['slot']} for a in draft['entries'] for v in a['history_adapter']['adapted_history_slots'] if v['kind']=='planned_slot_date_unresolved']
  bank_unresolved=draft['historical_bank']['status']=='unresolved';counts['bank_unresolved']+=bank_unresolved
  inventory={'selected_core_identity_raw_bound':9,'official_scope_raw_bound':True,'history_adapter_and_DOM_bound':True,'supporting_dependencies_bound':True,'planned_slots_unresolved':planned,'historical_bank_unresolved':bank_unresolved,'production_payload_schema_verified':False,'independent_complete_payload_SQL_insertion_audit':False,'all_registry_source_history_graph_verified':False,'broader_target_scope_verified':False,'ready_for_canonical_completed_row':False,'blockers':['production_payload_schema_and_adapter_not_verified','independent_complete_payload_SQL_insertion_audit_pending','all_registry_source_history_graph_pending','broader_target_scope_pending']+(['planned_slot_dates_explicit_IDs_unresolved'] if planned else [])+(['historical_bank_unresolved'] if bank_unresolved else [])}
  draft['evidence_integration']={'official_scope_raw':official_scope,'dependency_roots_used':{k:copy.deepcopy(deps['roots'][k]) for k in sorted({e['root'] for e in selected})},'readiness_inventory':inventory,'status':'selected_evidence_integrated_candidate_only'}
  events.append(draft);counts['scope_events']+=1
 assert counts=={'entries':126,'core_identity':126,'scope_events':14,'history_slots':378,'planned_unresolved':17,'prior':235,'meeting_context':126,'dependency_edges':21,'bank_unresolved':13}
 return {'schema':'rainbow_integrated_candidate_v1','events':events,'source_pool':s.pool,'input_pins':PINS,'counts':counts,'status':'selected_evidence_integrated_candidate_only','all_finals_complete':False,'canonical_SQL_inserts':0,'new_completed_races':0,'network_requests':0,'production_payload_insertion_audit':False,'original_component_recoverable':True}
def reject_completed_export(event):
 """Candidate schema is deliberately not accepted as a production payload."""
 inventory=event['evidence_integration']['readiness_inventory']
 raise ValueError('Candidate format; production export unavailable: '+','.join(inventory['blockers']))
def recover_original(document):
 out={'all_finals_complete':False,'events':copy.deepcopy(document['events']),'schema':'rainbow_component_drafts_v1'}
 for e in out['events']:
  e.pop('evidence_integration');e.pop('history_adapter_status')
  for a in e['entries']:a.pop('history_adapter');a.pop('evidence_integration')
 return out
def main():
 doc=integration();raw=enc(doc);OUT.mkdir(parents=True,exist_ok=True);p=OUT/'integrated_candidates.tar.xz'
 with tarfile.open(p,'w:xz',preset=9) as t:
  m=tarfile.TarInfo('integrated_candidates.json');m.size=len(raw);m.mtime=0;t.addfile(m,io.BytesIO(raw))
 with tarfile.open(p) as t:assert t.getnames()==['integrated_candidates.json'] and t.extractfile(t.getmembers()[0]).read()==raw
 report={'status':doc['status'],'counts':doc['counts'],'source_locators':len(doc['source_pool']),'archive':{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p.read_bytes()),'member':'integrated_candidates.json','member_bytes':len(raw),'member_sha256':sha(raw)},'input_pins':PINS,'canonical_SQL_inserts':0,'new_completed_races':0,'network_requests':0,'all_finals_complete':False,'production_payload_insertion_audit':False}
 (OUT/'integrated_candidate_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n');print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
