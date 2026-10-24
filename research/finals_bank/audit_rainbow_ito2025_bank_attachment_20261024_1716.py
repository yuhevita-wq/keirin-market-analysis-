"""Independent field mapping and whole-document preservation audit for bank attachment."""
import copy,hashlib,importlib.util,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/scope_review_1716'
BASE='63f7e29a955bb3daf179747ba958863ab43cbe472386d3d806e4a5d3ec8acfee'
DELTA='53348cb1c4e5a385370563c727de8f07ea61a85412c9102c8b8dfe46d97690fc'
KEY='keirin:2025-12-14:37:09'
def sha(b):return hashlib.sha256(b).hexdigest()
def enc(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def module(p,name):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def main():
 # The prior composition is pinned and was already independently audited. No new producer extraction is used here.
 p=ROOT/'recovery/adapt_rainbow_literal_table_1118.py';assert sha(p.read_bytes())=='b3bbc39aefea5963f7552cb833335dc5cb88b81ccee500949ef1ea38649dd992';old=module(p,'old1118_audit1716')
 bodies={}
 for path,member in old.MEMBERS.items():
  raw=(ROOT/path).read_bytes();assert sha(raw)==old.PINS[path]
  with tarfile.open(ROOT/path) as t:assert t.getnames()==[member];bodies[path]=t.extractfile(member).read()
 p=ROOT/'data/scope_review_1118/literal_table_profile_overlay.tar.xz';assert sha(p.read_bytes())=='3c4526bb77fb1047b57b5bf88da48a35cda461b3da9c012ed2e731dbf863fb48'
 with tarfile.open(p) as t:o=t.extractfile('literal_table_profile_overlay.json').read()
 baseline=old.attach(bodies,o);base=enc(baseline);assert sha(base)==BASE
 raw=(ROOT/'data/bank_sources_1227/ito2025_candidate_bank_delta.json').read_bytes();assert sha(raw)==DELTA;d=json.loads(raw);record=d['bank_record']
 assert sha((ROOT/'data/bank_sources_1227/ito20251211.pdf').read_bytes())=='181254589a44b6b9a7dfd35d78c4eb4aa6de9158fc4d0b217eef7a874f56b573'
 assert sha((ROOT/'data/bank_sources_1227/independent_ito2025_bank_audit.json').read_bytes())=='2324a315544ea302b348c56a7258fbb19b212b7755151a40d607d0d39cb3c842'
 assert record['geometry']=={'circumference_m':333,'minashi_straight_m':46.6,'home_width_m':11.0,'back_width_m':9.3,'center_width_m':7.8,'home_cant_dms':[3,26,1],'center_cant_dms':[34,41,9]}
 target=next(e for e in baseline['events'] if e['identity']['canonical_key']==KEY);assert target['identity']==d['identity'];assert record['valid_from']=='2025-12-11' and record['valid_until_exclusive']=='2025-12-15'
 assert target['payload']['race']['race_date']=='2025-12-14' and target['payload']['race']['race_no']==9 and target['identity']['venue_code']=='37'
 bank={}
 for dst,src in [('bank_version_id','bank_version_id'),('venue_code','venue_code'),('track','track'),('valid_from','valid_from'),('valid_until_exclusive','valid_until_exclusive'),('geometry','geometry'),('evidence','sources'),('validity_basis','validity_basis'),('status','review_status')]:bank[dst]=copy.deepcopy(record[src])
 bank['race_date']='2025-12-14'
 evidence={'delta_sha256':DELTA,'base_composed_sha256':BASE,'source_provider_count':1,'review_method':d['manual_visual_review']['method'],'official_news_proof':copy.deepcopy(d['official_news_proof']),'manual_visual_review':copy.deepcopy(d['manual_visual_review']),'source_record':copy.deepcopy(record),'canonical_save_pending':True,'completed_payload_verified':False}
 expected=copy.deepcopy(baseline);e=next(x for x in expected['events'] if x['identity']['canonical_key']==KEY);e['payload']['bank']=bank;e['dated_bank_attachment_evidence']=evidence
 def check(doc):
  assert doc==expected
  for a,b in zip(doc['events'],baseline['events']):
   assert a['blockers']==b['blockers'] and a['unmapped_fields']==b['unmapped_fields'];assert a['complete_payload'] is a['canonical_SQL_inserted'] is a['ready_for_completed_export'] is False
 producer=module(ROOT/'recovery/attach_rainbow_ito2025_bank_1716.py','producer_under_audit1716');actual=producer.attach(base,raw);check(actual)
 fragment=json.loads((OUT/'ito2025_bank_attachment.json').read_bytes());assert fragment=={'schema':'rainbow_exact_event_partial_bank_attachment_v1','identity':target['identity'],'payload_bank_fragment':bank,'attachment_evidence':evidence,'complete_payload':False,'canonical_SQL_inserted':False,'all_finals_complete':False}
 restored=copy.deepcopy(actual);e=next(x for x in restored['events'] if x['identity']['canonical_key']==KEY);e['payload'].pop('bank');e.pop('dated_bank_attachment_evidence');assert restored==baseline
 mutations=[]
 def reject(name,change):
  x=copy.deepcopy(actual);e=next(y for y in x['events'] if y['identity']['canonical_key']==KEY);change(x,e)
  try:check(x)
  except AssertionError:mutations.append(name)
  else:raise AssertionError('accepted '+name)
 reject('333_33_correction',lambda x,e:e['payload']['bank']['geometry'].update(circumference_m=333.33))
 reject('date_extension',lambda x,e:e['payload']['bank'].update(valid_until_exclusive='2026-01-01'))
 reject('wrong_race_date',lambda x,e:e['payload']['bank'].update(race_date='2025-12-13'))
 reject('wrong_source_SHA',lambda x,e:e['payload']['bank']['evidence'][0].update(sha256='0'*64))
 reject('false_second_provider',lambda x,e:e['dated_bank_attachment_evidence'].update(source_provider_count=2))
 reject('false_save_success',lambda x,e:e['dated_bank_attachment_evidence'].update(canonical_save_pending=False))
 reject('false_complete',lambda x,e:e.update(complete_payload=True))
 reject('blocker_removed',lambda x,e:e.update(blockers=[]))
 reject('entry_changed',lambda x,e:e['payload']['entries'][0].update(score='999'))
 reject('other_event_changed',lambda x,e:x['events'][0]['payload']['results'][0].update(finish_position=99))
 for label,b,r in [('base_byte',base+b' ',raw),('delta_byte',base,raw+b' ')]:
  try:producer.attach(b,r)
  except AssertionError:mutations.append(label)
  else:raise AssertionError('accepted '+label)
 report={'status':'independent_partial_bank_attachment_audit_passed','target':KEY,'base_composed_sha256':BASE,'delta_sha256':DELTA,'fragment_sha256':sha((OUT/'ito2025_bank_attachment.json').read_bytes()),'whole_document_expected_equal':True,'lossless_previous_composed_recovery':True,'mutations_rejected':mutations,'historical_scores_results_markets_identity_history_unchanged':True,'blockers_unmapped_fields_unchanged':True,'catalog_modified':False,'DB_changed':False,'canonical_SQL_inserts':0,'new_completed_races':0,'all_finals_complete':False,'full_payload_SQL_adapter_verified':False,'complete_payload_insertion_audit':False,'pending_canonical_save':True,'network_requests':0,'geometry_provider_count':1}
 (OUT/'independent_bank_attachment_audit.json').write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
