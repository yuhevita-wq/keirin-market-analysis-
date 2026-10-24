"""Audit derived legacy identity certificates against pinned proofs + event cards."""
import copy,datetime,gzip,hashlib,importlib.util,io,json,sqlite3,tarfile,unicodedata
from pathlib import Path
from lxml import html
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/scope_review_1037'
def h(b):return hashlib.sha256(b).hexdigest()
def js(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def clean(v):return ''.join(unicodedata.normalize('NFKC',v).split())
def full(v):return {'神奈':'神奈川','和歌':'和歌山','北海':'北海道'}.get(clean(v),clean(v))
def arc(path,sha,member):
 assert h((ROOT/path).read_bytes())==sha
 with tarfile.open(ROOT/path) as t:assert t.getnames()==[member];return t.extractfile(member).read()
def raw(ref):
 if 'raw_sha256' in ref:
  original=(ROOT/ref['path']).read_bytes();assert len(original)==ref['stored_bytes'] and h(original)==ref['stored_sha256'];b=gzip.decompress(original);assert len(b)==ref['raw_bytes'] and h(b)==ref['raw_sha256'];return b
 if 'archive' in ref:
  ar=ref['archive'];p=ar if isinstance(ar,str) else ar['path'];expected=ref['archive_sha256'] if isinstance(ar,str) else ar['sha256'];original=(ROOT/p).read_bytes();assert h(original)==expected
  if isinstance(ar,dict):assert len(original)==ar['bytes']
  with tarfile.open(fileobj=io.BytesIO(original)) as t:assert len(t.getnames())==len(set(t.getnames()));b=t.extractfile(ref['member']).read()
 else:b=(ROOT/ref['path']).read_bytes()
 assert len(b)==ref['bytes'] and h(b)==ref['sha256'];return b
def validate(doc,source):
 assert doc['schema']=='rainbow_exact_event_identity_field_overlay_v1' and len(doc['events'])==14 and doc['base_member_sha256']=='ff23321971f27f6608b572535ad1055dac21f5d07fd9d53c7efc3aefa94bc984'
 assert doc['all_finals_complete'] is doc['full_payload_SQL_adapter_verified'] is doc['complete_payload_insertion_audit'] is False and doc['canonical_SQL_inserts']==doc['new_completed_races']==doc['network_requests']==0
 for path,sha in doc['input_pins'].items():assert h((ROOT/path).read_bytes())==sha
 for event,original in zip(doc['events'],source['events']):
  ident=original['identity'];assert event['identity']==ident and event['complete_payload'] is event['canonical_SQL_inserted'] is False and len(event['entries'])==9
  refs=original['component_raw_provenance'];card=json.loads(raw(refs['card']));op=html.fromstring(raw(refs['independent_card']).decode());hidden={a.get('name'):a.get('value') for a in op.xpath('//input[@name]')};assert (hidden['joCode'],hidden['kaisaiBi'],int(hidden['raceNo']))==(ident['venue_code'],ident['day'].replace('-',''),ident['race_no'])
  for adapted,a in zip(event['entries'],original['entries']):
   car=a['car_no'];reg=a['registration_number'];p=a['evidence_integration']['core_identity'];certificate=adapted['rider_identity_evidence'];assert adapted['car_no']==car and adapted['registration_number']==reg
   c=card['race_athletes'][str(car)];assert c['un_id']==int(reg) and c['entry_num']==car and clean(c['name'])==clean(a['name_raw']) and c['graduate_period']==a['term'] and c['pref_name']==a['prefecture_raw']
   link=op.xpath('//a[@href=$v]',v='./PlayerDetail.do?playerCd='+reg);assert len(link)==1;td=link[0]
   while td.tag!='td':td=td.getparent()
   prefecture=td.getnext().text_content().strip();row=td.getparent();assert row.xpath('./td[@class=$v]',v='no'+str(car))[0].text_content().strip()==str(car) and full(prefecture)==full(a['prefecture_raw'])
   locatorlist=[source['source_pool'][sid] for sid in p['source_ids']]
   for loc in locatorlist:raw(loc)
   if p['kind']=='selected_official_identity':
    origin=p['original_proof'];assert origin['registration_number']==reg and clean(origin['name'])==clean(a['name_raw']) and origin['term']==a['term'] and full(origin['prefecture'])==full(c['pref_name']);failed=None;url=origin['source_url']
   else:
    assert p['kind']=='alternative_official_identity_plus_event_time_cards';alt=p['alternative_proof'];origin=alt['official_allocation'];failed=p['original_0115_failed_proof_retained'];url=origin['source_url'];target=alt['target'];assert (target['registration_number'],target['day'],target['car_no'],target['term'],target['prefecture_at_event'])==(reg,ident['day'],car,a['term'],a['prefecture_raw']);assert clean(target['name'])==clean(a['name_raw']);assert p['transfer_body_recovered'] is p['affiliation_interval_inferred'] is False and alt['original_transfer_page_body_status']=='still_unavailable_HTTP403_not_recovered_not_retried'
   end=(datetime.date.fromisoformat(ident['day'])+datetime.timedelta(days=1)).isoformat()
   joint={'registration_number':reg,'name':clean(a['name_raw']),'term':a['term'],'prefecture':a['prefecture_raw'],'source_url':url,'source_kind':'official_identity_plus_exact_event_prefecture_join','valid_from':ident['day'],'valid_until_exclusive':end,'applicability_kind':'derived_exact_event_only_not_source_affiliation_interval','official_identity_fields':['registration_number','name','term'],'historical_prefecture_sources':{k:refs[k] for k in ['card','independent_card']},'official_original_record':origin,'official_raw_locators':locatorlist,'official_prefecture_used_as_historical_prefecture':False}
   expected={'matching_fields':['registration_number','name','term','prefecture'],'official_sources':[joint],'use':'registration identity and exact reviewed event only; historical performance unchanged','exact_event':ident,'historical_prefecture_observation':{'chariloto_raw':c['pref_name'],'oddspark_raw':prefecture,'source_day':ident['day'],'affiliation_interval_inferred':False},'core_proof_kind':p['kind'],'original_core_proof':p,'original_failed_transfer_proof_retained':failed,'all_registry_source_history_graph_verified':False,'current_performance_imported':False}
   assert certificate==expected
   # The existing identity audit's field predicate, limited to this evidence
   # container; its whole-ledger audit is not invoked or relaxed.
   assert certificate['official_sources'] and all(x['registration_number']==reg and x['name']==clean(a['name_raw']) and x['term']==a['term'] and x['prefecture']==a['prefecture_raw'] and x['valid_from']<=ident['day']<x['valid_until_exclusive'] for x in certificate['official_sources'])
def main():
 sb=arc('data/scope_review_0552/integrated_candidates.tar.xz','c1a030bc66b78fec4bda307046402f9db15fa67dd386845b8d19862cd3f29e3a','integrated_candidates.json');assert h(sb)=='b9ed0533f9c8a654236a4c4c8d9d275161acdd6c154ad6c3efbaa444cb307e3f';source=json.loads(sb)
 bb=arc('data/scope_review_0835/legacy_payload_drafts.tar.xz','28ceabb5cafc8035193c22813b7a9617aeb1d9b84279db9fa942a23da5da9ab2','legacy_payload_drafts.json');base=json.loads(bb)
 ob=arc('data/scope_review_0916/raw_rows_profile_detail_overlay.tar.xz','b799216c9c87897d9f854bb913d17042343b0448df61696724a304d9a7e5b672','raw_rows_profile_detail_overlay.json')
 report=json.loads((OUT/'identity_field_review.json').read_bytes());ar=report['archive'];body=arc(ar['path'],ar['sha256'],ar['member']);assert len(body)==ar['member_bytes'] and h(body)==ar['member_sha256'];doc=json.loads(body);validate(doc,source)
 spec=importlib.util.spec_from_file_location('identity1037',ROOT/'recovery/adapt_rainbow_identity_field_1037.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 updated=mod.attach(bb,body,ob);oldspec=importlib.util.spec_from_file_location('raw0916',ROOT/'recovery/adapt_rainbow_raw_rows_0916.py');old=importlib.util.module_from_spec(oldspec);oldspec.loader.exec_module(old);original=old.attach(bb,ob)
 restored=copy.deepcopy(updated)
 for e,b in zip(restored['events'],original['events']):
  for a in e['payload']['entries']:a.pop('rider_identity_evidence')
  del e['identity_field_attachment_evidence'];e['unmapped_fields']=copy.deepcopy(b['unmapped_fields'])
 assert restored==original
 # Exact live DDL with all prior fields and the new evidence field, PARTIAL.
 mem=sqlite3.connect(':memory:');mem.execute(base['observed_legacy_contract']['races_DDL'])
 for e in updated['events']:
  p=e['payload'];assert e['sql_status']=='partial_adapter_draft' and e['ready_for_completed_export'] is e['complete_payload'] is False
  mem.execute('insert into races values (?,?,?,?,?,?,?,?,?,?)',(p['race_id'],e['identity']['day'],p['race']['source_url'],p['expected_label'],p['grade'],p['category'],'partial_adapter_draft',js(p).decode(),js(e['blockers']).decode(),None))
 for rid,p in mem.execute('select race_id,payload from races'):assert json.loads(p)==next(e['payload'] for e in updated['events'] if e['identity']['canonical_key']==rid)
 assert mem.execute("select count(*) from races where status in ('completed','collected')").fetchone()[0]==0;mem.close()
 cases=[('registration_ID',lambda v:v['official_sources'][0].__setitem__('registration_number','999999')),('name',lambda v:v['official_sources'][0].__setitem__('name','別人')),('term',lambda v:v['official_sources'][0].__setitem__('term','999')),('historical_prefecture',lambda v:v['official_sources'][0].__setitem__('prefecture','群馬')),('interval_extension',lambda v:v['official_sources'][0].__setitem__('valid_until_exclusive','9999-12-31')),('official_prefecture_temporal_reuse',lambda v:v['official_sources'][0].__setitem__('official_prefecture_used_as_historical_prefecture',True)),('current_performance_import',lambda v:v.__setitem__('current_performance_imported',True)),('full_registry_false_promotion',lambda v:v.__setitem__('all_registry_source_history_graph_verified',True))];mut=[]
 for name,fn in cases:
  bad=copy.deepcopy(doc);fn(bad['events'][0]['entries'][0]['rider_identity_evidence'])
  try:validate(bad,source)
  except (AssertionError,ValueError,KeyError):mut.append(name)
  else:raise AssertionError(name)
 bad=copy.deepcopy(doc);alt=next(a['rider_identity_evidence'] for e in bad['events'] for a in e['entries'] if a['rider_identity_evidence']['core_proof_kind']!='selected_official_identity');alt['original_core_proof']['transfer_body_recovered']=True
 try:validate(bad,source)
 except AssertionError:mut.append('unrecovered_transfer_false_success')
 else:raise AssertionError('false transfer success')
 for name,b,o in [('base_bytes',bb+b' ',body),('overlay_bytes',bb,body+b' ')]:
  try:mod.attach(b,o,ob)
  except AssertionError:mut.append(name)
  else:raise AssertionError(name)
 bad=copy.deepcopy(doc);bad['events'][0]['complete_payload']=True
 try:validate(bad,source)
 except AssertionError:mut.append('complete_flag')
 else:raise AssertionError('complete flag')
 assert all(a['rider_id_status']=='published_profile_link' for e in updated['events'] for a in e['payload']['entries'])
 out={'status':'independent_exact_event_identity_field_audit_passed','events':14,'legacy_identity_field_predicate_entries':126,'selected_official_rows':125,'alternative_official_rows':1,'failed_transfer_records_retained':1,'derived_exact_event_applicability_only':True,'source_affiliation_interval_inferred':False,'rider_status_promoted':False,'verified_composition_with_0916':True,'lossless_original_recovery':True,'partial_live_DDL_roundtrip_rows':14,'in_memory_completed_rows':0,'mutations_rejected':mut,'network_requests':0,'canonical_SQL_inserts':0,'new_completed_races':0,'all_finals_complete':False,'full_payload_SQL_adapter_verified':False,'complete_payload_insertion_audit':False,'second_SQLite_artifact_saved':False}
 (OUT/'independent_identity_field_audit.json').write_text(json.dumps(out,ensure_ascii=False,sort_keys=True,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
