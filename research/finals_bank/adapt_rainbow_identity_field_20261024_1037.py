"""Legacy identity-evidence container from explicit official + exact-event joins.

Derived applicability is one reviewed event, never a source affiliation interval.
"""
import copy,datetime,gzip,hashlib,io,json,tarfile,unicodedata
from pathlib import Path
from lxml import html
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/scope_review_1037'
PINS={'data/scope_review_0552/integrated_candidates.tar.xz':'c1a030bc66b78fec4bda307046402f9db15fa67dd386845b8d19862cd3f29e3a','data/scope_review_0835/legacy_payload_drafts.tar.xz':'28ceabb5cafc8035193c22813b7a9617aeb1d9b84279db9fa942a23da5da9ab2','data/scope_review_0916/raw_rows_profile_detail_overlay.tar.xz':'b799216c9c87897d9f854bb913d17042343b0448df61696724a304d9a7e5b672','recovery/adapt_rainbow_raw_rows_0916.py':'acda08d6f778e6b031720ebc86d99b2c740092e692276547eafec6e109ddd0bd'}
BASE='ff23321971f27f6608b572535ad1055dac21f5d07fd9d53c7efc3aefa94bc984'
def sha(b):return hashlib.sha256(b).hexdigest()
def enc(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def n(s):return ''.join(unicodedata.normalize('NFKC',s).split())
def pref(s):return {'神奈':'神奈川','和歌':'和歌山','北海':'北海道'}.get(n(s),n(s))
class Reader:
 def __init__(self):self.cache={}
 def get(self,ref):
  if 'raw_sha256' in ref:
   b=(ROOT/ref['path']).read_bytes();assert sha(b)==ref['stored_sha256'] and len(b)==ref['stored_bytes'];b=gzip.decompress(b);assert sha(b)==ref['raw_sha256'] and len(b)==ref['raw_bytes'];return b
  if 'archive' in ref:
   a=ref['archive'];path=a if isinstance(a,str) else a['path'];expected=ref['archive_sha256'] if isinstance(a,str) else a['sha256'];b=(ROOT/path).read_bytes();assert sha(b)==expected
   if isinstance(a,dict):assert len(b)==a['bytes']
   if path not in self.cache:
    with tarfile.open(fileobj=io.BytesIO(b)) as t:assert len(t.getnames())==len(set(t.getnames()));self.cache[path]={m.name:t.extractfile(m).read() for m in t.getmembers()}
   b=self.cache[path][ref['member']]
  else:b=(ROOT/ref['path']).read_bytes()
  assert sha(b)==ref['sha256'] and len(b)==ref['bytes'];return b
def load():
 for p,h in PINS.items():assert sha((ROOT/p).read_bytes())==h
 docs=[];bodies=[]
 for p,m,h in [('data/scope_review_0552/integrated_candidates.tar.xz','integrated_candidates.json','b9ed0533f9c8a654236a4c4c8d9d275161acdd6c154ad6c3efbaa444cb307e3f'),('data/scope_review_0835/legacy_payload_drafts.tar.xz','legacy_payload_drafts.json',BASE),('data/scope_review_0916/raw_rows_profile_detail_overlay.tar.xz','raw_rows_profile_detail_overlay.json','80c603d0d903387b5f08ab6a75c56db978803fa70776cf6a26da2f4dbd518956')]:
  with tarfile.open(ROOT/p) as t:assert t.getnames()==[m];b=t.extractfile(m).read()
  assert sha(b)==h;docs.append(json.loads(b));bodies.append(b)
 return docs,bodies
def identity_field(event,entry,reader,pool):
 ident=event['identity'];proof=entry['evidence_integration']['core_identity'];reg=entry['registration_number'];car=entry['car_no'];refs=event['component_raw_provenance']
 card=json.loads(reader.get(refs['card']));op=html.fromstring(reader.get(refs['independent_card']).decode('utf-8'));a=card['race_athletes'][str(car)]
 assert (card['box']['open_day'],card['box']['vel_code'],card['box']['race_num'])==(ident['day'],ident['venue_code'],ident['race_no'])
 for key,value in [('joCode',ident['venue_code']),('kaisaiBi',ident['day'].replace('-','')),('raceNo',str(ident['race_no']))]:assert op.xpath('//input[@name=$n]/@value',n=key)==[value]
 rows=op.xpath('//a[@href=$href]/ancestor::tr[1]',href='./PlayerDetail.do?playerCd='+reg);assert len(rows)==1;carcell=rows[0].xpath('./td[@class=$cls]',cls='no'+str(car));prefcells=rows[0].xpath('.//a[@href=$href]/ancestor::td[1]/following-sibling::td[1]',href='./PlayerDetail.do?playerCd='+reg);assert len(carcell)==len(prefcells)==1;prefcell=prefcells[0];assert int(n(carcell[0].text_content()))==car and pref(prefcell.text_content())==pref(entry['prefecture_raw'])
 assert a['un_id']==int(reg) and a['entry_num']==car and n(a['name'])==n(entry['name_raw']) and str(a['graduate_period'])==entry['term'] and a['pref_name']==entry['prefecture_raw']
 official_refs=[]
 for source_id in proof['source_ids']:reader.get(pool[source_id]);official_refs.append(copy.deepcopy(pool[source_id]))
 if proof['kind']=='selected_official_identity':
  original=proof['original_proof'];assert original['registration_number']==reg and n(original['name'])==n(entry['name_raw']) and original['term']==entry['term'] and pref(original['prefecture'])==pref(entry['prefecture_raw'])
  official_url=original['source_url'];original_record=copy.deepcopy(original);failed=None
 else:
  assert proof['kind']=='alternative_official_identity_plus_event_time_cards' and proof['affiliation_interval_inferred'] is proof['transfer_body_recovered'] is False
  alt=proof['alternative_proof'];target=alt['target'];assert target['registration_number']==reg and target['car_no']==car and target['day']==ident['day'] and target['term']==entry['term'] and n(target['name'])==n(entry['name_raw']) and target['prefecture_at_event']==entry['prefecture_raw']
  original_record=copy.deepcopy(alt['official_allocation']);official_url=original_record['source_url'];failed=copy.deepcopy(proof['original_0115_failed_proof_retained'])
  assert alt['original_transfer_page_body_status']=='still_unavailable_HTTP403_not_recovered_not_retried'
 day=datetime.date.fromisoformat(ident['day']);end=(day+datetime.timedelta(days=1)).isoformat()
 # Flat legacy columns describe this derived joint certificate; per-field origins
 # explicitly distinguish immutable official identity from historical affiliation.
 joint={'registration_number':reg,'name':n(entry['name_raw']),'term':entry['term'],'prefecture':entry['prefecture_raw'],'source_url':official_url,'source_kind':'official_identity_plus_exact_event_prefecture_join','valid_from':ident['day'],'valid_until_exclusive':end,'applicability_kind':'derived_exact_event_only_not_source_affiliation_interval','official_identity_fields':['registration_number','name','term'],'historical_prefecture_sources':{k:copy.deepcopy(refs[k]) for k in ['card','independent_card']},'official_original_record':original_record,'official_raw_locators':official_refs,'official_prefecture_used_as_historical_prefecture':False}
 return {'matching_fields':['registration_number','name','term','prefecture'],'official_sources':[joint],'use':'registration identity and exact reviewed event only; historical performance unchanged','exact_event':copy.deepcopy(ident),'historical_prefecture_observation':{'chariloto_raw':a['pref_name'],'oddspark_raw':prefcell.text_content().strip(),'source_day':ident['day'],'affiliation_interval_inferred':False},'core_proof_kind':proof['kind'],'original_core_proof':copy.deepcopy(proof),'original_failed_transfer_proof_retained':failed,'all_registry_source_history_graph_verified':False,'current_performance_imported':False}
def attach(basebytes,overlay_bytes,raw_rows_overlay_bytes=None):
 assert sha(basebytes)==BASE and sha(overlay_bytes)=='5df9fb07ef7c83304ca2aa09de1af625cff0cebe9f5ceac8527f2ec1692556fd';doc=json.loads(basebytes);overlay=json.loads(overlay_bytes)
 if raw_rows_overlay_bytes is not None:
  import importlib.util
  p=ROOT/'recovery/adapt_rainbow_raw_rows_0916.py';assert sha(p.read_bytes())==PINS['recovery/adapt_rainbow_raw_rows_0916.py'];sp=importlib.util.spec_from_file_location('verified_raw0916',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);doc=m.attach(basebytes,raw_rows_overlay_bytes)
 assert overlay['base_member_sha256']==BASE and overlay['all_finals_complete'] is False and len(doc['events'])==len(overlay['events'])==14
 for event,add in zip(doc['events'],overlay['events']):
  assert event['identity']==add['identity'] and add['complete_payload'] is add['canonical_SQL_inserted'] is False
  for a,p in zip(event['payload']['entries'],add['entries']):
   assert a['car_no']==p['car_no'] and a['rider_registration_number']==p['registration_number'];a['rider_identity_evidence']=copy.deepcopy(p['rider_identity_evidence']);event['unmapped_fields']['entries'][str(a['car_no'])].remove('rider_identity_evidence')
  event['identity_field_attachment_evidence']=copy.deepcopy(add)
 return doc
def main():
 (source,base,old),(sb,bb,ob)=load();reader=Reader();events=[]
 for e,b in zip(source['events'],base['events']):
  assert e['identity']==b['identity'];events.append({'identity':copy.deepcopy(e['identity']),'entries':[{'car_no':a['car_no'],'registration_number':a['registration_number'],'rider_identity_evidence':identity_field(e,a,reader,source['source_pool'])} for a in e['entries']],'complete_payload':False,'canonical_SQL_inserted':False})
 doc={'schema':'rainbow_exact_event_identity_field_overlay_v1','base_member_sha256':BASE,'input_pins':PINS,'events':events,'all_finals_complete':False,'canonical_SQL_inserts':0,'new_completed_races':0,'network_requests':0,'full_payload_SQL_adapter_verified':False,'complete_payload_insertion_audit':False};body=enc(doc);OUT.mkdir(parents=True,exist_ok=True);p=OUT/'identity_field_overlay.tar.xz'
 with tarfile.open(p,'w:xz',preset=9) as t:m=tarfile.TarInfo('identity_field_overlay.json');m.size=len(body);m.mtime=0;t.addfile(m,io.BytesIO(body))
 with tarfile.open(p) as t:assert t.getnames()==['identity_field_overlay.json'] and t.extractfile(t.getmembers()[0]).read()==body
 report={'status':'partial_exact_event_identity_field_attachment_only','events':14,'entries':126,'selected_official_identity_rows':125,'alternative_official_identity_rows':1,'unrecovered_transfer_records_retained':1,'applicability':'derived exact reviewed event join, not continuous official affiliation','rider_status_promoted':False,'full_registry_graph_verified':False,'archive':{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p.read_bytes()),'member':'identity_field_overlay.json','member_bytes':len(body),'member_sha256':sha(body)},'canonical_SQL_inserts':0,'new_completed_races':0,'all_finals_complete':False,'full_payload_SQL_adapter_verified':False,'complete_payload_insertion_audit':False}
 (OUT/'identity_field_review.json').write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
