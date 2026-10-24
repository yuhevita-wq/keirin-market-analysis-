"""Attach one reviewed historical bank to a partial draft; never export completed rows."""
import copy,hashlib,importlib.util,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/scope_review_1716'
BASE_SHA='63f7e29a955bb3daf179747ba958863ab43cbe472386d3d806e4a5d3ec8acfee'
DELTA_SHA='53348cb1c4e5a385370563c727de8f07ea61a85412c9102c8b8dfe46d97690fc'
CODE_SHA='b3bbc39aefea5963f7552cb833335dc5cb88b81ccee500949ef1ea38649dd992'
KEY='keirin:2025-12-14:37:09'
def sha(b):return hashlib.sha256(b).hexdigest()
def enc(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def compose():
 p=ROOT/'recovery/adapt_rainbow_literal_table_1118.py';assert sha(p.read_bytes())==CODE_SHA
 sp=importlib.util.spec_from_file_location('literal1118_bank1716',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 bodies={p:m.readarc(p)[name] for p,name in m.MEMBERS.items()}
 p=ROOT/'data/scope_review_1118/literal_table_profile_overlay.tar.xz';assert sha(p.read_bytes())=='3c4526bb77fb1047b57b5bf88da48a35cda461b3da9c012ed2e731dbf863fb48'
 with tarfile.open(p) as t:assert t.getnames()==['literal_table_profile_overlay.json'];overlay=t.extractfile(t.getnames()[0]).read()
 b=enc(m.attach(bodies,overlay));assert sha(b)==BASE_SHA;return b
def attach(base_bytes,delta_bytes):
 assert sha(base_bytes)==BASE_SHA and sha(delta_bytes)==DELTA_SHA
 original=json.loads(base_bytes);d=json.loads(delta_bytes);result=copy.deepcopy(original)
 assert d['identity']['canonical_key']==KEY and d['all_finals_complete'] is False and d['canonical_SQL_inserts']==0
 targets=[e for e in result['events'] if e['identity']==d['identity']];assert len(targets)==1;e=targets[0]
 assert e['complete_payload'] is e['canonical_SQL_inserted'] is e['ready_for_completed_export'] is False
 assert 'bank' not in e['payload'];r=d['bank_record'];day=e['identity']['day'];assert r['valid_from']<=day<r['valid_until_exclusive']
 assert (e['payload']['race']['race_date'],e['payload']['race']['race_no'],e['identity']['venue_code'])==(day,9,'37')
 e['payload']['bank']={'bank_version_id':r['bank_version_id'],'venue_code':r['venue_code'],'track':r['track'],'race_date':day,'valid_from':r['valid_from'],'valid_until_exclusive':r['valid_until_exclusive'],'geometry':copy.deepcopy(r['geometry']),'evidence':copy.deepcopy(r['sources']),'validity_basis':r['validity_basis'],'status':r['review_status']}
 e['dated_bank_attachment_evidence']={'delta_sha256':DELTA_SHA,'base_composed_sha256':BASE_SHA,'source_provider_count':1,'review_method':d['manual_visual_review']['method'],'official_news_proof':copy.deepcopy(d['official_news_proof']),'manual_visual_review':copy.deepcopy(d['manual_visual_review']),'source_record':copy.deepcopy(r),'canonical_save_pending':True,'completed_payload_verified':False}
 assert all(e['complete_payload'] is e['canonical_SQL_inserted'] is False for e in result['events'])
 return result
def main():
 b=compose();d=(ROOT/'data/bank_sources_1227/ito2025_candidate_bank_delta.json').read_bytes();after=attach(b,d);before=json.loads(b)
 e=next(e for e in after['events'] if e['identity']['canonical_key']==KEY)
 fragment={'schema':'rainbow_exact_event_partial_bank_attachment_v1','identity':e['identity'],'payload_bank_fragment':e['payload']['bank'],'attachment_evidence':e['dated_bank_attachment_evidence'],'complete_payload':False,'canonical_SQL_inserted':False,'all_finals_complete':False}
 restored=copy.deepcopy(after);target=next(e for e in restored['events'] if e['identity']['canonical_key']==KEY);target['payload'].pop('bank');target.pop('dated_bank_attachment_evidence');assert restored==before
 OUT.mkdir(exist_ok=True);raw=enc(fragment);(OUT/'ito2025_bank_attachment.json').write_bytes(raw)
 report={'status':'partial_bank_field_attachment_only_pending_canonical_save','events_unchanged_except_one_bank_and_evidence':14,'target':KEY,'fragment_bytes':len(raw),'fragment_sha256':sha(raw),'base_composed_sha256':BASE_SHA,'delta_sha256':DELTA_SHA,'lossless_previous_composed_recovery':True,'readiness_blockers_modified':False,'catalog_modified':False,'DB_changed':False,'canonical_SQL_inserts':0,'new_completed_races':0,'all_finals_complete':False,'full_payload_SQL_adapter_verified':False,'complete_payload_insertion_audit':False,'network_requests':0,'geometry_provider_count':1}
 (OUT/'bank_attachment_review.json').write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
