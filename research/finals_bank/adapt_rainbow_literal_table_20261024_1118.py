"""Literal historical NK entry table and explicit profile-text adapter.

No KDreams table reconstruction, hidden-code interpretation or completion write.
"""
import copy,hashlib,importlib.util,io,json,re,tarfile,unicodedata
from pathlib import Path
from lxml import html
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/scope_review_1118'
PINS={'data/scope_review_0835/legacy_payload_drafts.tar.xz': '28ceabb5cafc8035193c22813b7a9617aeb1d9b84279db9fa942a23da5da9ab2', 'data/scope_review_0916/raw_rows_profile_detail_overlay.tar.xz': 'b799216c9c87897d9f854bb913d17042343b0448df61696724a304d9a7e5b672', 'data/scope_review_1037/identity_field_overlay.tar.xz': 'e6873a344d8a39d958d458f842f3b2e35c2e95c9becc8cdacbcd449dc76b7f5c', 'data/scope_review_0730/entry_extension_overlay.tar.xz': '10cc8901c0d7674dad51ba1fa3353d4e266b6741552e6e7c4b4bcc646b6433e6', 'data/scope_review_0730/entry_extension_raw.tar.xz': '983e4f17e76ac5cea40967be16238268f7975408929b03aba140b38398331f72', 'recovery/adapt_rainbow_identity_field_1037.py': 'e035bb097acf6a63b6d415e0b7776733669db354d5b913526c91024e5e81e337', 'recovery/adapt_rainbow_raw_rows_0916.py': 'acda08d6f778e6b031720ebc86d99b2c740092e692276547eafec6e109ddd0bd'}
MEMBERS={'data/scope_review_0835/legacy_payload_drafts.tar.xz':'legacy_payload_drafts.json','data/scope_review_0916/raw_rows_profile_detail_overlay.tar.xz':'raw_rows_profile_detail_overlay.json','data/scope_review_1037/identity_field_overlay.tar.xz':'identity_field_overlay.json','data/scope_review_0730/entry_extension_overlay.tar.xz':'entry_extension_overlay.json'}
BASE='ff23321971f27f6608b572535ad1055dac21f5d07fd9d53c7efc3aefa94bc984'
OVERLAY_SHA='f4c8074ead1bbccf16ab80b8d0d5a0b0593bf67a95b1c5f7f831de5bcfe164e9'
def sha(b):return hashlib.sha256(b).hexdigest()
def enc(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def compact(s):return ''.join(unicodedata.normalize('NFKC',s).split())
def readarc(path):
 b=(ROOT/path).read_bytes();assert sha(b)==PINS[path]
 with tarfile.open(fileobj=io.BytesIO(b)) as t:
  assert len(t.getnames())==len(set(t.getnames()));return {m.name:t.extractfile(m).read() for m in t.getmembers()}
def load():
 for p,h in PINS.items():assert sha((ROOT/p).read_bytes())==h
 bodies={p:readarc(p)[m] for p,m in MEMBERS.items()};assert sha(bodies['data/scope_review_0835/legacy_payload_drafts.tar.xz'])==BASE
 ext=json.loads(bodies['data/scope_review_0730/entry_extension_overlay.tar.xz']);raw=readarc('data/scope_review_0730/entry_extension_raw.tar.xz');manifest=ext['raw_manifest']['members'];assert set(raw)==set(manifest)
 for k,b in raw.items():assert len(b)==manifest[k]['bytes'] and sha(b)==manifest[k]['sha256']
 return bodies,ext,raw
def extract(event,x,raw):
 assert event['identity']==x['identity'];key=x['NK_entry_member'].removesuffix('.html');b=raw[x['NK_entry_member']];meta=json.loads(raw[x['NK_meta_member']]);url='https://keirin.netkeiba.com/race/entry/?race_id='+key
 assert meta['status']==200 and meta['url']==meta['final_url']==url and meta['bytes']==len(b) and meta['sha256']==sha(b)
 doc=html.fromstring(b.decode('utf-8'));assert doc.xpath('//link[@rel="canonical"]/@href')==[url] and doc.xpath('//meta[@property="og:url"]/@content')==[url]
 tables=doc.xpath('//table[@id="RaceCard_Table_Static"]');assert len(tables)==1;table=tables[0];rows=table.xpath('.//tr');assert len(rows)==10 and len(rows[0].xpath('./th'))==23
 actual={'class':table.get('class','').split(),'rows':[[cell.text_content().strip() for cell in row.xpath('./th|./td')] for row in rows]}
 assert len(actual['rows'])==10 and all(len(row)==23 for row in actual['rows']);entries=[]
 for a in event['payload']['entries']:
  car=a['car_no'];reg=a['rider_registration_number'];r=table.xpath('./tbody/tr[@id=$v]|./tr[@id=$v]',v='player-'+str(car));assert len(r)==1;r=r[0];cells=r.xpath('./td');assert len(cells)==23
  anchor=cells[4].xpath('./a[@id=$v]',v='name_'+str(int(reg))+'_'+key);assert len(anchor)==1;anchor=anchor[0];assert compact(anchor.xpath('.//img[@class="PlayerPhotoImg01"]/@alt')[0])==compact(a['player_name'])
  pf=anchor.xpath('.//dd[@class="PlayerFrom"]');cl=anchor.xpath('.//dd[@class="PlayerClass"]');assert len(pf)==len(cl)==1
  prefnode=pf[0].xpath('./span');assert len(prefnode)==1
  pref=prefnode[0].text_content();age_match=re.fullmatch(re.escape(compact(pref))+r'(\d+)歳',compact(pf[0].text_content()));term_match=re.fullmatch(r'(\d+)期(.+)',compact(cl[0].text_content()));assert age_match and term_match
  age,term=age_match.group(1),term_match.group(1);assert compact(pref)==compact(a['prefecture']) and age==str(a['age']) and term==str(a['term'])
  # Explicitly formatted text from historical displayed components, not a raw KD token.
  value=compact(pref)+'/'+age+'/'+term
  entries.append({'car_no':car,'registration_number':reg,'player_profile':value,'proof':{'format':'NFKC_compact_prefecture/age/term_from_explicit_historical_display','raw_prefecture':pref,'raw_prefecture_age':pf[0].text_content(),'raw_term_class':cl[0].text_content(),'explicit_anchor':anchor.get('id'),'row_id':r.get('id'),'source_cell_index':4,'historical_components_verified_against_existing_entry':True,'current_profile_used':False}})
 return {'identity':copy.deepcopy(event['identity']),'racecard_tables':[actual],'entries':entries,'proof':{'provider':'netkeiba_historical_entry','source_url':url,'table_id':table.get('id'),'table_capture':'literal_direct_cell_text_including_hidden_text_not_semantic_feature_values','header_rows':1,'player_rows':9,'columns':23,'raw_html':{'member':x['NK_entry_member'],'bytes':len(b),'sha256':sha(b)},'raw_meta':{'member':x['NK_meta_member'],'bytes':len(raw[x['NK_meta_member']]),'sha256':sha(raw[x['NK_meta_member']])},'kd_table_reconstructed':False,'kd_specific_table_parser_compatibility_verified':False,'prediction_or_evaluation_codes_interpreted':False},'complete_payload':False,'canonical_SQL_inserted':False}
def previous(bodies):
 p=ROOT/'recovery/adapt_rainbow_identity_field_1037.py';assert sha(p.read_bytes())==PINS[str(p.relative_to(ROOT))];sp=importlib.util.spec_from_file_location('identity1037_for1118',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 return m.attach(bodies['data/scope_review_0835/legacy_payload_drafts.tar.xz'],bodies['data/scope_review_1037/identity_field_overlay.tar.xz'],bodies['data/scope_review_0916/raw_rows_profile_detail_overlay.tar.xz'])
def attach(bodies,overlay_bytes):
 assert sha(overlay_bytes)==OVERLAY_SHA;add=json.loads(overlay_bytes);result=previous(bodies);assert add['all_finals_complete'] is False and len(add['events'])==len(result['events'])==14
 for e,a in zip(result['events'],add['events']):
  assert e['identity']==a['identity'] and a['complete_payload'] is a['canonical_SQL_inserted'] is False
  e['payload']['racecard_tables']=copy.deepcopy(a['racecard_tables']);e['unmapped_fields']['top_level'].remove('racecard_tables')
  for entry,ext in zip(e['payload']['entries'],a['entries']):
   assert entry['car_no']==ext['car_no'] and entry['rider_registration_number']==ext['registration_number'];entry['player_profile']=ext['player_profile'];e['unmapped_fields']['entries'][str(entry['car_no'])].remove('player_profile')
  e['literal_table_profile_attachment_evidence']=copy.deepcopy(a)
 return result
def main():
 bodies,ext,raw=load();before=previous(bodies);events=[extract(e,x,raw) for e,x in zip(before['events'],ext['events'])];doc={'schema':'rainbow_literal_historical_table_profile_overlay_v1','input_pins':PINS,'base_member_sha256':BASE,'events':events,'all_finals_complete':False,'canonical_SQL_inserts':0,'new_completed_races':0,'network_requests':0,'full_payload_SQL_adapter_verified':False,'complete_payload_insertion_audit':False};body=enc(doc);assert sha(body)==OVERLAY_SHA
 after=attach(bodies,body);restored=copy.deepcopy(after)
 for e,old in zip(restored['events'],before['events']):
  e['payload'].pop('racecard_tables')
  for a in e['payload']['entries']:a.pop('player_profile')
  e.pop('literal_table_profile_attachment_evidence');e['unmapped_fields']=copy.deepcopy(old['unmapped_fields'])
 assert restored==before
 OUT.mkdir(parents=True,exist_ok=True);p=OUT/'literal_table_profile_overlay.tar.xz'
 with tarfile.open(p,'w:xz',preset=9) as t:m=tarfile.TarInfo('literal_table_profile_overlay.json');m.size=len(body);m.mtime=0;t.addfile(m,io.BytesIO(body))
 with tarfile.open(p) as t:assert t.getnames()==['literal_table_profile_overlay.json'] and t.extractfile(t.getmembers()[0]).read()==body
 report={'status':'partial_literal_historical_table_profile_attachment_only','events':14,'literal_tables':14,'header_rows':14,'player_rows':126,'columns_per_row':23,'profile_text_fields':126,'raw_archive_members_verified':28,'kd_table_reconstructed':False,'kd_specific_table_parser_compatibility_verified':False,'prediction_or_evaluation_codes_interpreted':False,'literal_table_text_includes_hidden_text':True,'lossless_original_recovery':True,'archive':{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p.read_bytes()),'member':'literal_table_profile_overlay.json','member_bytes':len(body),'member_sha256':sha(body)},'all_finals_complete':False,'canonical_SQL_inserts':0,'new_completed_races':0,'network_requests':0,'full_payload_SQL_adapter_verified':False,'complete_payload_insertion_audit':False}
 (OUT/'literal_table_profile_review.json').write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
