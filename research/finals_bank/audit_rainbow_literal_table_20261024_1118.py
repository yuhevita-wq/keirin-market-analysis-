"""Independent literal-table/profile audit using BeautifulSoup, not producer DOM."""
import copy,hashlib,importlib.util,io,json,re,sqlite3,tarfile,unicodedata
from pathlib import Path
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/scope_review_1118'
PINS={'data/scope_review_0835/legacy_payload_drafts.tar.xz': '28ceabb5cafc8035193c22813b7a9617aeb1d9b84279db9fa942a23da5da9ab2', 'data/scope_review_0916/raw_rows_profile_detail_overlay.tar.xz': 'b799216c9c87897d9f854bb913d17042343b0448df61696724a304d9a7e5b672', 'data/scope_review_1037/identity_field_overlay.tar.xz': 'e6873a344d8a39d958d458f842f3b2e35c2e95c9becc8cdacbcd449dc76b7f5c', 'data/scope_review_0730/entry_extension_overlay.tar.xz': '10cc8901c0d7674dad51ba1fa3353d4e266b6741552e6e7c4b4bcc646b6433e6', 'data/scope_review_0730/entry_extension_raw.tar.xz': '983e4f17e76ac5cea40967be16238268f7975408929b03aba140b38398331f72', 'recovery/adapt_rainbow_identity_field_1037.py': 'e035bb097acf6a63b6d415e0b7776733669db354d5b913526c91024e5e81e337', 'recovery/adapt_rainbow_raw_rows_0916.py': 'acda08d6f778e6b031720ebc86d99b2c740092e692276547eafec6e109ddd0bd'}
def h(b):return hashlib.sha256(b).hexdigest()
def js(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def norm(v):return ''.join(unicodedata.normalize('NFKC',v).split())
def arc(p):
 b=(ROOT/p).read_bytes();assert h(b)==PINS[p]
 with tarfile.open(fileobj=io.BytesIO(b)) as t:
  assert len(t.getnames())==len(set(t.getnames()));return {a.name:t.extractfile(a).read() for a in t.getmembers()}
def validate(doc,base,extension,raw):
 assert doc['schema']=='rainbow_literal_historical_table_profile_overlay_v1' and doc['base_member_sha256']=='ff23321971f27f6608b572535ad1055dac21f5d07fd9d53c7efc3aefa94bc984'
 assert doc['input_pins']==PINS and doc['all_finals_complete'] is doc['full_payload_SQL_adapter_verified'] is doc['complete_payload_insertion_audit'] is False
 assert doc['canonical_SQL_inserts']==doc['new_completed_races']==doc['network_requests']==0 and len(doc['events'])==len(base['events'])==len(extension['events'])==14
 for event,b,x in zip(doc['events'],base['events'],extension['events']):
  assert event['identity']==b['identity']==x['identity'] and event['complete_payload'] is event['canonical_SQL_inserted'] is False
  member=x['NK_entry_member'];key=member[:-5];body=raw[member];metab=raw[x['NK_meta_member']];meta=json.loads(metab);url='https://keirin.netkeiba.com/race/entry/?race_id='+key
  assert meta['status']==200 and meta['bytes']==len(body) and meta['sha256']==h(body) and meta['url']==meta['final_url']==url
  soup=BeautifulSoup(body.decode('utf-8'),'html.parser');assert [a['href'] for a in soup.find_all('link',rel='canonical')]==[url] and [a['content'] for a in soup.find_all('meta',property='og:url')]==[url]
  tables=soup.find_all('table',id='RaceCard_Table_Static');assert len(tables)==1;table=tables[0];rows=table.find_all('tr');assert len(rows)==10 and len(rows[0].find_all('th',recursive=False))==23
  literal={'class':table.get('class',[]),'rows':[[c.get_text().strip() for c in row.find_all(['th','td'],recursive=False)] for row in rows]};assert all(len(row)==23 for row in literal['rows'])
  assert event['racecard_tables']==[literal] and len(event['entries'])==9
  assert event['proof']=={'provider':'netkeiba_historical_entry','source_url':url,'table_id':'RaceCard_Table_Static','table_capture':'literal_direct_cell_text_including_hidden_text_not_semantic_feature_values','header_rows':1,'player_rows':9,'columns':23,'raw_html':{'member':member,'bytes':len(body),'sha256':h(body)},'raw_meta':{'member':x['NK_meta_member'],'bytes':len(metab),'sha256':h(metab)},'kd_table_reconstructed':False,'kd_specific_table_parser_compatibility_verified':False,'prediction_or_evaluation_codes_interpreted':False}
  for p,a in zip(event['entries'],b['payload']['entries']):
   car=a['car_no'];reg=a['rider_registration_number'];row=table.find_all('tr',id='player-'+str(car));assert len(row)==1;cells=row[0].find_all('td',recursive=False);assert len(cells)==23
   an=cells[4].find_all('a',id='name_'+str(int(reg))+'_'+key);assert len(an)==1;an=an[0];photo=an.find_all('img',class_='PlayerPhotoImg01');assert len(photo)==1 and norm(photo[0]['alt'])==norm(a['player_name'])
   fr=an.find_all('dd',class_='PlayerFrom');cl=an.find_all('dd',class_='PlayerClass');assert len(fr)==len(cl)==1;sp=fr[0].find_all('span',recursive=False);assert len(sp)==1;pref=sp[0].get_text();full=norm(fr[0].get_text());prefix=norm(pref);assert full.startswith(prefix) and full.endswith('歳');age=full[len(prefix):-1];termcl=norm(cl[0].get_text());term,cls=termcl.split('期');assert age.isdecimal() and term.isdecimal() and cls
   assert (prefix,age,term)==(norm(a['prefecture']),str(a['age']),str(a['term']))
   expected={'car_no':car,'registration_number':reg,'player_profile':prefix+'/'+age+'/'+term,'proof':{'format':'NFKC_compact_prefecture/age/term_from_explicit_historical_display','raw_prefecture':pref,'raw_prefecture_age':fr[0].get_text(),'raw_term_class':cl[0].get_text(),'explicit_anchor':an['id'],'row_id':row[0]['id'],'source_cell_index':4,'historical_components_verified_against_existing_entry':True,'current_profile_used':False}}
   assert p==expected
def module(name,p):
 assert h((ROOT/p).read_bytes())==('b3bbc39aefea5963f7552cb833335dc5cb88b81ccee500949ef1ea38649dd992' if p=='recovery/adapt_rainbow_literal_table_1118.py' else PINS[p]);s=importlib.util.spec_from_file_location(name,ROOT/p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 for p,v in PINS.items():assert h((ROOT/p).read_bytes())==v
 names={'data/scope_review_0835/legacy_payload_drafts.tar.xz':'legacy_payload_drafts.json','data/scope_review_0916/raw_rows_profile_detail_overlay.tar.xz':'raw_rows_profile_detail_overlay.json','data/scope_review_1037/identity_field_overlay.tar.xz':'identity_field_overlay.json','data/scope_review_0730/entry_extension_overlay.tar.xz':'entry_extension_overlay.json'}
 bodies={p:arc(p)[m] for p,m in names.items()};base=json.loads(bodies['data/scope_review_0835/legacy_payload_drafts.tar.xz']);ext=json.loads(bodies['data/scope_review_0730/entry_extension_overlay.tar.xz']);raw=arc('data/scope_review_0730/entry_extension_raw.tar.xz');manifest=ext['raw_manifest']['members'];assert set(raw)==set(manifest)
 for k,b in raw.items():assert len(b)==manifest[k]['bytes'] and h(b)==manifest[k]['sha256']
 p=OUT/'literal_table_profile_overlay.tar.xz'
 with tarfile.open(p) as t:assert t.getnames()==['literal_table_profile_overlay.json'];body=t.extractfile('literal_table_profile_overlay.json').read()
 assert h(body)=='f4c8074ead1bbccf16ab80b8d0d5a0b0593bf67a95b1c5f7f831de5bcfe164e9';doc=json.loads(body);validate(doc,base,ext,raw)
 producer=module('table1118','recovery/adapt_rainbow_literal_table_1118.py');identity=module('identity1037','recovery/adapt_rainbow_identity_field_1037.py');bb=bodies['data/scope_review_0835/legacy_payload_drafts.tar.xz'];before=identity.attach(bb,bodies['data/scope_review_1037/identity_field_overlay.tar.xz'],bodies['data/scope_review_0916/raw_rows_profile_detail_overlay.tar.xz']);updated=producer.attach(bodies,body);expected=copy.deepcopy(before)
 for e,a in zip(expected['events'],doc['events']):
  e['payload']['racecard_tables']=a['racecard_tables'];e['unmapped_fields']['top_level'].remove('racecard_tables')
  for entry,p in zip(e['payload']['entries'],a['entries']):entry['player_profile']=p['player_profile'];e['unmapped_fields']['entries'][str(entry['car_no'])].remove('player_profile')
  e['literal_table_profile_attachment_evidence']=a
 assert updated==expected;restored=copy.deepcopy(updated)
 for e,b in zip(restored['events'],before['events']):
  del e['payload']['racecard_tables'];del e['literal_table_profile_attachment_evidence']
  for a in e['payload']['entries']:del a['player_profile']
  e['unmapped_fields']=copy.deepcopy(b['unmapped_fields'])
 assert restored==before
 mem=sqlite3.connect(':memory:');mem.execute(base['observed_legacy_contract']['races_DDL'])
 for e in updated['events']:
  p=e['payload'];assert e['sql_status']=='partial_adapter_draft' and e['ready_for_completed_export'] is e['complete_payload'] is False
  mem.execute('insert into races values (?,?,?,?,?,?,?,?,?,?)',(p['race_id'],e['identity']['day'],p['race']['source_url'],p['expected_label'],p['grade'],p['category'],'partial_adapter_draft',js(p).decode(),js(e['blockers']).decode(),None))
 for rid,p in mem.execute('select race_id,payload from races'):assert json.loads(p)==next(e['payload'] for e in updated['events'] if e['identity']['canonical_key']==rid)
 assert mem.execute("select count(*) from races where status in ('completed','collected')").fetchone()[0]==0;mem.close()
 cases=[('table_header',lambda e:e['racecard_tables'][0]['rows'][0].__setitem__(3,'評価')),('table_cell',lambda e:e['racecard_tables'][0]['rows'][1].__setitem__(7,'999')),('table_class',lambda e:e['racecard_tables'][0].__setitem__('class',['racecard_table'])),('registration_ID',lambda e:e['entries'][0].__setitem__('registration_number','999999')),('profile_text',lambda e:e['entries'][0].__setitem__('player_profile','群馬/99/999')),('current_profile_import',lambda e:e['entries'][0]['proof'].__setitem__('current_profile_used',True)),('KD_parser_false_promotion',lambda e:e['proof'].__setitem__('kd_specific_table_parser_compatibility_verified',True)),('hidden_code_interpretation',lambda e:e['proof'].__setitem__('prediction_or_evaluation_codes_interpreted',True)),('complete_flag',lambda e:e.__setitem__('complete_payload',True))];rejected=[]
 for name,fn in cases:
  bad=copy.deepcopy(doc);fn(bad['events'][0])
  try:validate(bad,base,ext,raw)
  except (AssertionError,KeyError,ValueError):rejected.append(name)
  else:raise AssertionError(name)
 for name,bbad,obad in [('base_bytes',dict(bodies),body),('overlay_bytes',bodies,body+b' ')]:
  if name=='base_bytes':bbad['data/scope_review_0835/legacy_payload_drafts.tar.xz']+=b' '
  try:producer.attach(bbad,obad)
  except AssertionError:rejected.append(name)
  else:raise AssertionError(name)
 # Rehashed raw-source mutation: metadata alone cannot legitimize a changed ID.
 altered=copy.deepcopy(raw);key=ext['events'][0]['NK_entry_member'];target=base['events'][0]['payload']['entries'][0]['rider_registration_number'];old=('name_'+str(int(target))+'_').encode();assert old in altered[key];altered[key]=altered[key].replace(old,b'name_999999_',1);meta_key=ext['events'][0]['NK_meta_member'];meta=json.loads(altered[meta_key]);meta['sha256']=h(altered[key]);meta['bytes']=len(altered[key]);altered[meta_key]=js(meta)
 bad=copy.deepcopy(doc);bad['events'][0]['proof']['raw_html'].update(bytes=len(altered[key]),sha256=h(altered[key]));bad['events'][0]['proof']['raw_meta'].update(bytes=len(altered[meta_key]),sha256=h(altered[meta_key]))
 try:validate(bad,base,ext,altered)
 except (AssertionError,KeyError):rejected.append('source_ID_even_with_rehashed_meta')
 else:raise AssertionError('raw mutation')
 report={'status':'independent_literal_table_profile_audit_passed','literal_tables':14,'header_rows':14,'player_rows':126,'columns_per_row':23,'historical_profile_fields':126,'raw_archive_members_verified':28,'verified_composition_with_0916_1037':True,'lossless_original_recovery':True,'partial_live_DDL_roundtrip_rows':14,'in_memory_completed_rows':0,'mutations_rejected':rejected,'kd_table_reconstructed':False,'kd_specific_table_parser_compatibility_verified':False,'prediction_or_evaluation_codes_interpreted':False,'literal_hidden_text_retained_not_semantic_feature_import':True,'canonical_SQL_inserts':0,'new_completed_races':0,'network_requests':0,'all_finals_complete':False,'full_payload_SQL_adapter_verified':False,'complete_payload_insertion_audit':False,'second_SQLite_artifact_saved':False}
 (OUT/'independent_literal_table_profile_audit.json').write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
