"""Independent raw DOM reconstruction; partial payload insertion only."""
import copy,datetime,hashlib,importlib.util,json,sqlite3,tarfile,unicodedata
from pathlib import Path
from lxml import html
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/scope_review_0916'
def h(b):return hashlib.sha256(b).hexdigest()
def js(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def flat(s):return ''.join(unicodedata.normalize('NFKC',s).split())
def readarc(path,sha):
 assert h((ROOT/path).read_bytes())==sha
 with tarfile.open(ROOT/path) as t:
  assert len(t.getnames())==len(set(t.getnames()));return {m.name:t.extractfile(m).read() for m in t.getmembers()}
def validate(doc,base,extension,eraw,rraw):
 assert doc['schema']=='rainbow_raw_rows_profile_detail_overlay_v1' and doc['base_member_sha256']=='ff23321971f27f6608b572535ad1055dac21f5d07fd9d53c7efc3aefa94bc984'
 assert doc['all_finals_complete'] is doc['full_payload_SQL_adapter_verified'] is doc['complete_payload_insertion_audit'] is False
 assert doc['canonical_SQL_inserts']==doc['new_completed_races']==doc['network_requests']==0 and len(doc['events'])==14
 for d,b,x in zip(doc['events'],base['events'],extension['events']):
  assert d['identity']==b['identity']==x['identity'] and d['complete_payload'] is d['canonical_SQL_inserted'] is False
  key=x['NK_entry_member'][:-5];en=eraw[key+'.html'];rn=rraw[key+'_result.html'];em=json.loads(eraw[key+'.json']);rm=json.loads(rraw[key+'_result.meta.json'])
  for raw,meta,kind in [(en,em,'entry'),(rn,rm,'result')]:assert meta['sha256']==h(raw) and meta['bytes']==len(raw) and meta['status']==200 and meta['url']==meta['final_url']=='https://keirin.netkeiba.com/race/'+kind+'/?race_id='+key
  assert d['source_refs']=={'entry_html':{'member':key+'.html','bytes':len(en),'sha256':h(en)},'entry_meta':{'member':key+'.json','bytes':len(eraw[key+'.json']),'sha256':h(eraw[key+'.json'])},'result_html':{'member':key+'_result.html','bytes':len(rn),'sha256':h(rn)},'result_meta':{'member':key+'_result.meta.json','bytes':len(rraw[key+'_result.meta.json']),'sha256':h(rraw[key+'_result.meta.json'])}}
  ed=html.fromstring(en.decode());rd=html.fromstring(rn.decode());assert ed.xpath('//meta[@property="og:url"]/@content')==[em['url']] and rd.xpath('//input[@name="plp-raceid"]/@value')==[key]
  rows=rd.xpath('//table[@id="All_Result_Table"]/tbody/tr[td]') or rd.xpath('//table[@id="All_Result_Table"]/tr[td]');assert len(rows)==9
  assert len(d['entries'])==9
  for out,old in zip(d['entries'],b['payload']['entries']):
   car=old['car_no'];reg=old['rider_registration_number'];assert out['car_no']==car and out['registration_number']==reg
   tr=ed.get_element_by_id('player-'+str(car));td=list(tr.iterchildren('td'));assert len(td)==23
   anchor=ed.get_element_by_id('name_'+str(int(reg))+'_'+key);assert anchor in tr.iterdescendants() and flat(anchor.xpath('.//img/@alt')[0])==flat(old['player_name'])
   rawrow=json.loads(out['entry_fields']['raw_row_json']);expected={'provider':'netkeiba_historical_entry','cells':[''.join(c.itertext()).strip() for c in td],'row_class':tr.get('class','').split(),'headers':[flat(''.join(c.itertext())) for c in tr.getparent().getparent().iterdescendants('th')],'source_row_id':'player-'+str(car),'explicit_ID_anchor':'name_'+str(int(reg))+'_'+key}
   assert rawrow==expected and len(rawrow['headers'])==23 and rawrow['headers'][7:10]==['S','H','B']
   found=[r for r in rows if flat(r.xpath('./td')[2].text_content())==str(car)];assert len(found)==1;cells=found[0].xpath('./td');assert len(cells)==8
   link=cells[3].xpath('./a')[0].get('href');assert link=='https://keirin.netkeiba.com/db/profile/?id='+str(int(reg));assert flat(cells[3].xpath('.//img/@alt')[0])==flat(old['player_name'])
   detail=cells[4];br=detail.findall('br');assert len(br)<=1
   text='' if not br else ''.join(br[0].itersiblings()).strip() if len(list(br[0].itersiblings())) else (br[0].tail or '').strip()
   # Current saved tables use text tails, no following element siblings.
   assert not br or len(list(br[0].itersiblings()))==0
   assert out['result_fields']=={'result_comment':text}
   assert out['entry_fields']=={'raw_row_json':js(expected).decode(),'rider_source_id':link,'rider_id_status':'published_profile_link'}
   utc=datetime.datetime.fromisoformat(em['retrieved_at']).astimezone(datetime.timezone.utc).isoformat()
   assert out['proof']=={'entry_row_id':tr.get('id'),'entry_explicit_anchor':anchor.get('id'),'result_profile_href':link,'result_row_car':car,'result_detail_cell_index':4,'result_detail_cell_raw_text':''.join(detail.itertext()),'result_detail_br_present':bool(br),'detail_verification':'netkeiba_explicit_single_source_only' if text else 'netkeiba_explicit_blank_detail_not_inferred','entry_retrieved_at_original':em['retrieved_at'],'entry_retrieved_at_UTC':utc,'capture_time_not_assigned_to_composite_payload':True,'entry_anchor_href_raw':anchor.get('href')}
def main():
 bb=readarc('data/scope_review_0835/legacy_payload_drafts.tar.xz','28ceabb5cafc8035193c22813b7a9617aeb1d9b84279db9fa942a23da5da9ab2')['legacy_payload_drafts.json'];assert h(bb)=='ff23321971f27f6608b572535ad1055dac21f5d07fd9d53c7efc3aefa94bc984';base=json.loads(bb)
 extension=json.loads(readarc('data/scope_review_0730/entry_extension_overlay.tar.xz','10cc8901c0d7674dad51ba1fa3353d4e266b6741552e6e7c4b4bcc646b6433e6')['entry_extension_overlay.json'])
 eraw=readarc('data/scope_review_0730/entry_extension_raw.tar.xz','983e4f17e76ac5cea40967be16238268f7975408929b03aba140b38398331f72');rraw=readarc('data/scope_review_0831/independent_card_results_raw.tar.xz','b067a5fcb5629030f5440bc0efc88e368b753c4236a5a50fee8af372c270bbe2')
 manifest=(ROOT/'data/scope_review_0831/raw_manifest.json').read_bytes();assert h(manifest)=='47661ef1ca2ff32aa6cacf647b7db9c526a5652bbddc46f7ce5a615214c3d18f'
 for pool,man in [(eraw,extension['raw_manifest']['members']),(rraw,json.loads(manifest))]:
  assert pool.keys()==man.keys()
  for name,body in pool.items():assert len(body)==man[name]['bytes'] and h(body)==man[name]['sha256']
 rep=json.loads((OUT/'raw_rows_profile_detail_review.json').read_bytes());arc=rep['archive'];body=readarc(arc['path'],arc['sha256'])[arc['member']];assert len(body)==arc['member_bytes'] and h(body)==arc['member_sha256'];doc=json.loads(body);validate(doc,base,extension,eraw,rraw)
 spec=importlib.util.spec_from_file_location('attach0916',ROOT/'recovery/adapt_rainbow_raw_rows_0916.py');producer=importlib.util.module_from_spec(spec);spec.loader.exec_module(producer);updated=producer.attach(bb,body)
 expected=copy.deepcopy(base)
 for e,a in zip(expected['events'],doc['events']):
  proof={x['car_no']:x for x in a['entries']}
  for x in e['payload']['entries']:
   x.update(proof[x['car_no']]['entry_fields'])
   for field in ['raw_row_json','rider_source_id','rider_id_status']:e['unmapped_fields']['entries'][str(x['car_no'])].remove(field)
  for x in e['payload']['results']:x['result_comment']=proof[x['car_no']]['result_fields']['result_comment']
  e['unmapped_fields']['results']=[];e['raw_rows_attachment_evidence']=a
 assert updated==expected
 restored=copy.deepcopy(updated)
 for e,b in zip(restored['events'],base['events']):
  for x in e['payload']['entries']:
   for field in ['raw_row_json','rider_source_id','rider_id_status']:x.pop(field)
  for x in e['payload']['results']:x.pop('result_comment')
  del e['raw_rows_attachment_evidence'];e['unmapped_fields']=copy.deepcopy(b['unmapped_fields'])
 assert restored==base
 mem=sqlite3.connect(':memory:');mem.execute(base['observed_legacy_contract']['races_DDL'])
 for e in updated['events']:
  p=e['payload'];mem.execute('insert into races values (?,?,?,?,?,?,?,?,?,?)',(p['race_id'],e['identity']['day'],p['race']['source_url'],p['expected_label'],p['grade'],p['category'],'partial_adapter_draft',js(p).decode(),js(e['blockers']).decode(),None))
 for rid,p in mem.execute('select race_id,payload from races'):assert json.loads(p)==next(e['payload'] for e in expected['events'] if e['identity']['canonical_key']==rid)
 assert mem.execute("select count(*) from races where status in ('collected','completed')").fetchone()[0]==0;mem.close()
 mutations=[]
 cases=[('explicit_ID',lambda a:a.__setitem__('registration_number','999999')),('profile_href',lambda a:a['entry_fields'].__setitem__('rider_source_id','https://keirin.netkeiba.com/db/profile/?id=999999')),('detail_token',lambda a:a['result_fields'].__setitem__('result_comment','(病気欠場)')),('blank_detail_success',lambda a:a['proof'].__setitem__('detail_verification','independently_verified')),('raw_row',lambda a:a['entry_fields'].__setitem__('raw_row_json','{}')),('acquisition_time',lambda a:a['proof'].__setitem__('entry_retrieved_at_original','2023-06-06T00:00:00+09:00')),('composite_capture_invention',lambda a:a['entry_fields'].__setitem__('captured_at_utc',a['proof']['entry_retrieved_at_original'])),('profile_status_promotion',lambda a:a['entry_fields'].__setitem__('rider_id_status','all_registry_sources_verified'))]
 for name,fn in cases:
  bad=copy.deepcopy(doc);fn(bad['events'][0]['entries'][0])
  try:validate(bad,base,extension,eraw,rraw)
  except (AssertionError,ValueError,KeyError):mutations.append(name)
  else:raise AssertionError(name)
 for name,badbase,badbody in [('overlay_bytes',bb,body+b' '),('base_bytes',bb+b' ',body)]:
  try:producer.attach(badbase,badbody)
  except AssertionError:mutations.append(name)
  else:raise AssertionError(name)
 bad=copy.deepcopy(doc);bad['events'][0]['complete_payload']=True
 try:validate(bad,base,extension,eraw,rraw)
 except AssertionError:mutations.append('complete_flag')
 else:raise AssertionError('complete flag')
 counts={}
 for e in doc['events']:
  for a in e['entries']:s=a['result_fields']['result_comment'];counts[s]=counts.get(s,0)+1
 assert counts=={'(落車再乗)':3,'':119,'(失格)':2,'(事故)':2}
 out={'status':'independent_raw_rows_profile_detail_audit_passed','events':14,'explicit_profile_links':126,'raw_entry_rows':126,'result_detail_counts':counts,'detail_scope':'single source original tokens or explicit blank; no independent promotion','capture_times_assigned_to_composite':0,'lossless_base_recovery':True,'partial_live_DDL_in_memory_roundtrip':14,'in_memory_completed_rows':0,'mutations_rejected':mutations,'network_requests':0,'canonical_SQL_inserts':0,'new_completed_races':0,'all_finals_complete':False,'full_payload_SQL_adapter_verified':False,'complete_payload_insertion_audit':False,'second_SQLite_artifact_saved':False}
 (OUT/'independent_raw_rows_profile_detail_audit.json').write_text(json.dumps(out,ensure_ascii=False,sort_keys=True,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
