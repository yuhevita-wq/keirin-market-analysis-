"""Attach explicit raw rows/profile links/result details to blocked 0835 drafts."""
import copy,datetime,hashlib,io,json,tarfile,unicodedata
from pathlib import Path
from lxml import html
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/scope_review_0916'
PINS={'data/scope_review_0835/legacy_payload_drafts.tar.xz':'28ceabb5cafc8035193c22813b7a9617aeb1d9b84279db9fa942a23da5da9ab2','data/scope_review_0831/independent_card_results_raw.tar.xz':'b067a5fcb5629030f5440bc0efc88e368b753c4236a5a50fee8af372c270bbe2','data/scope_review_0831/raw_manifest.json':'47661ef1ca2ff32aa6cacf647b7db9c526a5652bbddc46f7ce5a615214c3d18f','data/scope_review_0730/entry_extension_overlay.tar.xz':'10cc8901c0d7674dad51ba1fa3353d4e266b6741552e6e7c4b4bcc646b6433e6','data/scope_review_0730/entry_extension_raw.tar.xz':'983e4f17e76ac5cea40967be16238268f7975408929b03aba140b38398331f72'}
BASE='ff23321971f27f6608b572535ad1055dac21f5d07fd9d53c7efc3aefa94bc984'
HEADERS=['枠','車','','本紙','選手名府県年齢期別級班','競走得点','脚質','S','H','B','逃げ','まくり','差し','マーク','1着','2着','3着','着外','勝率','2連対率','3連対率','ギヤ倍数','選手コメント']
def sha(b):return hashlib.sha256(b).hexdigest()
def enc(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def n(v):return ''.join(unicodedata.normalize('NFKC',v).split())
def archive(path,manifest=None):
 b=(ROOT/path).read_bytes();assert sha(b)==PINS[path]
 with tarfile.open(fileobj=io.BytesIO(b)) as t:
  names=t.getnames();assert len(names)==len(set(names));data={m.name:t.extractfile(m).read() for m in t.getmembers()}
 if manifest is not None:
  assert set(data)==set(manifest)
  for name,b in data.items():assert sha(b)==manifest[name]['sha256'] and len(b)==manifest[name]['bytes']
 return data
def load():
 for p,h in PINS.items():assert sha((ROOT/p).read_bytes())==h
 b=archive('data/scope_review_0835/legacy_payload_drafts.tar.xz')['legacy_payload_drafts.json'];assert sha(b)==BASE
 extension=json.loads(archive('data/scope_review_0730/entry_extension_overlay.tar.xz')['entry_extension_overlay.json'])
 eraw=archive('data/scope_review_0730/entry_extension_raw.tar.xz',extension['raw_manifest']['members'])
 rraw=archive('data/scope_review_0831/independent_card_results_raw.tar.xz',json.loads((ROOT/'data/scope_review_0831/raw_manifest.json').read_bytes()))
 return b,json.loads(b),extension,eraw,rraw
def event_overlay(e,x,eb,em,rb,rm,emb,rmb):
 ident=e['identity'];assert ident==x['identity'];key=x['NK_entry_member'].removesuffix('.html');assert len(key)==12
 for b,m,part in [(eb,em,'entry'),(rb,rm,'result')]:
  assert m['status']==200 and m['bytes']==len(b) and m['sha256']==sha(b)
  assert m['url']==m['final_url']=='https://keirin.netkeiba.com/race/'+part+'/?race_id='+key
 # Acquisition time is evidence only, never used as race date or composite capture time.
 at=datetime.datetime.fromisoformat(em['retrieved_at']);assert at.tzinfo is not None
 ed=html.fromstring(eb.decode('utf-8'));rd=html.fromstring(rb.decode('utf-8'))
 assert ed.xpath('//link[@rel="canonical"]/@href')==[em['url']] and rd.xpath('//input[@name="plp-raceid"]/@value')==[key]
 table=rd.xpath('//table[@id="All_Result_Table"]');assert len(table)==1
 assert [c.text_content().strip() for c in table[0].xpath('.//th')]==['着','枠番','車番','選手名','着差(個人情報)','上り','決','SB']
 resultrows={}
 for row in table[0].xpath('.//tr[td]'):
  cells=row.xpath('./td');assert len(cells)==8;car=int(n(cells[2].text_content()));assert car not in resultrows;resultrows[car]=(row,cells)
 out=[]
 for entry in e['payload']['entries']:
  car=entry['car_no'];reg=entry['rider_registration_number'];row=ed.xpath('//tr[@id=$id]',id='player-'+str(car));assert len(row)==1;row=row[0];cells=row.xpath('./td');assert len(cells)==23
  headers=[n(c.text_content()) for c in row.getparent().getparent().xpath('.//th')];assert headers==HEADERS
  anchor=row.xpath('.//a[@id=$id]',id='name_'+str(int(reg))+'_'+key);assert len(anchor)==1 and n(anchor[0].xpath('.//img/@alt')[0])==n(entry['player_name'])
  rr,rc=resultrows[car];links=rc[3].xpath('./a/@href');assert links==['https://keirin.netkeiba.com/db/profile/?id='+str(int(reg))]
  assert n(rc[3].xpath('.//img/@alt')[0])==n(entry['player_name'])
  br=rc[4].xpath('./br');assert len(br)<=1
  comment=(br[0].tail or '').strip() if br else '';assert (not comment or comment.startswith('(') and comment.endswith(')'))
  rawrow={'provider':'netkeiba_historical_entry','cells':[c.text_content().strip() for c in cells],'row_class':row.get('class','').split(),'headers':headers,'source_row_id':row.get('id'),'explicit_ID_anchor':anchor[0].get('id')}
  fields={'raw_row_json':enc(rawrow).decode(),'rider_source_id':links[0],'rider_id_status':'published_profile_link'}
  out.append({'car_no':car,'registration_number':reg,'entry_fields':fields,'result_fields':{'result_comment':comment},'proof':{'entry_row_id':row.get('id'),'entry_explicit_anchor':anchor[0].get('id'),'result_profile_href':links[0],'result_row_car':car,'result_detail_cell_index':4,'result_detail_cell_raw_text':rc[4].text_content(),'result_detail_br_present':bool(br),'detail_verification':'netkeiba_explicit_single_source_only' if comment else 'netkeiba_explicit_blank_detail_not_inferred','entry_retrieved_at_original':em['retrieved_at'],'entry_retrieved_at_UTC':at.astimezone(datetime.timezone.utc).isoformat(),'capture_time_not_assigned_to_composite_payload':True,'entry_anchor_href_raw':anchor[0].get('href')}})
 return {'identity':copy.deepcopy(ident),'entries':out,'source_refs':{'entry_html':{'member':x['NK_entry_member'],'bytes':len(eb),'sha256':sha(eb)},'entry_meta':{'member':x['NK_meta_member'],'bytes':len(emb),'sha256':sha(emb)},'result_html':{'member':key+'_result.html','bytes':len(rb),'sha256':sha(rb)},'result_meta':{'member':key+'_result.meta.json','bytes':len(rmb),'sha256':sha(rmb)}},'complete_payload':False,'canonical_SQL_inserted':False}
def attach(basebytes,overlay_bytes):
 assert sha(overlay_bytes)=='80c603d0d903387b5f08ab6a75c56db978803fa70776cf6a26da2f4dbd518956'
 overlay=json.loads(overlay_bytes)
 assert sha(basebytes)==BASE and overlay['base_member_sha256']==BASE and overlay['all_finals_complete'] is False
 result=json.loads(basebytes);assert len(result['events'])==len(overlay['events'])==14
 for event,add in zip(result['events'],overlay['events']):
  assert event['identity']==add['identity'] and add['complete_payload'] is add['canonical_SQL_inserted'] is False
  bycar={a['car_no']:a for a in add['entries']}
  for a in event['payload']['entries']:
   ext=bycar[a['car_no']];assert a['rider_registration_number']==ext['registration_number']
   assert set(ext['entry_fields'])=={'raw_row_json','rider_source_id','rider_id_status'};a.update(copy.deepcopy(ext['entry_fields']))
   for k in ext['entry_fields']:event['unmapped_fields']['entries'][str(a['car_no'])].remove(k)
  for a in event['payload']['results']:a.update(copy.deepcopy(bycar[a['car_no']]['result_fields']))
  event['unmapped_fields']['results'].remove('result_comment');event['raw_rows_attachment_evidence']=copy.deepcopy(add)
 return result
def main():
 bb,base,extension,eraw,rraw=load();events=[]
 for e,x in zip(base['events'],extension['events']):
  key=x['NK_entry_member'].removesuffix('.html');events.append(event_overlay(e,x,eraw[x['NK_entry_member']],json.loads(eraw[x['NK_meta_member']]),rraw[key+'_result.html'],json.loads(rraw[key+'_result.meta.json']),eraw[x['NK_meta_member']],rraw[key+'_result.meta.json']))
 doc={'schema':'rainbow_raw_rows_profile_detail_overlay_v1','base_member_sha256':BASE,'input_pins':PINS,'events':events,'all_finals_complete':False,'canonical_SQL_inserts':0,'new_completed_races':0,'network_requests':0,'full_payload_SQL_adapter_verified':False,'complete_payload_insertion_audit':False}
 updated=attach(bb,enc(doc))
 for e,old in zip(updated['events'],base['events']):
  for a in e['payload']['entries']:
   for k in ['raw_row_json','rider_source_id','rider_id_status']:del a[k]
  for a in e['payload']['results']:del a['result_comment']
  del e['raw_rows_attachment_evidence'];e['unmapped_fields']=copy.deepcopy(old['unmapped_fields'])
 assert updated==base
 OUT.mkdir(parents=True,exist_ok=True);body=enc(doc);p=OUT/'raw_rows_profile_detail_overlay.tar.xz'
 with tarfile.open(p,'w:xz',preset=9) as t:m=tarfile.TarInfo('raw_rows_profile_detail_overlay.json');m.size=len(body);m.mtime=0;t.addfile(m,io.BytesIO(body))
 with tarfile.open(p) as t:assert t.getnames()==['raw_rows_profile_detail_overlay.json'] and t.extractfile(t.getmembers()[0]).read()==body
 counts={}
 for e in events:
  for a in e['entries']:
   s=a['result_fields']['result_comment'];counts[s]=counts.get(s,0)+1
 report={'status':'partial_payload_raw_row_profile_detail_attachment_only','events':14,'entries':126,'result_rows':126,'result_detail_counts':counts,'new_entry_fields':['raw_row_json','rider_source_id','rider_id_status'],'capture_time_scope':'original acquisition metadata only; composite captured fields still absent','profile_link_scope':'actual result DOM href with explicit registration number; no guessed entry Javascript URL','source_archive_members_verified':84,'lossless_original_draft_recovery':True,'archive':{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p.read_bytes()),'member':'raw_rows_profile_detail_overlay.json','member_bytes':len(body),'member_sha256':sha(body)},'canonical_SQL_inserts':0,'new_completed_races':0,'all_finals_complete':False,'full_payload_SQL_adapter_verified':False,'complete_payload_insertion_audit':False}
 (OUT/'raw_rows_profile_detail_review.json').write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
