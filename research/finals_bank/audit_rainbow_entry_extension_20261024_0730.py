"""Independent DOM and exact-field audit of historical entry extensions."""
import copy,hashlib,io,json,re,sqlite3,tarfile,unicodedata
from collections import Counter
from decimal import Decimal,ROUND_HALF_UP
from pathlib import Path
from lxml import html
ROOT=Path(__file__).resolve().parents[1]
PINS={'data/scope_review_0552/integrated_candidates.tar.xz': 'c1a030bc66b78fec4bda307046402f9db15fa67dd386845b8d19862cd3f29e3a', 'data/scope_review_0552/integrated_candidate_review.json': '8d41909097f61ab60de8490f64f74a4facd5cc4a0ca8a580ccd45906a951a3ad', 'data/scope_review_0552/independent_integrated_candidate_audit.json': 'c7656fc4864e85692afac82910348f132f6b62264dd00a3877975d158f008537', 'data/scope_review_0636/payout_fragment_review.json': 'a8e27f3a622a6cd0bbaf7806241d84c1687a5f597ff3845fdd7ee189121ec98a'}
def h(b):return hashlib.sha256(b).hexdigest()
def j(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def n(s):return ''.join(unicodedata.normalize('NFKC',s).split())
def archive(path):
 with tarfile.open(ROOT/path) as t:
  assert len(t.getnames())==len(set(t.getnames()));return {m.name:t.extractfile(m).read() for m in t.getmembers()}
def verify_rows(event,extension,body,meta,card):
 key=event['observed_provider_keys']['netkeiba_race_id'];url='https://keirin.netkeiba.com/race/entry/?race_id='+key
 assert meta['url']==meta['final_url']==url and meta['status']==200 and meta['sha256']==h(body) and meta['bytes']==len(body)
 d=html.fromstring(body.decode('utf-8'));assert d.xpath('/html/head/link[@rel="canonical"]/@href')==[url];assert d.xpath('/html/head/meta[@property="og:url"]/@content')==[url]
 assert d.xpath('//li[@id="racing_num_'+key+'"]//a[contains(concat(" ",normalize-space(@class)," ")," Active ")]/@href')==[url]
 title=n(d.xpath('/html/head/title')[0].text_content());assert event['identity']['day'].replace('-','年',1).replace('-','月',1)+'日'+str(event['identity']['race_no'])+'R' in title
 assert len(extension['entries'])==len(d.xpath('//tr[starts-with(@id,"player-")]'))==9;counts=Counter()
 for idx,(old,a) in enumerate(zip(event['entries'],extension['entries'])):
  car=old['car_no'];reg=old['registration_number'];assert a['car_no']==car and a['registration_number']==reg and a['complete_payload'] is False
  row=d.xpath('//tr[@id="player-'+str(car)+'"]');assert len(row)==1;row=row[0];anchor=row.xpath('.//a[@id="name_'+str(int(reg))+'_'+key+'"]');assert len(anchor)==1;anchor=anchor[0]
  assert n(anchor.xpath('.//img[contains(@class,"PlayerPhotoImg")]/@alt')[0])==n(old['name_raw'])
  assert n(anchor.xpath('.//dd[@class="PlayerClass"]')[0].text_content())==old['term']+'期'+n(old['class_raw'])
  assert n(anchor.xpath('.//dd[@class="PlayerFrom"]')[0].text_content())==old['prefecture_raw']+str(old['age_at_race'])+'歳'
  headers=[n(c.text_content()) for c in row.getparent().getparent().xpath('.//th')]
  assert headers==['枠','車','','本紙','選手名府県年齢期別級班','競走得点','脚質','S','H','B','逃げ','まくり','差し','マーク','1着','2着','3着','着外','勝率','2連対率','3連対率','ギヤ倍数','選手コメント']
  cells=row.xpath('./td');assert len(cells)==23;raw=[v.text_content().strip() for v in cells];v=card['race_athletes'][str(car)];assert v['entry_num']==car and int(v['un_id'])==int(reg)
  assert n(raw[1])==str(car);s=n(raw[7]);b=n(raw[9]);w=n(raw[18]).removesuffix('%');two=n(raw[19]).removesuffix('%');three=n(raw[20]).removesuffix('%')
  assert int(s)==v['standing_count'] and int(b)==v['back_count'];assert Decimal(w)*10==v['average_win'] and Decimal(two)*10==v['average_rentai']
  r3='published_display_matches_at_one_decimal' if Decimal(three)==Decimal(str(v['average_rentai3_formatted'])).quantize(Decimal('0.1'),rounding=ROUND_HALF_UP) else 'provider_display_difference_unresolved'
  cm='matching_published_comments' if n(raw[22])==n(v['comment']) else 'provider_comment_gap_unresolved';counts.update([r3,cm])
  fields={'s_count':s,'b_count':b,'win_rate':w,'top2_rate':two}
  if r3=='published_display_matches_at_one_decimal':fields['top3_rate']=three
  assert a['legacy_entry_fields_verified']==fields
  expected={'S':{'chariloto_field':'standing_count','chariloto_raw':v['standing_count'],'NK_raw':raw[7]},'B':{'chariloto_field':'back_count','chariloto_raw':v['back_count'],'NK_raw':raw[9]},'win_rate':{'chariloto_field':'average_win','chariloto_raw':v['average_win'],'scale_divisor':10,'NK_raw':raw[18]},'top2_rate':{'chariloto_field':'average_rentai','chariloto_raw':v['average_rentai'],'scale_divisor':10,'NK_raw':raw[19]},'top3_rate':{'chariloto_field':'average_rentai3_formatted','chariloto_raw':v['average_rentai3_formatted'],'NK_raw':raw[20],'comparison':'Decimal printed Char value rounded HALF_UP to one decimal; no replacement of either original display','status':r3},'comment':{'chariloto_raw':v['comment'],'NK_raw':raw[22],'status':cm},'H':{'NK_raw':raw[8],'status':'explicit_single_source_only_no_Chariloto_counterpart_imported'}}
  assert a['new_field_proof']==expected
  assert a['DOM']=={'row_id':'player-'+str(car),'explicit_ID_anchor':anchor.get('id'),'headers':headers,'cell_indices':{'S':7,'H':8,'B':9,'win':18,'two':19,'three':20,'comment':22}}
 return counts
def check(doc,base,rawfiles):
 assert doc['schema']=='rainbow_historical_entry_extension_v1' and doc['base_integrated_candidate_member_sha256']=='b9ed0533f9c8a654236a4c4c8d9d275161acdd6c154ad6c3efbaa444cb307e3f' and doc['input_pins']==PINS
 assert doc['all_finals_complete'] is False and doc['current_performance_imported'] is False and doc['full_payload_SQL_adapter_verified'] is False and doc['canonical_SQL_inserts']==doc['new_completed_races']==0 and doc['new_raw_pages']==14
 assert len(doc['events'])==len(base['events'])==14;total=Counter()
 for ext,e in zip(doc['events'],base['events']):
  k=e['observed_provider_keys']['netkeiba_race_id'];assert ext['identity']==e['identity'] and ext['local_key']==e['identity']['canonical_key'];assert ext['complete_payload'] is False and ext['canonical_SQL_inserted'] is False
  assert ext['NK_entry_member']==k+'.html' and ext['NK_meta_member']==k+'.json' and ext['card_source']==e['component_raw_provenance']['card'] and ext['observed_link_source']==e['component_raw_provenance']['independent_result']
  ref=ext['card_source'];cb=(ROOT/ref['path']).read_bytes();assert len(cb)==ref['bytes'] and h(cb)==ref['sha256'];card=json.loads(cb)
  rr=ext['observed_link_source'];ab=(ROOT/rr['archive']['path']).read_bytes();assert h(ab)==rr['archive']['sha256'];rb=archive(rr['archive']['path'])[rr['member']];assert h(rb)==rr['sha256'];assert html.fromstring(rb.decode()).xpath('//a[@href=$url]',url='https://keirin.netkeiba.com/race/entry/?race_id='+k)
  total+=verify_rows(e,ext,rawfiles[k+'.html'],json.loads(rawfiles[k+'.json']),card)
 assert dict(total)==doc['counts']=={'published_display_matches_at_one_decimal':124,'provider_display_difference_unresolved':2,'matching_published_comments':108,'provider_comment_gap_unresolved':18}
 assert doc['core_field_matches']=={'S':126,'B':126,'win_rate':126,'top2_rate':126}
def main():
 for p,digest in PINS.items():assert h((ROOT/p).read_bytes())==digest
 basebytes=archive('data/scope_review_0552/integrated_candidates.tar.xz')['integrated_candidates.json'];assert h(basebytes)=='b9ed0533f9c8a654236a4c4c8d9d275161acdd6c154ad6c3efbaa444cb307e3f';base=json.loads(basebytes)
 review=json.loads((ROOT/'data/scope_review_0730/entry_extension_review.json').read_bytes());a=review['archive'];b=(ROOT/a['path']).read_bytes();assert len(b)==a['bytes'] and h(b)==a['sha256'];contents=archive(a['path']);assert list(contents)==[a['member']];body=contents[a['member']];assert h(body)==a['member_sha256'] and len(body)==a['member_bytes'];doc=json.loads(body)
 manifest=doc['raw_manifest'];rb=(ROOT/manifest['path']).read_bytes();assert len(rb)==manifest['bytes'] and h(rb)==manifest['sha256'];rawfiles=archive(manifest['path']);assert len(rawfiles)==28 and set(rawfiles)==set(manifest['members'])
 for p,b in rawfiles.items():assert h(b)==manifest['members'][p]['sha256'] and len(b)==manifest['members'][p]['bytes']
 check(doc,base,rawfiles)
 import review_rainbow_entry_extension_0730 as adapter
 attached=adapter.attach_verified_entry_extension(basebytes,body)
 for e,ext in zip(attached['events'],doc['events']):
  for entry,proof in zip(e['entries'],ext['entries']):
   assert entry.pop('production_entry_fields_fragment')==proof['legacy_entry_fields_verified']
   assert entry.pop('historical_entry_extension_proof')==proof
 assert attached==base
 c=sqlite3.connect(':memory:');c.execute("create table candidate_entry_extensions(key text,car integer,status text check(status='blocked'),payload text,primary key(key,car))")
 for e in doc['events']:
  for row in e['entries']:c.execute('insert into candidate_entry_extensions values(?,?,?,?)',(e['local_key'],row['car_no'],'blocked',j(row).decode()))
 assert [json.loads(v[0]) for v in c.execute('select payload from candidate_entry_extensions order by rowid')]==[row for e in doc['events'] for row in e['entries']];c.close()
 tests={
 'S':lambda x:x['events'][0]['entries'][0]['legacy_entry_fields_verified'].__setitem__('s_count','999'),
 'B':lambda x:x['events'][0]['entries'][0]['legacy_entry_fields_verified'].__setitem__('b_count','999'),
 'win_rate':lambda x:x['events'][0]['entries'][0]['legacy_entry_fields_verified'].__setitem__('win_rate','99.9'),
 'top2_rate':lambda x:x['events'][0]['entries'][0]['legacy_entry_fields_verified'].__setitem__('top2_rate','99.9'),
 'ID':lambda x:x['events'][0]['entries'][0].__setitem__('registration_number','099999'),
 'DOM_column':lambda x:x['events'][0]['entries'][0]['DOM']['cell_indices'].__setitem__('S',8),
 'top3_false_success':lambda x:next(a for e in x['events'] for a in e['entries'] if a['new_field_proof']['top3_rate']['status']=='provider_display_difference_unresolved')['legacy_entry_fields_verified'].__setitem__('top3_rate','53.9'),
 'comment_false_success':lambda x:next(a for e in x['events'] for a in e['entries'] if a['new_field_proof']['comment']['status']=='provider_comment_gap_unresolved')['new_field_proof']['comment'].__setitem__('status','matching_published_comments'),
 'current_profile_import':lambda x:x.__setitem__('current_performance_imported',True),
 'complete_flag':lambda x:x.__setitem__('all_finals_complete',True)}
 rejected=[]
 for name,change in tests.items():
  changed=copy.deepcopy(doc);change(changed)
  try:check(changed,base,rawfiles)
  except (AssertionError,KeyError):rejected.append(name)
  else:raise AssertionError('accepted mutation '+name)
 # A source mutation is rejected even after the raw metadata hash is recomputed.
 e=base['events'][0];ext=doc['events'][0];k=e['observed_provider_keys']['netkeiba_race_id'];orig=rawfiles[k+'.html'];meta=json.loads(rawfiles[k+'.json']);card=json.loads((ROOT/ext['card_source']['path']).read_bytes())
 for name,mutator in [('raw_explicit_ID',lambda d:d.xpath('//tr[@id="player-1"]//a[starts-with(@id,"name_")]')[0].set('id','name_99999_'+k)),('raw_S_field',lambda d:setattr(d.xpath('//tr[@id="player-1"]/td')[7],'text','999'))]:
  d=html.fromstring(orig.decode());mutator(d);b=html.tostring(d,encoding='utf-8');m=copy.deepcopy(meta);m.update(bytes=len(b),sha256=h(b))
  try:verify_rows(e,ext,b,m,card)
  except (AssertionError,KeyError):rejected.append(name)
  else:raise AssertionError('accepted raw mutation '+name)
 out={'status':'historical_entry_extension_independent_audit_passed','entries':126,'independent_S_B_win_top2':126,'top3_display_matches':124,'top3_display_differences_unresolved':2,'comment_matches':108,'comment_gaps_unresolved':18,'H_single_source_only':126,'verified_attachment_and_lossless_original_recovery':True,'blocked_in_memory_rows':126,'mutations_rejected':rejected,'full_payload_SQL_adapter_verified':False,'canonical_SQL_inserts':0,'new_completed_races':0,'second_SQLite_artifact_saved':False,'all_finals_complete':False}
 (ROOT/'data/scope_review_0730/independent_entry_extension_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True)+'\n');print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
