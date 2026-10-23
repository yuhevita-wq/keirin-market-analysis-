"""Verify newly sourced historical S/B/rates without importing current profiles."""
import copy,hashlib,io,json,re,tarfile,unicodedata
from collections import Counter
from decimal import Decimal,ROUND_HALF_UP
from pathlib import Path
from lxml import html
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/scope_review_0730';RAW=Path('new_entry_0730')
PINS={'data/scope_review_0552/integrated_candidates.tar.xz': 'c1a030bc66b78fec4bda307046402f9db15fa67dd386845b8d19862cd3f29e3a', 'data/scope_review_0552/integrated_candidate_review.json': '8d41909097f61ab60de8490f64f74a4facd5cc4a0ca8a580ccd45906a951a3ad', 'data/scope_review_0552/independent_integrated_candidate_audit.json': 'c7656fc4864e85692afac82910348f132f6b62264dd00a3877975d158f008537', 'data/scope_review_0636/payout_fragment_review.json': 'a8e27f3a622a6cd0bbaf7806241d84c1687a5f597ff3845fdd7ee189121ec98a'}
BASE_SHA='b9ed0533f9c8a654236a4c4c8d9d275161acdd6c154ad6c3efbaa444cb307e3f'
HEADERS=['枠','車','','本紙','選手名府県年齢期別級班','競走得点','脚質','S','H','B','逃げ','まくり','差し','マーク','1着','2着','3着','着外','勝率','2連対率','3連対率','ギヤ倍数','選手コメント']
def sha(b):return hashlib.sha256(b).hexdigest()
def enc(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def n(x):return ''.join(unicodedata.normalize('NFKC',x).split())
def base():
 for p,h in PINS.items():assert sha((ROOT/p).read_bytes())==h
 with tarfile.open(ROOT/'data/scope_review_0552/integrated_candidates.tar.xz') as t:b=t.extractfile('integrated_candidates.json').read()
 assert sha(b)==BASE_SHA;return json.loads(b)
def entries(e,raw,meta,card):
 k=e['observed_provider_keys']['netkeiba_race_id'];assert meta['status']==200 and meta['url']==meta['final_url']=='https://keirin.netkeiba.com/race/entry/?race_id='+k
 assert meta['sha256']==sha(raw) and meta['bytes']==len(raw);assert str(meta['headers'].get('Content-Length',len(raw)))==str(len(raw))
 doc=html.fromstring(raw.decode('utf-8','strict'));assert doc.xpath('//link[@rel="canonical"]/@href')==[meta['url']] and doc.xpath('//meta[@property="og:url"]/@content')==[meta['url']];assert doc.xpath('//li[@id=$id]/a[contains(@class,"Active")]/@href',id='racing_num_'+k)==[meta['url']];title=n(doc.xpath('/html/head/title')[0].text_content());assert e['identity']['day'].replace('-','年',1).replace('-','月',1)+'日'+str(e['identity']['race_no'])+'R' in title
 assert len(doc.xpath('//tr[starts-with(@id,"player-")]'))==9
 assert (card['box']['open_day'],card['box']['vel_code'],card['box']['race_num'])==(e['identity']['day'],e['identity']['venue_code'],e['identity']['race_no'])
 out=[]
 for a in e['entries']:
  car=a['car_no'];reg=a['registration_number'];old=card['race_athletes'][str(car)];row=doc.xpath('//tr[@id=$id]',id='player-'+str(car));assert len(row)==1;row=row[0]
  headers=[n(x.text_content()) for x in row.getparent().getparent().xpath('.//th')];assert headers==HEADERS
  anchor=row.xpath('.//a[@id=$id]',id='name_'+str(int(reg))+'_'+k);assert len(anchor)==1;anchor=anchor[0]
  names=anchor.xpath('.//img[contains(@class,"PlayerPhotoImg")]/@alt');assert len(names)==1 and n(names[0])==n(a['name_raw'])
  cl=n(anchor.xpath('.//dd[@class="PlayerClass"]')[0].text_content());fr=n(anchor.xpath('.//dd[@class="PlayerFrom"]')[0].text_content())
  assert cl==a['term']+'期'+n(a['class_raw']);assert fr==a['prefecture_raw']+str(a['age_at_race'])+'歳'
  assert int(old['un_id'])==int(reg) and old['entry_num']==car
  cells=row.xpath('./td');assert len(cells)==23;tokens=[c.text_content().strip() for c in cells];assert n(tokens[1])==str(car)
  for i in [7,8,9]:assert re.fullmatch('[0-9]+',n(tokens[i]))
  for i in [18,19,20]:assert re.fullmatch(r'[0-9]+(?:\.[0-9]+)?%',n(tokens[i]))
  s,b,win,two,three=[n(tokens[i]).rstrip('%') for i in [7,9,18,19,20]]
  assert int(s)==old['standing_count'] and int(b)==old['back_count']
  assert Decimal(win)==Decimal(old['average_win'])/10 and Decimal(two)==Decimal(old['average_rentai'])/10
  charthree=str(old['average_rentai3_formatted']);rounded=Decimal(charthree).quantize(Decimal('.1'),rounding=ROUND_HALF_UP)
  three_status='published_display_matches_at_one_decimal' if Decimal(three)==rounded else 'provider_display_difference_unresolved'
  comment_status='matching_published_comments' if n(tokens[22])==n(old['comment']) else 'provider_comment_gap_unresolved'
  fields={'s_count':s,'b_count':b,'win_rate':win,'top2_rate':two}
  if three_status=='published_display_matches_at_one_decimal':fields['top3_rate']=three
  out.append({'car_no':car,'registration_number':reg,'legacy_entry_fields_verified':fields,'new_field_proof':{'S':{'chariloto_field':'standing_count','chariloto_raw':old['standing_count'],'NK_raw':tokens[7]},'B':{'chariloto_field':'back_count','chariloto_raw':old['back_count'],'NK_raw':tokens[9]},'win_rate':{'chariloto_field':'average_win','chariloto_raw':old['average_win'],'scale_divisor':10,'NK_raw':tokens[18]},'top2_rate':{'chariloto_field':'average_rentai','chariloto_raw':old['average_rentai'],'scale_divisor':10,'NK_raw':tokens[19]},'top3_rate':{'chariloto_field':'average_rentai3_formatted','chariloto_raw':old['average_rentai3_formatted'],'NK_raw':tokens[20],'comparison':'Decimal printed Char value rounded HALF_UP to one decimal; no replacement of either original display','status':three_status},'comment':{'chariloto_raw':old['comment'],'NK_raw':tokens[22],'status':comment_status},'H':{'NK_raw':tokens[8],'status':'explicit_single_source_only_no_Chariloto_counterpart_imported'}},'DOM':{'row_id':'player-'+str(car),'explicit_ID_anchor':anchor.get('id'),'headers':headers,'cell_indices':{'S':7,'H':8,'B':9,'win':18,'two':19,'three':20,'comment':22}},'complete_payload':False})
 return out
def attach_verified_entry_extension(base_bytes,overlay_bytes):
 assert sha(base_bytes)==BASE_SHA and sha(overlay_bytes)=='3597429ce71d38b3a66ac777c083afaea4543bd3b86f33cfbd94c082d57c1503'
 result=json.loads(base_bytes);overlay=json.loads(overlay_bytes)
 assert result['all_finals_complete'] is overlay['all_finals_complete'] is False and len(result['events'])==len(overlay['events'])==14
 for event,extension in zip(result['events'],overlay['events']):
  assert event['identity']==extension['identity'] and event['identity']['canonical_key']==extension['local_key']
  for entry,proof in zip(event['entries'],extension['entries']):
   assert entry['car_no']==proof['car_no'] and entry['registration_number']==proof['registration_number'] and proof['complete_payload'] is False
   entry['production_entry_fields_fragment']=copy.deepcopy(proof['legacy_entry_fields_verified'])
   entry['historical_entry_extension_proof']=copy.deepcopy(proof)
 return result
def main():
 d=base();OUT.mkdir(parents=True,exist_ok=True);members={};events=[];counts=Counter()
 for e in d['events']:
  k=e['observed_provider_keys']['netkeiba_race_id'];raw=(RAW/(k+'.html')).read_bytes();mb=(RAW/(k+'.json')).read_bytes();meta=json.loads(mb);ref=e['component_raw_provenance']['card'];cb=(ROOT/ref['path']).read_bytes();assert sha(cb)==ref['sha256'] and len(cb)==ref['bytes'];card=json.loads(cb)
  # The entry URL must actually be linked from the saved target result HTML.
  rr=e['component_raw_provenance']['independent_result'];ar=(ROOT/rr['archive']['path']).read_bytes();assert sha(ar)==rr['archive']['sha256']
  with tarfile.open(fileobj=io.BytesIO(ar)) as t:rb=t.extractfile(rr['member']).read()
  assert len(rb)==rr['bytes'] and sha(rb)==rr['sha256'];rd=html.fromstring(rb.decode());assert rd.xpath('//a[@href=$u]',u=meta['url'])
  rows=entries(e,raw,meta,card);members[k+'.html']=raw;members[k+'.json']=mb
  for a in rows:counts[a['new_field_proof']['top3_rate']['status']]+=1;counts[a['new_field_proof']['comment']['status']]+=1
  events.append({'local_key':e['identity']['canonical_key'],'identity':copy.deepcopy(e['identity']),'NK_entry_member':k+'.html','NK_meta_member':k+'.json','card_source':copy.deepcopy(ref),'observed_link_source':copy.deepcopy(rr),'entries':rows,'complete_payload':False,'canonical_SQL_inserted':False})
 assert counts=={'published_display_matches_at_one_decimal':124,'provider_display_difference_unresolved':2,'matching_published_comments':108,'provider_comment_gap_unresolved':18}
 rawarchive=OUT/'entry_extension_raw.tar.xz'
 with tarfile.open(rawarchive,'w:xz',preset=9) as t:
  for name,b in sorted(members.items()):m=tarfile.TarInfo(name);m.size=len(b);m.mtime=0;t.addfile(m,io.BytesIO(b))
 with tarfile.open(rawarchive) as t:assert len(t.getnames())==len(set(t.getnames()))==28;assert {m.name:t.extractfile(m).read() for m in t.getmembers()}==members
 manifest={'path':str(rawarchive.relative_to(ROOT)),'bytes':rawarchive.stat().st_size,'sha256':sha(rawarchive.read_bytes()),'members':{k:{'bytes':len(v),'sha256':sha(v)} for k,v in sorted(members.items())}}
 doc={'schema':'rainbow_historical_entry_extension_v1','base_integrated_candidate_member_sha256':BASE_SHA,'input_pins':PINS,'raw_manifest':manifest,'events':events,'counts':dict(counts),'core_field_matches':{'S':126,'B':126,'win_rate':126,'top2_rate':126},'new_raw_pages':14,'all_finals_complete':False,'canonical_SQL_inserts':0,'new_completed_races':0,'current_performance_imported':False,'full_payload_SQL_adapter_verified':False}
 body=enc(doc);p=OUT/'entry_extension_overlay.tar.xz'
 with tarfile.open(p,'w:xz',preset=9) as t:m=tarfile.TarInfo('entry_extension_overlay.json');m.size=len(body);m.mtime=0;t.addfile(m,io.BytesIO(body))
 with tarfile.open(p) as t:assert t.getnames()==['entry_extension_overlay.json'] and t.extractfile(t.getmembers()[0]).read()==body
 review={k:copy.deepcopy(v) for k,v in doc.items() if k not in ['events']};review['archive']={'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p.read_bytes()),'member':'entry_extension_overlay.json','member_bytes':len(body),'member_sha256':sha(body)}
 (OUT/'entry_extension_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2,sort_keys=True)+'\n');print(json.dumps({'counts':dict(counts),'archive':review['archive'],'raw_archive':{k:v for k,v in manifest.items() if k!='members'}},ensure_ascii=False))
if __name__=='__main__':main()
