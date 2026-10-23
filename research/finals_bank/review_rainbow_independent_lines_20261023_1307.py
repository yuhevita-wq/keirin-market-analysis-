"""Compare the published line orders from two providers, preserving raw separators.
Run from workspace cwd after extracting 0831 and 1307 archives. No DB insertion.
"""
import pathlib,json,re,unicodedata,hashlib,urllib.parse,itertools,copy,tarfile
from lxml import html
ROOT=pathlib.Path('current_data/unpacked/finals_ai');BASE=ROOT/'data/scope_review_0457';RAW=pathlib.Path('raw_lines_1307');RESULT=pathlib.Path('raw_independent_0831');OUT=ROOT/'data/scope_review_1307'
def norm(s):return re.sub(r'\s+','',unicodedata.normalize('NFKC',str(s)))
def checked(p,meta):
 b=p.read_bytes();assert len(b)==meta['bytes'] and hashlib.sha256(b).hexdigest()==meta['sha256'];return b

def parse_lines(text,ath):
 # The fragment has no charset declaration. Decode strict UTF-8 before lxml.
 doc=html.fromstring(text);blocks=doc.xpath('//div[contains(concat(" ",normalize-space(@class)," ")," DeployBox ")]');assert len(blocks)==1
 groups=[];cur=[];tokens=[];labels=[]
 for node in blocks[0].xpath('.//div[contains(concat(" ",normalize-space(@class)," ")," Shaban_InBox ")]'):
  no=node.xpath('./span[contains(concat(" ",normalize-space(@class)," ")," Shaban_Num ")]/text()');sep=node.xpath('./span[@class="WakuSeparat"]')
  assert bool(no)!=bool(sep)
  if sep:
   assert len(sep)==1;tokens.append({'separator':True})
   if cur:groups.append(cur);cur=[]
   continue
  assert len(no)==1 and re.fullmatch('[1-9]',no[0]);car=int(no[0]);cur.append(car);names=node.xpath('./span[@class="Name"]/text()');assert len(names)==1 and norm(names[0])==norm(ath[str(car)]['short_name'])
  tokens.append({'car_no':car,'published_short_name':names[0]});labels.append(car)
 if cur:groups.append(cur)
 assert sorted(labels)==list(range(1,10)) and len(labels)==len(set(labels))==9
 return groups,tokens

def audit(e,raw_override=None,meta_override=None):
 k=e['day'].replace('-','')+e['venue_code']+str(e['race_no']).zfill(2)
 sources={}
 for v in e['source_files']:
  p=BASE/v['path'];sources[v['path']]=checked(p,v)
 cf=next(p for p in sources if p.endswith('card.html'));card=json.loads(sources[cf]);ath=card['race_athletes'];box=card['box'];assert (box['open_day'],box['vel_code'],box['race_num'])==(e['day'],e['venue_code'],e['race_no'])
 rf=next(p for p in sources if p.endswith('result.html') or p=='chariloto.html');c=html.fromstring(sources[rf]);sec=c.xpath('//*[@id="race'+str(e['race_no'])+'"]');assert len(sec)==1
 tables=sec[0].xpath('.//table[.//th[normalize-space()="周回予想"]]');assert len(tables)==1;cg=[];group=[]
 for cell in tables[0].xpath('.//table//tr/td'):
  no=cell.xpath('.//span[contains(concat(" ",@class," ")," square ")]')
  if no:assert len(no)==1;group.append(int(norm(no[0].text_content())))
  elif group:cg.append(group);group=[]
 if group:cg.append(group)
 assert cg==e['published_line_groups'] and sorted(itertools.chain.from_iterable(cg))==list(range(1,10))
 p=RESULT/(k+'_result.html');rm=json.loads(p.with_suffix('.meta.json').read_text());result=checked(p,rm);nk=html.fromstring(result);assert nk.xpath('//input[@name="plp-raceid"]/@value')==[k]
 script='\n'.join(nk.xpath('//script/text()'));assert 'function api_get_narabi_yoso' in script and 'https://keirin.netkeiba.com/race/api_get_narabi_yoso.html' in script and "_data.race_id    = _race_id" in script
 ids=[]
 for row in nk.xpath('//table[@id="All_Result_Table"]//tr[td]'):
  cs=row.xpath('./td');car=int(norm(cs[2].text_content()));qs=urllib.parse.parse_qs(urllib.parse.urlparse(cs[3].xpath('.//a/@href')[0]).query);rid=int(qs['id'][0]);assert rid==int(ath[str(car)]['un_id']);assert [norm(x) for x in cs[3].xpath('.//img/@alt')]==[norm(ath[str(car)]['name'])];ids.append({'car_no':car,'explicit_registration_number':str(rid).zfill(6)})
 assert sorted(x['car_no'] for x in ids)==list(range(1,10))
 p=RAW/(k+'.html');meta=meta_override or json.loads(p.with_suffix('.json').read_text());url=urllib.parse.urlparse(meta['url']);assert url.scheme=='https' and url.netloc=='keirin.netkeiba.com' and url.path=='/race/api_get_narabi_yoso.html';assert urllib.parse.parse_qs(url.query)=={'input':['UTF-8'],'output':['html'],'race_id':[k]} and meta['status']==200 and meta['final_url']==meta['url']
 b=raw_override if raw_override is not None else checked(p,meta)
 assert len(b)==meta['bytes'] and hashlib.sha256(b).hexdigest()==meta['sha256'];ng,tokens=parse_lines(b.decode('utf-8'),ath);assert ng==cg
 return {'day':e['day'],'venue_code':e['venue_code'],'race_no':e['race_no'],'race_key':k,'groups_chariloto':cg,'groups_netkeiba':ng,'netkeiba_raw_tokens':tokens,'explicit_result_identities':ids,'netkeiba_line_meta':meta,'result_page_raw_sha256':rm['sha256'],'chariloto_source_files':e['source_files'],'ordered_group_sequence_exact':True,'all_9_cars_once':True,'provider_publication_comparison_only':True,'canonical_db_inserted':False}

def main():
 events=json.loads((BASE/'rainbow_scope_independent_review.json').read_text())['events'];out=[audit(e) for e in events];mut=[];e=events[2];k=e['day'].replace('-','')+e['venue_code']+str(e['race_no']).zfill(2);p=RAW/(k+'.html');b=p.read_bytes();m=json.loads(p.with_suffix('.json').read_text())
 changes={'wrong_car':b.replace(b'Shaban_Num">4</span>',b'Shaban_Num">1</span>',1),'removed_separator':b.replace(b'WakuSeparat',b'UnknownSeparator',1),'wrong_name':b.replace('望月一'.encode(),'別名'.encode(),1),'changed_order':b.replace(b'Shaban_Num">4</span>',b'Shaban_Num">7</span>',1).replace(b'Shaban_Num">7</span><span class="Name">'+ '菅原大'.encode(),b'Shaban_Num">4</span><span class="Name">'+'菅原大'.encode(),1)}
 for name,x in changes.items():
  assert x!=b;mm=copy.deepcopy(m);mm.update(bytes=len(x),sha256=hashlib.sha256(x).hexdigest())
  try:audit(e,x,mm)
  except AssertionError:mut.append({'mutation':name,'rejected_after_recomputed_raw_hash':True})
  else:raise AssertionError('accepted mutation '+name)
 mm=copy.deepcopy(m);mm['url']=mm['url'].replace(k,'202312053710');mm['final_url']=mm['url']
 try:audit(e,b,mm)
 except AssertionError:mut.append({'mutation':'wrong_requested_race_key','rejected':True})
 else:raise AssertionError('accepted wrong race')
 report={'all_finals_complete':False,'canonical_db_changed':False,'events_verified':len(out),'car_identities_verified':sum(len(x['explicit_result_identities']) for x in out),'events':out,'mutations':mut,'note':'Independent publisher comparison of historical published predictions, not a guarantee of actual race formation. Car IDs are verified in saved actual result rows; short labels never infer registration IDs. UTF-8 fragments decoded explicitly; all raw separator spans including repeated blank separators retained. Shared upstream provenance is not asserted.'}
 (OUT/'independent_lines_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));manifest={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in RAW.iterdir()};(OUT/'lines_raw_manifest.json').write_text(json.dumps(manifest,indent=2))
 with tarfile.open(OUT/'independent_lines_raw.tar.xz','w:xz') as t:
  for p in sorted(RAW.iterdir()):t.add(p,arcname=p.name)
 with tarfile.open(OUT/'independent_lines_raw.tar.xz') as t:
  assert set(t.getnames())==set(manifest)
  for name,v in manifest.items():
   x=t.extractfile(name).read();assert len(x)==v['bytes'] and hashlib.sha256(x).hexdigest()==v['sha256']
 print(json.dumps({'events_verified':len(out),'identities':report['car_identities_verified'],'mutations_rejected':len(mut),'raw_members':len(manifest),'tar_bytes':(OUT/'independent_lines_raw.tar.xz').stat().st_size}))
if __name__=='__main__':main()
