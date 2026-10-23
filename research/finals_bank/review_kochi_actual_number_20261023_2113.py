"""Verify one legacy encoded race-number observation against actual provider DOM.
No general subtraction rule, no canonical DB mutation, no complete-race approval.
"""
import pathlib,json,re,hashlib,sqlite3,tarfile,io,copy,urllib.parse,unicodedata
from lxml import html
ROOT=pathlib.Path('current_data/unpacked/finals_ai');OUT=ROOT/'data/scope_review_2113';RID='7420260531030107';NK='202606037407'
DBSHA='04c4de11b6937c3b3a499f042410786cfdde4c2892ca0af46999ca059c5e2d91'
PAYSHA='b28159b667c61970e8f105fca8f4b7935b72ddd475b23c959cd607c9f4ea14fe'
def sha(b):return hashlib.sha256(b).hexdigest()
def norm(v):return re.sub(r'\s+','',unicodedata.normalize('NFKC',str(v)))
def body(files,key):
 b=files[key+'.html'];m=json.loads(files[key+'.json']);assert m['status']==200 and m['bytes']==len(b) and m['sha256']==sha(b)
 return html.fromstring(b.decode('utf8')),m

def verify(files,payload):
 assert payload['race_id']==RID and payload['race']['race_id']==RID and payload['race']['race_date']=='2026-06-03' and payload['race']['track']=='高知競輪'
 assert payload['race']['race_no']==107 # original field intentionally retained
 ath={int(e['car_no']):e for e in payload['entries']};ref={int(e['car_no']):e for e in payload['results']};assert set(ath)==set(ref)==set(range(1,8))
 nk,nm=body(files,'nk_kochi_actual');op,om=body(files,'op_kochi_card')
 assert nm['url']==nm['final_url']=='https://keirin.netkeiba.com/race/result/?race_id='+NK
 assert om['url']==om['final_url']=='https://www.oddspark.com/keirin/RaceList.do?joCode=74&kaisaiBi=20260603&raceNo=7'
 assert nk.xpath('//input[@name="plp-raceid"]/@value')==[NK]
 nt=norm(nk.findtext('.//title'));ot=norm(op.findtext('.//title'))
 assert '2026年06月03日7Rチャレンジ決勝' in nt and '高知競輪' in nt and norm(payload['race']['event_name']) in nt
 assert '2026年6月3日(水)高知競輪7R' in ot and 'A級チ決勝' in ot
 for field,value in [('joCode','74'),('kaisaiBi','20260603'),('raceNo','7')]:assert op.xpath('//input[@id="'+field+'"]/@value')==[value]
 rows=[r for r in op.xpath('//table[@summary="出走表"]/tr[td]') if any(re.fullmatch(r'no[1-9]',c.get('class','')) for c in r.xpath('./td'))];assert len(rows)==7
 o={}
 for row in rows:
  cs=row.xpath('./td');num=[c for c in cs if re.fullmatch(r'no[1-9]',c.get('class',''))];assert len(num)==1;car=int(norm(num[0].text_content()));assert car not in o;a=ath[car]
  links=cs[-8].xpath('.//a');assert len(links)==1;u=urllib.parse.urlparse(links[0].get('href'));assert u.path.endswith('/PlayerDetail.do');num=urllib.parse.parse_qs(u.query)['playerCd'][0];reg=a['rider_registration_number'];assert re.fullmatch(r'[0-9]{6}',reg) and int(num)==int(reg)
  official=urllib.parse.urlparse(a['rider_source_id']);assert official.hostname in ['keirin.jp','www.keirin.jp'] and urllib.parse.parse_qs(official.query)['snum']==[reg]
  assert norm(links[0].text_content())==norm(a['player_name']);at=re.search(r'(\d+)歳/(\d+)期',norm(cs[-8].text_content()));assert at and at.groups()==(str(a['age']),str(a['term']));assert norm(cs[-6].text_content())==a['class']=='A3'
  assert norm(cs[-7].text_content())==norm(a['prefecture']);o[car]={'registration_number':reg,'name':norm(a['player_name']),'term':str(a['term']),'op_profile_href':links[0].get('href')}
 assert set(o)==set(ath)
 checked=[]
 for row in nk.xpath('//table[@id="All_Result_Table"]//tr[td]'):
  cs=row.xpath('./td');assert len(cs)==8;car=int(norm(cs[2].text_content()));assert car in o and car not in [x['car_no'] for x in checked];a=ath[car]
  href=cs[3].xpath('.//a/@href');assert len(href)==1;u=urllib.parse.urlparse(href[0]);assert u.hostname=='keirin.netkeiba.com' and u.path=='/db/profile/';reg=urllib.parse.parse_qs(u.query)['id'][0];assert int(reg)==int(a['rider_registration_number'])
  assert [norm(s) for s in cs[3].xpath('.//img/@alt')]==[norm(a['player_name'])];info=norm(cs[3].text_content());cl=re.search(r'(\d+)期([ASL][123])',info);age=re.search(r'(\d+)歳',info);assert cl and age and cl.groups()==(str(a['term']),'A3') and age[1]==str(a['age'])
  rank=re.sub(r'着$','',norm(cs[0].text_content()));assert rank==str(ref[car]['finish_text']);assert cs[5].text_content().strip()==str(ref[car]['last200'])
  checked.append(dict(o[car],car_no=car,nk_profile_href=href[0],finish_text=rank,last_lap_raw=cs[5].text_content().strip()))
 assert len(checked)==7 and {x['car_no'] for x in checked}==set(range(1,8))
 return {'actual_tuple':{'day':'2026-06-03','venue_code':'74','race_no':7},'legacy_id_retained':RID,'raw_race_no_retained':107,'source_specific_mapping':'exact_legacy_row_to_observed_actual_provider_tuple','riders':sorted(checked,key=lambda x:x['car_no']),'NK_title':nk.findtext('.//title'),'OP_title':op.findtext('.//title'),'NK_source':nm,'OP_source':om,'general_100_subtraction_rule':False,'KD_encoding_cause_claimed':False,'historical_score_values_changed':False}

def main():
 OUT.mkdir(exist_ok=True)
 db=ROOT/'data/finals.sqlite';assert db.stat().st_size==281206784 and sha(db.read_bytes())==DBSHA;assert not any(pathlib.Path(str(db)+s).exists() for s in ['-wal','-shm','-journal'])
 with sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True) as c:
  assert c.execute('pragma integrity_check').fetchall()==[('ok',)] and c.execute('select count(*) from races').fetchone()[0]==5354
  raw=c.execute('select payload from races where race_id=?',(RID,)).fetchone()[0];assert sha(raw.encode())==PAYSHA;p=json.loads(raw)
 if (OUT/'raw.tar.xz').exists():
  with tarfile.open(OUT/'raw.tar.xz') as t:
   names=t.getnames();assert len(names)==len(set(names))==4;files={m.name:t.extractfile(m).read() for m in t.getmembers()}
 else:files={f.name:f.read_bytes() for f in pathlib.Path('raw_legacy_2113').iterdir() if f.is_file()}
 assert set(files)=={'nk_kochi_actual.html','nk_kochi_actual.json','op_kochi_card.html','op_kochi_card.json'}
 review=verify(files,p)
 # Mutation metadata is recomputed; all wrong semantic evidence must still fail.
 mutations=[]
 for key,kind in [('nk_kochi_actual','hidden_key'),('nk_kochi_actual','explicit_profile_ID'),('nk_kochi_actual','finish'),('op_kochi_card','race_number'),('op_kochi_card','explicit_profile_ID'),('nk_kochi_actual','title_date')]:
  d,m=body(files,key)
  if kind=='hidden_key':d.xpath('//input[@name="plp-raceid"]')[0].set('value','202606037408')
  elif kind=='explicit_profile_ID':
   a=d.xpath('//table[@id="All_Result_Table"]//a')[0] if key.startswith('nk') else d.xpath('//a[contains(@href,"playerCd=015882")]')[0];a.set('href',a.get('href').replace('15882','15883'))
  elif kind=='finish':d.xpath('//table[@id="All_Result_Table"]//tr[td]')[0].xpath('./td')[0].text='7着'
  elif kind=='race_number':d.xpath('//input[@id="raceNo"]')[0].set('value','8')
  else:d.find('.//title').text=d.findtext('.//title').replace('2026年06月03日','2026年06月04日')
  b=html.tostring(d,encoding='utf8');m.update(sha256=sha(b),bytes=len(b));changed=dict(files);changed[key+'.html']=b;changed[key+'.json']=json.dumps(m).encode()
  try:verify(changed,p)
  except AssertionError:mutations.append(key+':'+kind)
  else:raise AssertionError('bad mutation accepted '+kind)
 bad=copy.deepcopy(p);bad['entries'][0]['rider_registration_number']='015883'
 try:verify(files,bad)
 except AssertionError:mutations.append('legacy_explicit_registration_ID')
 else:raise AssertionError('bad legacy ID accepted')
 assert len(mutations)==7
 source=ROOT/'data/scope_review_1307/history_postponement_delta_review.json';assert sha(source.read_bytes())=='1b39580e5d88f9822426d8f4eeac7fa5aedaa359aed02996db2a01909a58ddec';assert 'https://keirin.netkeiba.com/db/result/?race_id='+NK+'&player_id=15882' in source.read_text()
 if not (OUT/'raw.tar.xz').exists():
  with tarfile.open(OUT/'raw.tar.xz','w:xz',preset=9) as t:
   for name,b in sorted(files.items()):
    i=tarfile.TarInfo(name);i.size=len(b);t.addfile(i,io.BytesIO(b))
 manifest={name:{'bytes':len(b),'sha256':sha(b)} for name,b in sorted(files.items())}
 with tarfile.open(OUT/'raw.tar.xz') as t:assert {m.name:{'bytes':m.size,'sha256':sha(t.extractfile(m).read())} for m in t.getmembers()}==manifest
 review.update(status='one_legacy_actual_number_mapping_independently_verified',raw_members=manifest,archive={'bytes':(OUT/'raw.tar.xz').stat().st_size,'sha256':sha((OUT/'raw.tar.xz').read_bytes())},legacy_payload_sha256=PAYSHA,mutations_rejected=mutations,canonical_DB_changed=False,SQL_rows_inserted=0,new_completed_finals=0,all_finals_complete=False,remaining_legacy_number_mappings=4,prior_1918_report_unchanged=True,note='Only this exact legacy row and observed actual tuple. No raw number/ID/stat rewrite, no bank-completion increase, no claim of complete SQL insertion adapter.')
 assert sha(db.read_bytes())==DBSHA
 (OUT/'kochi_actual_number_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2,sort_keys=True));print(json.dumps({k:v for k,v in review.items() if k not in ['riders','NK_source','OP_source','raw_members']},ensure_ascii=False))
if __name__=='__main__':main()
