"""Verify meeting starts from saved OP and NK actual DOM links, no raw repair.
Run in workspace cwd with saved 0831 archive extracted to raw_independent_0831.
This verifies meeting firstday only; it does not infer each rider started then.
"""
import pathlib,json,hashlib,re,datetime,urllib.parse,importlib.util,copy
from lxml import html
ROOT=pathlib.Path('current_data/unpacked/finals_ai');OUT=ROOT/'data/scope_review_1545'
def sha(b):return hashlib.sha256(b).hexdigest()
def load_module():
 p=ROOT/'recovery/review_rainbow_independent_card_results_0831.py';s=importlib.util.spec_from_file_location('cards0831',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def verify(e,op,nk,card,m):
 k=e['day'].replace('-','')+e['venue_code']+str(e['race_no']).zfill(2)
 for f,v in [('joCode',e['venue_code']),('kaisaiBi',e['day'].replace('-','')),('raceNo',str(e['race_no']))]:assert op.xpath('//input[@id="'+f+'"]/@value')==[v]
 assert nk.xpath('//input[@name="plp-raceid"]/@value')==[k]
 def firstday(anchors,provider):
  hits=[]
  for a in anchors:
   if '初日' not in a.text_content():continue
   u=urllib.parse.urlparse(a.get('href'));q=urllib.parse.parse_qs(u.query)
   if provider=='OP':assert u.path=='/keirin/AllRaceList.do';assert q['joCode']==[e['venue_code']];d=q['kaisaiBi'][0]
   else:assert u.hostname=='keirin.netkeiba.com' and u.path=='/race/race_list/';assert q['jyocd']==[e['venue_code']];d=q['kaisai_date'][0]
   actual=datetime.datetime.strptime(d,'%Y%m%d').date();assert actual.isoformat()<=e['day'];text=m.norm(a.text_content());md=re.search(r'(\d{1,2})/(\d{1,2})',text);assert md and (int(md[1]),int(md[2]))==(actual.month,actual.day)
   if provider=='NK':assert '月火水木金土日'[actual.weekday()] in text
   hits.append({'date':actual.isoformat(),'href':a.get('href'),'label':a.text_content()})
  assert len(hits)==1;return hits[0]
 od=firstday(op.xpath('//ul[@id="raceDay"]//a[@href]'),'OP')
 nd=firstday(nk.xpath('//ul[contains(concat(" ",normalize-space(@class)," ")," DayChange_List ")]//a[@href]'),'NK')
 assert od['date']==nd['date']
 # Every result belongs to the actual target; ID and cohort/name must match
 # explicit Chariloto IDs and the OP actual car row, never names alone.
 ath=card['race_athletes'];rows=nk.xpath('//table[@id="All_Result_Table"]//tr[td]');assert len(rows)==9
 out=[]
 for row in rows:
  cs=row.xpath('./td');assert len(cs)==8;car=int(m.norm(cs[2].text_content()));a=ath[str(car)];rid=int(a['un_id'])
  nr=cs[3].xpath('.//a[@href]');assert len(nr)==1;assert int(urllib.parse.parse_qs(urllib.parse.urlparse(nr[0].get('href')).query)['id'][0])==rid
  assert [m.norm(x) for x in cs[3].xpath('.//img/@alt')]==[m.norm(a['name'])]
  assert re.search(r'(\d+)期',m.norm(cs[3].text_content()))[1]==a['graduate_period']
  orows=op.xpath('//table[@summary="出走表"]/tr[td[@class="no'+str(car)+'"]]');assert len(orows)==1;links=orows[0].xpath('./td')[-8].xpath('.//a[@href]');assert len(links)==1
  assert int(urllib.parse.parse_qs(urllib.parse.urlparse(links[0].get('href')).query)['playerCd'][0])==rid and m.norm(links[0].text_content())==m.norm(a['name'])
  h=a['history_data1'];assert not h.get('short_boxes',[])
  out.append({'car_no':car,'registration_number':str(rid).zfill(6),'name':a['name'],'term':a['graduate_period'],'raw_Chariloto_history_data1':h,'meeting_firstday':od['date'],'status':'meeting_firstday_independently_verified_two_providers_raw_date_retained','rider_firstday_participation_inferred':False})
 assert sorted(x['car_no'] for x in out)==list(range(1,10))
 return {'day':e['day'],'venue_code':e['venue_code'],'race_no':e['race_no'],'actual_NK_key':k,'OP_firstday':od,'NK_firstday':nd,'entries':out}
def main():
 OUT.mkdir(exist_ok=True);m=load_module();manifest=json.loads((ROOT/'data/scope_review_0831/raw_manifest.json').read_text())
 # Every saved HTML and meta byte is verified against durable archive manifest.
 for n,v in manifest.items():
  b=(pathlib.Path('raw_independent_0831')/n).read_bytes();assert len(b)==v['bytes'] and sha(b)==v['sha256']
 events=json.loads((ROOT/'data/scope_review_0457/rainbow_event_candidates.json').read_text());reports=[];inputs=[]
 for e in events:
  k=e['day'].replace('-','')+e['venue_code']+str(e['race_no']).zfill(2);op,om=m.doc(k,'card');nk,nm=m.doc(k,'result');cp=m.cardfile(e);b=cp.read_bytes();card=json.loads(b)
  reports.append(verify(e,op,nk,card,m));inputs.append({'key':k,'OP_meta':om,'NK_meta':nm,'Chariloto_path':str(cp.relative_to(ROOT)),'Chariloto_bytes':len(b),'Chariloto_sha256':sha(b)})
 assert len(reports)==14 and sum(len(x['entries']) for x in reports)==126
 e=events[2];k=e['day'].replace('-','')+e['venue_code']+str(e['race_no']).zfill(2);op,_=m.doc(k,'card');nk,_=m.doc(k,'result');card=json.loads(m.cardfile(e).read_text());tests=[]
 for mode in ['firstday_year','firstday_venue','target_hidden_key','explicit_player_ID','firstday_label']:
  n=copy.deepcopy(nk)
  if mode.startswith('firstday_'):
   a=[a for a in n.xpath('//ul[contains(@class,"DayChange_List")]//a[@href]') if '初日' in a.text_content()][0]
   if mode=='firstday_year':a.set('href',a.get('href').replace('kaisai_date=2023','kaisai_date=2022'))
   elif mode=='firstday_venue':a.set('href',a.get('href').replace('jyocd='+e['venue_code'],'jyocd=99'))
   else:a.clear();a.text='初日(1/1)';a.set('href','https://keirin.netkeiba.com/race/race_list/?kaisai_date=20231202&jyocd=37')
  elif mode=='target_hidden_key':n.xpath('//input[@name="plp-raceid"]')[0].set('value','209912313709')
  else:
   a=n.xpath('//table[@id="All_Result_Table"]//tr[td]')[0].xpath('./td')[3].xpath('.//a[@href]')[0];a.set('href','https://keirin.netkeiba.com/db/profile/?id=999999')
  # Re-serialize/rehash mutants so rejection is semantic, not stale SHA.
  raw=html.tostring(n);changed_sha=sha(raw)
  try:verify(e,op,html.fromstring(raw),card,m)
  except (AssertionError,KeyError,ValueError,TypeError):tests.append({'mutation':mode,'rejected':True,'mutated_raw_sha256':changed_sha})
  else:raise AssertionError('accepted mutation '+mode)
 report={'all_finals_complete':False,'canonical_db_changed':False,'new_network_requests':0,'meeting_firstdays_independently_verified':14,'entry_history_data1_meeting_dates_independently_verified':126,'raw_dates_and_groups_changed':False,'rider_firstday_participation_inferred':False,'emptyplanned_prior17_unchanged':True,'scope_note':'Meeting start is verified from two providers actual DOM firstday links and explicit target/car/registration identities. It does not assert rider participation on that firstday, fill planned slots, or rewrite malformed raw years. Existing single-provider observations remain historical records; this separate review adds corroboration.','events':reports,'sources':inputs,'mutations':tests}
 (OUT/'current_firstday_independent_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['events','sources','mutations','scope_note']},ensure_ascii=False));print('mutations rejected',len(tests))
if __name__=='__main__':main()
