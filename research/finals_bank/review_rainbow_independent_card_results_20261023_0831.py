"""Independent historical card and result checks. Does not alter canonical DB."""
import pathlib,json,re,unicodedata,decimal,hashlib,urllib.parse
from lxml import html
ROOT=pathlib.Path('current_data/unpacked/finals_ai');BASE=ROOT/'data/scope_review_0457';RAW=pathlib.Path('raw_independent_0831')
def norm(s):return re.sub(r'\s+','',unicodedata.normalize('NFKC',str(s)))
def cardfile(e):
 k=e['day'].replace('-','')+'_'+e['venue_code']
 if e['race_no']!=9:k+='_r'+str(e['race_no'])
 return BASE/'chariloto_api_card.html' if e['day']=='2023-12-05' else BASE/'other_rainbows'/(k+'_card.html')
def doc(k,part):
 p=RAW/(k+'_'+part+'.html');m=json.loads(p.with_suffix('.meta.json').read_text());b=p.read_bytes();assert m['sha256']==hashlib.sha256(b).hexdigest() and m['bytes']==len(b);assert m['status']==200
 return html.fromstring(b),m
aliases={'神奈':'神奈川','和歌':'和歌山','北海':'北海道'}
def pref(s):return aliases.get(norm(s),norm(s))
def audit(e):
 k=e['day'].replace('-','')+e['venue_code']+str(e['race_no']).zfill(2);ath=json.loads(cardfile(e).read_text())['race_athletes'];op,om=doc(k,'card');nk,nm=doc(k,'result');checked=[]
 for field,want in [('joCode',e['venue_code']),('kaisaiBi',e['day'].replace('-','')),('raceNo',str(e['race_no']))]:assert op.xpath('//input[@id="'+field+'"]/@value')==[want]
 assert nk.xpath('//input[@name="plp-raceid"]/@value')==[k]
 rows=[row for row in op.xpath('//table[@summary="出走表"]/tr[td]') if any(re.fullmatch(r'no[1-9]',c.get('class','')) for c in row.xpath('./td'))];assert len(rows)==9
 cars=[]
 for row in rows:
  cs=row.xpath('./td');no=[c for c in cs if re.fullmatch(r'no[1-9]',c.get('class',''))];assert len(no)==1;car=int(norm(no[0].text_content()));cars.append(car);a=ath[str(car)]
  href=cs[-8].xpath('.//a/@href');assert len(href)==1;rid=urllib.parse.parse_qs(urllib.parse.urlparse(href[0]).query)['playerCd'][0];assert int(rid)==int(a['un_id']);name=norm(cs[-8].xpath('.//a')[0].text_content());assert name==norm(a['name'])
  at=re.search(r'(\d+)歳/(\d+)期',norm(cs[-8].text_content()));assert at and (int(at[1]),int(at[2]))==(a['age'],int(a['graduate_period']))
  assert pref(cs[-7].text_content())==pref(a['pref_name']);assert norm(cs[-6].text_content())==norm(a['class_name']+a['group_name'])
  gearstyle=unicodedata.normalize('NFKC',' '.join(cs[-5].itertext()));gears=re.findall(r'\d+\.\d+',gearstyle);assert a['gear_ratio'] in gears and norm(a['leg_quality']) in norm(gearstyle)
  stats=norm(cs[-4].text_content());score=re.search(r'競走得点:([0-9.]+)',stats);finish=re.search(r'着順:(\d+)-(\d+)-(\d+)-(\d+)',stats);moves=re.search(r'決まり手:(\d+)-(\d+)-(\d+)-(\d+)',stats)
  assert score and finish and moves,stats
  assert decimal.Decimal(score[1])==decimal.Decimal(a['average_point'])/1000
  assert list(map(int,finish.groups()))==[a[f] for f in ['first_count','second_count','third_count','out_rank_count']]
  assert list(map(int,moves.groups()))==[a[f] for f in ['nige_count','makuri_count','sashi_count','mark_count']]
  checked.append({'car_no':car,'registration_number':rid,'name':name,'age':a['age'],'term':a['graduate_period'],'score_exact':score[1],'finish_counts':list(map(int,finish.groups())),'winning_move_counts':list(map(int,moves.groups())),'gear_raw':gearstyle,'identity_score_counts_gear_style_verified':True})
 assert sorted(cars)==list(range(1,10))
 results=[]
 reference={x['car_no']:x for x in e['results']}
 for row in nk.xpath('//table[@id="All_Result_Table"]//tr[td]'):
  cs=row.xpath('./td');assert len(cs)==8;car=int(norm(cs[2].text_content()));a=ath[str(car)];rank=norm(cs[0].text_content());rank=re.sub(r'着$','',rank);assert re.fullmatch(r'[1-9]|失|落|棄|欠',rank),rank
  assert rank==reference[car]['finish_text'],(car,rank,reference[car])
  href=cs[3].xpath('.//a/@href');assert len(href)==1;rid=urllib.parse.parse_qs(urllib.parse.urlparse(href[0]).query)['id'][0];assert int(rid)==int(a['un_id'])
  assert [norm(v) for v in cs[3].xpath('.//img/@alt')]==[norm(a['name'])]
  info=norm(cs[3].text_content());assert norm(a['name']) in info;age=re.search(r'(\d+)歳',info);cl=re.search(r'(\d+)期([ASL][123])',info);assert age and cl;assert (int(age[1]),int(cl[1]),cl[2])==(a['age'],int(a['graduate_period']),norm(a['class_name']+a['group_name']))
  results.append({'car_no':car,'registration_number':rid,'finish_text':rank,'margin_raw':cs[4].text_content().strip(),'last_lap_raw':cs[5].text_content().strip(),'move_raw':cs[6].text_content().strip(),'SB_raw':cs[7].text_content().strip()})
 assert sorted(x['car_no'] for x in results)==list(range(1,10)) and len(results)==9
 return {'day':e['day'],'venue_code':e['venue_code'],'race_no':e['race_no'],'race_id_netkeiba':k,'card_source':om,'result_source':nm,'historical_entries_verified':checked,'results_verified':results,'result_car_rank_exact_equal_chariloto':True,'canonical_db_inserted':False}
if __name__=='__main__':
 events=json.loads((BASE/'rainbow_scope_independent_review.json').read_text())['events'];out=[];err=[]
 for e in events:
  try:out.append(audit(e))
  except Exception as x:err.append({'day':e['day'],'venue':e['venue_code'],'error':repr(x)})
 report={'all_finals_complete':False,'canonical_db_changed':False,'events_verified':len(out),'entry_stats_verified':sum(len(e['historical_entries_verified']) for e in out),'result_entries_verified':sum(len(e['results_verified']) for e in out),'events':out,'errors':err,'unresolved':['malformed Chariloto historical dates and prior race histories','line independent verification','canonical key policy and SQL insertion audit','historical bank applicability','full period final-equivalent label scope']}
 pathlib.Path('review_0831.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['events','unresolved']},ensure_ascii=False))
