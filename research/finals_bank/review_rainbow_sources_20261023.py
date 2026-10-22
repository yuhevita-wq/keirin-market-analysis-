"""Review published Rainbow sources without inserting unverified records."""
import pathlib,json,re,unicodedata,sqlite3,itertools,math,hashlib,os
from lxml import html
P=pathlib.Path(__file__).resolve().parent
ROOT=pathlib.Path(os.environ.get('FINALS_ROOT',P.parent/'current_data/unpacked/finals_ai')).resolve()
def norm(s):return re.sub(r'\s+','',unicodedata.normalize('NFKC',str(s)))
def read_json(n):return json.loads(n.read_text())
def get_files(ev):
 d,v=ev['day'],ev['venue_code'];k=d.replace('-','')+'_'+v
 if ev['race_no']!=9:k+='_r'+str(ev['race_no'])
 if d=='2023-12-05':return [P/'chariloto_api_card.html',P/'chariloto_api_odds.html',P/'chariloto.html']
 return [P/'other_rainbows'/(k+'_'+part+'.html') for part in ['card','odds']]+[P/'other_rainbows'/(d.replace('-','')+'_'+v+'_result.html')]
def market_check(data,cars):
 out={}
 assert data['has_error'] is False
 for bet,ordered in [('sanrentan',True),('sanrenfuku',False)]:
  rows=[x for x in data['odds'] if x['bet_type']==bet]
  keys=[tuple(map(int,re.findall(r'\d+',x['deme']))) for x in rows]
  expected=set(itertools.permutations(cars,3) if ordered else itertools.combinations(cars,3))
  assert len(keys)==len(set(keys)) and set(keys)==expected
  assert all(x['is_valid'] is True and x['on_final'] is True and math.isfinite(float(x['rate'])) and float(x['rate'])>0 and float(x['rate'])==float(x['normalize_rate']) for x in rows)
  out[bet]={'count':len(keys),'exact_set':True,'duplicates':0,'positive_finite':True,'all_final':True}
 return out
registry=read_json(ROOT/'data/registry_official_sources.json')
c=sqlite3.connect(f'file:{(ROOT/"data/finals.sqlite").resolve()}?mode=ro',uri=True)
assert c.execute('pragma integrity_check').fetchone()[0]=='ok'
events=read_json(P/'rainbow_event_candidates.json');out=[]
for ev in events:
 d,v=ev['day'],ev['venue_code'];rn=ev['race_no'];cf,of,rf=get_files(ev)
 card,odds=read_json(cf),read_json(of);box=card['box'];assert (box['open_day'],box['vel_code'],box['race_num'])==(d,v,rn)
 assert box['is_girls'] is False and box['is_end_day'] is True and box['on_abort'] is False and box['is_invalid'] is False
 program=card['race_program'];assert (program['open_day'],program['vel_code'],program['race_num'])==(d,v,rn)
 expectedlabel='A級F' if ev['category']=='A12' else 'チ・F'
 assert norm(program['event_name'])==expectedlabel,program['event_name']
 athletes=card['race_athletes'];cars=sorted(int(x) for x in athletes);assert cars==list(range(1,10))
 for a in athletes.values():assert (a['open_day'],a['vel_code'],a['race_num'])==(d,v,rn)
 doc=html.fromstring(rf.read_bytes());sections=doc.xpath('//*[@id="race'+str(rn)+'"]');assert len(sections)==1
 sec=sections[0];tables=sec.xpath('.//table[.//th[normalize-space()="選手名"] and .//th[normalize-space()="着"]]');assert len(tables)==1
 results=[]
 for row in tables[0].xpath('./tbody/tr'):
  cells=[norm(x.text_content()) for x in row.xpath('./td')];assert len(cells)==12
  result={'finish_text':cells[0],'car_no':int(cells[1]),'name':cells[2],'age':int(cells[3]),'pref_short':cells[4],'term':int(cells[5]),'class':cells[6],'margin':cells[7],'last_lap':cells[8],'winning_move':cells[9],'SB':cells[10]}
  a=athletes[str(result['car_no'])]
  assert norm(a['name'])==result['name'] and a['age']==result['age'] and int(a['graduate_period'])==result['term'] and norm(a['class_name']+a['group_name'])==result['class']
  assert norm(a['pref_name']).startswith(result['pref_short']) or result['pref_short']==norm(a['pref_name'])
  results.append(result)
 assert sorted(x['car_no'] for x in results)==cars and len(results)==9
 assert all(re.fullmatch(r'[1-9]|失|落|棄|欠',x['finish_text']) for x in results)
 assert results[0]['finish_text']=='1'
 groups=[];group=[]
 line=sec.xpath('.//table[.//th[normalize-space()="周回予想"]]')
 assert len(line)==1
 for cell in line[0].xpath('.//table//tr/td'):
  n=cell.xpath('.//span[contains(concat(" ",@class," ")," square ")]')
  if n:group.append(int(norm(n[0].text_content())))
  elif group:groups.append(group);group=[]
 if group:groups.append(group)
 assert sorted(itertools.chain.from_iterable(groups))==cars
 identities=[];history_anomalies=[]
 for car,a in athletes.items():
  rid=str(a['un_id']).zfill(6);matches=[s for s in registry if s['registration_number']==rid and norm(s['name'])==norm(a['name']) and str(s['term'])==str(a['graduate_period']) and norm(s['prefecture'])=={'神奈':'神奈川','和歌':'和歌山','北海':'北海道'}.get(norm(a['pref_name']),norm(a['pref_name']))]
  identities.append({'car_no':int(car),'registration_number':rid,'name':a['name'],'term':a['graduate_period'],'prefecture_raw':a['pref_name'],'prefecture':{'神奈':'神奈川','和歌':'和歌山','北海':'北海道'}.get(norm(a['pref_name']),norm(a['pref_name'])),'prefecture_alias_policy':'explicit three unambiguous publication abbreviations; registration ID/name/term must still match','status':'verified_explicit_provider_ID_name_term_prefecture' if matches else 'unresolved','official_sources':matches})
  for hk in ['history_data1','history_data2','history_data3']:
   h=a.get(hk,{}) or {};date=h.get('open_from','')
   if date and int(date[:4])<int(d[:4])-10:history_anomalies.append({'car_no':int(car),'field':hk+'.open_from','raw_value':date,'reason':'decades_before_race_date; excluded from usable history'})
 m=market_check(odds,cars)
 label=norm(sec.xpath('./div/div/p')[0].text_content())
 matches=c.execute('select race_id,label from races where day=? and substr(race_id,1,2)=? and cast(substr(race_id,-4) as integer)=?',(d,v,rn)).fetchall()
 excluded=c.execute('select race_id,label from excluded_races where day=? and substr(race_id,1,2)=? and cast(substr(race_id,-4) as integer)=?',(d,v,rn)).fetchall()
 canc=c.execute('select event_card_url,label from cancellations where day=? and race_no=?',(d,rn)).fetchall();canc=[x for x in canc if re.search(r'/'+v+r'\d{12}/',x[0])]
 assert not matches and not excluded and not canc
 existing_s=c.execute('select race_id from races where day=? and substr(race_id,1,2)=? and cast(substr(race_id,-4) as integer)=12',(d,v)).fetchall();assert len(existing_s)==1
 candidate=existing_s[0][0][:-4]+str(rn).zfill(4)
 rec={**ev,'status':'confirmed_missing_final_full_independent_collection_pending','candidate_kdreams_race_id':candidate,'candidate_id_confirmation':'constructed from independently matching same-event existing S final prefix and published actual published race number; actual KDreams target-race href not yet observed','source_race_key':{'provider':'chariloto','day':d,'venue_code':v,'race_no':rn,'race_info_id':program['id']},'actual_race_held':True,'provider_label':label,'published_program':program,'source_files':[{ 'path':str(f.relative_to(P)), 'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size} for f in [cf,of,rf]],'results':results,'published_line_groups':groups,'markets':m,'entry_identities':identities,'identity_verified':sum(x['status']!='unresolved' for x in identities),'history_anomalies':history_anomalies,'unresolved':['actual KDreams target-race ID/link or independent canonical race-key policy','independent historical entry/statistic verification; malformed prior-race dates are not usable','independent full odds cross-provider comparison','historical bank applicability for this candidate not yet independently applied'],'canonical_db_inserted':False,'all_finals_complete':False}
 out.append(rec)
summary={'all_finals_complete':False,'existing_male_collected':5353,'existing_ledger_rows':5354,'missing_rainbow_finals_confirmed':len(out),'male_scope_lower_bound':5353+len(out),'candidate_entries':sum(len(x['entry_identities']) for x in out),'explicit_IDs_officially_matched':sum(x['identity_verified'] for x in out),'historical_history_anomalies':sum(len(x['history_anomalies']) for x in out),'full_final_market_sets_verified':sum(len(x['markets']) for x in out),'canonical_db_changed':False,'events':out,'scope_note':'14 Rainbow finals independently compared against existing races/excluded_races/cancellations. This is a lower bound; other labels and single races still need scope audit. Raw malformed history dates preserved and not silently repaired.'}
(P/'rainbow_scope_independent_review.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='events'},ensure_ascii=False,indent=2))
