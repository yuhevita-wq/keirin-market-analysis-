"""Extract explicit historical dates; independently compare saved OP prior slots.
No date offsets, no DB insertion, and no forced equivalence of abnormal results.
Run from workspace cwd after extracting saved OP card archive.
"""
import pathlib,json,re,unicodedata,hashlib,urllib.parse,datetime,collections
from lxml import html
ROOT=pathlib.Path('current_data/unpacked/finals_ai');BASE=ROOT/'data/scope_review_0457';NK=pathlib.Path('raw_history_0914');OP=pathlib.Path('raw_independent_0831')
def norm(s):return re.sub(r'\s+','',unicodedata.normalize('NFKC',str(s)))
def date(s):return datetime.datetime.strptime(s,'%Y%m%d').date().isoformat()
def group_rows(row,rid,day):
 groups=[];g=None
 for cell in row:
  if not isinstance(cell.tag,str):continue
  if cell.tag=='th':
   links=cell.xpath('.//a[contains(@href,"kaisai_group_id=")]/@href')
   if not links:continue
   assert len(links)==1;key=urllib.parse.parse_qs(urllib.parse.urlparse(links[0]).query)['kaisai_group_id'][0];assert re.fullmatch(r'\d{10}',key);start=date(key[:8]);assert start<day
   names=cell.xpath('.//span[contains(concat(" ",normalize-space(@class)," ")," JyoName ")]/text()');assert len(names)==1
   g={'group_id':key,'start_date':start,'venue_code':key[8:10],'venue_name':names[0],'group_href':links[0],'results':[]};groups.append(g)
  elif cell.tag=='td' and g is not None:
   anchors=cell.xpath('.//a[contains(@href,"/db/result/")]')
   for a in anchors:
    qs=urllib.parse.parse_qs(urllib.parse.urlparse(a.get('href')).query);k=qs['race_id'][0];assert re.fullmatch(r'\d{12}',k) and int(qs['player_id'][0])==rid;actual=date(k[:8]);assert g['start_date']<=actual<day and k[8:10]==g['venue_code']
    ranks=a.xpath('.//span[contains(concat(" ",normalize-space(@class)," ")," result_no ")]');assert len(ranks)==1;rank=norm(ranks[0].text_content());assert re.fullmatch(r'[1-9]|失|落|棄|欠|',rank),rank
    g['results'].append({'race_id':k,'date':actual,'race_no':int(k[10:]),'finish_text':rank,'href':a.get('href')})
 assert len({g['group_id'] for g in groups})==len(groups)
 return groups

def op_slot(cell):
 text=unicodedata.normalize('NFKC',cell.text_content());dates=list(re.finditer(r'(?<!\d)(\d{1,2})/\s*(\d{1,2})',text));out=[]
 for i,m in enumerate(dates):
  tail=text[m.end():dates[i+1].start() if i+1<len(dates) else len(text)];rank=re.search(r'([1-9])\s*着',tail)
  if rank:finish=rank[1]
  else:
   abnormal=re.findall(r'[失落棄欠]',tail);assert len(abnormal)==1,(text,tail);finish=abnormal[0]
  out.append({'month':int(m[1]),'day':int(m[2]),'finish_text':finish,'raw':m[0]+tail})
 return {'raw':text,'venue_header':norm(text[:dates[0].start()]) if dates else norm(text),'results':out}

def postponement_notice():
 p=NK/'kochi_postponement.html';m=json.loads(p.with_suffix('.meta.json').read_text());b=p.read_bytes();assert len(b)==m['bytes'] and hashlib.sha256(b).hexdigest()==m['sha256'];t=norm(html.fromstring(b).text_content())
 assert '2026/06/02' in t and '6月2日(火)高知競輪3日目は台風のため開催中止、3日(水)に順延となりました。' in t
 return m

def match_slot(slot,groups):
 if not slot['results']:return {'status':'unresolved_empty_OP_slot','raw':slot['raw']}
 md=[(r['month'],r['day']) for r in slot['results']];matches=[]
 for original in groups:
  g=dict(original);g['results']=list(original['results'])
  if original['group_id']=='2026053174' and any(r['race_id']=='202606027407' and r['finish_text']=='' for r in original['results']):
   notice=postponement_notice();g['excluded_non_result_rows']=[r for r in original['results'] if r['race_id']=='202606027407' and r['finish_text']==''];g['postponement_evidence']=notice;g['results']=[r for r in original['results'] if r not in g['excluded_non_result_rows']]
  actual=[(int(r['date'][5:7]),int(r['date'][8:10])) for r in g['results']]
  if actual==md and slot['venue_header'].startswith(norm(g['venue_name'])):matches.append(g)
 if len(matches)!=1:return {'status':'unresolved_date_venue_sequence','OP_slot':slot,'candidate_groups':groups[:3]}
 g=matches[0];different=[]
 for op,nk in zip(slot['results'],g['results']):
  if op['finish_text']!=nk['finish_text']:different.append({'date':nk['date'],'OP':op['finish_text'],'NK':nk['finish_text']})
 return {'status':'independently_verified_dates_and_results' if not different else 'dates_matched_result_text_discrepancy_unresolved','OP_slot':slot,'NK_group':g,'result_differences':different}

def cardfile(e):
 k=e['day'].replace('-','')+'_'+e['venue_code']
 if e['race_no']!=9:k+='_r'+str(e['race_no'])
 return BASE/'chariloto_api_card.html' if e['day']=='2023-12-05' else BASE/'other_rainbows'/(k+'_card.html')

def review():
 events=json.loads((BASE/'rainbow_event_candidates.json').read_text());out=[];errors=[]
 for e in events:
  k=e['day'].replace('-','')+e['venue_code']+str(e['race_no']).zfill(2);p=NK/(k+'.html');m=json.loads(p.with_suffix('.meta.json').read_text());b=p.read_bytes();assert len(b)==m['bytes'] and hashlib.sha256(b).hexdigest()==m['sha256'];doc=html.fromstring(b)
  title=norm(doc.xpath('//title/text()')[0]);assert e['day'].replace('-','年',1).replace('-','月')+'日' in title or (str(int(e['day'][:4]))+'年'+e['day'][5:7]+'月'+e['day'][8:]+'日') in title
  opdoc=html.fromstring((OP/(k+'_card.html')).read_bytes());card=json.loads(cardfile(e).read_text());ath=card['race_athletes'];rows=doc.xpath('//tr[starts-with(@id,"player-")]');assert len(rows)==9
  entries=[]
  starts=opdoc.xpath('//a[contains(@href,"AllRaceList.do") and contains(.,"初日")]/@href');assert len(starts)==1;qs=urllib.parse.parse_qs(urllib.parse.urlparse(starts[0]).query);assert qs['joCode']==[e['venue_code']];startdate=date(qs['kaisaiBi'][0]);assert startdate<=e['day']
  for row in rows:
   car=int(row.get('id').split('-')[1]);a=ath[str(car)];rid=int(a['un_id']);assert row.xpath('.//a[@id="name_'+str(rid)+'_'+k+'"]');assert [norm(s) for s in row.xpath('.//img[starts-with(@id,"entry_player_photo_")]/@alt')]==[norm(a['name'])]
   groups=group_rows(row,rid,e['day']);oprow=opdoc.xpath('//table[@summary="出走表"]/tr[td[@class="no'+str(car)+'"]]');assert len(oprow)==1;cs=oprow[0].xpath('./td');prior=[match_slot(op_slot(c),groups) for c in cs[-2:]]
   slots=[]
   for field in ['history_data2','history_data3']:
    h=a[field];raw=h.get('open_from','');boxes=h.get('short_boxes',[])
    if not boxes:slots.append({'field':field,'raw_date':raw,'status':'empty_planned_slot_date_unresolved','raw_history':h});continue
    candidates=[g for g in groups if g['venue_code']==h['vel_code'] and g['start_date'][5:].replace('-','')==raw[-4:] and [r['race_no'] for r in g['results']]==[x['race_num'] for x in boxes] and len(g['results'])==len(boxes)]
    matching=[]
    for g in candidates:
     if all((str(x['rank'])==n['finish_text'] if int(x['rank'])>0 else norm(x.get('note',''))==n['finish_text']) for x,n in zip(boxes,g['results'])):matching.append(g)
    verified=[g for g in matching if any(s['status']=='independently_verified_dates_and_results' and s['NK_group']['group_id']==g['group_id'] for s in prior)]
    slots.append({'field':field,'raw_date':raw,'raw_history':h,'status':'independently_verified_explicit_full_date' if len(verified)==1 else 'unresolved_slot_mapping_or_abnormal_text','independent_group':verified[0] if len(verified)==1 else None,'candidate_full_dates':[g['start_date'] for g in matching]})
   entries.append({'car_no':car,'registration_number':str(rid).zfill(6),'name':a['name'],'OP_actual_current_firstday':{'date':startdate,'href':starts[0],'raw_Chariloto_date':a['history_data1']['open_from'],'status':'one_independent_source_explicit_date_not_silent_repair'},'prior_OP_slots':prior,'Chariloto_prior_slots':slots,'NK_all_raw_groups':groups})
  out.append({**e,'source_meta':m,'entries':entries})
 report={'all_finals_complete':False,'canonical_db_changed':False,'events':out,'errors':errors,'entry_identities_verified':sum(len(e['entries']) for e in out),'independent_prior_slots_status':dict(collections.Counter(s['status'] for e in out for a in e['entries'] for s in a['prior_OP_slots'])),'Chariloto_slot_date_status':dict(collections.Counter(s['status'] for e in out for a in e['entries'] for s in a['Chariloto_prior_slots'])),'scope_note':'Explicit historic dates only. Raw malformed dates retained. Abnormal rank tokens, blank slots, and postponed-day candidates are not silently repaired. Single-source current firstday observations are not counted as fully independent verification.'}
 pathlib.Path('history_review_0914.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['events','scope_note']},ensure_ascii=False))
if __name__=='__main__':review()
