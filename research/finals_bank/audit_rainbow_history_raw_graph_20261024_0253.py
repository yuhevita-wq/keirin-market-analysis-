"""Independent verification of raw graph nodes against pinned overlay and DOM."""
from pathlib import Path
import copy,hashlib,io,json,tarfile,unicodedata
from collections import Counter
from lxml import html
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/scope_review_0253'
def sha(b):return hashlib.sha256(b).hexdigest()
def payload():
 b=(ROOT/'data/scope_review_0106/history_adapter_overlay.tar.xz').read_bytes();assert sha(b)=='c81b9260ef8e3fb8f5d2186fc6477e45fdd953cec5c1a4118d69513628290248'
 with tarfile.open(fileobj=io.BytesIO(b)) as t:p=t.extractfile('history_adapter_overlay.json').read()
 assert sha(p)=='53a19c888f8063062d7541ee8879e95796d2999c6c57d5e597260a0c8d765522';return json.loads(p)
def audit(g):
 base=payload();assert g['all_finals_complete'] is False and g['canonical_SQL_inserts']==g['new_completed_races']==g['network_requests']==0
 assert g['base_history_overlay']['member_sha256']=='53a19c888f8063062d7541ee8879e95796d2999c6c57d5e597260a0c8d765522'
 raw={};arcs={}
 for h,ref in g['source_pool'].items():
  assert h==ref['sha256']
  if 'archive' in ref:
   a=ref['archive'];name=a['path']
   if name not in arcs:
    blob=(ROOT/name).read_bytes();assert len(blob)==a['bytes'] and sha(blob)==a['sha256']
    with tarfile.open(fileobj=io.BytesIO(blob)) as t:
     assert len(t.getnames())==len(set(t.getnames()));arcs[name]={n:t.extractfile(n).read() for n in t.getnames()}
   b=arcs[name][ref['member']]
  else:b=(ROOT/ref['path']).read_bytes()
  assert len(b)==ref['bytes'] and sha(b)==h;raw[h]=b
 assert len(raw)==55 and len(g['events'])==14
 kinds=Counter();priorrows=0;opc=0;details=0;alt=0
 for ei,(event,original) in enumerate(zip(g['events'],base['events'])):
  assert event['local_key']==original['local_key'] and len(event['entries'])==9
  hist=html.fromstring(raw[original['history_source']['sha256']].decode('utf-8'));op=html.fromstring(raw[original['independent_card_source']['sha256']].decode('utf-8'));res=html.fromstring(raw[original['firstday_NK_result_source']['sha256']].decode('utf-8'))
  for ai,(entry,old) in enumerate(zip(event['entries'],original['entries'])):
   car=old['car_no'];reg=old['registration_number'];assert entry['car_no']==car and entry['registration_number']==reg and entry['NK_row_id']=='player-'+str(car)
   row=hist.xpath('//tr[@id=$id]',id=entry['NK_row_id']);assert len(row)==1;row=row[0]
   assert len(row.xpath('.//a[starts-with(@id,$prefix)]',prefix='name_'+str(int(reg))+'_'))==1
   player=op.xpath('//a[contains(@href,$reg)]',reg='playerCd='+reg);assert len(player)==1;r=player[0]
   while r.tag!='tr':r=r.getparent()
   cells=r.xpath('./td');assert len(entry['OP_original_cells'])==2
   for j,p in enumerate(entry['OP_original_cells']):
    oldslot=old['original_OP_prior_slots'][j];txt=oldslot['OP_slot']['raw'] if 'OP_slot' in oldslot else oldslot['raw']
    assert p=={'cell_from_end':j-2,'raw_text_sha256':sha(txt.encode()),'original_status':oldslot['status']}
    assert unicodedata.normalize('NFKC',cells[j-2].text_content())==txt;opc+=1
   assert len(entry['slots'])==3
   for si,(node,slot) in enumerate(zip(entry['slots'],old['adapted_history_slots'])):
    assert node['overlay_pointer']==['events',ei,'entries',ai,'adapted_history_slots',si]
    assert all(node[k]==slot[k] for k in ['slot','kind','actual_start_date'])
    assert node['original_history_sha256']==sha(json.dumps(slot['original'],ensure_ascii=False,sort_keys=True,separators=(',',':')).encode())
    assert node['history_raw_sha256']==original['history_source']['sha256'] and node['OP_card_raw_sha256']==original['independent_card_source']['sha256']
    kinds[node['kind']]+=1
    if slot['kind']=='actual_prior_meeting':
     group=slot['independent_group'];dom=node['NK_group_DOM'];assert dom['group_href']==group['group_href']
     a=row.xpath('./th/a[@href=$href]',href=dom['group_href']);assert len(a)==1
     actual=[];following=a[0].getparent().getnext()
     while following is not None and following.tag=='td':
      links=following.xpath('./a[contains(@href,"/db/result/")]')
      for link in links:
       spans=link.xpath('.//span[contains(@class,"result_no")]');assert len(spans)==1
       actual.append({'href':link.get('href'),'finish_text':spans[0].text_content().strip()})
      following=following.getnext()
     assert dom['original_raw_result_rows']==actual
     expected=group['results']+group.get('excluded_non_result_rows',[])
     assert {a['href']:a['finish_text'] for a in actual}=={a['href']:a['finish_text'] for a in expected} and len(actual)==len(expected)
     assert dom['excluded_non_result_rows_retained']==group.get('excluded_non_result_rows',[]);priorrows+=len(actual)
    elif slot['kind']=='planned_slot_date_unresolved':
     assert node['actual_start_date'] is None and node['planned_date_independent_source'] is None and not slot['original']['short_boxes']
    else:
     assert node['kind']=='current_meeting_context_not_participation' and node['firstday_links']==slot['proof'] and node['actual_start_date'] is None
     assert node['NK_result_raw_sha256']==original['firstday_NK_result_source']['sha256']
     for label,d in [('OP_firstday',op),('NK_firstday',res)]:assert d.xpath('//a[@href=$href]',href=node['firstday_links'][label]['href'])
   assert entry['abnormal_detail_raw_sha256']==[p['source']['sha256'] for p in old['abnormal_detail_sources']]
   for p in old['abnormal_detail_sources']:
    d=html.fromstring(raw[p['source']['sha256']]);proof=p['detail'];tables=d.xpath('//table[@id="All_Result_Table"]');assert len(tables)==1
    rows=tables[0].xpath('.//tr[.//a[@href=$href]]',href='https://keirin.netkeiba.com/db/profile/?id='+str(int(reg)))
    assert len(rows)==1 and '落車棄権' in rows[0].text_content();details+=1
   expected=[] if old['OP_empty_cell_alternative'] is None else [p['sha256'] for p in old['OP_empty_cell_alternative']['sources']]
   assert entry['OP_empty_cell_alternative_raw_sha256']==expected;assert all(h in raw for h in expected);alt+=len(expected)
 assert kinds=={'current_meeting_context_not_participation':126,'actual_prior_meeting':235,'planned_slot_date_unresolved':17}
 assert priorrows==692 and opc==252 and details==8 and alt==5
 assert g['counts']=={'meeting_context_slots':126,'prior_slots':235,'planned_slots_unresolved':17,'prior_raw_rows':692,'OP_original_cells':252,'abnormal_detail_links':8,'alternative_gap_source_links':5}
 return {'raw_sources':55,'prior_slots':235,'prior_original_rows':692,'meeting_context_slots':126,'planned_slots_unresolved':17,'OP_original_cells':252,'abnormal_detail_links':8,'alternative_gap_sources':5,'canonical_SQL_inserts':0,'complete_payload_insertion_audit':False,'all_finals_complete':False}
def main():
 review=json.loads((OUT/'history_raw_graph_review.json').read_text());a=review['archive'];b=(ROOT/a['path']).read_bytes();assert len(b)==a['bytes'] and sha(b)==a['sha256']
 with tarfile.open(fileobj=io.BytesIO(b)) as t:
  assert t.getnames()==[a['member']];p=t.extractfile(a['member']).read()
 assert len(p)==a['member_bytes'] and sha(p)==a['member_sha256'];g=json.loads(p);result=audit(g);rejected=[]
 def mutate(name,fn):
  x=copy.deepcopy(g);fn(x)
  try:audit(x)
  except (AssertionError,KeyError,TypeError,ValueError,IndexError):rejected.append(name);return
  raise AssertionError('mutation accepted '+name)
 mutate('explicit_ID',lambda x:x['events'][0]['entries'][0].update(registration_number='099999'))
 mutate('group_href',lambda x:x['events'][0]['entries'][0]['slots'][1]['NK_group_DOM'].update(group_href='https://example.com'))
 mutate('raw_finish',lambda x:x['events'][0]['entries'][0]['slots'][1]['NK_group_DOM']['original_raw_result_rows'][0].update(finish_text='9'))
 mutate('original_year_proof',lambda x:x['events'][0]['entries'][0]['slots'][1].update(original_history_sha256='0'*64))
 mutate('planned_success_promotion',lambda x:next(n for e in x['events'] for a in e['entries'] for n in a['slots'] if n['kind']=='planned_slot_date_unresolved').update(kind='actual_prior_meeting',actual_start_date='2023-01-01'))
 mutate('source_bytes',lambda x:next(iter(x['source_pool'].values())).update(bytes=1))
 mutate('original_OP_status',lambda x:x['events'][0]['entries'][0]['OP_original_cells'][0].update(original_status='completed'))
 mutate('completion_flag',lambda x:x.update(all_finals_complete=True))
 result['mutations_rejected']=rejected;(OUT/'independent_history_raw_graph_audit.json').write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
