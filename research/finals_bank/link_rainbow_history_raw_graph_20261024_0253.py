"""Offline graph from existing candidate history slots to exact saved raw bytes.
No promotion of planned slots, no production SQL writes, no year correction.
"""
import hashlib,io,json,tarfile,copy,unicodedata
from pathlib import Path
from lxml import html
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/scope_review_0253'
BASE='data/scope_review_0106/history_adapter_overlay.tar.xz'
BASE_SHA='c81b9260ef8e3fb8f5d2186fc6477e45fdd953cec5c1a4118d69513628290248'
MEMBER_SHA='53a19c888f8063062d7541ee8879e95796d2999c6c57d5e597260a0c8d765522'
def sha(b):return hashlib.sha256(b).hexdigest()
def norm(s):return ''.join(s.split())
class Reader:
 def __init__(self):self.arcs={};self.pool={};self.docs={}
 def raw(self,ref):
  if 'archive' in ref:
   a=ref['archive'];key=a['path']
   if key not in self.arcs:
    b=(ROOT/key).read_bytes();assert sha(b)==a['sha256'] and len(b)==a['bytes']
    with tarfile.open(fileobj=io.BytesIO(b)) as t:
     names=t.getnames();assert len(names)==len(set(names));self.arcs[key]={n:t.extractfile(n).read() for n in names}
   b=self.arcs[key][ref['member']]
  else:b=(ROOT/ref['path']).read_bytes()
  assert sha(b)==ref['sha256'] and len(b)==ref['bytes']
  h=ref['sha256'];self.pool[h]=copy.deepcopy(ref);return b
 def doc(self,ref):
  b=self.raw(ref);h=sha(b)
  if h not in self.docs:self.docs[h]=html.fromstring(b.decode('utf-8','strict'))
  return self.docs[h]
def load_base():
 raw=(ROOT/BASE).read_bytes();assert sha(raw)==BASE_SHA
 with tarfile.open(fileobj=io.BytesIO(raw)) as t:
  assert t.getnames()==['history_adapter_overlay.json'];b=t.extractfile('history_adapter_overlay.json').read()
 assert sha(b)==MEMBER_SHA
 return json.loads(b)
def group_node(row,g,registration):
 anchors=row.xpath('./th/a[@href=$href]',href=g['group_href']);assert len(anchors)==1
 raw_results=[];cur=anchors[0].getparent().getnext()
 while cur is not None and cur.tag=='td':
  a=cur.xpath('./a[contains(@href,"/db/result/")]')
  if a:
   assert len(a)==1;r=a[0].xpath('.//span[contains(concat(" ",normalize-space(@class)," ")," result_no ")]');assert len(r)==1
   raw_results.append({'href':a[0].get('href'),'finish_text':r[0].text_content().strip()})
  cur=cur.getnext()
 expected=g['results']+g.get('excluded_non_result_rows',[])
 assert len(raw_results)==len(expected)
 assert {x['href']:x['finish_text'] for x in raw_results}=={x['href']:x['finish_text'] for x in expected}
 assert all('player_id='+str(int(registration)) in x['href'] for x in raw_results)
 return {'group_href':g['group_href'],'original_raw_result_rows':raw_results,'excluded_non_result_rows_retained':copy.deepcopy(g.get('excluded_non_result_rows',[]))}
def build():
 base=load_base();reader=Reader();events=[];counts={'meeting_context_slots':0,'prior_slots':0,'planned_slots_unresolved':0,'prior_raw_rows':0,'OP_original_cells':0,'abnormal_detail_links':0,'alternative_gap_source_links':0}
 for ei,event in enumerate(base['events']):
  history=reader.doc(event['history_source']);card=reader.doc(event['independent_card_source']);result=reader.doc(event['firstday_NK_result_source']);entries=[]
  for ai,entry in enumerate(event['entries']):
   car=entry['car_no'];reg=entry['registration_number'];rows=history.xpath('//tr[@id=$id]',id='player-'+str(car));assert len(rows)==1;row=rows[0]
   assert len(row.xpath('.//a[starts-with(@id,$id)]',id='name_'+str(int(reg))+'_'))==1
   op=card.xpath('//a[contains(@href,$id)]',id='playerCd='+reg);assert len(op)==1
   op_row=op[0]
   while op_row.tag!='tr':op_row=op_row.getparent()
   cells=op_row.xpath('./td');assert len(cells)>=8
   op_proofs=[]
   for j,old in enumerate(entry['original_OP_prior_slots']):
    original=old['OP_slot']['raw'] if 'OP_slot' in old else old['raw']
    assert unicodedata.normalize('NFKC',cells[-2+j].text_content())==original
    op_proofs.append({'cell_from_end':-2+j,'raw_text_sha256':sha(original.encode()),'original_status':old['status']});counts['OP_original_cells']+=1
   nodes=[]
   for si,s in enumerate(entry['adapted_history_slots']):
    n={'overlay_pointer':['events',ei,'entries',ai,'adapted_history_slots',si],'slot':s['slot'],'kind':s['kind'],'original_history_sha256':sha(json.dumps(s['original'],ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()),'history_raw_sha256':event['history_source']['sha256'],'OP_card_raw_sha256':event['independent_card_source']['sha256'],'actual_start_date':s['actual_start_date']}
    if s['kind']=='actual_prior_meeting':
     n['NK_group_DOM']=group_node(row,s['independent_group'],reg);counts['prior_slots']+=1;counts['prior_raw_rows']+=len(n['NK_group_DOM']['original_raw_result_rows'])
    elif s['kind']=='current_meeting_context_not_participation':
     assert s['actual_start_date'] is None and s['rider_firstday_participation_inferred'] is False
     for label,doc in [('OP_firstday',card),('NK_firstday',result)]:
      proof=s['proof'][label];a=doc.xpath('//a[@href=$href]',href=proof['href']);assert len(a)>=1 and any(norm(proof['label'])==norm(x.text_content()) for x in a)
     n['firstday_links']=copy.deepcopy(s['proof']);n['NK_result_raw_sha256']=event['firstday_NK_result_source']['sha256'];counts['meeting_context_slots']+=1
    else:
     assert s['kind']=='planned_slot_date_unresolved' and s['actual_start_date'] is None and s['independent_group'] is None and not s['original']['short_boxes']
     n['planned_date_independent_source']=None;counts['planned_slots_unresolved']+=1
    nodes.append(n)
   details=[]
   for detail in entry['abnormal_detail_sources']:
    raw=reader.raw(detail['source']);d=html.fromstring(raw);proof=detail['detail'];rs=d.xpath('//table[@id="All_Result_Table"]//tr[.//a[@href=$href]]',href='https://keirin.netkeiba.com/db/profile/?id='+str(int(reg)));assert len(rs)==1 and '落車棄権' in rs[0].text_content();details.append(detail['source']['sha256']);counts['abnormal_detail_links']+=1
   alt=entry['OP_empty_cell_alternative'];altrefs=[]
   if alt:
    for ref in alt['sources']:reader.raw(ref);altrefs.append(ref['sha256']);counts['alternative_gap_source_links']+=1
    assert entry['original_OP_prior_slots'][1]['status']=='unresolved_empty_OP_slot'
   entries.append({'car_no':car,'registration_number':reg,'NK_row_id':'player-'+str(car),'OP_original_cells':op_proofs,'slots':nodes,'abnormal_detail_raw_sha256':details,'OP_empty_cell_alternative_raw_sha256':altrefs})
  events.append({'local_key':event['local_key'],'entries':entries})
 assert counts['meeting_context_slots']==126 and counts['prior_slots']==235 and counts['planned_slots_unresolved']==17 and counts['OP_original_cells']==252
 return {'schema':'rainbow_history_raw_graph_v1','status':'candidate_history_DOM_locator_graph_only','base_history_overlay':{'path':BASE,'sha256':BASE_SHA,'member':'history_adapter_overlay.json','member_sha256':MEMBER_SHA},'counts':counts,'source_pool':reader.pool,'events':events,'unresolved':['17 planned dates/explicit IDs','015612 official transfer raw notice','13 candidate historical banks','alias/postponement dependency subgraphs not yet integrated','complete payload schema/readiness/isolated complete insertion audit','broader scope and existing 5123 historical banks'],'canonical_SQL_inserts':0,'new_completed_races':0,'network_requests':0,'all_finals_complete':False}
def main():
 OUT.mkdir(exist_ok=True);g=build();payload=json.dumps(g,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode();p=OUT/'history_raw_graph.tar.xz'
 with tarfile.open(p,'w:xz',preset=9) as t:
  m=tarfile.TarInfo('history_raw_graph.json');m.size=len(payload);m.mtime=0;t.addfile(m,io.BytesIO(payload))
 with tarfile.open(p) as t:assert t.extractfile('history_raw_graph.json').read()==payload
 report={'archive':{'path':str(p.relative_to(ROOT)),'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size,'member':'history_raw_graph.json','member_sha256':sha(payload),'member_bytes':len(payload)},'counts':g['counts'],'source_pool_unique':len(g['source_pool']),'status':g['status'],'unresolved':g['unresolved'],'canonical_SQL_inserts':0,'new_completed_races':0,'network_requests':0,'all_finals_complete':False}
 (OUT/'history_raw_graph_review.json').write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
