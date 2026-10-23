"""Independent root/edge proof audit for saved history dependency subgraphs."""
from pathlib import Path
import copy,hashlib,io,json,tarfile,unicodedata,re
from collections import Counter
from lxml import html
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/scope_review_0414'
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def clean(s):return ''.join(unicodedata.normalize('NFKC',s).split())
def audit(g):
 assert g['all_finals_complete'] is False and g['canonical_SQL_inserts']==g['new_completed_races']==g['network_requests']==0 and g['planned_slots_unresolved']==17
 assert g['base_overlay_member_sha256']=='53a19c888f8063062d7541ee8879e95796d2999c6c57d5e597260a0c8d765522'
 assert g['base_history_DOM_graph_member_sha256']=='5d5529ee8b7eb6be9b730bc44f0b75ded9ebfb062d90f486a5e0305ea9b466e0'
 raw={};arcs={}
 for h,r in g['source_pool'].items():
  assert h==r['sha256']
  if 'archive' in r:
   a=r['archive'];p=a['path']
   if p not in arcs:
    b=(ROOT/p).read_bytes();assert len(b)==a['bytes'] and sha(b)==a['sha256']
    with tarfile.open(fileobj=io.BytesIO(b)) as t:
     assert len(t.getnames())==len(set(t.getnames()));arcs[p]={n:t.extractfile(n).read() for n in t.getnames()}
   b=arcs[p][r['member']]
  else:b=(ROOT/r['path']).read_bytes()
  assert len(b)==r['bytes'] and sha(b)==h;raw[h]=b
 assert len(raw)==15 and set(g['roots'])=={'venue37_alias','venue13_alias','kochi_postponement'}
 # Read exact base members independently, never invoke old writer scripts.
 with tarfile.open(ROOT/'data/scope_review_0106/history_adapter_overlay.tar.xz') as t:b=t.extractfile('history_adapter_overlay.json').read()
 assert sha(b)==g['base_overlay_member_sha256'];base=json.loads(b)
 with tarfile.open(ROOT/'data/scope_review_0253/history_raw_graph.tar.xz') as t:b=t.extractfile('history_raw_graph.json').read()
 assert sha(b)==g['base_history_DOM_graph_member_sha256']
 for p,h in g['review_pins'].items():assert sha((ROOT/p).read_bytes())==h
 def source(root,needle):
  ids=g['roots'][root]['raw_sources'];found=[h for h in ids if needle(g['source_pool'][h])];assert len(found)==1;return raw[found[0]],found[0]
 r=g['roots']['venue37_alias'];assert r['venue_code']=='37' and r['names']==['伊東','伊東温泉'] and r['proof_event']=={'day':'2023-12-05','venue_code':'37','race_no':9}
 c=json.loads(source('venue37_alias',lambda x:x.get('path','').endswith('chariloto_api_card.html'))[0]);assert (c['box']['open_day'],c['box']['vel_code'],c['box']['vel_name'],c['box']['race_num'])==('2023-12-05','37','伊東',9)
 o=html.fromstring(source('venue37_alias',lambda x:x.get('member')=='202312053709_card.html')[0]);n=html.fromstring(source('venue37_alias',lambda x:x.get('member')=='202312053709_result.html')[0])
 assert '伊東競輪' in o.xpath('//title/text()')[0] and '伊東温泉競輪' in n.xpath('//title/text()')[0] and n.xpath('//input[@name="plp-raceid"]/@value')==['202312053709']
 for k,v in [('joCode','37'),('kaisaiBi','20231205'),('raceNo','9')]:assert o.xpath('//input[@id=$k]/@value',k=k)==[v]
 r=g['roots']['venue13_alias'];assert r['venue_code']=='13' and r['names']==['平','いわき平'] and r['explicit_registration']=='015049' and r['term']=='107' and r['proof_event']=={'day':'2023-11-26','venue_code':'13','race_no':9} and r['NK_actual_group_id']=='2023112413'
 c=json.loads(source('venue13_alias',lambda x:x.get('member')=='chariloto.html')[0]);a=c['race_athletes']['2'];assert (c['box']['open_day'],c['box']['vel_code'],c['box']['vel_name'],c['box']['race_num'])==('2023-11-26','13','いわき平',9)
 assert a['un_id']==15049 and clean(a['name'])=='菅原大也' and a['graduate_period']=='107' and a['history_data1']==r['raw_Chariloto_history1']
 assert a['history_data1']['vel_code']=='' and a['history_data1']['open_from']=='19881124' and clean(a['history_data1']['vel_name'])=='平' and [(x['race_num'],x['rank']) for x in a['history_data1']['short_boxes']]==[(5,2),(5,1)]
 o=html.fromstring(source('venue13_alias',lambda x:x.get('member')=='oddspark.html')[0]);player=o.xpath('//a[contains(@href,"PlayerDetail.do?playerCd=015049")]');assert len(player)==1;row=player[0]
 while row.tag!='tr':row=row.getparent()
 assert row.xpath('./td[@class="no2"]') and '107期' in row.text_content();text=unicodedata.normalize('NFKC',row.xpath('./td')[-3].text_content());assert re.findall(r'(11/\s*(?:24|25)).*?([12])\s*着',text,re.S)==[('11/24','2'),('11/25','1')]
 for k,v in [('joCode','13'),('kaisaiBi','20231126'),('raceNo','9')]:assert o.xpath('//input[@id=$k]/@value',k=k)==[v]
 n=html.fromstring(source('venue13_alias',lambda x:x.get('member')=='202312053709.html')[0]);row=n.xpath('//tr[@id="player-7"]')[0];assert row.xpath('.//a[@id="name_15049_202312053709"]')
 th=row.xpath('./th[a[@href="https://keirin.netkeiba.com/db/race_program/?kaisai_group_id=2023112413"]]');assert len(th)==1 and th[0].xpath('.//span[@class="JyoName"]/text()')==['いわき平']
 for key,finish in [('202311241305','2'),('202311251305','1'),('202311261309','1')]:assert row.xpath('.//a[@href=$href]//span[contains(@class,"result_no")]/text()',href='https://keirin.netkeiba.com/db/result/?race_id='+key+'&player_id=15049')==[finish]
 r=g['roots']['kochi_postponement'];assert (r['venue_code'],r['notice_date'],r['postponed_from'],r['postponed_to'])==('74','2026-06-02','2026-06-02','2026-06-03')
 body,notice_sha=source('kochi_postponement',lambda x:x.get('member')=='kochi_postponement.html');meta=json.loads(source('kochi_postponement',lambda x:x.get('member')=='kochi_postponement.meta.json')[0]);assert meta['sha256']==notice_sha and meta['bytes']==len(body) and meta['url']=='https://sp.oddspark.com/info/2026/06/62-1.html' and meta['status']==200
 txt=clean(html.fromstring(body,parser=html.HTMLParser(encoding='utf-8')).text_content());assert '2026/06/02' in txt and '6月2日(火)高知競輪3日目は台風のため開催中止、3日(水)に順延となりました。' in txt
 # Reconstruct only expected applicability edges, independently of new producer.
 expected=[]
 for ei,event in enumerate(base['events']):
  for ai,entry in enumerate(event['entries']):
   for field in ['original_OP_prior_slots','adapted_history_slots']:
    for si,slot in enumerate(entry[field]):
     group=slot.get('NK_group') if field=='original_OP_prior_slots' else slot.get('independent_group')
     if not group:continue
     names=[]
     if field=='original_OP_prior_slots':
      if (group['venue_code'],group['venue_name'])==('37','伊東温泉') and slot['OP_slot']['venue_header'] in ('伊東F1','伊東F2'):names.append('venue37_alias')
      if (group['venue_code'],group['venue_name'])==('13','いわき平') and slot['OP_slot']['venue_header'] in ('平F1','平F2'):names.append('venue13_alias')
     elif slot['original']['vel_code']=='' and clean(slot['original']['vel_name'])=='平' and (group['venue_code'],group['venue_name'])==('13','いわき平'):names.append('venue13_alias')
     if group.get('excluded_non_result_rows'):
      assert (event['local_key'],entry['registration_number'],group['group_id'])==('keirin:2026-06-30:23:09','015882','2026053174')
      assert group['excluded_non_result_rows']==[{'date':'2026-06-02','finish_text':'','href':'https://keirin.netkeiba.com/db/result/?race_id=202606027407&player_id=15882','race_id':'202606027407','race_no':7}]
      assert [r['date'] for r in group['results']]==['2026-05-31','2026-06-01','2026-06-03'] and group['postponement_evidence']['sha256']==notice_sha;names.append('kochi_postponement')
     for name in names:expected.append({'root':name,'overlay_pointer':['events',ei,'entries',ai,field,si],'local_key':event['local_key'],'car_no':entry['car_no'],'registration_number':entry['registration_number'],'group_id':group['group_id'],'group_sha256':sha(canon(group))})
 assert sorted(map(canon,g['edges']))==sorted(map(canon,expected)) and len(expected)==21
 counts=Counter(e['root'] for e in expected);assert counts=={'venue37_alias':5,'venue13_alias':14,'kochi_postponement':2}
 return {'roots_independently_verified':3,'raw_source_locators':15,'applicability_edges':21,'venue37_OP_slots':5,'venue13_OP_slots':7,'venue13_Chariloto_slots':7,'postponement_OP_and_Chariloto_slots':2,'planned_slots_unresolved':17,'original_history_and_blank_row_preserved':True,'canonical_SQL_inserts':0,'complete_payload_insertion_audit':False,'all_finals_complete':False}
def main():
 r=json.loads((OUT/'dependency_subgraphs_review.json').read_bytes());a=r['archive'];b=(ROOT/a['path']).read_bytes();assert sha(b)==a['sha256'] and len(b)==a['bytes']
 with tarfile.open(fileobj=io.BytesIO(b)) as t:
  assert t.getnames()==[a['member']];p=t.extractfile(a['member']).read()
 assert sha(p)==a['member_sha256'] and len(p)==a['member_bytes'];g=json.loads(p);result=audit(g);mut=[]
 def reject(name,fn):
  x=copy.deepcopy(g);fn(x)
  try:audit(x)
  except (AssertionError,KeyError,IndexError,ValueError,TypeError):mut.append(name);return
  raise AssertionError('mutation accepted '+name)
 reject('alias_code',lambda x:x['roots']['venue13_alias'].update(venue_code='14'))
 reject('alias_name',lambda x:x['roots']['venue37_alias'].update(names=['伊東','別府']))
 reject('proof_explicit_ID',lambda x:x['roots']['venue13_alias'].update(explicit_registration='015050'))
 reject('raw_history_year',lambda x:x['roots']['venue13_alias']['raw_Chariloto_history1'].update(open_from='20231124'))
 reject('postponement_date',lambda x:x['roots']['kochi_postponement'].update(postponed_from='2026-06-01'))
 reject('edge_registration',lambda x:x['edges'][0].update(registration_number='099999'))
 reject('edge_group_SHA',lambda x:x['edges'][0].update(group_sha256='0'*64))
 reject('source_bytes',lambda x:next(iter(x['source_pool'].values())).update(bytes=1))
 reject('planned_success_promotion',lambda x:x.update(planned_slots_unresolved=0))
 reject('completion_flag',lambda x:x.update(all_finals_complete=True))
 result['mutations_rejected']=mut;(OUT/'independent_dependency_subgraphs_audit.json').write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
