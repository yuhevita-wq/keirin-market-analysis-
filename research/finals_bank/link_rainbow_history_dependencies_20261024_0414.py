"""Attach saved venue-alias and postponement proof subgraphs to candidate history.
Offline, no canonical SQL, no source edits or unresolved-slot promotion.
"""
from pathlib import Path
import copy,hashlib,io,json,tarfile,sys,re,unicodedata
from collections import Counter
from lxml import html
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'recovery'))
from link_rainbow_history_raw_graph_0253 import Reader,load_base
OUT=ROOT/'data/scope_review_0414'
PINS={'data/scope_review_1028/venue37_alias_evidence.json':'d6ce9e30873a9d1b39681fcc211d258697ad5fe5d589a14605f787798b2ec546','data/scope_review_1206/raw_manifest.json':'146de6afa92ea44f6ec56e35bdd768c64cadde9218ab9c7dd7f0f567cd7e7c53','data/scope_review_0831/raw_manifest.json':'47661ef1ca2ff32aa6cacf647b7db9c526a5652bbddc46f7ce5a615214c3d18f','data/scope_review_0914/raw_manifest.json':'f1c162c631e4dcbf593907bb4656c34a1cc282b700fc4bf144d1eae06cfb30af','data/scope_review_0253/history_raw_graph_review.json':'becd72b7cbafb9f0a5d2af1987b4b590caf1303168e7bfffd8088e4a398a9315'}
ARCHIVES={'cards':('data/scope_review_0831/independent_card_results_raw.tar.xz','b067a5fcb5629030f5440bc0efc88e368b753c4236a5a50fee8af372c270bbe2','data/scope_review_0831/raw_manifest.json'),'history':('data/scope_review_0914/independent_history_raw.tar.xz','6e9c07f6b552b16ba501fd1520347e85ac0dbf6d16d7f3d3020b4da8404c4985','data/scope_review_0914/raw_manifest.json'),'alias13':('data/scope_review_1206/venue13_raw.tar.xz','8e74d4f274e72e59aa26d888dccd5b9c08d7a133c54979f578aa0b9bc8f5a9c4','data/scope_review_1206/raw_manifest.json')}
def sha(b):return hashlib.sha256(b).hexdigest()
def norm(s):return ''.join(unicodedata.normalize('NFKC',s).split())
def enc(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
class Inputs:
 def __init__(self):self.r=Reader();self.loaded={}
 def direct(self,p,h=None):
  b=(ROOT/p).read_bytes();assert h is None or sha(b)==h;ref={'path':p,'sha256':sha(b),'bytes':len(b)};self.r.raw(ref);return b,ref
 def member(self,label,name):
  path,h,mp=ARCHIVES[label];manifest=json.loads((ROOT/mp).read_bytes())
  if label not in self.loaded:
   b=(ROOT/path).read_bytes();assert sha(b)==h
   with tarfile.open(fileobj=io.BytesIO(b)) as t:
    assert len(t.getnames())==len(set(t.getnames())) and set(t.getnames())==set(manifest)
    for n in t.getnames():
     raw=t.extractfile(n).read();assert len(raw)==manifest[n]['bytes'] and sha(raw)==manifest[n]['sha256']
   self.loaded[label]={'path':path,'sha256':h,'bytes':len(b),'members':len(manifest)}
  ref={'archive':self.loaded[label],'member':name,**manifest[name]};return self.r.raw(ref),ref
 def pair(self,label,body,meta):
  raw,ref=self.member(label,body);m,mref=self.member(label,meta);d=json.loads(m);assert d['status']==200 and d['sha256']==sha(raw) and d['bytes']==len(raw);return raw,ref,d,mref

def roots(i):
 ev=json.loads(i.direct('data/scope_review_1028/venue37_alias_evidence.json',PINS['data/scope_review_1028/venue37_alias_evidence.json'])[0]);raw={};refs=[]
 for label,s in ev['sources'].items():
  if 'archive' in s:
   body=s['member'];b,ref,m,mref=i.pair('cards',body,body.replace('.html','.meta.json'));refs.extend([ref['sha256'],mref['sha256']])
  else:
   b,ref=i.direct(s['path']);mb,mref=i.direct(s['path'].replace('.html','.json'));m=json.loads(mb);assert m['status']==200 and m['sha256']==sha(b) and m['bytes']==len(b);refs.extend([ref['sha256'],mref['sha256']])
  assert sha(b)==s['sha256'] and len(b)==s['bytes'];raw[label]=b
 c=json.loads(raw['Chariloto']);o=html.fromstring(raw['OddsPark']);n=html.fromstring(raw['netkeiba'])
 assert (c['box']['vel_code'],c['box']['vel_name'],c['box']['open_day'],c['box']['race_num'])==('37','伊東','2023-12-05',9)
 for k,v in [('joCode','37'),('kaisaiBi','20231205'),('raceNo','9')]:assert o.xpath('//input[@id=$k]/@value',k=k)==[v]
 assert '伊東競輪' in o.xpath('//title/text()')[0] and n.xpath('//input[@name="plp-raceid"]/@value')==['202312053709'] and '伊東温泉競輪' in n.xpath('//title/text()')[0]
 r37={'kind':'explicit_same_event_venue_alias','venue_code':'37','names':['伊東','伊東温泉'],'proof_event':{'day':'2023-12-05','venue_code':'37','race_no':9},'raw_sources':refs,'scope':'venue name only; no ID inference or temporal score/geometry extension'}
 cb,cr,cm,cmr=i.pair('alias13','chariloto.html','chariloto.json');ob,orr,om,omr=i.pair('alias13','oddspark.html','oddspark.json');hb,hr,hm,hmr=i.pair('history','202312053709.html','202312053709.meta.json');c=json.loads(cb);o=html.fromstring(ob);n=html.fromstring(hb)
 assert (c['box']['vel_code'],c['box']['vel_name'],c['box']['open_day'],c['box']['race_num'])==('13','いわき平','2023-11-26',9)
 a=c['race_athletes']['2'];assert a['un_id']==15049 and norm(a['name'])=='菅原大也' and a['graduate_period']=='107' and a['history_data1']['vel_code']=='' and norm(a['history_data1']['vel_name'])=='平'
 assert a['history_data1']['open_from']=='19881124' and [(r['race_num'],r['rank']) for r in a['history_data1']['short_boxes']]==[(5,2),(5,1)]
 for k,v in [('joCode','13'),('kaisaiBi','20231126'),('raceNo','9')]:assert o.xpath('//input[@id=$k]/@value',k=k)==[v]
 link=o.xpath('//a[contains(@href,"PlayerDetail.do?playerCd=015049")]');assert len(link)==1;row=link[0]
 while row.tag!='tr':row=row.getparent()
 assert row.xpath('./td[@class="no2"]') and '107期' in row.text_content()
 nr=n.xpath('//tr[@id="player-7"]');assert len(nr)==1 and nr[0].xpath('.//a[@id="name_15049_202312053709"]')
 group=nr[0].xpath('./th/a[@href="https://keirin.netkeiba.com/db/race_program/?kaisai_group_id=2023112413"]');assert len(group)==1 and group[0].xpath('.//span[@class="JyoName"]/text()')==['いわき平']
 for key,finish in [('202311241305','2'),('202311251305','1'),('202311261309','1')]:assert nr[0].xpath('.//a[@href=$u]//span[contains(@class,"result_no")]/text()',u='https://keirin.netkeiba.com/db/result/?race_id='+key+'&player_id=15049')==[finish]
 r13={'kind':'explicit_same_code_event_rider_series_alias','venue_code':'13','names':['平','いわき平'],'proof_event':{'day':'2023-11-26','venue_code':'13','race_no':9},'explicit_registration':'015049','term':'107','raw_Chariloto_history1':copy.deepcopy(a['history_data1']),'NK_actual_group_id':'2023112413','raw_sources':[r['sha256'] for r in [cr,cmr,orr,omr,hr,hmr]],'scope':'venue alias only; original empty venue code and 1988 date unchanged'}
 nb,nr,nm,nmr=i.pair('history','kochi_postponement.html','kochi_postponement.meta.json');nb.decode('utf-8','strict');text=norm(html.fromstring(nb,parser=html.HTMLParser(encoding='utf-8')).text_content());assert '2026/06/02' in text and '6月2日(火)高知競輪3日目は台風のため開催中止、3日(水)に順延となりました。' in text and nm['url']=='https://sp.oddspark.com/info/2026/06/62-1.html'
 rp={'kind':'dated_postponement_notice','venue_code':'74','notice_date':'2026-06-02','postponed_from':'2026-06-02','postponed_to':'2026-06-03','raw_sources':[nr['sha256'],nmr['sha256']],'scope':'only group2026053174 blank actual202606027407; retain blank raw row, no numeric finish inference'}
 return {'venue37_alias':r37,'venue13_alias':r13,'kochi_postponement':rp}

def edges(base):
 out=[]
 def add(ei,ai,field,si,root,g):out.append({'root':root,'overlay_pointer':['events',ei,'entries',ai,field,si],'local_key':base['events'][ei]['local_key'],'car_no':base['events'][ei]['entries'][ai]['car_no'],'registration_number':base['events'][ei]['entries'][ai]['registration_number'],'group_id':g['group_id'],'group_sha256':sha(enc(g))})
 for ei,e in enumerate(base['events']):
  for ai,a in enumerate(e['entries']):
   for si,s in enumerate(a['original_OP_prior_slots']):
    g=s.get('NK_group')
    if not g:continue
    h=s['OP_slot']['venue_header']
    if g['venue_code']=='37' and g['venue_name']=='伊東温泉' and h in ['伊東F1','伊東F2']:add(ei,ai,'original_OP_prior_slots',si,'venue37_alias',g)
    if g['venue_code']=='13' and g['venue_name']=='いわき平' and h in ['平F1','平F2']:add(ei,ai,'original_OP_prior_slots',si,'venue13_alias',g)
    if g.get('excluded_non_result_rows'):add(ei,ai,'original_OP_prior_slots',si,'kochi_postponement',g)
   for si,s in enumerate(a['adapted_history_slots']):
    if s['kind']!='actual_prior_meeting':continue
    g=s['independent_group'];h=s['original']
    if h['vel_code']=='' and norm(h['vel_name'])=='平' and g['venue_code']=='13' and g['venue_name']=='いわき平':add(ei,ai,'adapted_history_slots',si,'venue13_alias',g)
    if g.get('excluded_non_result_rows'):add(ei,ai,'adapted_history_slots',si,'kochi_postponement',g)
 return out

def main():
 for p,h in PINS.items():assert sha((ROOT/p).read_bytes())==h
 base=load_base();i=Inputs();rs=roots(i);es=edges(base);counts=Counter((x['root'],x['overlay_pointer'][4]) for x in es)
 assert counts=={('venue37_alias','original_OP_prior_slots'):5,('venue13_alias','original_OP_prior_slots'):7,('venue13_alias','adapted_history_slots'):7,('kochi_postponement','original_OP_prior_slots'):1,('kochi_postponement','adapted_history_slots'):1}
 for edge in es:
  if edge['root']=='kochi_postponement':
   assert edge['local_key']=='keirin:2026-06-30:23:09' and edge['registration_number']=='015882' and edge['group_id']=='2026053174'
   p=edge['overlay_pointer'];s=base['events'][p[1]]['entries'][p[3]][p[4]][p[5]];g=s['NK_group'] if p[4]=='original_OP_prior_slots' else s['independent_group'];assert len(g['excluded_non_result_rows'])==1 and g['excluded_non_result_rows'][0]['race_id']=='202606027407' and g['excluded_non_result_rows'][0]['finish_text']=='' and len(g['results'])==3
 payload={'schema':'rainbow_history_dependency_subgraphs_v1','base_overlay_member_sha256':'53a19c888f8063062d7541ee8879e95796d2999c6c57d5e597260a0c8d765522','base_history_DOM_graph_member_sha256':'5d5529ee8b7eb6be9b730bc44f0b75ded9ebfb062d90f486a5e0305ea9b466e0','review_pins':PINS,'roots':rs,'source_pool':i.r.pool,'edges':es,'status':'candidate_history_dependency_subgraphs_only','planned_slots_unresolved':17,'canonical_SQL_inserts':0,'new_completed_races':0,'network_requests':0,'all_finals_complete':False,'remaining':['unified candidate payload and complete production schema/readiness/independent insertion audit','17 planned dates/explicit IDs','13 candidate banks','broader scope and existing 5123 bank gaps']}
 OUT.mkdir(exist_ok=True);b=enc(payload);p=OUT/'dependency_subgraphs.tar.xz'
 with tarfile.open(p,'w:xz',preset=9) as t:
  m=tarfile.TarInfo('dependency_subgraphs.json');m.size=len(b);m.mtime=0;t.addfile(m,io.BytesIO(b))
 with tarfile.open(p) as t:assert t.getnames()==['dependency_subgraphs.json'] and t.extractfile('dependency_subgraphs.json').read()==b
 report={'archive':{'path':str(p.relative_to(ROOT)),'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size,'member':'dependency_subgraphs.json','member_sha256':sha(b),'member_bytes':len(b)},'roots':3,'raw_sources':len(i.r.pool),'edges':len(es),'edge_counts':{k+':'+f:v for (k,f),v in counts.items()},'status':payload['status'],'canonical_SQL_inserts':0,'new_completed_races':0,'network_requests':0,'all_finals_complete':False,'remaining':payload['remaining']};(OUT/'dependency_subgraphs_review.json').write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
