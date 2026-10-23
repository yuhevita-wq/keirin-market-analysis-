"""Resolve one candidate identity using an alternative official registration PDF.
Historical affiliation is separately matched in the saved event-time cards.
No claim that a 2022 allocation proves participation or a continuous affiliation.
"""
from pathlib import Path
import copy,hashlib,io,json,tarfile
import fitz
from lxml import html
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/scope_review_0323'
PDF_SHA='f3a87ac9c4528d8f7a3752a325fd50f4171ccc3ae2f894add18e2436bec04155'
SCOPE_SHA='53552457be52018be9f3b6c8c9258e1acb9fa92d8bab5568431dc06605591bdb'
def sha(b):return hashlib.sha256(b).hexdigest()
def norm(s):return ''.join(s.split())
def check_pdf(raw,registration='015612',name='小榑佑弥',term='121',prefecture='新潟'):
 assert sha(raw)==PDF_SHA
 d=fitz.open(stream=raw,filetype='pdf');assert len(d)==1;lines=d[0].get_text().splitlines()
 assert '新登録選手('+term+'期)2022年8月開催別あっせん一覧' in norm(lines[0])
 hits=[]
 for j,s in enumerate(lines):
  if s.strip()==str(int(registration)):
   assert norm(lines[j+1])==norm(name) and norm(lines[j+2])==prefecture
   hits.append({'page_one_based':1,'line_zero_based':j,'original_lines':lines[j:j+3]})
 assert len(hits)==2
 return hits
class Source:
 def __init__(self):self.arcs={}
 def get(self,ref):
  if 'archive' in ref:
   a=ref['archive'];n=a['path']
   if n not in self.arcs:
    b=(ROOT/n).read_bytes();assert len(b)==a['bytes'] and sha(b)==a['sha256']
    with tarfile.open(fileobj=io.BytesIO(b)) as t:
     assert len(t.getnames())==len(set(t.getnames()));self.arcs[n]={x:t.extractfile(x).read() for x in t.getnames()}
   b=self.arcs[n][ref['member']]
  else:b=(ROOT/ref['path']).read_bytes()
  assert sha(b)==ref['sha256'] and len(b)==ref['bytes'];return b

def build():
 manifest=json.loads((OUT/'raw_manifest.json').read_bytes());src=Source();b=src.get(manifest['archive'])
 with tarfile.open(fileobj=io.BytesIO(b)) as t:
  assert len(t.getnames())==len(set(t.getnames()))==2 and set(t.getnames())==set(manifest['members']);files={n:t.extractfile(n).read() for n in t.getnames()}
 for n,raw in files.items():assert {'sha256':sha(raw),'bytes':len(raw)}==manifest['members'][n]
 raw=files['allocation121.pdf'];hits=check_pdf(raw);meta=json.loads(files['allocation121.meta.json'])
 assert meta['status']==200 and meta['sha256']==sha(raw) and meta['bytes']==len(raw) and meta['single_attempt'] is True
 assert meta['url']==meta['final_url']=='https://keirin.jp/pc/dfw/portal/guest/news/2022khn/06/pdf/20220628_04_02.pdf'
 sb=(ROOT/'data/scope_review_0457/rainbow_scope_independent_review.json').read_bytes();assert sha(sb)==SCOPE_SHA;scope=json.loads(sb)
 e=next(e for e in scope['events'] if (e['day'],e['venue_code'],e['race_no'])==('2024-06-30','23',9));entry=next(a for a in e['entry_identities'] if a['car_no']==9)
 assert entry['registration_number']=='015612' and norm(entry['name'])=='小榑佑弥' and entry['term']=='121' and entry['prefecture']=='新潟'
 component_ref={'path':'data/scope_review_2308/component_drafts.tar.xz','bytes':60080,'sha256':'aed23e460c08eccba068b28f45b405f02168f908192e19ba9020d3d8bdf7f2ea'}
 with tarfile.open(fileobj=io.BytesIO(src.get(component_ref))) as t:
  cb=t.extractfile('component_drafts.json').read();assert sha(cb)=='eec2b7f45058a277ce6e0ab17b898735d416d17783cef807fe4a0193fc4b833c'
 c=next(d for d in json.loads(cb)['events'] if d['identity']['day']=='2024-06-30');refs={k:c['component_raw_provenance'][k] for k in ['card','independent_card','independent_result']}
 card=json.loads(src.get(refs['card']));assert card['box']['open_day']=='2024-06-30' and card['box']['vel_code']=='23' and card['race_program']['race_num']==9
 a=card['race_athletes']['9'];assert a['un_id']==15612 and norm(a['name'])=='小榑佑弥' and a['graduate_period']=='121' and a['pref_name']=='新潟'
 op=html.fromstring(src.get(refs['independent_card']));rows=op.xpath('//table[@summary="出走表"]/tr[td[@class="no9"]]');assert len(rows)==1;cells=rows[0].xpath('./td')
 assert rows[0].xpath('.//a[@href="./PlayerDetail.do?playerCd=015612"]') and norm(cells[-8].text_content())=='小榑佑弥24歳／121期' and cells[-7].text_content()=='新潟'
 for k,v in [('joCode','23'),('kaisaiBi','20240630'),('raceNo','9')]:assert op.xpath('//input[@name=$k and @value=$v]',k=k,v=v)
 nk=html.fromstring(src.get(refs['independent_result']));assert nk.xpath('//input[@name="plp-raceid" and @value="202406302309"]')
 nr=nk.xpath('//table[@id="All_Result_Table"]//tr[.//a[@href="https://keirin.netkeiba.com/db/profile/?id=15612"]]');assert len(nr)==1 and nr[0].xpath('.//img[@alt="小榑佑弥"]')
 prior_raw=(ROOT/'data/scope_review_0115/identity_scope_provenance_review.json').read_bytes();assert sha(prior_raw)=='4b4819ab2b8da379f0835490bbde6e59defdfbad3bd1d1eb075767d5bf71083f';prior=json.loads(prior_raw)
 prev=next(x for event in prior['events'] if event['day']=='2024-06-30' for x in event['entries'] if x['registration_number']=='015612');assert prev['status']=='raw_transfer_source_unresolved' and prev['locator'] is None
 return {'schema':'rainbow_alternative_official_identity_proof_v1','status':'candidate_identity_raw_source_gap_resolved_by_alternative_primary_evidence','target':{'local_key':'keirin:2024-06-30:23:09','day':'2024-06-30','venue_code':'23','race_no':9,'car_no':9,'registration_number':'015612','name':'小榑佑弥','term':'121','prefecture_at_event':'新潟'},'official_allocation':{'locator':{'archive':manifest['archive'],'member':'allocation121.pdf','sha256':sha(raw),'bytes':len(raw)},'source_url':meta['url'],'original_identity_rows':hits,'scope':'explicit registration ID/name/term and published 2022 prefecture only; no participation or continuous affiliation inferred'},'event_time_sources':refs,'original_review_scope_sha256':SCOPE_SHA,'prior_identity_provenance_sha256':sha(prior_raw),'prior_0115_identity_provenance_counts_retained':prior['counts'],'candidate_identity_core_raw_coverage_after_delta':{'previously_bound':125,'alternative_primary_resolved':1,'total_bound':126,'unresolved_core_identity':0},'original_transfer_page_body_status':'still_unavailable_HTTP403_not_recovered_not_retried','historical_prefecture_basis':'saved 2024-06-30 same-car explicit-ID Chariloto and OddsPark card fields agree; 2022 PDF not extended as an affiliation interval','unresolved_remaining':['17 planned dates/explicit IDs','13 candidate historical banks','alias/postponement full provenance subgraphs','complete production payload/readiness/isolated complete insertion audit','broader scope and existing 5123 bank gaps'],'canonical_SQL_inserts':0,'new_completed_races':0,'all_finals_complete':False,'current_performance_imported':False}
if __name__=='__main__':
 r=build();(OUT/'alternative_identity_review.json').write_text(json.dumps(r,ensure_ascii=False,sort_keys=True,indent=2)+'\n');print(json.dumps(r['candidate_identity_core_raw_coverage_after_delta']))
