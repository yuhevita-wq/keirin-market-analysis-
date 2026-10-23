"""Map independently matched saved payouts to the existing payload.payouts fields.

Only a payload fragment; no production race row or completion promotion.
"""
import copy,hashlib,io,json,re,tarfile,unicodedata,sqlite3
from decimal import Decimal
from pathlib import Path
from lxml import html
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/scope_review_0636'
PINS={'pipeline.py': '122cd6a9b06821618e539e84b2fc98a031f6df065f2332c9a88e8b6b969b4d81', 'data/scope_review_0552/integrated_candidates.tar.xz': 'c1a030bc66b78fec4bda307046402f9db15fa67dd386845b8d19862cd3f29e3a', 'data/scope_review_0552/integrated_candidate_review.json': '8d41909097f61ab60de8490f64f74a4facd5cc4a0ca8a580ccd45906a951a3ad', 'data/scope_review_0552/independent_integrated_candidate_audit.json': 'c7656fc4864e85692afac82910348f132f6b62264dd00a3877975d158f008537'}
MAP={'Wakuren':('frame_quinella','2枠複',2,False),'Wakutan':('frame_exacta','2枠単',2,True),'Umaren':('quinella','2車複',2,False),'Umatan':('exacta','2車単',2,True),'Wide':('wide','ワイド',2,False),'Fuku3':('trio','3連複',3,False),'Tan3':('trifecta','3連単',3,True)}
CHAR={'2枠連':{'複':'Wakuren','単':'Wakutan'},'2車連':{'複':'Umaren','単':'Umatan'},'3連勝':{'複':'Fuku3','単':'Tan3'},'ワイド':{'ワイド':'Wide'}}
def sha(b):return hashlib.sha256(b).hexdigest()
def enc(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def norm(x):return unicodedata.normalize('NFKC',''.join(x.split()))
class Sources:
 def __init__(self):self.arcs={}
 def raw(self,ref):
  if 'archive' in ref:
   a=ref['archive'];p=a['path'];b=(ROOT/p).read_bytes();assert sha(b)==a['sha256'] and len(b)==a['bytes']
   if p not in self.arcs:
    with tarfile.open(fileobj=io.BytesIO(b)) as t:
     assert len(t.getnames())==len(set(t.getnames()));self.arcs[p]={m.name:t.extractfile(m).read() for m in t.getmembers()}
   b=self.arcs[p][ref['member']]
  else:b=(ROOT/ref['path']).read_bytes()
  assert sha(b)==ref['sha256'] and len(b)==ref['bytes'];return b
def candidate():
 for p,h in PINS.items():assert sha((ROOT/p).read_bytes())==h
 with tarfile.open(ROOT/'data/scope_review_0552/integrated_candidates.tar.xz') as t:
  assert t.getnames()==['integrated_candidates.json'];b=t.extractfile(t.getmembers()[0]).read()
 assert sha(b)=='b9ed0533f9c8a654236a4c4c8d9d275161acdd6c154ad6c3efbaa444cb307e3f'
 return json.loads(b)
def combo(s,n,ordered,frame=False):
 s=norm(s);assert re.fullmatch(r'[1-9](?:[=>-][1-9]){'+str(n-1)+'}',s),s
 parts=[int(v) for v in re.split('[=>-]',s)];assert all(v<= (6 if frame else 9) for v in parts)
 if not frame:assert len(set(parts))==len(parts)
 return tuple(parts if ordered else sorted(parts))
def amount(s):
 s=norm(s);assert re.fullmatch(r'[1-9][0-9,]*円',s);return int(s[:-1].replace(',',''))
def parse_char(doc,race_no):
 sections=doc.xpath('//section[@id=$id]',id='race'+str(race_no));assert len(sections)==1;sec=sections[0]
 assert re.search(r'第\s*'+str(race_no)+r'\s*レース',sec.text_content())
 out={};tables=sec.xpath('.//table');used=[]
 for ti,tb in enumerate(tables):
  th=tb.xpath('.//tr/th[contains(@class,"bg-accent")]')
  if not th:continue
  title=norm(th[0].text_content())
  if title not in CHAR:continue
  used.append(title)
  for ri,tr in enumerate(tb.xpath('.//tr[td]')):
   td=tr.xpath('./td');label='ワイド' if title=='ワイド' else norm(td[0].text_content());assert label in CHAR[title]
   cls=CHAR[title][label];code,_,n,ordered=MAP[cls]
   for pi,p in enumerate(tr.xpath('.//p')):
    sp=p.xpath('./span');assert len(sp)==1;token=(p.text or '').strip();pay=sp[0].text_content();cars=combo(token,n,ordered,code.startswith('frame_'));key=(code,cars);assert key not in out
    out[key]={'payout_yen':amount(pay),'raw_combination':token,'raw_amount':pay,'raw_text':p.text_content(),'DOM':{'section':'race'+str(race_no),'table_index':ti,'row_index':ri,'paragraph_index':pi}}
 assert sorted(used)==sorted(CHAR)
 return out
def parse_nk(doc,race_key):
 assert doc.xpath('//input[@name="plp-raceid"]/@value')==[race_key]
 ts=doc.xpath('//table[contains(concat(" ",normalize-space(@class)," ")," Payout_Detail_Table ")]');assert len(ts)==1;out={}
 for ri,tr in enumerate(ts[0].xpath('.//tr[td]')):
  classes=tr.get('class','').split();assert len(classes)==1 and classes[0] in MAP;cls=classes[0];code,ticket,n,ordered=MAP[cls]
  cells={c.get('class'):c for c in tr.xpath('./td')};assert set(cells)=={'Result','Payout','Ninki'}
  token=cells['Result'].text_content().strip();pay=cells['Payout'].text_content().strip();pop=cells['Ninki'].text_content().strip();assert re.fullmatch(r'[1-9][0-9]*人気',norm(pop))
  cars=combo(token,n,ordered,code.startswith('frame_'));key=(code,cars);assert key not in out
  out[key]={'ticket_type':ticket,'payout_yen':amount(pay),'popularity':int(norm(pop)[:-2]),'raw_combination':token,'raw_amount':pay,'raw_popularity':pop,'raw_text':tr.text_content(),'DOM':{'table_class':'Payout_Detail_Table','row_index':ri}}
 return out
def fragment(event,sources):
 ident=event['identity'];key=ident['canonical_key'];refs=event['component_raw_provenance'];charraw=sources.raw(refs['result']);nkraw=sources.raw(refs['independent_result']);cardraw=sources.raw(refs['card']);card=json.loads(cardraw)
 assert (card['box']['open_day'],card['box']['vel_code'],card['box']['race_num'])==(ident['day'],ident['venue_code'],ident['race_no'])
 cdoc=html.fromstring(charraw.decode('utf-8','strict'));assert cdoc.xpath('//section[@id=$id]//a[@href=$href]',id='race'+str(ident['race_no']),href='/keirin/athletes/'+ident['day']+'/'+ident['venue_code']+'/'+str(ident['race_no'])+'?odds=1');char=parse_char(cdoc,ident['race_no']);nk=parse_nk(html.fromstring(nkraw.decode('utf-8','strict')),event['observed_provider_keys']['netkeiba_race_id'])
 assert set(char)==set(nk);assert len(char)==9 and {c:sum(k[0]==c for k in char) for c,_,_,_ in MAP.values()}=={c:(3 if c=='wide' else 1) for c,_,_,_ in MAP.values()}
 rows=[];details=[]
 for code,cars in sorted(char):
  c=char[code,cars];n=nk[code,cars];assert c['payout_yen']==n['payout_yen'];ordered=code in ['frame_exacta','exacta','trifecta'];separator='-' if ordered else '='
  rows.append({'ticket_type':n['ticket_type'],'bet_code':code,'combination':separator.join(map(str,cars)),'payout_yen':n['payout_yen'],'popularity':n['popularity'],'status':'paid','raw_text':n['raw_text']})
  details.append({'bet_code':code,'numbers':list(cars),'chariloto':c,'netkeiba':n,'independently_verified_fields':['combination','payout_yen'],'popularity_verification':'netkeiba_explicit_single_source_only_not_independently_confirmed'})
 results=event['results'];top=[]
 for rank in ['1','2','3']:
  matches=[r['car_no'] for r in results if r['finish_text']==rank];assert len(matches)==1;top.append(matches[0])
 for code,nums in [('trifecta',tuple(top)),('trio',tuple(sorted(top))),('exacta',tuple(top[:2])),('quinella',tuple(sorted(top[:2])))]:assert (code,nums) in char
 assert {k[1] for k in char if k[0]=='wide'}=={tuple(sorted([top[i],top[j]])) for i,j in [(0,1),(0,2),(1,2)]}
 frames=[card['race_status']['waku'+str(v)+'_entry_num'] for v in top[:2]]
 assert ('frame_exacta',tuple(frames)) in char and ('frame_quinella',tuple(sorted(frames))) in char
 for market,code,nums in [('sanrentan','trifecta',top),('sanrenfuku','trio',sorted(top))]:
  selected=[x for x in event['markets'][market] if x['cars']==nums];assert len(selected)==1 and Decimal(selected[0]['odds_decimal'])*100==char[code,tuple(nums)]['payout_yen']
 return {'local_key':key,'identity':copy.deepcopy(ident),'payload_payouts_fragment':rows,'evidence':details,'raw_sources':{k:copy.deepcopy(refs[k]) for k in ['result','independent_result','card']},'status':'candidate_payload_payouts_fragment_only','complete_payload':False,'canonical_SQL_inserted':False}
def attach_verified_payout_fragment(base_bytes,overlay_bytes):
 """Attach payout fragment to a copy of the exact saved integrated candidate."""
 assert sha(base_bytes)=='b9ed0533f9c8a654236a4c4c8d9d275161acdd6c154ad6c3efbaa444cb307e3f'
 assert sha(overlay_bytes)=='b468330e89e778acc6cd481dcc3fab654d3a1bc56fcb8a5e202491439c4ccd33'
 base=json.loads(base_bytes);overlay=json.loads(overlay_bytes)
 assert len(base['events'])==len(overlay['events'])==14 and base['all_finals_complete'] is overlay['all_finals_complete'] is False
 for e,p in zip(base['events'],overlay['events']):
  assert e['identity']==p['identity'] and e['identity']['canonical_key']==p['local_key']
  assert p['complete_payload'] is False and p['canonical_SQL_inserted'] is False
  e['production_payload_fragments']={'payouts':copy.deepcopy(p['payload_payouts_fragment'])}
  e['payout_fragment_evidence']=copy.deepcopy(p)
 return base
def contract():
 c=sqlite3.connect((ROOT/'data/finals.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
 schema=c.execute("select sql from sqlite_master where name='races'").fetchone()[0]
 row=c.execute("select race_id,payload from races where category='A12' and json_array_length(json_extract(payload,'$.payouts'))>0 order by day,race_id limit 1").fetchone();c.close();p=json.loads(row[1]);sample=p['payouts'][0]
 assert set(sample)=={'ticket_type','bet_code','combination','payout_yen','popularity','status','raw_text'}
 return {'races_SQL_schema':schema,'pipeline_source_sha256':PINS['pipeline.py'],'sample_legacy_race_id':row[0],'sample_payload_sha256':sha(row[1].encode()),'sample_payload_top_level_fields':sorted(p),'observed_payout_fields':sorted(sample),'compatibility_scope':'payload.payouts field names and observed value types only; full payload SQL adapter and completion audit remain pending'}
def main():
 d=candidate();s=Sources();out={'schema':'rainbow_payout_fragment_overlay_v1','base_integrated_candidate_member_sha256':'b9ed0533f9c8a654236a4c4c8d9d275161acdd6c154ad6c3efbaa444cb307e3f','events':[fragment(e,s) for e in d['events']],'contract':contract(),'input_pins':PINS,'canonical_SQL_inserts':0,'new_completed_races':0,'all_finals_complete':False,'network_requests':0}
 raw=enc(out);OUT.mkdir(parents=True,exist_ok=True);p=OUT/'payout_fragments.tar.xz'
 with tarfile.open(p,'w:xz',preset=9) as t:m=tarfile.TarInfo('payout_fragments.json');m.size=len(raw);m.mtime=0;t.addfile(m,io.BytesIO(raw))
 with tarfile.open(p) as t:assert t.getnames()==['payout_fragments.json'] and t.extractfile(t.getmembers()[0]).read()==raw
 report={'status':'candidate_payload_payouts_fragment_only','events':14,'payout_rows':sum(len(e['payload_payouts_fragment']) for e in out['events']),'independently_verified_fields':['combination','payout_yen'],'popularity_single_source_retained':True,'contract':out['contract'],'archive':{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p.read_bytes()),'member':'payout_fragments.json','member_bytes':len(raw),'member_sha256':sha(raw)},'canonical_SQL_inserts':0,'new_completed_races':0,'full_payload_SQL_adapter_verified':False,'all_finals_complete':False}
 (OUT/'payout_fragment_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n');print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
