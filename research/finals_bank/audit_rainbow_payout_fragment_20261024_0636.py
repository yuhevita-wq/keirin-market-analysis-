"""Independent saved-DOM audit and exact races-DDL fragment roundtrip.
This is not a full production-payload insertion audit.
"""
import copy,hashlib,io,itertools,json,re,sqlite3,tarfile,unicodedata
from decimal import Decimal
from pathlib import Path
from lxml import html
ROOT=Path(__file__).resolve().parents[1]
PINS={'pipeline.py': '122cd6a9b06821618e539e84b2fc98a031f6df065f2332c9a88e8b6b969b4d81', 'data/scope_review_0552/integrated_candidates.tar.xz': 'c1a030bc66b78fec4bda307046402f9db15fa67dd386845b8d19862cd3f29e3a', 'data/scope_review_0552/integrated_candidate_review.json': '8d41909097f61ab60de8490f64f74a4facd5cc4a0ca8a580ccd45906a951a3ad', 'data/scope_review_0552/independent_integrated_candidate_audit.json': 'c7656fc4864e85692afac82910348f132f6b62264dd00a3877975d158f008537'}
def h(b):return hashlib.sha256(b).hexdigest()
def js(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def compact(s):return ''.join(unicodedata.normalize('NFKC',s).split())
def arc(path):
 with tarfile.open(ROOT/path) as t:
  ns=t.getnames();assert len(ns)==len(set(ns));return {m.name:t.extractfile(m).read() for m in t.getmembers()}
cache={}
def source(ref):
 if 'archive' in ref:
  a=ref['archive'];b=(ROOT/a['path']).read_bytes();assert len(b)==a['bytes'] and h(b)==a['sha256']
  if a['path'] not in cache:cache[a['path']]=arc(a['path'])
  b=cache[a['path']][ref['member']]
 else:b=(ROOT/ref['path']).read_bytes()
 assert len(b)==ref['bytes'] and h(b)==ref['sha256'];return b
def parse_number_sequence(s,ordered,frame=False):
 vals=[int(x) for x in re.findall('[0-9]+',compact(s))];assert len(vals) in [2,3]
 assert all(1<=x<=(6 if frame else 9) for x in vals)
 if not frame:assert len(set(vals))==len(vals)
 return tuple(vals if ordered else sorted(vals))
def independent_expected(event):
 refs=event['component_raw_provenance'];c=html.fromstring(source(refs['result']).decode('utf-8'));n=html.fromstring(source(refs['independent_result']).decode('utf-8'));card=json.loads(source(refs['card']));ident=event['identity'];rno=ident['race_no']
 assert n.xpath('//input[@name="plp-raceid"]/@value')==[event['observed_provider_keys']['netkeiba_race_id']]
 sec=c.xpath('//section[@id="race'+str(rno)+'"]');assert len(sec)==1
 section=sec[0];assert section.xpath('.//a[@href=$href]',href='/keirin/athletes/'+ident['day']+'/'+ident['venue_code']+'/'+str(rno)+'?odds=1');head=section.xpath('.//p[contains(concat(" ",normalize-space(@class)," ")," text-huge ")]');assert any(re.search(r'第\s*'+str(rno)+r'\s*レース',v.text_content()) for v in head)
 char={};nk={};labels={('2枠連','複'):'frame_quinella',('2枠連','単'):'frame_exacta',('2車連','複'):'quinella',('2車連','単'):'exacta',('3連勝','複'):'trio',('3連勝','単'):'trifecta',('ワイド','ワイド'):'wide'}
 ordered={'frame_exacta','exacta','trifecta'}
 for tb in section.xpath('.//table'):
  headers=tb.xpath('.//th');title=compact(headers[0].text_content()) if headers else ''
  if title not in {'2枠連','2車連','3連勝','ワイド'}:continue
  for row in tb.xpath('.//tr[td]'):
   cells=row.xpath('./td');sub='ワイド' if title=='ワイド' else compact(cells[0].text_content());code=labels[title,sub]
   for p in row.xpath('.//p'):
    spans=p.xpath('./span');assert len(spans)==1;combo=(p.text or '').strip();amount=spans[0].text_content();assert re.fullmatch('[0-9,]+円',compact(amount));nums=parse_number_sequence(combo,code in ordered,code.startswith('frame_'));key=(code,nums);assert key not in char;char[key]=(int(compact(amount)[:-1].replace(',','')),combo,amount,p.text_content())
 types={'Wakuren':('frame_quinella','2枠複'),'Wakutan':('frame_exacta','2枠単'),'Umaren':('quinella','2車複'),'Umatan':('exacta','2車単'),'Wide':('wide','ワイド'),'Fuku3':('trio','3連複'),'Tan3':('trifecta','3連単')}
 tables=n.xpath('//table[@summary="払戻し"]');assert len(tables)==1
 for row in tables[0].xpath('.//tr[td]'):
  code,ticket=types[row.get('class')];combo=row.xpath('./td[@class="Result"]')[0].text_content().strip();amount=row.xpath('./td[@class="Payout"]')[0].text_content().strip();pop=row.xpath('./td[@class="Ninki"]')[0].text_content().strip();assert re.fullmatch('[0-9,]+円',compact(amount)) and re.fullmatch('[0-9]+人気',compact(pop));nums=parse_number_sequence(combo,code in ordered,code.startswith('frame_'));key=(code,nums);assert key not in nk;nk[key]=(int(compact(amount)[:-1].replace(',','')),int(compact(pop)[:-2]),ticket,combo,amount,pop,row.text_content())
 assert set(char)==set(nk) and len(char)==9;assert all(char[k][0]==nk[k][0] for k in char)
 assert {code:sum(k[0]==code for k in char) for code,_ in types.values()}=={code:(3 if code=='wide' else 1) for code,_ in types.values()}
 top=[next(r['car_no'] for r in event['results'] if r['finish_text']==str(rank)) for rank in [1,2,3]]
 expected_codes={'trifecta':{tuple(top)},'trio':{tuple(sorted(top))},'exacta':{tuple(top[:2])},'quinella':{tuple(sorted(top[:2]))},'wide':{tuple(sorted(c)) for c in itertools.combinations(top,2)}}
 fs=[card['race_status']['waku'+str(car)+'_entry_num'] for car in top[:2]];expected_codes.update(frame_exacta={tuple(fs)},frame_quinella={tuple(sorted(fs))})
 for code,values in expected_codes.items():assert {k[1] for k in char if k[0]==code}==values
 for market,code,cars in [('sanrentan','trifecta',top),('sanrenfuku','trio',sorted(top))]:
  prices=[a for a in event['markets'][market] if a['cars']==cars];assert len(prices)==1 and Decimal(prices[0]['odds_decimal'])*100==char[code,tuple(cars)][0]
 return char,nk
def check(doc,base):
 assert doc['schema']=='rainbow_payout_fragment_overlay_v1' and doc['base_integrated_candidate_member_sha256']=='b9ed0533f9c8a654236a4c4c8d9d275161acdd6c154ad6c3efbaa444cb307e3f'
 assert doc['input_pins']==PINS and doc['all_finals_complete'] is False and doc['canonical_SQL_inserts']==doc['new_completed_races']==doc['network_requests']==0 and len(doc['events'])==14
 total=0
 for e,original in zip(doc['events'],base['events']):
  assert e['local_key']==original['identity']['canonical_key'] and e['identity']==original['identity'];assert e['status']=='candidate_payload_payouts_fragment_only' and e['complete_payload'] is False and e['canonical_SQL_inserted'] is False
  assert e['raw_sources']=={k:original['component_raw_provenance'][k] for k in ['result','independent_result','card']}
  char,nk=independent_expected(original);expected=[]
  for code,nums in sorted(char):
   amount,pop,ticket,token,pay,poptoken,txt=nk[code,nums];sep='-' if code in {'exacta','trifecta','frame_exacta'} else '='
   expected.append({'ticket_type':ticket,'bet_code':code,'combination':sep.join(map(str,nums)),'payout_yen':amount,'popularity':pop,'status':'paid','raw_text':txt})
  assert e['payload_payouts_fragment']==expected and len(e['evidence'])==9
  for ev,(code,nums) in zip(e['evidence'],sorted(char)):
   assert ev['bet_code']==code and ev['numbers']==list(nums) and ev['independently_verified_fields']==['combination','payout_yen'] and ev['popularity_verification']=='netkeiba_explicit_single_source_only_not_independently_confirmed'
   cc=ev['chariloto'];nn=ev['netkeiba'];assert (cc['payout_yen'],cc['raw_combination'],cc['raw_amount'],cc['raw_text'])==char[code,nums];assert (nn['payout_yen'],nn['popularity'],nn['ticket_type'],nn['raw_combination'],nn['raw_amount'],nn['raw_popularity'],nn['raw_text'])==nk[code,nums]
   assert cc['DOM']['section']=='race'+str(original['identity']['race_no']) and nn['DOM']['table_class']=='Payout_Detail_Table'
  total+=len(expected)
 assert total==126;return total
def main():
 for p,digest in PINS.items():assert h((ROOT/p).read_bytes())==digest
 basebytes=arc('data/scope_review_0552/integrated_candidates.tar.xz')['integrated_candidates.json'];assert h(basebytes)=='b9ed0533f9c8a654236a4c4c8d9d275161acdd6c154ad6c3efbaa444cb307e3f';base=json.loads(basebytes)
 report=json.loads((ROOT/'data/scope_review_0636/payout_fragment_review.json').read_bytes());a=report['archive'];ab=(ROOT/a['path']).read_bytes();assert h(ab)==a['sha256'] and len(ab)==a['bytes'];members=arc(a['path']);assert list(members)==[a['member']];payload=members[a['member']];assert h(payload)==a['member_sha256'] and len(payload)==a['member_bytes'];doc=json.loads(payload);rows=check(doc,base)
 original=sqlite3.connect((ROOT/'data/finals.sqlite').resolve().as_uri()+'?mode=ro',uri=True);schema=original.execute("select sql from sqlite_master where name='races'").fetchone()[0];sample=original.execute('select payload from races where race_id=?',(doc['contract']['sample_legacy_race_id'],)).fetchone()[0];original.close();assert h(sample.encode())==doc['contract']['sample_payload_sha256'];assert schema==doc['contract']['races_SQL_schema'];assert sorted(json.loads(sample))==doc['contract']['sample_payload_top_level_fields'];assert sorted(json.loads(sample)['payouts'][0])==doc['contract']['observed_payout_fields'];assert doc['contract']['pipeline_source_sha256']==PINS['pipeline.py']
 import adapt_rainbow_payout_fragment_0636 as adapter
 attached=adapter.attach_verified_payout_fragment(basebytes,payload)
 for e,old,fragment in zip(attached['events'],base['events'],doc['events']):
  assert e.pop('production_payload_fragments')=={'payouts':fragment['payload_payouts_fragment']}
  assert e.pop('payout_fragment_evidence')==fragment and e==old
 assert attached==base
 c=sqlite3.connect(':memory:');c.execute(schema)
 for e in doc['events']:
  ident=e['identity'];payload={'payouts':e['payload_payouts_fragment'],'fragment_status':'payouts_only_not_complete'};c.execute('insert into races values(?,?,?,?,?,?,?,?,?,?)',(e['local_key'],ident['day'],None,None,None,None,'partial_fragment',js(payload).decode(),None,None))
 saved=c.execute('select status,payload from races order by rowid').fetchall();assert len(saved)==14 and all(status=='partial_fragment' for status,_ in saved);assert [json.loads(p)['payouts'] for _,p in saved]==[e['payload_payouts_fragment'] for e in doc['events']];c.close()
 tests={
 'payout_yen':lambda x:x['events'][0]['payload_payouts_fragment'][0].__setitem__('payout_yen',1),
 'combination':lambda x:x['events'][0]['payload_payouts_fragment'][0].__setitem__('combination','1-1'),
 'popularity':lambda x:x['events'][0]['payload_payouts_fragment'][0].__setitem__('popularity',999),
 'drop_row':lambda x:x['events'][0]['payload_payouts_fragment'].pop(),
 'wrong_race_section':lambda x:x['events'][0]['evidence'][0]['chariloto']['DOM'].__setitem__('section','race10'),
 'raw_source_SHA':lambda x:x['events'][0]['raw_sources']['result'].__setitem__('sha256','0'*64),
 'false_popularity_independence':lambda x:x['events'][0]['evidence'][0].__setitem__('popularity_verification','independently_verified'),
 'full_payload_promotion':lambda x:x['events'][0].__setitem__('complete_payload',True),
 'canonical_insert_promotion':lambda x:x['events'][0].__setitem__('canonical_SQL_inserted',True),
 'completion_flag':lambda x:x.__setitem__('all_finals_complete',True)}
 rejected=[]
 for name,change in tests.items():
  x=copy.deepcopy(doc);change(x)
  try:check(x,base)
  except (AssertionError,KeyError):rejected.append(name)
  else:raise AssertionError('mutation accepted '+name)
 out={'status':'independent_payout_fragment_audit_passed','events':14,'paid_rows':rows,'bet_types':7,'independently_verified_fields':['combination','payout_yen'],'popularity_independent_verification':False,'existing_races_DDL_in_memory_fragment_roundtrip':14,'saved_fragment_status':'partial_fragment','verified_attachment_and_lossless_original_recovery':True,'full_payload_SQL_adapter_verified':False,'complete_payload_insertion_audit':False,'canonical_SQL_inserts':0,'new_completed_races':0,'second_SQLite_artifact_saved':False,'mutations_rejected':rejected,'all_finals_complete':False}
 (ROOT/'data/scope_review_0636/independent_payout_fragment_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True)+'\n');print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
