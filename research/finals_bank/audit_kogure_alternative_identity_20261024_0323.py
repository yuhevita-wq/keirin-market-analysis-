"""Independent PDF-word geometry and historic-card identity audit; no SQL writes."""
from pathlib import Path
import copy,hashlib,io,json,tarfile
import fitz
from lxml import html
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/scope_review_0323'
def sha(b):return hashlib.sha256(b).hexdigest()
def plain(s):return ''.join(s.split())
def audit(review):
 target=review['target'];assert target=={'local_key':'keirin:2024-06-30:23:09','day':'2024-06-30','venue_code':'23','race_no':9,'car_no':9,'registration_number':'015612','name':'小榑佑弥','term':'121','prefecture_at_event':'新潟'}
 assert review['all_finals_complete'] is False and review['canonical_SQL_inserts']==review['new_completed_races']==0 and review['current_performance_imported'] is False
 assert review['original_transfer_page_body_status']=='still_unavailable_HTTP403_not_recovered_not_retried'
 assert review['candidate_identity_core_raw_coverage_after_delta']=={'previously_bound':125,'alternative_primary_resolved':1,'total_bound':126,'unresolved_core_identity':0}
 loc=review['official_allocation']['locator'];a=loc['archive'];b=(ROOT/a['path']).read_bytes();assert sha(b)==a['sha256'] and len(b)==a['bytes']
 with tarfile.open(fileobj=io.BytesIO(b)) as t:
  assert len(t.getnames())==len(set(t.getnames()))==2;pdf=t.extractfile(loc['member']).read();metadata=json.load(t.extractfile('allocation121.meta.json'))
 assert sha(pdf)==loc['sha256']=='f3a87ac9c4528d8f7a3752a325fd50f4171ccc3ae2f894add18e2436bec04155' and len(pdf)==loc['bytes']==86312
 assert metadata['status']==200 and metadata['sha256']==sha(pdf) and metadata['bytes']==len(pdf) and metadata['url']==review['official_allocation']['source_url']
 document=fitz.open(stream=pdf,filetype='pdf');assert len(document)==1;page=document[0];assert plain(page.get_text()).startswith('新登録選手(121期)2022年8月開催別あっせん一覧')
 words=page.get_text('words');matches=[]
 for word in words:
  if word[4]=='15612':
   same=sorted([w for w in words if abs((w[1]+w[3])/2-(word[1]+word[3])/2)<1],key=lambda w:w[0]);i=same.index(word)
   assert plain(same[i+1][4])=='小榑佑弥' and same[i+2][4]=='新潟'
   matches.append({'registration_bbox':list(word[:4]),'name_bbox':list(same[i+1][:4]),'prefecture_bbox':list(same[i+2][:4])})
 assert len(matches)==2
 original=review['official_allocation']['original_identity_rows'];lines=page.get_text().splitlines();assert len(original)==2
 for p in original:
  assert p['page_one_based']==1 and lines[p['line_zero_based']:p['line_zero_based']+3]==p['original_lines']
  assert p['original_lines']==['15612','小　榑　佑　弥','新潟']
 pinned=(ROOT/'data/scope_review_0115/identity_scope_provenance_review.json').read_bytes();assert sha(pinned)==review['prior_identity_provenance_sha256']=='4b4819ab2b8da379f0835490bbde6e59defdfbad3bd1d1eb075767d5bf71083f'
 old=json.loads(pinned);assert old['counts']==review['prior_0115_identity_provenance_counts_retained'];assert old['counts']['saved_gzip_identity_rows']+old['counts']['recovered_allocation_identity_rows']==125
 scope=(ROOT/'data/scope_review_0457/rainbow_scope_independent_review.json').read_bytes();assert sha(scope)==review['original_review_scope_sha256']=='53552457be52018be9f3b6c8c9258e1acb9fa92d8bab5568431dc06605591bdb'
 pool={}
 for key,ref in review['event_time_sources'].items():
  if 'archive' in ref:
   ar=ref['archive'];blob=(ROOT/ar['path']).read_bytes();assert sha(blob)==ar['sha256'] and len(blob)==ar['bytes']
   with tarfile.open(fileobj=io.BytesIO(blob)) as t:raw=t.extractfile(ref['member']).read()
  else:raw=(ROOT/ref['path']).read_bytes()
  assert sha(raw)==ref['sha256'] and len(raw)==ref['bytes'];pool[key]=raw
 card=json.loads(pool['card']);p=card['race_athletes']['9'];assert (p['open_day'],p['vel_code'],p['race_num'],p['entry_num'],p['un_id'],plain(p['name']),p['graduate_period'],p['pref_name'])==('2024-06-30','23',9,9,15612,'小榑佑弥','121','新潟')
 op=html.fromstring(pool['independent_card']);player=op.xpath('//a[@href="./PlayerDetail.do?playerCd=015612"]');assert len(player)==1;row=player[0]
 while row.tag!='tr':row=row.getparent()
 cells=row.xpath('./td');assert row.xpath('./td[@class="no9"]') and plain(cells[-8].text_content())=='小榑佑弥24歳／121期' and cells[-7].text_content()=='新潟'
 for k,v in [('joCode','23'),('kaisaiBi','20240630'),('raceNo','9')]:assert op.xpath('//input[@name=$k and @value=$v]',k=k,v=v)
 nk=html.fromstring(pool['independent_result']);assert nk.xpath('//input[@name="plp-raceid" and @value="202406302309"]');row=nk.xpath('//table[@id="All_Result_Table"]//tr[.//a[@href="https://keirin.netkeiba.com/db/profile/?id=15612"]]');assert len(row)==1 and row[0].xpath('.//img[@alt="小榑佑弥"]')
 return {'alternative_official_identity_verified':1,'PDF_explicit_identity_row_occurrences':2,'PDF_independent_geometry':matches,'original_0115_report_preserved':True,'previous125_plus_alternative1_core_identity_raw_coverage':126,'original_transfer_raw_body_recovered':False,'affiliation_interval_inferred':False,'historic_card_performance_changed':False,'canonical_SQL_inserts':0,'new_completed_races':0,'all_finals_complete':False}
def main():
 review=json.loads((OUT/'alternative_identity_review.json').read_bytes());result=audit(review);rejects=[]
 def reject(name,fn):
  x=copy.deepcopy(review);fn(x)
  try:audit(x)
  except (AssertionError,ValueError,KeyError,IndexError):rejects.append(name);return
  raise AssertionError('mutation accepted '+name)
 reject('explicit_registration',lambda x:x['target'].update(registration_number='015613'))
 reject('name',lambda x:x['target'].update(name='別人'))
 reject('term',lambda x:x['target'].update(term='123'))
 reject('historical_prefecture',lambda x:x['target'].update(prefecture_at_event='群馬'))
 reject('event_date',lambda x:x['target'].update(day='2025-10-11'))
 reject('PDF_identity_row',lambda x:x['official_allocation']['original_identity_rows'][0]['original_lines'].__setitem__(2,'群馬'))
 reject('PDF_raw_sha',lambda x:x['official_allocation']['locator'].update(sha256='0'*64))
 reject('false_transfer_recovery',lambda x:x.update(original_transfer_page_body_status='recovered'))
 reject('completion_flag',lambda x:x.update(all_finals_complete=True))
 result['mutations_rejected']=rejects;(OUT/'independent_alternative_identity_audit.json').write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='PDF_independent_geometry'}))
if __name__=='__main__':main()
