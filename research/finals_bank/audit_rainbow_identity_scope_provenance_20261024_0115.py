"""Separate raw-byte audit of partial candidate proof locators; no SQL writes."""
from pathlib import Path
import copy,gzip,hashlib,io,json,re,tarfile,unicodedata
import fitz
from lxml import html
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/scope_review_0115'
def digest(b):return hashlib.sha256(b).hexdigest()
def clean(s):return ''.join(unicodedata.normalize('NFKC',s).split())
def audit(report):
 scope_bytes=(ROOT/'data/scope_review_0457/rainbow_scope_independent_review.json').read_bytes()
 assert digest(scope_bytes)==report['scope_sha256']=='53552457be52018be9f3b6c8c9258e1acb9fa92d8bab5568431dc06605591bdb'
 original=json.loads(scope_bytes);assert report['all_finals_complete'] is False and report['canonical_SQL_inserts']==report['new_completed_races']==0
 assert len(report['events'])==len(original['events'])==14
 archive=(OUT/'identity_source_recovery.tar.xz').read_bytes()
 with tarfile.open(fileobj=io.BytesIO(archive)) as t:
  assert len(t.getnames())==len(set(t.getnames()))==5
  members={n:t.extractfile(n).read() for n in t.getnames()}
 counts=[0,0,0];pref_gaps=[]
 for ev,source in zip(report['events'],original['events']):
  assert tuple(ev[k] for k in ('day','venue_code','race_no'))==tuple(source[k] for k in ('day','venue_code','race_no'))
  assert len(ev['entries'])==len(source['entry_identities'])==9
  for e,old in zip(ev['entries'],source['entry_identities']):
   assert all(e[k]==old[k] for k in ('car_no','registration_number','name','term','prefecture'))
   loc=e['locator'];reg=e['registration_number']
   if e['status']=='identity_columns_bound_to_saved_raw':
    assert isinstance(loc,dict)
    assert any(s.get('raw_sha256')==loc['raw_sha256'] and s['source_url']==e['source_url'] for s in old['official_sources'])
    blob=(ROOT/loc['path']).read_bytes();assert len(blob)==loc['stored_bytes'] and digest(blob)==loc['stored_sha256']
    raw=gzip.decompress(blob);assert digest(raw)==loc['raw_sha256'] and len(raw)==loc['raw_bytes']
    rows=html.fromstring(raw).xpath('//tr');cells=rows[e['row_index']].xpath('./td');a=[clean(c.text_content()) for c in cells]
    assert a[3]==reg and clean(e['name']) in a[2] and a[5]==str(e['term'])+'期'
    pref=a[6]
    if cells[2].get('rowspan')=='2':
     paired=rows[e['row_index']+1].xpath('./td');assert len(paired)==11;pref=clean(paired[1].text_content())
    assert e['raw_prefecture']==pref;same=pref in {clean(e['prefecture']),clean(e['prefecture'])+'県',clean(e['prefecture'])+'府',clean(e['prefecture'])+'都'};assert e['prefecture_raw_matches_review']==same
    assert e['verified_columns']==['registration_number','name','term']+(['prefecture'] if same else [])
    if not same:
     assert e['prefecture_provenance_blocker'];pref_gaps.append({'day':ev['day'],'car_no':e['car_no'],'registration_number':reg,'selected_raw_prefecture':pref,'original_review_prefecture':e['prefecture']})
    counts[0]+=1
   elif e['status']=='identity_columns_bound_to_recovered_allocation_raw':
    assert digest(archive)==loc['archive_sha256'];raw=members[loc['member']];assert digest(raw)==loc['sha256'] and len(raw)==loc['bytes']
    with fitz.open(stream=raw,filetype='pdf') as doc:
     proof=e['pdf_evidence'];lines=doc[proof['page_one_based']-1].get_text().splitlines();j=proof['line_zero_based']
     assert lines[j:j+3]==proof['original_lines'];assert lines[j].strip()==str(int(reg)) and clean(lines[j+1])==clean(e['name']) and clean(lines[j+2])==clean(e['prefecture'])
     assert str(e['term'])+'期' in clean(doc[0].get_text())
    counts[1]+=1
   else:
    assert e['status']=='raw_transfer_source_unresolved' and reg=='015612' and loc is None and e['existing_review_sources']==old['official_sources']
    failure=json.loads(members['transfer2025.json']);assert '403' in failure['error'] and e['blockers'];counts[2]+=1
  assert len(ev['official_scope_raw'])==len(source['official_scope_evidence'])
  for loc,proof in zip(ev['official_scope_raw'],source['official_scope_evidence']):
   assert loc['original_scope_proof']==proof and loc['path']=='data/scope_review_0457/'+proof['path']
   raw=(ROOT/loc['path']).read_bytes();assert digest(raw)==loc['sha256']==proof['sha256'] and len(raw)==loc['bytes']
 assert counts==[107,18,1]
 return {'identity_raw_bound':125,'identity_raw_unresolved':1,'scope_raw_events_bound':14,'selected_registry_prefecture_gaps':pref_gaps,'complete_row_export_authorized':False,'canonical_SQL_inserts':0,'all_finals_complete':False}
def main():
 report=json.loads((OUT/'identity_scope_provenance_review.json').read_text());result=audit(report);mutations=[]
 def reject(name,fn):
  x=copy.deepcopy(report);fn(x)
  try:audit(x)
  except (AssertionError,KeyError,IndexError,ValueError):mutations.append(name);return
  raise AssertionError('mutation accepted '+name)
 reject('explicit_ID',lambda x:x['events'][0]['entries'][0].update(registration_number='099999'))
 reject('raw_locator_sha',lambda x:x['events'][0]['entries'][0]['locator'].update(raw_sha256='0'*64))
 reject('raw_prefecture_false_promotion',lambda x:x['events'][0]['entries'][0].update(raw_prefecture='未確認'))
 reject('allocation_name',lambda x:next(e for v in x['events'] for e in v['entries'] if 'pdf_evidence' in e)['pdf_evidence']['original_lines'].__setitem__(1,'別人'))
 reject('scope_raw_sha',lambda x:x['events'][0]['official_scope_raw'][0].update(sha256='0'*64))
 reject('unresolved_promoted',lambda x:next(e for v in x['events'] for e in v['entries'] if e['status']=='raw_transfer_source_unresolved').update(status='identity_columns_bound_to_saved_raw'))
 reject('completion_flag',lambda x:x.update(all_finals_complete=True))
 result['mutations_rejected']=mutations
 (OUT/'independent_identity_scope_provenance_audit.json').write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='selected_registry_prefecture_gaps'},ensure_ascii=False));print('prefecture gaps:',len(result['selected_registry_prefecture_gaps']))
if __name__=='__main__':main()
