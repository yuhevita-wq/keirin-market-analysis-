"""Resolve candidate identity/scope proof bytes; does not export completed races."""
from pathlib import Path
import gzip,hashlib,json,tarfile,re,unicodedata,io
from urllib.parse import urlparse,parse_qs
import fitz
from lxml import html
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/scope_review_0115'
SCOPE_SHA='53552457be52018be9f3b6c8c9258e1acb9fa92d8bab5568431dc06605591bdb'
PDF_SHA={'125':'a5fd6970eb56eee3ed514fc73e3ab21ee17c61aae51a610639ce428a76e2e428','127':'e5b948b40b4d669ff7a9808686251ec4c7299c486f262d4af3a028a9eab1c725'}
def sha(b):return hashlib.sha256(b).hexdigest()
def norm(s):return re.sub(r'\s+','',unicodedata.normalize('NFKC',s))
def pinned(path,h):
 b=(ROOT/path).read_bytes();assert sha(b)==h,(path,'SHA');return b
def run():
 scope=json.loads(pinned('data/scope_review_0457/rainbow_scope_independent_review.json',SCOPE_SHA))
 idx=json.loads((OUT/'registry_raw_index.json').read_text());manifest=json.loads((OUT/'raw_manifest.json').read_text())
 archive=(OUT/'identity_source_recovery.tar.xz').read_bytes();assert sha(archive)==manifest['archive_sha256']
 with tarfile.open(fileobj=io.BytesIO(archive)) as t:
  names=t.getnames();assert len(names)==len(set(names)) and set(names)==set(manifest['members'])
  files={n:t.extractfile(n).read() for n in names}
 for n,b in files.items():assert {'bytes':len(b),'sha256':sha(b)}==manifest['members'][n]
 pdfs={}
 for term,h in PDF_SHA.items():
  b=files['allocation'+term+'.raw'];assert sha(b)==h and b.startswith(b'%PDF-')
  doc=fitz.open(stream=b,filetype='pdf');pages=[p.get_text().splitlines() for p in doc]
  assert term+'期' in norm('\n'.join(pages[0]))
  pdfs[term]=pages
 events=[];counts={'saved_gzip_identity_rows':0,'recovered_allocation_identity_rows':0,'unresolved_identity_rows':0,'official_scope_events':0}
 for event in scope['events']:
  entries=[]
  for e in event['entry_identities']:
   reg=e['registration_number'];record={'car_no':e['car_no'],'registration_number':reg,'name':e['name'],'term':e['term'],'prefecture':e['prefecture'],'source_scope_pointer':{'day':event['day'],'venue_code':event['venue_code'],'car_no':e['car_no']}}
   sources=[s for s in e['official_sources'] if s.get('raw_sha256')]
   if sources:
    s=sources[0];loc=idx[s['raw_sha256']][0];stored=(ROOT/loc['path']).read_bytes()
    assert len(stored)==loc['stored_bytes'] and sha(stored)==loc['stored_sha256']
    raw=gzip.decompress(stored);assert len(raw)==loc['raw_bytes'] and sha(raw)==s['raw_sha256']==loc['raw_sha256']
    doc=html.fromstring(raw);matches=[]
    for rowno,tr in enumerate(doc.xpath('//tr')):
     cells=tr.xpath('./td');texts=[norm(c.text_content()) for c in cells]
     if len(texts)>=7 and texts[3]==reg:
      assert norm(e['name']) in texts[2] and texts[5]==str(e['term'])+'期',(event['day'],reg,e['name'],e['term'],texts)
      pref=texts[6]
      if cells[2].get('rowspan')=='2':
       paired=tr.getnext();assert paired is not None and len(paired.xpath('./td'))==11
       pref=norm(paired.xpath('./td')[1].text_content())
      matches.append((rowno,pref))
    assert len(matches)==1
    record.update(status='identity_columns_bound_to_saved_raw',locator=loc,source_url=s['source_url'],row_index=matches[0][0],raw_prefecture=matches[0][1],prefecture_raw_matches_review=(matches[0][1] in {norm(e['prefecture']),norm(e['prefecture'])+'県',norm(e['prefecture'])+'府',norm(e['prefecture'])+'都'}),verified_columns=['registration_number','name','term']+(['prefecture'] if matches[0][1] in {norm(e['prefecture']),norm(e['prefecture'])+'県',norm(e['prefecture'])+'府',norm(e['prefecture'])+'都'} else []),prefecture_provenance_blocker=(None if matches[0][1] in {norm(e['prefecture']),norm(e['prefecture'])+'県',norm(e['prefecture'])+'府',norm(e['prefecture'])+'都'} else 'selected_raw_prefecture_missing_or_different_original_review_not_overwritten'))
    counts['saved_gzip_identity_rows']+=1
   elif str(e['term']) in pdfs:
    term=str(e['term']);hits=[]
    for p,lines in enumerate(pdfs[term]):
     for j,line in enumerate(lines):
      if line.strip()==str(int(reg)):
       assert norm(lines[j+1])==norm(e['name']) and norm(lines[j+2])==norm(e['prefecture'])
       hits.append({'page_one_based':p+1,'line_zero_based':j,'original_lines':lines[j:j+3]})
    assert len(hits)==1
    record.update(status='identity_columns_bound_to_recovered_allocation_raw',locator={'archive':'data/scope_review_0115/identity_source_recovery.tar.xz','archive_sha256':manifest['archive_sha256'],'member':'allocation'+term+'.raw','sha256':PDF_SHA[term],'bytes':len(files['allocation'+term+'.raw'])},source_url=json.loads(files['allocation'+term+'.json'])['url'],pdf_evidence=hits[0],verified_columns=['registration_number','name','term','prefecture'])
    counts['recovered_allocation_identity_rows']+=1
   else:
    assert reg=='015612' and event['day']=='2024-06-30'
    record.update(status='raw_transfer_source_unresolved',locator=None,existing_review_sources=e['official_sources'],blockers=['original_dated_transfer_notice_body_not_located','single_recovery_attempt_HTTP403_not_success'],recovery_metadata_member='transfer2025.json')
    counts['unresolved_identity_rows']+=1
   entries.append(record)
  evidence=[]
  for proof in event['official_scope_evidence']:
   path='data/scope_review_0457/'+proof['path'];raw=pinned(path,proof['sha256']);doc=html.fromstring(raw.decode('utf-8','strict'));text=norm(doc.text_content())
   if 'required_text' in proof:
    assert all(norm(token) in text for token in proof['required_text']),(event['day'],proof,text[:500])
   if 'result_href' in proof:
    href=proof['result_href'];q=parse_qs(urlparse(href).query)
    assert q['KCD']==[str(event['venue_code'])] and q['KBI']==[event['day'].replace('-','')] and int(q['RNO'][0])==event['race_no']
    rows=doc.xpath('//tr[.//a[@href=$href]]',href=href);assert len(rows)==1
    assert norm(event['results'][0]['name']) in norm(rows[0].text_content())
   evidence.append({'path':path,'sha256':sha(raw),'bytes':len(raw),'original_scope_proof':proof})
  assert evidence;counts['official_scope_events']+=1
  events.append({'day':event['day'],'venue_code':event['venue_code'],'race_no':event['race_no'],'entries':entries,'official_scope_raw':evidence})
 assert counts=={'saved_gzip_identity_rows':107,'recovered_allocation_identity_rows':18,'unresolved_identity_rows':1,'official_scope_events':14},counts
 return {'schema':'rainbow_identity_scope_raw_provenance_v1','status':'partial_candidate_provenance_only','scope_sha256':SCOPE_SHA,'counts':counts,'events':events,'no_current_performance_imported':True,'new_completed_races':0,'canonical_SQL_inserts':0,'all_finals_complete':False,'remaining_blockers':['015612_transfer_raw_body_unresolved','history_complete_raw_locator_graph_pending','complete_payload_schema_readiness_and_independent_insertion_audit_pending','17_planned_slots_unresolved','13_candidate_historical_banks_unresolved','broader_scope_and_5123_existing_banks_unresolved']}
if __name__=='__main__':
 result=run();(OUT/'identity_scope_provenance_review.json').write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\n');print(json.dumps(result['counts']))
