"""Verify one historical bank patch in an in-memory backup. Canonical files are read-only."""
import copy,hashlib,json,sqlite3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/scope_review_1811'
DB_SHA='04c4de11b6937c3b3a499f042410786cfdde4c2892ca0af46999ca059c5e2d91'
CAT_SHA='14a7cb0d95b566dce54927c81f534a077991be307bc0eebc21ba572c4fddaba3'
DELTA_SHA='53348cb1c4e5a385370563c727de8f07ea61a85412c9102c8b8dfe46d97690fc'
RID='3720251211040012'
def sha(b):return hashlib.sha256(b).hexdigest()
def enc(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def snapshot(c):
 result={}
 for (name,) in c.execute("select name from sqlite_master where type='table' order by name"):
  assert name in ['races','days','banks','cancellations','excluded_races'];ncols=len(c.execute('pragma table_info('+name+')').fetchall());h=hashlib.sha256();n=0
  for row in c.execute('select * from '+name+' order by '+','.join(str(i+1) for i in range(ncols))):
   b=enc(row);h.update(len(b).to_bytes(8,'big'));h.update(b);n+=1
  result[name]={'rows':n,'length_framed_sorted_rows_sha256':h.hexdigest()}
 return result
def choose(records,rid,day,track):
 hits=[r for r in records if r['venue_code']==rid[:2] and r['track']==track and r['valid_from']<=day<r['valid_until_exclusive']];assert len(hits)<=1;return hits[0] if hits else None
def bank_value(record,day,track):
 return {'bank_version_id':record['bank_version_id'],'venue_code':record['venue_code'],'track':track,'race_date':day,'valid_from':record['valid_from'],'valid_until_exclusive':record['valid_until_exclusive'],'geometry':copy.deepcopy(record['geometry']),'evidence':copy.deepcopy(record['sources']),'validity_basis':record['validity_basis'],'status':record['review_status']}
def apply_memory(c,original_catalog,record):
 assert c.execute('pragma database_list').fetchall()==[(0,'main','')], 'Only in-memory database accepted'
 catalog=copy.deepcopy(original_catalog);assert not any(x['bank_version_id']==record['bank_version_id'] for x in catalog['records']);catalog['records'].append(copy.deepcopy(record))
 # Validate source-period collisions across the whole proposed catalog.
 for i,a in enumerate(catalog['records']):
  for b in catalog['records'][i+1:]:
   if a['venue_code']==b['venue_code']:assert max(a['valid_from'],b['valid_from'])>=min(a['valid_until_exclusive'],b['valid_until_exclusive'])
 changed=[];oldcount=newcount=0
 for rid,day,raw in c.execute("select race_id,day,payload from races where status!='excluded' order by race_id").fetchall():
  d=json.loads(raw);track=d['race']['track'];old=choose(original_catalog['records'],rid,day,track);new=choose(catalog['records'],rid,day,track);oldcount+=old is not None;newcount+=new is not None
  if old is not None:assert d.get('bank')==bank_value(old,day,track)
  if old==new:continue
  assert old is None and rid==RID and new==record and d.get('bank') is None
  assert (day,d['race']['race_date'],track,str(d['race']['race_no']),d['race']['race_type'])==('2025-12-14','2025-12-14','伊東競輪','12','Ｓ級決勝')
  d['bank']=bank_value(new,day,track);updated=json.dumps(d,ensure_ascii=False);c.execute('update races set payload=? where race_id=?',(updated,rid));changed.append({'race_id':rid,'before_payload_sha256':sha(raw.encode()),'after_payload_sha256':sha(updated.encode()),'bank':d['bank']})
 assert len(changed)==1 and oldcount==230 and newcount==231;c.commit();return catalog,changed
def main(pdf):
 db=ROOT/'data/finals.sqlite';cb=(ROOT/'data/historical_bank_catalog.json').read_bytes();dbytes=(ROOT/'data/bank_sources_1227/ito2025_candidate_bank_delta.json').read_bytes()
 assert db.stat().st_size==281206784 and sha(db.read_bytes())==DB_SHA and sha(cb)==CAT_SHA and sha(dbytes)==DELTA_SHA
 d=json.loads(dbytes);record=d['bank_record'];assert record['bank_version_id']=='37-ito20251211-reviewed';assert sha(pdf.read_bytes())==record['sources'][0]['sha256'] and pdf.stat().st_size==8404111
 source=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True);assert source.execute('pragma integrity_check').fetchall()==[('ok',)];before=snapshot(source);memory=sqlite3.connect(':memory:');source.backup(memory)
 catalog,changes=apply_memory(memory,json.loads(cb),record);after=snapshot(memory);assert memory.execute('pragma integrity_check').fetchall()==[('ok',)]
 for name in before:
  assert before[name]['rows']==after[name]['rows']
  if name!='races':assert before[name]==after[name]
 originals=source.execute('select * from races order by race_id').fetchall();modified=memory.execute('select * from races order by race_id').fetchall();diff=[]
 for a,b in zip(originals,modified):
  if a==b:continue
  assert a[0]==b[0]==RID and a[:7]==b[:7] and a[8:]==b[8:];p=json.loads(a[7]);q=json.loads(b[7]);newbank=q.pop('bank');assert p==q;assert newbank==changes[0]['bank'];diff.append(a[0])
 assert diff==[RID];source.close();memory.close();assert sha(db.read_bytes())==DB_SHA
 report={'schema':'isolated_historical_bank_patch_review_v1','status':'isolated_memory_patch_verified_not_canonical_applied','canonical_DB_sha256':DB_SHA,'original_catalog_sha256':CAT_SHA,'source_delta_sha256':DELTA_SHA,'proposed_catalog_canonical_JSON_sha256':sha(enc(catalog)),'proposed_catalog_added_record':record,'changes':changes,'before_tables':before,'after_tables':after,'changed_race_rows':diff,'existing_SQL_bank_before':230,'isolated_SQL_bank_after':231,'existing_SQL_bank_unresolved_before':5123,'isolated_SQL_bank_unresolved_after':5122,'canonical_SQL_bank_verified':230,'canonical_SQL_bank_unresolved':5123,'canonical_DB_changed':False,'catalog_file_changed':False,'canonical_SQL_inserts':0,'new_completed_races':0,'all_finals_complete':False,'complete_payload_insertion_audit':False,'second_SQLite_artifact_saved':False,'network_requests':0,'raw_source_provider_count':1}
 OUT.mkdir(exist_ok=True);(OUT/'isolated_bank_patch_review.json').write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n');print(json.dumps({'changed_rows':diff,'isolated_bank_coverage':231,'canonical_bank_coverage':230,'canonical_DB_changed':False}))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--pdf',type=Path,required=True);main(p.parse_args().pdf)
