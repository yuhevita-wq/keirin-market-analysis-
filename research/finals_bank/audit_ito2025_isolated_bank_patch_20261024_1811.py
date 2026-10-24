"""Independent all-row/catalog delta audit. Does not call the patch producer."""
import copy,hashlib,json,sqlite3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/scope_review_1811';RID='3720251211040012'
def sha(b):return hashlib.sha256(b).hexdigest()
def encode(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def tables(c):
 result={}
 for (name,) in c.execute("select name from sqlite_master where type='table' order by name"):
  cols=c.execute('pragma table_info('+name+')').fetchall();rows=c.execute('select * from '+name+' order by '+','.join(str(i+1) for i in range(len(cols)))).fetchall();h=hashlib.sha256()
  for row in rows:
   raw=encode(row);h.update(len(raw).to_bytes(8,'big')+raw)
  result[name]={'rows':len(rows),'length_framed_sorted_rows_sha256':h.hexdigest()}
 return result
def main():
 db=ROOT/'data/finals.sqlite';catalog_bytes=(ROOT/'data/historical_bank_catalog.json').read_bytes();delta_bytes=(ROOT/'data/bank_sources_1227/ito2025_candidate_bank_delta.json').read_bytes()
 assert sha(db.read_bytes())=='04c4de11b6937c3b3a499f042410786cfdde4c2892ca0af46999ca059c5e2d91';assert sha(catalog_bytes)=='14a7cb0d95b566dce54927c81f534a077991be307bc0eebc21ba572c4fddaba3';assert sha(delta_bytes)=='53348cb1c4e5a385370563c727de8f07ea61a85412c9102c8b8dfe46d97690fc'
 catalog=json.loads(catalog_bytes);record=json.loads(delta_bytes)['bank_record'];proposed=copy.deepcopy(catalog);proposed['records'].append(copy.deepcopy(record));assert len(catalog['records'])==55 and len(proposed['records'])==56
 c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True);assert c.execute('pragma integrity_check').fetchall()==[('ok',)];before=tables(c);memory=sqlite3.connect(':memory:');c.backup(memory)
 oldcovered=[];newcovered=[];changed=[]
 for rid,day,raw in c.execute("select race_id,day,payload from races where status!='excluded' order by race_id"):
  payload=json.loads(raw);track=payload['race']['track'];old=[x for x in catalog['records'] if (x['venue_code'],x['track'])==(rid[:2],track) and x['valid_from']<=day<x['valid_until_exclusive']];new=[x for x in proposed['records'] if (x['venue_code'],x['track'])==(rid[:2],track) and x['valid_from']<=day<x['valid_until_exclusive']];assert len(old)<=1 and len(new)<=1
  if old:oldcovered.append(rid)
  if new:newcovered.append(rid)
  if old!=new:changed.append(rid)
 assert changed==[RID] and len(oldcovered)==230 and len(newcovered)==231 and set(newcovered)-set(oldcovered)=={RID}
 row=c.execute('select * from races where race_id=?',(RID,)).fetchone();payload=json.loads(row[7]);assert payload.get('bank') is None and payload['race']['race_date']==row[1]=='2025-12-14' and str(payload['race']['race_no'])=='12' and payload['race']['track']=='伊東競輪'
 bank=dict(bank_version_id=record['bank_version_id'],venue_code='37',track='伊東競輪',race_date='2025-12-14',valid_from='2025-12-11',valid_until_exclusive='2025-12-15',geometry=record['geometry'],evidence=record['sources'],validity_basis=record['validity_basis'],status='verified_within_source_event')
 assert record['valid_from']==bank['valid_from'] and record['valid_until_exclusive']==bank['valid_until_exclusive'];payload['bank']=bank;after_raw=json.dumps(payload,ensure_ascii=False);memory.execute('update races set payload=? where race_id=?',(after_raw,RID));memory.commit();assert memory.execute('pragma integrity_check').fetchall()==[('ok',)];after=tables(memory)
 for name in before:
  assert before[name]['rows']==after[name]['rows']
  if name!='races':assert before[name]==after[name]
 expected_changes=[{'race_id':RID,'before_payload_sha256':sha(row[7].encode()),'after_payload_sha256':sha(after_raw.encode()),'bank':bank}]
 expected={'before_tables':before,'after_tables':after,'changes':expected_changes,'changed_race_rows':[RID],'existing_SQL_bank_before':230,'isolated_SQL_bank_after':231,'existing_SQL_bank_unresolved_before':5123,'isolated_SQL_bank_unresolved_after':5122,'canonical_SQL_bank_verified':230,'canonical_SQL_bank_unresolved':5123,'canonical_DB_changed':False,'catalog_file_changed':False,'canonical_SQL_inserts':0,'new_completed_races':0,'all_finals_complete':False,'complete_payload_insertion_audit':False,'second_SQLite_artifact_saved':False,'proposed_catalog_canonical_JSON_sha256':sha(encode(proposed)),'proposed_catalog_added_record':record}
 def validate(report):
  for k,v in expected.items():assert report[k]==v,k
 report=json.loads((OUT/'isolated_bank_patch_review.json').read_bytes());validate(report);mutations=[]
 for name,key,value in [('false_canonical_apply','canonical_DB_changed',True),('false_full_completion','all_finals_complete',True),('extra_row','changed_race_rows',[RID,'3720251211040009']),('inflated_coverage','isolated_SQL_bank_after',232),('dropped_unresolved','isolated_SQL_bank_unresolved_after',0),('false_catalog_applied','catalog_file_changed',True),('false_complete_insertion_audit','complete_payload_insertion_audit',True)]:
  bad=copy.deepcopy(report);bad[key]=value
  try:validate(bad)
  except AssertionError:mutations.append(name)
  else:raise AssertionError(name)
 for name,change in [('geometry_changed',lambda x:x['changes'][0]['bank']['geometry'].update(circumference_m=333.33)),('source_SHA_changed',lambda x:x['changes'][0]['bank']['evidence'][0].update(sha256='0'*64)),('period_extended',lambda x:x['changes'][0]['bank'].update(valid_until_exclusive='2026-01-01'))]:
  bad=copy.deepcopy(report);change(bad)
  try:validate(bad)
  except AssertionError:mutations.append(name)
  else:raise AssertionError(name)
 c.close();memory.close();assert sha(db.read_bytes())==report['canonical_DB_sha256'] and sha((ROOT/'data/historical_bank_catalog.json').read_bytes())==report['original_catalog_sha256']
 result={'status':'independent_isolated_bank_patch_audit_passed','review_sha256':sha((OUT/'isolated_bank_patch_review.json').read_bytes()),'changed_race_rows':[RID],'isolated_SQL_bank_after':231,'isolated_SQL_bank_unresolved_after':5122,'canonical_SQL_bank_verified':230,'canonical_SQL_bank_unresolved':5123,'all5table_counts_preserved':True,'non_races_tables_byte_value_digests_unchanged':True,'producer_mapping_or_selector_imported':False,'mutations_rejected':mutations,'canonical_DB_changed':False,'catalog_file_changed':False,'canonical_SQL_inserts':0,'new_completed_races':0,'all_finals_complete':False,'complete_payload_insertion_audit':False,'second_SQLite_artifact_saved':False,'canonical_save_pending':True}
 (OUT/'independent_isolated_bank_patch_audit.json').write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
