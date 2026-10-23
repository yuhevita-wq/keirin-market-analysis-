"""Replace only SQLite's ZIP compressed record with a stronger lossless candidate.
Logical SQLite bytes, member names, all raw sources, and every other local record
are retained. Standard Python ZIP decoding + all-member SHA + normal RO SQL
comparison must pass before atomic replacement. Never use an unverified DB.
"""
import pathlib,zipfile,hashlib,copy,sqlite3,os,json,importlib.util,argparse
DB='finals_ai/data/finals.sqlite';EXPECTED='04c4de11b6937c3b3a499f042410786cfdde4c2892ca0af46999ca059c5e2d91'
def sha(b):return hashlib.sha256(b).hexdigest()
def record(z,info,end):
 z.fp.seek(info.header_offset);b=z.fp.read(end-info.header_offset);assert len(b)==end-info.header_offset;return b

def optimize(path,candidate,root,report):
 temporary=path.with_suffix('.strong_verified.zip');work=root/'data/finals.sqlite';assert work.stat().st_size==281206784 and sha(work.read_bytes())==EXPECTED
 assert not any(work.with_name(work.name+x).exists() for x in ['-wal','-shm','-journal'])
 spec=importlib.util.spec_from_file_location('builder',root/'recovery/build_unchanged_db_checkpoint.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);tables=m.sql(work)
 expected={};changed=[];old_size=path.stat().st_size;unchanged_local_records=0
 with zipfile.ZipFile(path) as src,zipfile.ZipFile(candidate) as cand,zipfile.ZipFile(temporary,'w') as out:
  assert cand.namelist()==[DB];ci=cand.getinfo(DB);dbraw=cand.read(DB);assert len(dbraw)==281206784 and sha(dbraw)==EXPECTED and cand.getinfo(DB).compress_type==zipfile.ZIP_LZMA
  assert len(src.namelist())==len(set(src.namelist())) and DB in src.namelist();assert src.read(DB)==dbraw and ci.compress_size<src.getinfo(DB).compress_size
  infos=sorted(src.infolist(),key=lambda x:x.header_offset)
  for i,info in enumerate(infos):
   raw=src.read(info);expected[info.filename]=sha(raw)
   if info.filename==DB:
    b=record(cand,ci,cand.start_dir);replacement=copy.copy(ci);replacement.external_attr=info.external_attr;replacement.create_system=info.create_system;changed.append({'member':DB,'old_compressed_bytes':info.compress_size,'new_compressed_bytes':ci.compress_size,'raw_bytes':len(raw),'raw_sha256':expected[DB]})
   else:
    end=infos[i+1].header_offset if i+1<len(infos) else src.start_dir;b=record(src,info,end);replacement=copy.copy(info);unchanged_local_records+=1
   replacement.header_offset=out.fp.tell();out.fp.write(b);out.filelist.append(replacement);out.NameToInfo[replacement.filename]=replacement;out.start_dir=out.fp.tell();out._didModify=True
  out.comment=src.comment
 with zipfile.ZipFile(temporary) as check:
  assert len(check.namelist())==len(set(check.namelist()))==len(expected) and set(check.namelist())==set(expected)
  for name,want in expected.items():assert sha(check.read(name))==want,name
  stage=report.parent/'strong_reextracted.sqlite';b=check.read(DB)
  with stage.open('wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
  assert m.sql(stage)==tables
 assert work.stat().st_size==281206784 and sha(work.read_bytes())==EXPECTED and m.sql(work)==tables
 result={'all_finals_complete':False,'canonical_DB_bytes_changed':False,'SQLite_raw_bytes':281206784,'SQLite_raw_sha256':EXPECTED,'normal_RO_SQL':'ok','all5table_sorted_row_hashes_equal_working':True,'all_member_full_SHA_equal_preoptimization':True,'members':len(expected),'missing_extra_duplicate':0,'unchanged_other_local_records':unchanged_local_records,'recompressed':changed,'bytes_before':old_size,'bytes_after':temporary.stat().st_size,'saved_bytes':old_size-temporary.stat().st_size,'compression':'ZIP method14 LZMA1 preset9 EXTREME,64MiB dictionary; ordinary standard ZIP read supports exact recovery','original_sources_deleted':0,'table_fingerprints':tables}
 with temporary.open('rb') as f:os.fsync(f.fileno())
 temporary.replace(path);report.write_text(json.dumps(result,ensure_ascii=False,indent=2));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('zip',type=pathlib.Path);p.add_argument('candidate',type=pathlib.Path);p.add_argument('root',type=pathlib.Path);p.add_argument('report',type=pathlib.Path);a=p.parse_args();r=optimize(a.zip,a.candidate,a.root,a.report);print(json.dumps({k:v for k,v in r.items() if k not in ['table_fingerprints']},ensure_ascii=False))
