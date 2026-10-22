"""Preserve verified closed-backup SQLite bytes and unchanged compressed records.

Allowed only when the entire working SQLite is byte-identical to the materialized
baseline. No sidecars, hidden files, secondary DBs, omitted evidence or duplicate
members. Validate all members and normal read-only SQL before atomic install.
"""
import argparse,copy,hashlib,json,os,pathlib,sqlite3,tempfile,zipfile

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def allowed(p):
 return not any(x.startswith('.') or x=='__pycache__' for x in p.parts) and p.suffix!='.pyc' and not p.name.endswith(('-journal','-wal','-shm')) and (p.suffix!='.sqlite' or p==pathlib.Path('data/finals.sqlite'))
def sql(path):
 c=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
 assert c.execute('pragma integrity_check').fetchall()==[('ok',)]
 tables={}
 for (name,) in c.execute("select name from sqlite_master where type='table' order by name"):
  assert name in ['races','days','banks','cancellations','excluded_races']
  h=hashlib.sha256();n=0
  for row in c.execute('select * from '+name+' order by '+','.join(str(i+1) for i in range(len(c.execute('pragma table_info('+name+')').fetchall())))):
   h.update(json.dumps(row,ensure_ascii=False,separators=(',',':')).encode());h.update(b'\n');n+=1
  tables[name]={'rows':n,'sha256_rows':h.hexdigest()}
 c.close();return tables
def build(root,baseline,output):
 db=root/'data/finals.sqlite'
 assert not any(pathlib.Path(str(db)+s).exists() for s in ['-journal','-wal','-shm'])
 before=sha(db);working_tables=sql(db)
 working={'finals_ai/'+str(p.relative_to(root)):p for p in root.rglob('*') if p.is_file() and allowed(p.relative_to(root))}
 expected={name:sha(p) for name,p in working.items()};changes=[];added=[]
 tmp=output.with_suffix('.building.zip')
 with zipfile.ZipFile(baseline) as src,zipfile.ZipFile(tmp,'w') as dst:
  infos=sorted(src.infolist(),key=lambda x:x.header_offset);names=src.namelist();assert len(names)==len(set(names));assert set(names)<=set(working),'baseline evidence missing from worktree'
  bdb=src.read('finals_ai/data/finals.sqlite');assert hashlib.sha256(bdb).hexdigest()==before,'DB change forbidden on this route';del bdb
  for index,info in enumerate(infos):
   n=info.filename;oldhash=hashlib.sha256(src.read(n)).hexdigest()
   if oldhash==expected[n]:
    end=infos[index+1].header_offset if index+1<len(infos) else src.start_dir;src.fp.seek(info.header_offset);record=src.fp.read(end-info.header_offset);assert len(record)==end-info.header_offset
    new=copy.copy(info);new.header_offset=dst.fp.tell();dst.fp.write(record);dst.filelist.append(new);dst.NameToInfo[n]=new;dst.start_dir=dst.fp.tell();dst._didModify=True
   else:
    assert n!='finals_ai/data/finals.sqlite';dst.write(working[n],n,compress_type=zipfile.ZIP_LZMA);changes.append(n)
  for n in sorted(set(working)-set(names)):
   dst.write(working[n],n,compress_type=zipfile.ZIP_LZMA);added.append(n)
 with zipfile.ZipFile(tmp) as z,tempfile.TemporaryDirectory() as td:
  assert len(z.namelist())==len(expected) and set(z.namelist())==set(expected)
  assert [n for n in z.namelist() if n.endswith('.sqlite')]==['finals_ai/data/finals.sqlite']
  assert not any(n.endswith(('-journal','-wal','-shm')) for n in z.namelist())
  for n,h in expected.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
  z.extract('finals_ai/data/finals.sqlite',td);extracted=pathlib.Path(td)/'finals_ai/data/finals.sqlite'
  assert sha(extracted)==before;assert sql(extracted.resolve())==working_tables
 assert sha(db)==before
 with tmp.open('rb') as f:os.fsync(f.fileno())
 tmp.replace(output)
 return {'zip_bytes':output.stat().st_size,'members':len(expected),'db_sha256':before,'db_bytes':db.stat().st_size,'normal_RO_integrity':'ok','all5tables_equal_baseline_and_working':True,'tables':working_tables,'all_nonSQLite_member_hashes_equal_complete_worktree':True,'missing_extra_duplicate_members':0,'sidecars':False,'changed_members':changes,'added_members':added,'preserved_verified_closed_backup_DB_byte_for_byte':True,'all_finals_complete':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--baseline',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--report',type=pathlib.Path,required=True);a=p.parse_args();r=build(a.root.resolve(),a.baseline.resolve(),a.output.resolve());a.report.write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ['added_members','tables']},ensure_ascii=False))
