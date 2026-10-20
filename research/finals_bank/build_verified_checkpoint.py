"""Build a checkpoint from a closed SQLite backup; never package recovery sidecars."""
import argparse, pathlib, sqlite3, tempfile, zipfile, hashlib, shutil, json

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def include(path):
    if path.suffix=='.sqlite' and path!=pathlib.Path('data/finals.sqlite'):return False
    return not any(p.startswith('.') or p=='__pycache__' for p in path.parts) and path.suffix!='.pyc' and not path.name.endswith(('-journal','-wal','-shm'))

def build(root,output):
    db=root/'data/finals.sqlite'
    # Do not silently ignore recoverable or pending transactions.
    for suffix in ['-journal','-wal','-shm']:
        side=pathlib.Path(str(db)+suffix)
        if side.exists() and side.stat().st_size:raise RuntimeError('SQLite sidecar requires separate review: '+str(side))
    with tempfile.TemporaryDirectory(prefix='checkpoint_snapshot_') as td:
        snapshot=pathlib.Path(td)/'snapshot.sqlite'
        source=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True)
        assert source.execute('pragma integrity_check').fetchall()==[('ok',)]
        dest=sqlite3.connect(snapshot);source.backup(dest);dest.close();source.close()
        c=sqlite3.connect(snapshot.as_uri()+'?immutable=1',uri=True)
        assert c.execute('pragma integrity_check').fetchall()==[('ok',)]
        assert c.execute('select count(*) from races').fetchone()[0]==5354;c.close()
        tmp=output.with_suffix('.building.zip')
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in sorted(root.rglob('*')):
                rel=p.relative_to(root)
                if p.is_file() and include(rel):
                    z.write(snapshot if rel==pathlib.Path('data/finals.sqlite') else p,'finals_ai/'+str(rel))
        with zipfile.ZipFile(tmp) as z:
            assert not any(n.endswith(('-journal','-wal','-shm')) for n in z.namelist())
            z.extract('finals_ai/data/finals.sqlite',td)
        restored=pathlib.Path(td)/'finals_ai/data/finals.sqlite'
        assert digest(restored)==digest(snapshot)
        c=sqlite3.connect(restored.as_uri()+'?mode=ro',uri=True)
        assert c.execute('pragma integrity_check').fetchall()==[('ok',)]
        count=c.execute('select count(*) from races').fetchone()[0];c.close()
        tmp.replace(output)
        return {'zip_bytes':output.stat().st_size,'db_sha256':digest(snapshot),'ledger_rows':count,'integrity_check':'ok','no_sqlite_sidecars':True}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.root.resolve(),a.output.resolve())))
