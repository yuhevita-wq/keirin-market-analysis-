import argparse,base64,gzip,json,pathlib,sqlite3,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import pipeline as p

def main():
    a=argparse.ArgumentParser();a.add_argument('shard',type=int);x=a.parse_args()
    data=p.ROOT/'data';c=p.db_open(data)
    seed=json.loads(gzip.decompress(base64.b64decode((p.ROOT/'pending_seed.b64').read_text())))
    selected=[r for r in seed if int(r[0])%8==x.shard]
    c.executemany('insert or ignore into races values(?,?,?,?,?,?,?,?,?,?)',selected);c.commit()
    print('shard',x.shard,'expected',len(selected),flush=True)

if __name__=='__main__':main()
