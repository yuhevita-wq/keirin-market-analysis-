"""Offline review of archived official event: partial geometry, never bank approval."""
import pathlib,json,hashlib,tarfile,io,sqlite3
from lxml import html
ROOT=pathlib.Path('current_data/unpacked/finals_ai')
OUT=ROOT/'data/bank_research_1813'
EXPECTED={'maebashi_2024_news.html':'70f2b71ba01dee5af4bf2c084d2c55d543c1b0c97e305a8187ef70cef3d997f8','maebashi74_cdx.html':'70d98dc6336ca38d78629a985f8c6934c6488c40d76cefd76d21090e136396ba','maebashi74_archive.html':'5210721e657e26679eb522dee3fe54d0bea3d42fda48d06d8376ee5bbf491c25'}
def check(files):
 for n,h in EXPECTED.items():
  b=files[n];assert hashlib.sha256(b).hexdigest()==h
  m=json.loads(files[n.replace('.html','.json')]);assert m['status']==200 and m['bytes']==len(b) and m['sha256']==h
 news=html.fromstring(files['maebashi_2024_news.html'].decode('utf8'));assert 'https://www.maebashi-keirin.jp/miyamaokan_74th/' in [x.strip() for x in news.xpath('//a/@href')]
 cdx=json.loads(files['maebashi74_cdx.html']);assert any(r[:2]==['20240524233811','https://www.maebashi-keirin.jp/miyamaokan_74th/'] for r in cdx[1:])
 doc=html.fromstring(files['maebashi74_archive.html'].decode('utf8'));s=doc.text_content()
 for token in ['74','2024','6/4','バンク周長：335m','ゴールまでの見なし直線：46.7m','最大カント：36度0分0秒']:assert token in s,token
 m=json.loads(files['maebashi74_archive.json']);headers={k.lower():v for k,v in m['headers'].items()};assert headers['memento-datetime']=='Fri, 24 May 2024 23:38:11 GMT'
 err=json.loads(files['maebashi74_event.json']);assert '404' in err['error'] and 'sha256' not in err
 return {'source_capture':'2024-05-24T23:38:11Z','venue_code':'22','target_actual_date':'2024-06-04','target_race_no':9,'status':'partial_historical_geometry_not_applied','observed':{'circumference_m':'335','minashi_m':'46.7','max_cant_raw':'36度0分0秒'},'missing':['home_width','back_width','center_width','home_cant'],'full7_verified':False,'period_extension':False,'database_changed':False,'SQL_insertions':0,'all_finals_complete':False,'live_event_HTTP404_retained':True,'no_same_route_retry':True}
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'raw.tar.xz').exists():
  with tarfile.open(OUT/'raw.tar.xz') as t:
   names=t.getnames();assert len(names)==len(set(names));files={m.name:t.extractfile(m).read() for m in t.getmembers()}
 else:files={p.name:p.read_bytes() for p in pathlib.Path('bank_raw_1813').iterdir() if p.is_file()}
 review=check(files);mutations=[]
 for n in EXPECTED:
  changed=dict(files);changed[n]=files[n]+b' '
  try:check(changed)
  except AssertionError:mutations.append(n)
  else:raise AssertionError('altered evidence accepted')
 review['hash_mutations_rejected']=mutations
 if not (OUT/'raw.tar.xz').exists():
  with tarfile.open(OUT/'raw.tar.xz','w:xz',preset=9) as t:
   for n,b in sorted(files.items()):
    i=tarfile.TarInfo(n);i.size=len(b);t.addfile(i,io.BytesIO(b))
 manifest={n:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in files.items()}
 with tarfile.open(OUT/'raw.tar.xz') as t:assert {m.name:{'bytes':m.size,'sha256':hashlib.sha256(t.extractfile(m).read()).hexdigest()} for m in t.getmembers()}==manifest
 review['raw_members']=manifest;review['raw_archive']={'bytes':(OUT/'raw.tar.xz').stat().st_size,'sha256':hashlib.sha256((OUT/'raw.tar.xz').read_bytes()).hexdigest()}
 db=ROOT/'data/finals.sqlite';assert db.stat().st_size==281206784 and hashlib.sha256(db.read_bytes()).hexdigest()=='04c4de11b6937c3b3a499f042410786cfdde4c2892ca0af46999ca059c5e2d91'
 assert not any(pathlib.Path(str(db)+s).exists() for s in ['-wal','-shm','-journal'])
 with sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True) as c:assert c.execute('pragma integrity_check').fetchall()==[('ok',)] and c.execute('select count(*) from races').fetchone()[0]==5354
 (OUT/'review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2));print(json.dumps(review,ensure_ascii=False))
if __name__=='__main__':main()
