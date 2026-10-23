"""Bounded, cached fetch of actual saved-result entry links for missing fields."""
import concurrent.futures,datetime,hashlib,io,json,pathlib,time,tarfile,urllib.request
from lxml import html
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=pathlib.Path('new_entry_0730');OUT.mkdir(exist_ok=True)
with tarfile.open(ROOT/'data/scope_review_0552/integrated_candidates.tar.xz') as t:
 raw=t.extractfile('integrated_candidates.json').read()
assert hashlib.sha256(raw).hexdigest()=='b9ed0533f9c8a654236a4c4c8d9d275161acdd6c154ad6c3efbaa444cb307e3f'
events=json.loads(raw)['events'];jobs=[]
for e in events:
 ref=e['component_raw_provenance']['independent_result']
 with tarfile.open(ROOT/ref['archive']['path']) as t:b=t.extractfile(ref['member']).read()
 assert len(b)==ref['bytes'] and hashlib.sha256(b).hexdigest()==ref['sha256'];d=html.fromstring(b.decode('utf-8','strict'));k=e['observed_provider_keys']['netkeiba_race_id'];url='https://keirin.netkeiba.com/race/entry/?race_id='+k
 assert d.xpath('//a[@href=$href]',href=url);jobs.append((k,url,ref))
def fetch(job):
 k,url,ref=job;meta=OUT/(k+'.json');p=OUT/(k+'.html')
 if meta.exists():
  m=json.loads(meta.read_bytes())
  if 'error' not in m:assert p.stat().st_size==m['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==m['sha256']
  return m
 m={'url':url,'key':k,'retry_this_run':False,'observed_link_source':ref,'purpose':'new independent historical S/B and rate/comment field evidence; not a completed race'};start=time.monotonic()
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0','Accept-Encoding':'identity'}),timeout=25) as r:
   chunks=[];m.update(status=r.status,final_url=r.url,headers=dict(r.headers))
   while True:
    b=r.read(16384)
    if not b:break
    chunks.append(b)
    if time.monotonic()-start>55:raise TimeoutError('whole55sec')
  b=b''.join(chunks);assert len(b)>10000;p.write_bytes(b);m.update(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
 except Exception as e:m['error']=repr(e)
 meta.write_text(json.dumps(m,ensure_ascii=False,indent=2));return m
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(fetch,jobs))
 print(json.dumps({'success':sum('error' not in m for m in results),'errors':[m for m in results if 'error' in m]},ensure_ascii=False))
