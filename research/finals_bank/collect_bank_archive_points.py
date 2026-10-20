"""Collect explicit archive capture points; never infer a continuous validity interval.
Seeds must come from observed CDX results. Completed and failed attempts are cached.
"""
import urllib.request,pathlib,lxml.html,json,concurrent.futures,hashlib,gzip,email.utils,datetime

def decode_wire(raw):
    count=0
    while raw.startswith(b'\x1f\x8b'):
        if count>=3:raise ValueError('too many gzip transport layers')
        raw=gzip.decompress(raw);count+=1
    return raw,count

def extract(raw,track):
    h=lxml.html.fromstring(raw)
    for el in h.xpath('//script|//style'):el.drop_tree()
    if track=='tamano':
        labels=('1周','直線カント','最大カント','みなし直線','ホーム幅員','バック幅員','センター幅員')
        lines=[' '.join(s.split()) for s in h.text_content().splitlines() if s.strip()]
        vals=[s for s in lines if any(s.startswith(x) for x in labels)]
        if len(vals)!=7:raise ValueError(('expected seven bank fields',vals))
        return vals
    if track=='kokura':
        vals=[[' '.join(c.text_content().split()) for c in tr.xpath('./th|./td')] for tr in h.xpath('//table//tr')]
        if len(vals)!=17 or vals[0]!=['周長','400m']:raise ValueError('unexpected Kokura geometry table')
        return vals
    raise ValueError('unknown parser')

def collect(data,retry_failed=False):
    out=data/'bank_archives_0714';out.mkdir(exist_ok=True);seeds=json.loads((out/'seed_inventory.json').read_text())
    old=json.loads((out/'observations.json').read_text()) if (out/'observations.json').exists() else []
    cached={(x['source_url'],x['requested_timestamp']):x for x in old}
    def fetch(s):
        name=s['track'];ts=s['timestamp'];orig=s['source_url'];key=(orig,ts)
        previous=cached.get(key)
        if previous and previous.get('raw_file') and (data/previous['raw_file']).exists():
            wire=gzip.decompress((data/previous['raw_file']).read_bytes());assert hashlib.sha256(wire).hexdigest()==previous['sha256'];r=dict(previous)
        elif previous and not retry_failed:return previous
        else:
            try:
                url=f'https://web.archive.org/web/{ts}id_/{orig}';response=urllib.request.urlopen(url,timeout=20);wire=response.read();memento=response.headers.get('Memento-Datetime')
                if not memento:raise ValueError('capture timestamp missing')
                actual_dt=email.utils.parsedate_to_datetime(memento);actual_ts=actual_dt.strftime('%Y%m%d%H%M%S')
                if not actual_dt.tzinfo:raise ValueError('capture timezone missing')
                fname=f'{name}_{actual_ts}.html.gz';(out/fname).write_bytes(gzip.compress(wire,mtime=0))
                r={'track':name,'source_url':orig,'requested_timestamp':ts,'actual_archive_url':response.url,'memento_datetime':memento,'actual_timestamp':actual_ts,'raw_file':'bank_archives_0714/'+fname,'sha256':hashlib.sha256(wire).hexdigest()}
            except Exception as e:return {**s,'source_url':orig,'requested_timestamp':ts,'status':'failed_no_retry','error':str(e)}
        try:
            wire=gzip.decompress((data/r['raw_file']).read_bytes());body,n=decode_wire(wire);r['extracted']=extract(body,name);r['transport_gzip_layers']=n
            r['actual_timestamp']=email.utils.parsedate_to_datetime(r['memento_datetime']).strftime('%Y%m%d%H%M%S')
            r['status']='captured_point_observation_only';r['continuous_applicability']='not_proven'
            decoded=pathlib.Path(r['raw_file']).name.replace('.html.gz','.decoded.html.gz');(out/decoded).write_bytes(gzip.compress(body,mtime=0));r['decoded_file']='bank_archives_0714/'+decoded;r['decoded_sha256']=hashlib.sha256(body).hexdigest()
        except Exception as e:r['status']='parse_failed_no_retry';r['error']=str(e);r['continuous_applicability']='not_proven'
        return r
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:result=list(ex.map(fetch,seeds))
    (out/'observations.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data',type=pathlib.Path,required=True);p.add_argument('--retry-failed',action='store_true');a=p.parse_args();r=collect(a.data,a.retry_failed);print({'observations':len(r),'decoded':sum(x['status']=='captured_point_observation_only' for x in r),'continuous_periods_proven':0,'all_finals_complete':False})
