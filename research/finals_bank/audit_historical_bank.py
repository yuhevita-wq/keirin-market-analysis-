"""Independent comparison against saved baseline: only bank may change."""
import pathlib,json,sqlite3,zipfile,tempfile,hashlib,gzip,math,collections,datetime
def audit(data,baseline):
    cat=json.loads((data/'historical_bank_catalog.json').read_text())['records'];byid={r['bank_version_id']:r for r in cat};verified=collections.Counter();unresolved=0
    for r in cat:
        for s in r['sources']:
            p=data/s['raw_file'];raw=p.read_bytes();raw=gzip.decompress(raw) if p.suffix=='.gz' else raw
            assert hashlib.sha256(raw).hexdigest()==s['sha256']
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(baseline) as z:z.extract('finals_ai/data/finals.sqlite',td)
        b=sqlite3.connect(pathlib.Path(td)/'finals_ai/data/finals.sqlite');a=sqlite3.connect(data/'finals.sqlite')
        for c in (a,b):assert c.execute('pragma integrity_check').fetchall()==[('ok',)]
        original={x[0]:x[1:] for x in b.execute('select race_id,day,status,payload from races')};assert len(original)==5354
        for rid,day,status,raw in a.execute('select race_id,day,status,payload from races'):
            od,os,op=original.pop(rid);assert (day,status)==(od,os);d=json.loads(raw);old=json.loads(op);bank=d.pop('bank',None);old_bank=old.pop('bank',None);assert d==old,(rid,'nonbank payload altered')
            if status=='excluded':assert bank==old_bank;continue
            applicable=[r for r in cat if r['track']==d['race']['track'] and r['venue_code']==rid[:2] and r['valid_from']<=day<r['valid_until_exclusive']]
            if not bank:assert not applicable;unresolved+=1;continue
            assert len(applicable)==1 and byid[bank['bank_version_id']]==applicable[0]
            r=applicable[0];assert bank['race_date']==day and bank['geometry']==r['geometry'] and bank['evidence']==r['sources']
            assert (bank['valid_from'],bank['valid_until_exclusive'])==(r['valid_from'],r['valid_until_exclusive'])
            g=bank['geometry'];assert 250<=g['circumference_m']<=500 and 0<g['minashi_straight_m']<100
            for k in ['home_width_m','back_width_m','center_width_m']:assert math.isfinite(g[k]) and 0<g[k]<20
            for k in ['home_cant_dms','center_cant_dms']:
                x,y,z=g[k];assert 0<=x<90 and 0<=y<60 and 0<=z<60
            verified[d['race']['track']]+=1
        assert not original and sum(verified.values())+unresolved==5353;a.close();b.close()
    report={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'independent_verification':True,'baseline':str(baseline),'verified_bank_races':sum(verified.values()),'verified_by_track':dict(verified),'unresolved_bank_races':unresolved,'integrity_check':'ok','all_nonbank_payload_preserved':True,'all_finals_complete':False}
    (data/'historical_bank_independent_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(report)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data',type=pathlib.Path,required=True);p.add_argument('--baseline-zip',type=pathlib.Path,required=True);v=p.parse_args();audit(v.data,v.baseline_zip)
