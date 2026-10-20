"""Independently audit identity additions and preservation against a prior checkpoint."""
import argparse, collections, datetime, itertools, json, math, pathlib, sqlite3, tempfile, zipfile
ID_FIELDS={'rider_source_id','rider_registration_number','rider_id_status','rider_identity_evidence'}
def without_identity(p):
    for e in p.get('entries',[]):
        for k in ID_FIELDS:e.pop(k,None)
    return p
def audit(data,baseline_zip):
    with tempfile.TemporaryDirectory() as tmp:
        prior=pathlib.Path(tmp)/'prior.sqlite'
        with zipfile.ZipFile(baseline_zip) as z:prior.write_bytes(z.read('finals_ai/data/finals.sqlite'))
        a=sqlite3.connect(prior);b=sqlite3.connect(data/'finals.sqlite')
        assert a.execute('pragma integrity_check').fetchall()==[('ok',)]
        assert b.execute('pragma integrity_check').fetchall()==[('ok',)]
        counts=collections.Counter();gaps=[];errors=[]
        for rid,day,status,raw in b.execute('select race_id,day,status,payload from races'):
            counts['ledger_total']+=1;p=json.loads(raw);old=a.execute('select payload from races where race_id=?',(rid,)).fetchone()
            assert old is not None
            assert without_identity(json.loads(old[0]))==without_identity(json.loads(raw)),('nonidentity change',rid)
            if status=='excluded':counts['excluded']+=1;continue
            counts['in_scope']+=1;es=p.get('entries',[]);cars=[int(e['car_no']) for e in es]
            assert len(cars)==len(set(cars)) and len(cars)>=3,(rid,'entries')
            assert p.get('results') and p.get('race',{}).get('race_date')==day and '2023-01-01'<=day<='2026-10-19',(rid,'date/results')
            active=p.get('odds_scope',{}).get('active_cars',cars)
            for market,comb in [('trifecta',itertools.permutations(active,3)),('trio',itertools.combinations(sorted(active),3))]:
                rows=p.get(market,[]);keys=[tuple(r['cars']) for r in rows]
                assert set(keys)==set(comb) and len(keys)==len(set(keys)),(rid,market,'set')
                assert all(r.get('status')=='available' and type(r.get('odds')) in (int,float) and math.isfinite(r['odds']) and r['odds']>0 for r in rows),(rid,market,'value')
                counts[market+'_verified']+=1
            line=p.get('line',{});line_cars={int(x) for x in line.get('riders',[])}
            assert line.get('status')=='published' and line_cars in (set(cars),set(active)),(rid,'line')
            if set(cars)!=set(active):counts['withdrawal_line_entered_field' if line_cars==set(cars) else 'withdrawal_line_active_field']+=1
            counts['published_line_verified']+=1
            missing=0
            for e in es:
                if not e.get('rider_source_id'):missing+=1;continue
                n=e['rider_registration_number'];assert len(n)==6 and n.isdigit()
                evidence=e['rider_identity_evidence']['official_sources']
                name=''.join(e['player_name'].split());term=''.join(x for x in str(e['term']) if x.isdigit());pref=''.join(e['prefecture'].split())
                assert evidence and all(r['registration_number']==n and r['name']==name and r['term']==term and r['prefecture']==pref for r in evidence),(rid,'identity evidence')
            counts['unresolved_identity_entries']+=missing;counts['verified_identity_entries']+=len(es)-missing
            if missing:gaps.append({'race_id':rid,'errors':['rider_source_id']})
            else:counts['races_with_all_identities_verified']+=1
        assert counts['ledger_total']==a.execute('select count(*) from races').fetchone()[0]
        a.close();b.close()
    report={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all_finals_complete':False,'counts':dict(counts),'gaps':gaps,'historical_bank_applicability':'unresolved','integrity_check':'ok','nonidentity_payload_preserved':True,'baseline':str(baseline_zip)}
    (data/'merged_integrity.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print({k:v for k,v in report.items() if k!='gaps'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=pathlib.Path,required=True);p.add_argument('--baseline-zip',type=pathlib.Path,required=True);v=p.parse_args();audit(v.data,v.baseline_zip)
