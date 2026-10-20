"""Apply source-period bank specifications only, using actual venue and race date."""
import pathlib,json,sqlite3,datetime,collections,shutil,tempfile,os
REQUIRED=('circumference_m','minashi_straight_m','home_width_m','back_width_m','center_width_m','home_cant_dms','center_cant_dms')
def select(records,race_id,day,track):
    hits=[r for r in records if r['venue_code']==race_id[:2] and r['track']==track and r['valid_from']<=day<r['valid_until_exclusive']]
    if len(hits)>1:raise ValueError(('overlapping bank source periods',race_id))
    return hits[0] if hits else None
def validate(records):
    for r in records:
        assert r['valid_from']<r['valid_until_exclusive']
        assert r['sources'] and set(REQUIRED)<=set(r['geometry'])
        for k in REQUIRED:
            v=r['geometry'][k]
            if k.endswith('_dms'):assert len(v)==3 and 0<=v[0]<90 and 0<=v[1]<60 and 0<=v[2]<60
            else:assert isinstance(v,(float,int)) and 0<v<1000
    for i,a in enumerate(records):
        for b in records[i+1:]:
            if a['venue_code']==b['venue_code']:assert max(a['valid_from'],b['valid_from'])>=min(a['valid_until_exclusive'],b['valid_until_exclusive'])
def apply(data):
    records=json.loads((data/'historical_bank_catalog.json').read_text())['records'];validate(records)
    db=data/'finals.sqlite';fd,tmp=tempfile.mkstemp(prefix='bank_work_',suffix='.sqlite',dir=data);os.close(fd);shutil.copy2(db,tmp)
    try:
        c=sqlite3.connect(tmp);assert c.execute('pragma integrity_check').fetchall()==[('ok',)]
        covered=collections.Counter();missing=collections.defaultdict(list)
        for rid,day,raw in c.execute("select race_id,day,payload from races where status!='excluded'").fetchall():
            d=json.loads(raw);track=d['race']['track'];r=select(records,rid,day,track)
            if not r:missing[track].append({'race_id':rid,'day':day,'reason':'No verified bank specification covering this venue and race date'});continue
            value={'bank_version_id':r['bank_version_id'],'venue_code':r['venue_code'],'track':track,'race_date':day,'valid_from':r['valid_from'],'valid_until_exclusive':r['valid_until_exclusive'],'geometry':r['geometry'],'evidence':r['sources'],'validity_basis':r['validity_basis'],'status':r['review_status']}
            if d.get('bank') not in (None,value):raise ValueError(('existing bank differs',rid))
            d['bank']=value;c.execute('update races set payload=? where race_id=?',(json.dumps(d,ensure_ascii=False),rid));covered[track]+=1
        c.commit();assert c.execute('pragma integrity_check').fetchall()==[('ok',)];c.close();os.replace(tmp,db)
        report={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'verified_races':sum(covered.values()),'covered_by_track':dict(covered),'unresolved_races':sum(map(len,missing.values())),'remaining_by_track':{k:{'count':len(v),'first_day':min(x['day'] for x in v),'last_day':max(x['day'] for x in v)} for k,v in sorted(missing.items())},'unresolved':dict(missing),'all_finals_complete':False}
        (data/'historical_bank_integrity.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print({k:v for k,v in report.items() if k not in ['unresolved','remaining_by_track']})
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data',type=pathlib.Path,required=True);apply(p.parse_args().data)
