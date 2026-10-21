"""Extract all seven geometry values from explicitly dated official event programs.

Requires locally saved, visually reviewed source PDFs and their manifest. No network.
Only the event period printed on the same page as the bank table is applicable.
"""
import datetime, hashlib, json, pathlib, re, subprocess, unicodedata
from apply_historical_bank import select, validate

def parse(text):
    s=re.sub(r'\s+','',unicodedata.normalize('NFKC',text))
    assert '防府けいりん' in s and '【最終日】' in s
    dates=re.findall(r'令和(\d+)年(\d+)月(\d+)日\([日月火水木金土]\)[~〜](?:(\d+)月)?(\d+)日\([日月火水木金土]\)',s)
    assert len(set(dates))==1, ('ambiguous or missing event header',dates)
    y,m,d,em,ed=dates[0];y=int(y)+2018;m=int(m);em=int(em or m)
    start=datetime.date(y,m,int(d));end=datetime.date(y,em,int(ed))
    assert start<=end and (end-start).days<=5
    g={}
    for label,key in [('周長','circumference_m'),('見なし直線距離','minashi_straight_m'),('ホーム幅員','home_width_m'),('バック幅員','back_width_m'),('センター幅員','center_width_m')]:
        vals=re.findall(re.escape(label)+r'(\d+(?:\.\d+)?)m',s)
        assert vals and len(set(map(float,vals)))==1,(label,vals)
        g[key]=float(vals[0])
    for label,key in [('最大カント','center_cant_dms'),('直線部路面傾斜','home_cant_dms')]:
        vals=re.findall(re.escape(label)+r'(\d+)°(\d+)[’′\x27](\d+)[”″\x22]',s)
        assert vals and len(set(vals))==1,(label,vals)
        g[key]=list(map(int,vals[0]))
    return start.isoformat(),(end+datetime.timedelta(days=1)).isoformat(),g

def build(data):
    manifest=json.loads((data/'bank_sources_1248/manifest.json').read_text())
    records=[]
    for source in manifest['sources']:
        p=data/source['raw_file'];raw=p.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==source['sha256']
        txt=subprocess.check_output(['pdftotext','-f','2','-l','2','-layout',str(p),'-']).decode()
        start,end,g=parse(txt)
        assert start==source['expected_start'] and end==source['expected_until_exclusive']
        records.append({'bank_version_id':'63-program-'+start,'track':'防府競輪','venue_code':'63','valid_from':start,'valid_until_exclusive':end,'geometry':g,'sources':[source],
          'validity_basis':'Official event program: explicit Reiwa year and event date range on the same page as all seven bank dimensions. Restricted to this event only; no bridging between events.',
          'review_status':'verified_within_source_period','visual_review':'PDF page 2 rendered and reviewed for event header, venue, final class and bank table label/value correspondence.'})
    validate(records)
    for r in records:
        assert select([r],'63test',r['valid_from'],'防府競輪')==r
        assert select([r],'63test',r['valid_until_exclusive'],'防府競輪') is None
        assert select([r],'11test',r['valid_from'],'防府競輪') is None
        assert select([r],'63test',r['valid_from'],'函館競輪') is None
    catp=data/'historical_bank_catalog.json';cat=json.loads(catp.read_text())
    ids={r['bank_version_id'] for r in records}
    old=[r for r in cat['records'] if r['bank_version_id'] not in ids]
    validate(old+records);cat['records']=old+records
    catp.write_text(json.dumps(cat,ensure_ascii=False,indent=2))
    return records

def test():
    fixture='防府けいりん 令和8年5月17日(日)~5月19日(火) 【最終日】 周長333m 見なし直線距離42.5m 最大カント34°41’9” 直線部路面傾斜4°34’26” ホーム幅員10.2m バック幅員9.1m センター幅員7.4m'
    assert parse(fixture)[0:2]==('2026-05-17','2026-05-20')
    assert parse(fixture.replace('42.5m','41.5m'))[2]['minashi_straight_m']==41.5
    for bad in [fixture.replace('令和8年',''),fixture.replace('周長333m',''),fixture+' 周長400m',fixture.replace('防府けいりん','小倉けいりん')]:
        try:parse(bad)
        except AssertionError:pass
        else:raise AssertionError('invalid source accepted')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data',type=pathlib.Path,required=True);v=p.parse_args();test();print(json.dumps(build(v.data),ensure_ascii=False))
