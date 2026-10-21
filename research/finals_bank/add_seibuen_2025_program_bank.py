"""Use the official dated Seibuen75 leaflet only for its Aug28-31 event."""
import argparse,datetime,hashlib,json,pathlib,re,subprocess,unicodedata
from apply_historical_bank import validate,select

def compact(s):return re.sub(r'\s+','',unicodedata.normalize('NFKC',s))

def geometry(text):
    s=compact(text);g={}
    for label,key in [('周長','circumference_m'),('見なし直線距離','minashi_straight_m'),('ホーム幅員','home_width_m'),('バック幅員','back_width_m'),('センター幅員','center_width_m')]:
        v=re.findall(re.escape(label)+r'(\d+(?:\.\d+)?)m',s)
        assert len(v)==1,(label,v);g[key]=float(v[0])
    for label,key in [('センター部路面傾斜','center_cant_dms'),('直線部路面傾斜','home_cant_dms')]:
        v=re.findall(re.escape(label)+r'(\d+)°(\d+)′(\d+)′′',s)
        assert len(v)==1,(label,v);g[key]=list(map(int,v[0]))
    return g

def event_dates(text,timetable=None):
    s=unicodedata.normalize('NFKC',text)
    assert '西武園競輪ゴールド・ウイング賞GIII' in compact(s)
    years=re.findall(r'(?<!\d)(20\d{2})(?!\d)',s[:180]);assert years==['2025'],years
    pairs=re.findall(r'(?<!\d)(\d{1,2})月(\d{1,2})日',unicodedata.normalize('NFKC',timetable if timetable is not None else text))
    assert pairs==[('8','28'),('8','29'),('8','30'),('8','31')],pairs
    dates=[datetime.date(int(years[0]),int(m),int(d)) for m,d in pairs]
    assert all(b-a==datetime.timedelta(days=1) for a,b in zip(dates,dates[1:]))
    return dates[0].isoformat(),(dates[-1]+datetime.timedelta(days=1)).isoformat()

def add(data):
    path=data/'bank_research_1724/seibuen2025.pdf';raw=path.read_bytes()
    meta=json.loads((data/'bank_research_1724/seibuen2025.json').read_text())
    assert hashlib.sha256(raw).hexdigest()==meta['sha256']
    text=subprocess.check_output(['pdftotext','-layout',str(path),'-']).decode()
    timetable=subprocess.check_output(['pdftotext','-f','1','-l','1','-x','900','-y','685','-W','270','-H','130','-layout',str(path),'-']).decode()
    start,end=event_dates(text,timetable)
    table=subprocess.check_output(['pdftotext','-f','1','-l','1','-x','430','-y','675','-W','150','-H','125','-layout',str(path),'-']).decode()
    g=geometry(table)
    src={'source_url':meta['url'],'raw_file':'bank_research_1724/seibuen2025.pdf','sha256':meta['sha256'],'bytes':len(raw),'location':'PDF page1 lower-left bank table, x430 y675 W150 H125 PDFpoints; top header and lower-right daily timetable explicitly identify 2025Aug28-31.'}
    record={'bank_version_id':'26-seibuen75-20250828','track':'西武園競輪','venue_code':'26','valid_from':start,'valid_until_exclusive':end,'geometry':g,'sources':[src],'review_status':'verified_within_source_event','validity_basis':'Official Seibuen75 event leaflet explicitly lists2025Aug28-31. Apply only within that event; the June5 statistic sampling note and July1 entrant list date are not geometry validity periods.','visual_review':'Full PDF rendered and visually checked; all seven labelled geometry values and event header confirmed.'}
    catpath=data/'historical_bank_catalog.json';cat=json.loads(catpath.read_text());records=[x for x in cat['records'] if x['bank_version_id']!=record['bank_version_id']]+[record];validate(records)
    assert select([record],'2620250828040012','2025-08-31','西武園競輪')==record
    for rid,day,track in [('4420250828040012','2025-08-31','西武園競輪'),('2620250828040012','2025-08-27','西武園競輪'),('2620250828040012','2025-09-01','西武園競輪'),('2620250828040012','2025-08-31','大宮競輪')]:assert select([record],rid,day,track) is None
    cat['records']=records;catpath.write_text(json.dumps(cat,ensure_ascii=False,indent=2));return record

def test():
    s='周長400m 見なし直線距離47.6m センター部路面傾斜29°26′54″ 直線部路面傾斜2°51′45″ ホーム幅員11m バック幅員10m センター幅員7.5m'
    assert geometry(s)['home_width_m']==11
    assert geometry(s.replace('47.6m','48.2m'))['minashi_straight_m']==48.2
    for bad in [s.replace('ホーム幅員11m',''),s+'ホーム幅員12m']:
        try:geometry(bad)
        except AssertionError:pass
        else:raise AssertionError('Missing or ambiguous source accepted')
    for bad in ['西武園競輪ゴールド・ウイング賞GIII 2024 8月28日8月29日8月30日8月31日','西武園競輪ゴールド・ウイング賞GIII 2025 8月28日8月30日']:
        try:event_dates(bad)
        except AssertionError:pass
        else:raise AssertionError('Ungrounded event dates accepted')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True,type=pathlib.Path);args=p.parse_args();test();print(json.dumps(add(args.data),ensure_ascii=False))
