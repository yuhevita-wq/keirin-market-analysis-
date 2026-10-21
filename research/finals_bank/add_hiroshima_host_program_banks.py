"""Event-only bank facts from official Hiroshima-hosted programs.

The host is not the venue: three programs explicitly say in Tamano/61.
Only bank table and cover date crops are parsed; rider statistics are ignored.
"""
import argparse,datetime,hashlib,json,pathlib,re,subprocess,unicodedata
from apply_historical_bank import validate,select

def norm(s):
    for ch in ['´','′','’']:s=s.replace(ch,"'")
    for ch in ['″','”']:s=s.replace(ch,'"')
    # Nov2024 PDF encodes its visually checked degree symbol as U+309A.
    return re.sub(r'\s+','',unicodedata.normalize('NFKC',s)).replace('\u309a','°')

def event(text):
    s=norm(text)
    assert '広島市営競輪' in s
    if '広島市営競輪in玉野' in s:code,track='61','玉野競輪'
    else:
        assert 'in' not in s;code,track='62','広島競輪'
    m=re.search(r'(20\d{2})年(\d{1,2})月(\d{1,2})([月火水木金土日])(\d{1,2})([月火水木金土日])(\d{1,2})([月火水木金土日])',s)
    if m:y,mo,a,wa,b,wb,c,wc=m.groups()
    else:
        # December cover uses a positioned year/month after the day labels.
        m=re.search(r'(\d{1,2})([月火水木金土日])(\d{1,2})([月火水木金土日])(\d{1,2})([月火水木金土日])(20\d{2})(\d{1,2})/',s)
        assert m,'missing explicit three-day cover';a,wa,b,wb,c,wc,y,mo=m.groups()
    dates=[datetime.date(int(y),int(mo),int(d)) for d in [a,b,c]]
    assert dates[1]-dates[0]==dates[2]-dates[1]==datetime.timedelta(days=1)
    assert ['月火水木金土日'[d.weekday()] for d in dates]==[wa,wb,wc]
    return code,track,dates[0].isoformat(),(dates[-1]+datetime.timedelta(days=1)).isoformat()

def geometry(text,track):
    s=norm(text);label=track+'場バンク概要';assert s.count(label)==1
    s=s.split(label,1)[1][:260]
    def one(pattern):
        a=re.findall(pattern,s);assert len(a)==1,(pattern,a);return a[0]
    g={}
    for key,label in [('circumference_m','1周'),('minashi_straight_m','みなし直線'),('home_width_m','ホーム幅員'),('back_width_m','バック幅員'),('center_width_m','センター幅員')]:
        g[key]=float(one(label+r'([0-9]+(?:\.[0-9]+)?)m'))
    for key,label in [('home_cant_dms','直線カント'),('center_cant_dms','最大カント')]:
        g[key]=list(map(int,one(label+r'(\d+)°(\d+)[´′’\x27](\d+)["″”]')))
    return g

def add(data):
    folder=data/'bank_sources_2127';man=json.loads((folder/'events.json').read_text());records=[]
    for item in man:
        p=data/item['raw_file'];b=p.read_bytes();assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256']
        cover=subprocess.check_output(['pdftotext','-f','1','-l','1','-r','72','-x','0','-y','0','-W','595','-H','210','-layout',str(p),'-']).decode()
        code,track,start,end=event(cover)
        assert (code,track,start,end)==tuple(item['visually_reviewed_event']), 'cover/review disagreement'
        full=subprocess.check_output(['pdftotext','-layout',str(p),'-']).decode()
        g=geometry(full,track)
        source={k:item[k] for k in ['source_url','raw_file','sha256','bytes']};source['location']='page1 explicit year/three weekdays/venue; page2 labelled7-value bank table. Full pages visually reviewed.'
        r={'bank_version_id':code+'-hiroshima-host-program-'+start.replace('-',''),'venue_code':code,'track':track,'valid_from':start,'valid_until_exclusive':end,'geometry':g,'sources':[source],'review_status':'verified_within_source_event','validity_basis':'Cover explicitly gives three-day event year and actual venue, corroborated by weekday calculation. Applies only to this event; host Hiroshima does not imply venue62. Rider data as-of dates are not shape periods.','visual_review':'Both original PDF pages rendered and reviewed: cover dates, actual venue, and all7 bank-label/value correspondence.'}
        validate([r]);rid=code+start.replace('-','')+'030012'
        assert select([r],rid,start,track)==r
        for badrid,day,badtrack in [('00'+rid[2:],start,track),(rid,end,track),(rid,(datetime.date.fromisoformat(start)-datetime.timedelta(days=1)).isoformat(),track),(rid,start,'別場競輪')]:assert select([r],badrid,day,badtrack) is None
        records.append(r)
    p=data/'historical_bank_catalog.json';cat=json.loads(p.read_text());ids={r['bank_version_id'] for r in records};cat['records']=[r for r in cat['records'] if r['bank_version_id'] not in ids]+records;validate(cat['records']);p.write_text(json.dumps(cat,ensure_ascii=False,indent=2));return records

def test():
    s='広島市営競輪in玉野FI2025年5月12月13火14水';assert event(s)==('61','玉野競輪','2025-05-12','2025-05-15')
    assert event('広島市営競輪FI7日8月9火202512/')==('62','広島競輪','2025-12-07','2025-12-10')
    for bad in [s.replace('2025年',''),s.replace('14水','15水'),s.replace('2025年','2024年')]:
        try:event(bad)
        except (AssertionError,ValueError):pass
        else:raise AssertionError('invalid cover accepted')
    s='玉野競輪場バンク概要1周400m直線カント3°26´1"最大カント30°37´33"みなし直線47.9mホーム幅員10.3mバック幅員9.3mセンター幅員7.5m'
    assert geometry(s,'玉野競輪')['back_width_m']==9.3
    assert geometry(s.replace('9.3m','9.4m'),'玉野競輪')['back_width_m']==9.4
    for bad in [s.replace('バック幅員9.3m',''),s+'バック幅員9.4m']:
        try:geometry(bad,'玉野競輪')
        except AssertionError:pass
        else:raise AssertionError('missing/conflicting bank accepted')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=pathlib.Path,required=True);a=p.parse_args();test();print(json.dumps(add(a.data),ensure_ascii=False))
