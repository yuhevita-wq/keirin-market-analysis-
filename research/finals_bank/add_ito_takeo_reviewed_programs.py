"""Apply four SHA-locked visual source reviews, seven male finals only.

Geometry and cover years are manual visual transcription, not an automatic
geometry parser. Ito companion page2 header dates independently checked.
Takeo daily paper applies for September16 only, never the September1 point.
"""
import argparse,copy,datetime,hashlib,json,pathlib,re,sqlite3,subprocess,unicodedata
from apply_historical_bank import REQUIRED,validate,select
FINGERPRINT='7685409265b657b95d3eae99d0a9361e2b17ac7abf4711bc9d72fdeec0036f81'

def record(r):
    start=datetime.date.fromisoformat(r['valid_from']);end=datetime.date.fromisoformat(r['valid_until_exclusive'])
    days=[start+datetime.timedelta(days=i) for i in range((end-start).days)]
    assert len(days) in (1,3)
    assert ['月火水木金土日'[d.weekday()] for d in days]==r['weekdays']
    assert (r['track'],r['venue_code']) in [('伊東競輪','37'),('武雄競輪','84')]
    assert r['key'].startswith('ito' if r['venue_code']=='37' else 'takeo')
    assert set(r['geometry'])==set(REQUIRED)
    assert r['review_method'].startswith('Manual visual transcription')
    assert all(i.startswith(r['venue_code']) for i in r['expected_race_ids'])
    return {'bank_version_id':r['venue_code']+'-'+r['key'],'track':r['track'],'venue_code':r['venue_code'],'valid_from':r['valid_from'],'valid_until_exclusive':r['valid_until_exclusive'],'geometry':r['geometry'],'sources':[{'source_url':r['source_url'],'raw_file':r['raw_file'],'sha256':r['sha256'],'bytes':r['bytes'],'location':r['visual_location']+'; SHA-locked manual review '+FINGERPRINT}],'review_status':'verified_within_source_event' if len(days)==3 else 'verified_on_source_date','validity_basis':'Dated official source with all seven geometry labels; only the explicitly reviewed event or exact paper date applies. No interpolation between sources.','visual_review':r['review_method']+' '+r['visual_location']}

def add(data):
    b=(data/'bank_sources_0200/reviewed_records.json').read_bytes();assert hashlib.sha256(b).hexdigest()==FINGERPRINT
    reviews=json.loads(b);records=[];c=sqlite3.connect(f'file:{data}/finals.sqlite?mode=ro',uri=True)
    rows=c.execute('select race_id,day,payload,status from races').fetchall()
    for r in reviews:
        raw=(data/r['raw_file']).read_bytes();assert len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256']
        rec=record(r);validate([rec])
        if r['venue_code']=='37':
            h=subprocess.check_output(['pdftotext','-f','2','-l','2','-layout',str(data/r['raw_file']),'-']).decode()
            h=re.sub(r'\s+','',unicodedata.normalize('NFKC',h)).replace('・祝','')
            s=datetime.date.fromisoformat(r['valid_from']);dates=[s+datetime.timedelta(days=i) for i in range(3)]
            expected='FIナイター'+str(s.month)+'月'+''.join(str(d.day)+'('+w+')' for d,w in zip(dates,r['weekdays']))
            assert h.startswith(expected),(h[:100],expected)
        hits=[]
        for rid,day,payload,status in rows:
            d=json.loads(payload)
            if status!='excluded' and select([rec],rid,day,d['race']['track']):
                assert d.get('bank') is None;hits.append(rid)
        assert sorted(hits)==sorted(r['expected_race_ids']),(r['key'],hits)
        rid=hits[0]
        for badid,badday,badtrack in [('99'+rid[2:],r['valid_from'],r['track']),(rid,(datetime.date.fromisoformat(r['valid_from'])-datetime.timedelta(days=1)).isoformat(),r['track']),(rid,r['valid_until_exclusive'],r['track']),(rid,r['valid_from'],'玉野競輪')]:
            assert select([rec],badid,badday,badtrack) is None
        for mutate in ['missing_geometry','wrong_weekday','wrong_venue']:
            bad=copy.deepcopy(r)
            if mutate=='missing_geometry':bad['geometry'].pop('center_width_m')
            elif mutate=='wrong_weekday':bad['weekdays'][0]='X'
            else:bad['venue_code']='61'
            try:record(bad)
            except AssertionError:pass
            else:raise AssertionError(('bad review accepted',mutate))
        records.append(rec)
    c.close();p=data/'historical_bank_catalog.json';cat=json.loads(p.read_text());validate(cat['records']+records)
    assert not {r['bank_version_id'] for r in records}&{r['bank_version_id'] for r in cat['records']}
    cat['records']+=records;p.write_text(json.dumps(cat,ensure_ascii=False,indent=2));return [r['bank_version_id'] for r in records]

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=pathlib.Path,required=True);print(add(p.parse_args().data))
