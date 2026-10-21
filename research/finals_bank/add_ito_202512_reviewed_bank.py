"""Add only the visually reviewed official Ito December17-19,2025 event.

Geometry is manual visual transcription, not an automatic PDF geometry parser.
The SHA-locked review is tied to the original PDF and a separately checked
page2 event header. Printed circumference333m is retained without rounding.
"""
import argparse,copy,datetime,hashlib,json,pathlib,re,sqlite3,subprocess,unicodedata
from apply_historical_bank import REQUIRED,validate,select

FINGERPRINT='d472645a691b0d3e38fe4b9e97508b85ddc570d196de373cbd523d3d6e63be3b'

def check_review(r,raw,header):
    assert hashlib.sha256(raw).hexdigest()==r['sha256']
    assert len(raw)==r['bytes']
    assert r['track']=='伊東競輪' and r['venue_code']=='37'
    assert set(r['geometry'])==set(REQUIRED)
    days=[datetime.date.fromisoformat(s) for s in r['days']]
    assert len(days)==3 and days[0].isoformat()==r['start']
    assert all(d.year==r['year'] for d in days)
    assert all(b-a==datetime.timedelta(days=1) for a,b in zip(days,days[1:]))
    assert (days[-1]+datetime.timedelta(days=1)).isoformat()==r['until_exclusive']
    assert ['月火水木金土日'[d.weekday()] for d in days]==r['weekdays']
    s=re.sub(r'\s+','',unicodedata.normalize('NFKC',header))
    expected='FIナイター'+str(days[0].month)+'月'+''.join(str(d.day)+'('+w+')' for d,w in zip(days,r['weekdays']))
    assert s.startswith(expected),(s[:100],expected)
    assert r['review_method'].startswith('Manual visual transcription:')

def build(r):
    return {'bank_version_id':'37-ito-program-20251217','venue_code':r['venue_code'],'track':r['track'],'valid_from':r['start'],'valid_until_exclusive':r['until_exclusive'],'geometry':r['geometry'],'sources':[{'source_url':r['source_url'],'raw_file':r['raw_file'],'sha256':r['sha256'],'bytes':r['bytes'],'location':'PDF page1 dated cover and phone37;page2 lower-right complete bank diagram; manual reviewed_transcription.json SHA '+FINGERPRINT}],'review_status':'verified_within_source_event','validity_basis':'Official event program explicitly identifies2025December17-19 and Ito. Only that event applies; no interpolation to other dates. Circumference333m retained as printed.','visual_review':r['review_method']}

def run(data):
    p=data/'bank_sources_0049/reviewed_transcription.json';b=p.read_bytes()
    assert hashlib.sha256(b).hexdigest()==FINGERPRINT
    r=json.loads(b);raw=(data/r['raw_file']).read_bytes()
    header=subprocess.check_output(['pdftotext','-f','2','-l','2','-layout',str(data/r['raw_file']),'-']).decode()
    check_review(r,raw,header);record=build(r);validate([record])
    for day in r['days']:
        assert select([record],r['expected_race_ids'][0],day,r['track'])==record
    for rid,day,track in [('6120251217030011','2025-12-19',r['track']),(r['expected_race_ids'][0],'2025-12-16',r['track']),(r['expected_race_ids'][0],'2025-12-20',r['track']),(r['expected_race_ids'][0],'2025-12-19','玉野競輪')]:
        assert select([record],rid,day,track) is None
    c=sqlite3.connect(f'file:{data}/finals.sqlite?mode=ro',uri=True)
    hits=[]
    for rid,day,payload,status in c.execute('select race_id,day,payload,status from races'):
        d=json.loads(payload)
        if status!='excluded' and select([record],rid,day,d['race']['track']):
            assert d.get('bank') is None;hits.append(rid)
    assert sorted(hits)==sorted(r['expected_race_ids']);c.close()
    # Meaningful source-boundary and missing-attribute rejection checks.
    for key,val in [('track','玉野競輪'),('venue_code','61'),('year',2024),('weekdays',['木','金','土'])]:
        bad=copy.deepcopy(r);bad[key]=val
        try:check_review(bad,raw,header)
        except AssertionError:pass
        else:raise AssertionError(('Invalid review accepted',key))
    bad=copy.deepcopy(r);bad['geometry'].pop('center_width_m')
    try:check_review(bad,raw,header)
    except AssertionError:pass
    else:raise AssertionError('Missing geometry accepted')
    cp=data/'historical_bank_catalog.json';cat=json.loads(cp.read_text())
    assert all(x['bank_version_id']!=record['bank_version_id'] for x in cat['records'])
    cat['records'].append(record);validate(cat['records'])
    cp.write_text(json.dumps(cat,ensure_ascii=False,indent=2));return record

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=pathlib.Path,required=True)
    print(json.dumps(run(p.parse_args().data),ensure_ascii=False))
