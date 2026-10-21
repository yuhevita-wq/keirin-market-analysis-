"""Apply a visually reviewed outlined official leaflet, with explicit provenance.

This is a manual primary-source transcription, NOT an automatic PDF text parser.
The review fingerprint binds the numeric values to the reviewed raw PDF evidence.
"""
import argparse,datetime,hashlib,json,pathlib,re,subprocess,unicodedata
from apply_historical_bank import validate,select

def dates(text):
    s=re.sub(r'\s+','',unicodedata.normalize('NFKC',text))
    m=re.search(r'(20\d{2})年(\d{1,2})月(\d{1,2})日\(木\)〜(\d{1,2})日\(日\)の4日間、大宮競輪場を舞台に、東日本競輪発祥77周年記念「倉茂記念杯」',s)
    assert m,'missing independent event date/venue prose'
    y,month,first,last=map(int,m.groups());a=datetime.date(y,month,first);b=datetime.date(y,month,last)
    assert b-a==datetime.timedelta(days=3)
    return a.isoformat(),(b+datetime.timedelta(days=1)).isoformat()

def add(data):
    raw=(data/'bank_sources_1903/reviewed_transcription.json').read_bytes()
    # Replacing this review requires a fresh visual review, never an inferred value.
    expected=(data/'bank_sources_1903/review_fingerprint.txt').read_text().strip()
    assert hashlib.sha256(raw).hexdigest()==expected,'review changed without fresh visual attestation'
    r=json.loads(raw)
    assert r['transcription_method']=='visual_from_outlined_primary_PDF_not_text_parser'
    for s in r['sources']:
        b=(data/s['raw_file']).read_bytes();assert len(b)==s['bytes'] and hashlib.sha256(b).hexdigest()==s['sha256']
    companion=data/'bank_sources_1903/omiya2026-2.pdf'
    text=subprocess.check_output(['pdftotext','-layout',str(companion),'-']).decode()
    assert dates(text)==(r['valid_from'],r['valid_until_exclusive'])
    assert set(r['geometry'])=={'circumference_m','minashi_straight_m','home_width_m','back_width_m','center_width_m','home_cant_dms','center_cant_dms'}
    validate([r]);assert select([r],'2520260115040012','2026-01-18','大宮競輪')==r
    for rid,day,track in [('2620260115040012','2026-01-18','大宮競輪'),('2520260115040012','2026-01-14','大宮競輪'),('2520260115040012','2026-01-19','大宮競輪'),('2520260115040012','2026-01-18','西武園競輪')]:assert select([r],rid,day,track) is None
    p=data/'historical_bank_catalog.json';c=json.loads(p.read_text());c['records']=[x for x in c['records'] if x['bank_version_id']!=r['bank_version_id']]+[r];validate(c['records']);p.write_text(json.dumps(c,ensure_ascii=False,indent=2));return r

def test():
    s='2026年1月15日(木)〜18日(日)の4日間、大宮競輪場を舞台に、東日本競輪発祥77周年記念「倉茂記念杯」'
    assert dates(s)==('2026-01-15','2026-01-19')
    for bad in [s.replace('18日','19日'),s.replace('大宮','西武園'),s.replace('2026年','')]:
        try:dates(bad)
        except AssertionError:pass
        else:raise AssertionError('ungrounded event accepted')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True,type=pathlib.Path);args=p.parse_args();test();print(json.dumps(add(args.data),ensure_ascii=False))
