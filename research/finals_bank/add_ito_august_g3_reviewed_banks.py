"""Three exact male finals from two visually reviewed official Ito sources.

Geometry is SHA-locked manual visual transcription. G3 year and final day
come from the explicit official companion article, not the PDF filename.
"""
import argparse,copy,datetime,hashlib,json,pathlib,re,sqlite3,subprocess,unicodedata
from apply_historical_bank import REQUIRED,validate,select
FINGERPRINT='702d8c95587e74cea04d4389df6bc10265ee984f6d00e5cee460dae06e1f7bb0'

def compact(s):return re.sub(r'\s+','',unicodedata.normalize('NFKC',s))
def g3_date_proof(html):
    s=compact(re.sub('<[^>]*>',' ',html))
    assert '2026年9月16日に開催します伊東温泉競輪' in s
    assert 'MIDNIGHTGIII×HPCJCTIPSTAR杯' in s
    assert '9RS級決勝出場選手インタビュー(2026.09.16)' in s

def add(data):
    raw=(data/'bank_sources_0242/reviewed_records.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==FINGERPRINT
    reviews=json.loads(raw);records=[];c=sqlite3.connect(f'file:{data}/finals.sqlite?mode=ro',uri=True)
    rows=c.execute('select race_id,day,payload,status from races').fetchall()
    for r in reviews:
        pdf=(data/r['raw_file']).read_bytes();assert len(pdf)==r['bytes'] and hashlib.sha256(pdf).hexdigest()==r['sha256']
        assert r['track']=='伊東競輪' and r['venue_code']=='37'
        assert set(r['geometry'])==set(REQUIRED) and r['review_method'].startswith('Manual visual transcription')
        start=datetime.date.fromisoformat(r['valid_from']);end=datetime.date.fromisoformat(r['valid_until_exclusive'])
        days=[start+datetime.timedelta(days=i) for i in range((end-start).days)]
        assert ['月火水木金土日'[d.weekday()] for d in days]==r['weekdays']
        sources=[{k:r[k] for k in ['source_url','raw_file','sha256','bytes']}];sources[0]['location']=r['visual_location']+' ReviewSHA '+FINGERPRINT
        if r['key']=='ito20260816':
            assert (start.isoformat(),end.isoformat())==('2026-08-16','2026-08-19')
            h=subprocess.check_output(['pdftotext','-f','2','-l','2','-layout',str(data/r['raw_file']),'-']).decode()
            assert compact(h).startswith('FIIナイター8月16(日)17(月)18(火)')
        elif r['key']=='ito20260914':
            assert (start.isoformat(),end.isoformat())==('2026-09-16','2026-09-17')
            s=r['companion'];b=(data/s['raw_file']).read_bytes();assert len(b)==s['bytes'] and hashlib.sha256(b).hexdigest()==s['sha256']
            html=b.decode();g3_date_proof(html);sources.append(s)
            for bad in [html.replace('2026年9月16日','2025年9月16日'),html.replace('S級','A級')]:
                try:g3_date_proof(bad)
                except AssertionError:pass
                else:raise AssertionError('Changed companion event accepted')
        else:raise AssertionError('Unreviewed source')
        rec={'bank_version_id':'37-'+r['key']+'-reviewed','track':r['track'],'venue_code':r['venue_code'],'valid_from':r['valid_from'],'valid_until_exclusive':r['valid_until_exclusive'],'geometry':r['geometry'],'sources':sources,'review_status':'verified_within_source_event','validity_basis':'Only explicitly reviewed source event, or exact GIII finalday with official companion year/date/race identification. No period interpolation. Printed333m retained.','visual_review':r['review_method']+' '+r['visual_location']};validate([rec])
        hits=[]
        for rid,day,payload,status in rows:
            d=json.loads(payload)
            if status!='excluded' and select([rec],rid,day,d['race']['track']):
                assert d.get('bank') is None;hits.append(rid)
        assert sorted(hits)==sorted(r['expected_race_ids']),(r['key'],hits)
        for rid,day,track in [('6133333333333333',r['valid_from'],r['track']),(hits[0],(start-datetime.timedelta(days=1)).isoformat(),r['track']),(hits[0],r['valid_until_exclusive'],r['track']),(hits[0],r['valid_from'],'玉野競輪')]:assert select([rec],rid,day,track) is None
        bad=copy.deepcopy(rec);bad['geometry'].pop('center_width_m')
        try:validate([bad])
        except AssertionError:pass
        else:raise AssertionError('Missing geometry accepted')
        records.append(rec)
    c.close();p=data/'historical_bank_catalog.json';cat=json.loads(p.read_text());validate(cat['records']+records)
    assert not {r['bank_version_id'] for r in records}&{r['bank_version_id'] for r in cat['records']}
    cat['records']+=records;p.write_text(json.dumps(cat,ensure_ascii=False,indent=2));return records

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=pathlib.Path,required=True);print([r['bank_version_id'] for r in add(p.parse_args().data)])
