"""Apply visually reviewed original official programs; not an OCR inference.

Dates of all six and geometry of two are manual primary-source transcriptions.
Four tables also have an independent PDF-label parser corroboration.
"""
import argparse,datetime,hashlib,json,pathlib,sqlite3,subprocess
from apply_historical_bank import validate,select
from add_hiroshima_host_program_banks import geometry

def checked_record(r):
    assert r['venue_code']=='61' and r['track']=='玉野競輪'
    assert r['date_method']=='manual_visual_from_explicit_cover_year_and_three_dates'
    first=datetime.date.fromisoformat(r['valid_from']);last=datetime.date.fromisoformat(r['valid_until_exclusive'])
    assert last-first==datetime.timedelta(days=3)
    assert ['月火水木金土日'[(first+datetime.timedelta(days=i)).weekday()] for i in range(3)]==r['reviewed_weekdays']
    assert set(r['geometry'])=={'circumference_m','minashi_straight_m','home_width_m','back_width_m','center_width_m','home_cant_dms','center_cant_dms'}
    validate([r])
    rid=r['expected_race_ids'][0]
    assert select([r],rid,first.isoformat(),'玉野競輪')==r
    for testid,day,track in [('62'+rid[2:],first.isoformat(),'玉野競輪'),(rid,(first-datetime.timedelta(days=1)).isoformat(),'玉野競輪'),(rid,last.isoformat(),'玉野競輪'),(rid,first.isoformat(),'広島競輪')]:assert select([r],testid,day,track) is None

def add(data):
    folder=data/'bank_sources_2209';raw=(folder/'reviewed_records.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==(folder/'review_fingerprint.txt').read_text().strip(),'manual review modified without new attestation'
    records=json.loads(raw);assert len(records)==6
    c=sqlite3.connect('file:'+str((data/'finals.sqlite').resolve())+'?mode=ro',uri=True)
    for r in records:
        checked_record(r)
        for s in r['sources']:
            p=data/s['raw_file'];b=p.read_bytes();assert len(b)==s['bytes'] and hashlib.sha256(b).hexdigest()==s['sha256']
            if r['geometry_method']=='manual_visual_primary_PDF_table_with_independent_text_corroboration':
                text=subprocess.check_output(['pdftotext','-layout',str(p),'-']).decode();assert geometry(text,r['track'])==r['geometry']
            else:assert r['geometry_method']=='manual_visual_primary_PDF_table_not_automatic_text_parser'
        ids=[]
        for rid,day,payload in c.execute('select race_id,day,payload from races where day>=? and day<?',(r['valid_from'],r['valid_until_exclusive'])):
            d=json.loads(payload)
            if select([r],rid,day,d['race']['track']):ids.append(rid)
        assert sorted(ids)==r['expected_race_ids'] and len(ids)==2
    c.close()
    p=data/'historical_bank_catalog.json';cat=json.loads(p.read_text());ids={r['bank_version_id'] for r in records};cat['records']=[r for r in cat['records'] if r['bank_version_id'] not in ids]+records;validate(cat['records']);p.write_text(json.dumps(cat,ensure_ascii=False,indent=2));return records

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=pathlib.Path,required=True);a=p.parse_args();print(json.dumps({'reviewed_records':len(add(a.data))}))
