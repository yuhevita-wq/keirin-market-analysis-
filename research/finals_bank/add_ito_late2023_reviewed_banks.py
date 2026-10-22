"""Six exact male finals from three visually reviewed official Ito 2023 sources.

Geometry is SHA-locked manual visual transcription. Explicit2023year is visually read from each cover; page2eventheaders and weekdays are independently checked.
"""
import argparse,copy,datetime,hashlib,json,pathlib,re,sqlite3,subprocess,unicodedata
from apply_historical_bank import REQUIRED,validate,select
FINGERPRINT='b5168d49f2a5ea570c690ad1cc7f70e4167529bb2fb9c7fbc1213bc59048faff'

def compact(s):return re.sub(r'\s+','',unicodedata.normalize('NFKC',s))
def add(data):
    raw=(data/'bank_sources_0231/reviewed_records.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==FINGERPRINT
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
        assert r['key'] in {'ito20230718','ito20230830','ito20231030'}
        assert (end-start).days==3 and start.year==2023
        h=subprocess.check_output(['pdftotext','-f','2','-l','2','-r','72','-x','385','-y','35','-W','330','-H','40','-layout',str(data/r['raw_file']),'-']).decode()
        assert compact(h).startswith(r['header'])
        cover=subprocess.check_output(['pdftotext','-f','1','-l','1','-layout',str(data/r['raw_file']),'-']).decode()
        assert '2023' in compact(cover) and r['month'] in compact(cover)
        for bad in [compact(h).replace(r['month'],'99月',1),compact(h).replace('('+r['weekdays'][0]+')','(誤)',1)]:
            assert not compact(bad).startswith(r['header'])
        rec={'bank_version_id':'37-'+r['key']+'-reviewed','track':r['track'],'venue_code':r['venue_code'],'valid_from':r['valid_from'],'valid_until_exclusive':r['valid_until_exclusive'],'geometry':r['geometry'],'sources':sources,'review_status':'verified_within_source_event','validity_basis':'Only explicitly reviewed 2023 source event, cover year and three weekdays plus independent page2 header. No period interpolation. Printed333m retained.','visual_review':r['review_method']+' '+r['visual_location']};validate([rec])
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
