"""Apply the explicitly dated 2023 Hofu G3 program to actual Tamano venue only."""
import pathlib,json,hashlib,subprocess,re,unicodedata
from apply_historical_bank import select,validate

def compact(s):return re.sub(r'\s+','',unicodedata.normalize('NFKC',s))
def geometry(text):
    s=compact(text);g={}
    for label,key in [('1周','circumference_m'),('見なし直線','minashi_straight_m'),('ホーム幅員','home_width_m'),('バック幅員','back_width_m'),('センター幅員','center_width_m')]:
        v=re.findall(re.escape(label)+r'(\d+(?:\.\d+)?)m',s)
        assert len(v)==1,(label,v);g[key]=float(v[0])
    for label,key in [('最大カント','center_cant_dms'),('直線カント','home_cant_dms')]:
        v=re.findall(re.escape(label)+r'(\d+)°(\d+)′(\d+)′′',s)
        # NFKC maps double-prime to two primes.
        assert len(v)==1,(label,v);g[key]=list(map(int,v[0]))
    return g

def build(data):
    src=json.loads((data/'bank_sources_1316/hofu74_capture.json').read_text())
    p=data/src['raw_file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==src['sha256']
    alltext=compact(subprocess.check_output(['pdftotext','-layout',str(p),'-']).decode())
    for token in ['防府競輪2023.9.27','玉野競輪場を借上げての開催','周防国府杯','11.2木〜11.5日4日間開催']:
        assert token in alltext,('missing event provenance',token)
    text=subprocess.check_output(['pdftotext','-f','1','-l','1','-x','790','-y','395','-W','230','-H','200','-layout',str(p),'-']).decode()
    g=geometry(text)
    r={'bank_version_id':'61-hofu74-20231102','track':'玉野競輪','venue_code':'61','valid_from':'2023-11-02','valid_until_exclusive':'2023-11-06','geometry':g,'sources':[src],'review_status':'verified_within_source_period',
       'validity_basis':'Official Hofu74 program explicitly dated2023.9.27 with Nov2-5 event, actual Tamano borrowed venue and telephone61. Geometry applies to this event only. Archive capture2024 is not event date. Printed statistical sampling period is not geometry validity.',
       'visual_review':'PDF page1 full event cover/dated heading and bank table crop rendered and checked. Actual venue is Tamano, not Hofu.'}
    validate([r]);assert select([r],'6120231102040012','2023-11-05','玉野競輪')==r
    for rid,day,track in [('6320231102040012','2023-11-05','防府競輪'),('6120231102040012','2023-11-06','玉野競輪'),('6120231102040012','2023-11-01','玉野競輪')]:assert select([r],rid,day,track) is None
    catp=data/'historical_bank_catalog.json';cat=json.loads(catp.read_text());old=[x for x in cat['records'] if x['bank_version_id']!=r['bank_version_id']];validate(old+[r]);cat['records']=old+[r];catp.write_text(json.dumps(cat,ensure_ascii=False,indent=2));return r

def test():
    s='1周400m 見なし直線47.9m 最大カント30°37′33″ 直線カント3°26′1″ ホーム幅員10.3m バック幅員9.3m センター幅員7.5m'
    assert geometry(s)['minashi_straight_m']==47.9
    assert geometry(s.replace('47.9m','48.1m'))['minashi_straight_m']==48.1
    for bad in [s.replace('ホーム幅員10.3m',''),s+' ホーム幅員10.4m']:
        try:geometry(bad)
        except AssertionError:pass
        else:raise AssertionError('incomplete or ambiguous geometry accepted')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data',type=pathlib.Path,required=True);v=p.parse_args();test();print(json.dumps(build(v.data),ensure_ascii=False))
