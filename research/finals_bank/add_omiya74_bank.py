"""Parse primary bank labels; event year is separately visually attested.

The outer leaflet has an isolated stale Reiwa4 timetable heading. The explicit
2023 cover and all four weekday/date pairs, plus the companion74th event text,
establish this event. No year is taken from a filename or current rider data.
"""
import argparse,datetime,hashlib,json,pathlib,re,subprocess,unicodedata
from apply_historical_bank import validate,select

def geometry(text):
    s=re.sub(r'\s+','',unicodedata.normalize('NFKC',text))
    def one(pattern):
        a=re.findall(pattern,s);assert len(a)==1,(pattern,a);return a[0]
    g={}
    for key,label in [('circumference_m','周長'),('minashi_straight_m','見なし直線距離'),('home_width_m','ホーム幅員'),('back_width_m','バック幅員'),('center_width_m','センター幅員')]:
        g[key]=float(one(label+r'([0-9]+(?:\.[0-9]+)?)m'))
    for key,label in [('home_cant_dms','直線部路面傾斜'),('center_cant_dms','センター部路面傾斜')]:
        g[key]=list(map(int,one(label+r'(\d+)°(\d+)′(\d+)′′')))
    return g

def add(data):
    folder=data/'bank_sources_1916';raw=(folder/'event_visual_review.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==(folder/'review_fingerprint.txt').read_text().strip()
    review=json.loads(raw);assert review['method']=='manual_visual_review_of_explicit_2023_cover_and_four_dates'
    assert review['venue']=='大宮競輪'
    assert hashlib.sha256((data/review['source']).read_bytes()).hexdigest()==review['sha256']
    first=datetime.date(review['year'],review['month'],review['first_day']);last=datetime.date(review['year'],review['month'],review['last_day'])
    assert last-first==datetime.timedelta(days=3) and [ (first+datetime.timedelta(days=i)).weekday() for i in range(4)]==[3,4,5,6]
    inner=folder/'leaflet2.pdf'
    full=subprocess.check_output(['pdftotext','-layout',str(inner),'-']).decode()
    compact=re.sub(r'\s+','',unicodedata.normalize('NFKC',full))
    assert all('1/'+str(i) in compact for i in range(19,23))
    text=subprocess.check_output(['pdftotext','-f','1','-l','1','-r','72','-x','300','-y','620','-W','180','-H','190','-layout',str(inner),'-']).decode()
    g=geometry(text)
    sources=[]
    for name,location in [('leaflet1.pdf','page1 explicit2023cover andJan19-22dates, Omiya74th title; stale isolatedReiwa4heading rejected'),('leaflet2.pdf','page1 labelled bank table; timetableJan19-22 and74th event prose')]:
        b=(folder/name).read_bytes();sources.append({'source_url':'https://www.keirin-saitama.jp/omiya/wp-content/uploads/pdf/'+name,'raw_file':'bank_sources_1916/'+name,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'location':location})
    r={'bank_version_id':'25-omiya74-20230119','venue_code':'25','track':'大宮競輪','valid_from':first.isoformat(),'valid_until_exclusive':(last+datetime.timedelta(days=1)).isoformat(),'geometry':g,'sources':sources,'review_status':'verified_within_source_event','validity_basis':review['corroboration'],'visual_review':'Cover explicit2023Jan19-22 and74th anniversary visually reviewed; enlarged companion table confirms all7 parsed labels. No annual extension.','geometry_method':'automatic_label_parser_on_PDF_table_crop','date_method':review['method']}
    validate([r]);assert select([r],'2520230119040012','2023-01-22','大宮競輪')==r
    for rid,day,track in [('2620230119040012','2023-01-22','大宮競輪'),('2520230119040012','2023-01-18','大宮競輪'),('2520230119040012','2023-01-23','大宮競輪'),('2520230119040012','2023-01-22','西武園競輪')]:assert select([r],rid,day,track) is None
    p=data/'historical_bank_catalog.json';c=json.loads(p.read_text());c['records']=[x for x in c['records'] if x['bank_version_id']!=r['bank_version_id']]+[r];validate(c['records']);p.write_text(json.dumps(c,ensure_ascii=False,indent=2));return r

def test():
    s='周長500m 見なし直線距離66.7m ホーム幅員10.3m バック幅員9.3m センター幅員7.5m 直線部路面傾斜3°26′1″ センター部路面傾斜26°16′40″'
    assert geometry(s)['back_width_m']==9.3
    assert geometry(s.replace('9.3m','9.4m'))['back_width_m']==9.4
    for bad in [s.replace('バック幅員9.3m',''),s+' バック幅員9.4m']:
        try:geometry(bad)
        except AssertionError:pass
        else:raise AssertionError('missing or conflicting label accepted')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=pathlib.Path,required=True);a=p.parse_args();test();print(json.dumps(add(a.data),ensure_ascii=False))
