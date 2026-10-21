"""Recover geometry from saved official event tables; never bridge between events."""
import argparse,pathlib,json,gzip,hashlib,re,datetime
from lxml import html

def compact(s):return re.sub(r'\s+','',s)
def parse(raw):
    doc=html.fromstring(raw);rows={}
    for tr in doc.xpath('//table//tr'):
        hs=tr.xpath('./th');ts=tr.xpath('./td')
        if len(hs)==1 and len(ts)==1:
            key=compact(hs[0].text_content());value=compact(ts[0].text_content())
            if key in ['周長','長軸','短軸','見なし直線距離','路面傾斜','幅員']:
                if key in rows:assert rows[key]==value,('conflicting source cells',key)
                rows[key]=value
    def number(key):
        m=re.fullmatch(r'(\d+(?:\.\d+)?)メートル',rows[key]);assert m,(key,rows[key]);return float(m[1])
    g={k:number(label) for k,label in [('circumference_m','周長'),('long_axis_m','長軸'),('short_axis_m','短軸'),('minashi_straight_m','見なし直線距離')]}
    for key,label in [('home_cant_dms','直線部'),('center_cant_dms','センター部')]:
        m=re.search(label+r'(\d+)°(\d+)[’′](\d+)[”″]',rows['路面傾斜']);assert m;g[key]=list(map(int,m.groups()))
    for key,label in [('home_width_m','ホーム'),('back_width_m','バック'),('center_width_m','センター')]:
        m=re.search(label+r'(\d+(?:\.\d+)?)メートル',rows['幅員']);assert m;g[key]=float(m[1])
    return g,compact(doc.text_content())

def add(data):
    seeds=json.loads((data/'bank_sources_0936/events.json').read_text());catpath=data/'historical_bank_catalog.json';cat=json.loads(catpath.read_text());added=[]
    for seed in seeds:
        path=data/seed['raw_file'];raw=gzip.decompress(path.read_bytes());assert hashlib.sha256(raw).hexdigest()==seed['sha256']
        geometry,text=parse(raw)
        for token in seed['date_evidence_tokens']:assert compact(token) in text,('date not grounded',seed['source_url'],token)
        record={'bank_version_id':seed['bank_version_id'],'track':'小田原競輪','venue_code':'36','valid_from':seed['valid_from'],'valid_until_exclusive':seed['valid_until_exclusive'],
            'geometry':geometry,'sources':[{k:seed[k] for k in ['source_url','raw_file','sha256','bytes','location']}],
            'validity_basis':'Official event-specific timetable and dated banner-submission notice identify this event only. Geometry extracted from that event page tables. No extension across intervening dates or2024-2025 reconstruction.',
            'review_status':'verified_within_source_event','date_evidence_tokens':seed['date_evidence_tokens']}
        old=[x for x in cat['records'] if x['bank_version_id']==record['bank_version_id']]
        if old:assert old==[record]
        else:cat['records'].append(record);added.append(record['bank_version_id'])
    from apply_historical_bank import validate,select
    validate(cat['records'])
    for s in seeds:
        assert select(cat['records'],'36-test',s['valid_from'],'小田原競輪')
        assert select(cat['records'],'61-test',s['valid_from'],'小田原競輪') is None
        assert select(cat['records'],'36-test',s['valid_until_exclusive'],'小田原競輪') is None
    catpath.write_text(json.dumps(cat,ensure_ascii=False,indent=2));print({'event_records_added':added})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=pathlib.Path,required=True);add(p.parse_args().data)
