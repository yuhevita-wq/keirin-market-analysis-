"""Parse dated partial bank evidence without applying it to races or date intervals."""
import pathlib,json,re,unicodedata,hashlib,datetime,argparse
from pypdf import PdfReader

def normalize(t):return re.sub(r'\s+','',unicodedata.normalize('NFKC',t))
def parse_ito(text):
    t=normalize(text);m=re.search(r'競輪場施設[（(]令和(\d+)年(\d+)月(\d+)日現在',t);assert m,'Missing explicit as-of date'
    asof=datetime.date(2018+int(m[1]),int(m[2]),int(m[3])).isoformat()
    def scalar(pattern):
        m=re.search(pattern,t);assert m,pattern;return float(m[1])
    g={'circumference_m':scalar(r'周長[:：](\d+(?:\.\d+)?)m'),
       'home_width_m':scalar(r'ホームストレッチ(\d+(?:\.\d+)?)m'),
       'back_width_m':scalar(r'バックストレッチ(\d+(?:\.\d+)?)m'),
       'center_width_m':scalar(r'センター(\d+(?:\.\d+)?)m')}
    for key,label in [('home_cant_dms','直線部'),('center_cant_dms','曲線部')]:
        m=re.search(label+r'(\d+)度(\d+)分(\d+)秒',t);assert m;g[key]=list(map(int,m.groups()))
    return asof,g
def parse_kokura(text):
    t=normalize(text);m=re.search(r'令和(\d+)年(\d+)月(\d+)日',t);assert m
    day=datetime.date(2018+int(m[1]),int(m[2]),int(m[3])).isoformat()
    # This PDF draws the two labels before the two values. Handle that exact
    # verified header order rather than searching arbitrary racecard numbers.
    pair=re.search(r'みなし直線距離周長(\d+)m(\d+(?:\.\d+)?)m',t);assert pair,'Unrecognized bank-header layout'
    return day,{'circumference_m':float(pair[1]),'minashi_straight_m':float(pair[2])}
def build(data):
    root=data/'bank_sources_1015';manifest=json.loads((root/'manifest.json').read_text());out=[]
    for source in manifest:
        p=root/source['saved_file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==source['saved_sha256']
        r=PdfReader(p);page=r.pages[source['geometry_saved_page_1based']-1];day,g=(parse_ito if source['venue_code']=='37' else parse_kokura)(page.extract_text())
        assert day==source['expected_source_date'];out.append({'venue_code':source['venue_code'],'track':source['track'],'source_date':day,'geometry':g,'source':source,'application_status':'not_applied','continuous_applicability':'not_proven','missing_fields':sorted(set(['circumference_m','minashi_straight_m','home_width_m','back_width_m','center_width_m','home_cant_dms','center_cant_dms'])-g.keys()),'note':'A dated observation is not a certificate of geometry throughout the fiscal year or between archive points.'})
    result={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'observations':out,'canonical_db_modified':False,'resolved_races_added':0,'all_finals_complete':False}
    (data/'historical_bank_point_evidence_1015.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print([(x['track'],x['source_date'],x['missing_fields']) for x in out])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=pathlib.Path,required=True);build(p.parse_args().data)
