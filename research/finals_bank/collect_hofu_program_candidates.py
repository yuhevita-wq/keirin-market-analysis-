"""Bounded source-only probe from known ledger meetings; no race data writes.

Candidate URLs follow the three observed official pYYYYMMDD.pdf paths. A URL
pattern is discovery only: explicit PDF dates/venue/all geometry still required.
Both successes and failures are cached; never automatically retry failures.
"""
import concurrent.futures,datetime,hashlib,json,pathlib,sqlite3,subprocess,urllib.request
from add_hofu_program_banks import parse

def collect(data):
    out=data/'hofu_program_candidates_1248';out.mkdir(exist_ok=True)
    with sqlite3.connect(f'file:{data / "finals.sqlite"}?mode=ro',uri=True) as c:
        dates=sorted({r[0][2:10] for r in c.execute("select race_id from races where race_id like '63%' and status!='excluded'")})
    known=json.loads((data/'bank_sources_1248/manifest.json').read_text())['sources']
    known={r['expected_start'].replace('-',''):r for r in known}
    def one(d):
        meta=out/(d+'.json')
        if meta.exists():return json.loads(meta.read_text())
        u=f'https://www.hofukeirin.jp/wp-content/themes/hofukeirin2023/pdf/p{d}.pdf'
        r={'meeting_start_candidate':d,'source_url':u,'url_discovery':'Derived from observed official filename convention; not evidence of event date','at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'application_status':'not_applied','visual_review':'pending'}
        try:
            if d in known:
                p=data/known[d]['raw_file'];b=p.read_bytes()
            else:
                with urllib.request.urlopen(u,timeout=15) as response:b=response.read(20000000)
                assert b.startswith(b'%PDF-') and len(b)<20000000
                p=out/(d+'.pdf');p.write_bytes(b)
            r.update(raw_file=str(p.relative_to(data)),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),status='retrieved')
            info=subprocess.check_output(['pdfinfo',str(p)]).decode()
            import re
            page_count=int(re.search(r'^Pages:\s*(\d+)',info,re.M).group(1))
            for page in range(1,min(page_count,6)+1):
                txt=subprocess.check_output(['pdftotext','-f',str(page),'-l',str(page),'-layout',str(p),'-'],stderr=subprocess.DEVNULL).decode()
                if not txt.strip():break
                try:
                    start,end,g=parse(txt)
                    assert start.replace('-','')==d
                    r.update(parsed_start=start,parsed_until_exclusive=end,geometry=g,bank_page=page,parse_status='all_seven_fields_and_explicit_event_dates')
                    break
                except (AssertionError,ValueError):pass
            if 'parse_status' not in r:r['parse_status']='needs_layout_or_date_review'
        except Exception as e:r.update(status='failed',error=type(e).__name__+': '+str(e))
        meta.write_text(json.dumps(r,ensure_ascii=False,indent=2));return r
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(one,dates))
    (out/'inventory.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    print(json.dumps({'meetings':len(rows),'retrieved':sum(r['status']=='retrieved' for r in rows),'parsed':sum('geometry' in r for r in rows),'applied':0},ensure_ascii=False))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data',type=pathlib.Path,required=True);collect(p.parse_args().data)
