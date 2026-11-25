"""One-off Jul-Oct 2026 archive, using existing authorized encrypted odds workflow.
No production changes. No public plaintext odds; no betting, no automatic restart.
"""
import csv,gzip,json,hashlib,math,os,subprocess,time,sys
from collections import Counter,defaultdict
from datetime import date,timedelta
from pathlib import Path
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from odds_market import collect_all_odds as c
from keirin_market_analysis import historical_final_odds as h

START=date(2026,7,1);END=date(2026,10,31)
DEST=Path("data/odds_h2_2026_encrypted")
CERT=Path("odds_market/vault_public_cert.pem")
TEMP=Path(os.environ["RUNNER_TEMP"])/"odds_h2_private"
def seal(day,kind,rows,run):
    if not rows:return None
    p=TEMP/(day.isoformat()+"-"+kind+".csv.gz")
    p.parent.mkdir(parents=True,exist_ok=True)
    dest=DEST/kind/day.isoformat()/(run+".p7m")
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():raise RuntimeError("refusing overwrite")
    try:
        with gzip.open(p,"wt",encoding="utf-8",newline="") as f:
            w=csv.writer(f);w.writerow(c.FIELDS);w.writerows(rows)
        if kind=="all_bets" and sum(map(len,c.complete_saved_races(p).values()))!=len(rows):
            raise ValueError("full book integrity check failed")
        tmp=dest.with_suffix(".tmp")
        subprocess.run(["openssl","cms","-encrypt","-binary","-aes-256-gcm","-in",str(p),"-out",str(tmp),"-outform","DER",str(CERT)],check=True,capture_output=True)
        if tmp.stat().st_size<500:raise ValueError("short ciphertext")
        tmp.replace(dest)
        return {"path":str(dest),"sha256":hashlib.sha256(dest.read_bytes()).hexdigest(),"races":len({r[0] for r in rows}),"rows":len(rows)}
    finally:p.unlink(missing_ok=True)
def known_cars():
    out=defaultdict(set)
    with gzip.open("data/rider_condition_2025_202610_full/rider_races_2026.csv.gz","rt",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if "2026-07-01"<=r["race_date"]<="2026-10-31":
                out[r["race_id"]].add(int(r["car_no"]))
    return {k:sorted(v) for k,v in out.items()}
def already():
    all_ids=set();two_ids=set()
    for p in Path("data/odds_universe_v1/data/2026_future_block1").rglob("summary.json"):
        two_ids.update(json.loads(p.read_text()).get("complete_race_ids",[]))
    for p in DEST.glob("audit/*.json"):
        a=json.loads(p.read_text())
        all_ids.update(a.get("all_ids",[]));two_ids.update(a.get("two_ids",[]))
    return all_ids,two_ids
def fallback(html,rid,day,cars):
    if not cars or not 5<=len(cars)<=9:raise ValueError("no verified entrant universe")
    soup=BeautifulSoup(html,"lxml");label=soup.get_text(" ",strip=True)
    race_type=c.race_type_from_page(soup)
    match=c.DATE_RX.search(label)
    if not match or date(*map(int,match.groups()))!=day:raise ValueError("page date mismatch")
    header=h.parse_market_header(soup)
    out=[]
    import itertools
    for bet,k,func,expected in [("3連単","3rentan",h.parse_trifecta,set(itertools.permutations(cars,3))),("3連複","3renpuku",h.parse_trio,set(itertools.combinations(cars,3)))]:
        a=func(soup,cars)
        if set(a)!=expected:raise ValueError("incomplete book")
        stamp=header[bet]["odds_as_of"]
        if not stamp:raise ValueError("missing odds time")
        for key,(val,status) in sorted(a.items()):
            if status!="available" or not math.isfinite(float(val)) or float(val)<1:raise ValueError("unavailable odds")
            out.append((rid,race_type,k,"-".join(map(str,key)),val,val,stamp))
    return out
def main():
    month=int(sys.argv[1])
    if month not in (7,8,9,10):raise ValueError("scope")
    run=os.environ["GITHUB_RUN_ID"]+"-"+os.environ["GITHUB_RUN_ATTEMPT"]
    start=date(2026,month,1);end=min(END,date(2026,month+1,1)-timedelta(days=1))
    assert START<=start<=end<=END
    cars=known_cars();prior_full,prior_two=already()
    session=requests.Session();session.headers.update({"User-Agent":"Keirin-Research-Odds-Collector/1.0"})
    for i in range((end-start).days+1):
        day=start+timedelta(days=i);counts=Counter();full=[];partial=[];errors={};halt=False
        all_ids=[];two_ids=[]
        try:urls=c.discover(session,day,2)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code in (401,403,429):raise SystemExit("Provider refused access; stopped without retry")
            urls={};errors["index"]=type(e).__name__
        except Exception as e:urls={};errors["index"]=type(e).__name__
        counts["discovered"]=len(urls)
        for rid,url in sorted(urls.items()):
            if rid in prior_full:counts["already_all"]+=1;continue
            try:
                if urlparse(url).hostname!="keirin.kdreams.jp":raise ValueError("unexpected host")
                time.sleep(2);resp=session.get(url,timeout=35);resp.raise_for_status()
                try:
                    rows=c.parse_all(resp.text,rid)
                    if rid in cars:
                        got={int(v) for r in rows if r[2]=="3rentan" for v in r[3].split("-")}
                        if got!=set(cars[rid]):raise ValueError("entrant universe differs")
                    full.extend(rows);all_ids.append(rid);prior_full.add(rid);counts["new_all"]+=1
                except c.ExcludedRace:counts["girls"]+=1
                except Exception as err:
                    counts["all_incomplete"]+=1
                    if rid in prior_two:counts["already_two"]+=1;continue
                    try:
                        rows=fallback(resp.text,rid,day,cars.get(rid))
                        partial.extend(rows);two_ids.append(rid);prior_two.add(rid);counts["new_two_only"]+=1
                    except c.ExcludedRace:counts["girls"]+=1
                    except Exception as err2:counts["unresolved"]+=1;errors[rid]=str(err2)[:180]
            except requests.HTTPError as e:
                status=e.response.status_code if e.response is not None else 0
                counts["http_error"]+=1;errors[rid]="HTTP "+str(status)
                if status in (401,403,429):halt=True;break
            except Exception as e:counts["other_error"]+=1;errors[rid]=type(e).__name__
        sealed=[x for x in [seal(day,"all_bets",full,run),seal(day,"two_bets_only",partial,run)] if x]
        audit={"date":day.isoformat(),"counts":dict(counts),"all_ids":all_ids,"two_ids":two_ids,"errors":errors,"ciphertexts":sealed,"scope":"2026-07-01..2026-10-31; final odds only; encrypted, no plain market data"}
        dest=DEST/"audit"/(day.isoformat()+"-"+run+".json");dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(audit,ensure_ascii=False,indent=2))
        print("DAY",day,json.dumps(dict(counts)),flush=True)
        if halt:raise SystemExit("Provider refused access; stopped without retry")
if __name__=="__main__":main()
