"""Authorized monthly wide/trio archive. Draft release assets, never public raw data."""
import argparse, csv, gzip, hashlib, itertools, json, math, os, re, sys, time
from datetime import date,timedelta
from pathlib import Path
from collections import defaultdict
import requests
from bs4 import BeautifulSoup
from odds_market import collect_all_odds as source

FIELDS = ("race_date", "race_id", "race_type", "bet_type", "selection", "odds_min", "odds_max", "odds_time_jst", "price_phase")

def parse_two(html,rid,day):
    soup=BeautifulSoup(html,"html.parser")
    label=soup.get_text(" ",strip=True)
    rt=source.race_type_from_page(soup)
    if "確定オッズ" not in label:
        raise ValueError("not explicitly final")
    match=source.DATE_RX.search(label)
    if not match or date(*map(int,match.groups()))!=day:
        raise ValueError("source date mismatch")
    stamps=re.findall(r"(20\d\d/\d\d/\d\d)\s+(\d\d:\d\d)現在",label)
    if not stamps:
        raise ValueError("missing source timestamp")
    stamp=stamps[0][0].replace("/","-")+"T"+stamps[0][1]+":00+09:00"
    tables=[source.rows(t) for t in soup.find_all("table")]
    for n in (9,8,7,6,5,4,3):
        exp=source.expected(n)
        # Independently establish the field from the full ordered market.
        ordered={}
        for rs in tables:
            got=source.first_place_block(rs,n)
            if got:
                for k,v in got.items():
                    source.record_map(ordered,k,v)
        if set(ordered)!=exp["3rentan"]:
            continue
        trio=source.three_combinations_from_rank_tables(soup,n)
        if set(trio)!=exp["3renpuku"]:
            raise ValueError("incomplete trio")
        candidates=[]
        for rs in tables:
            grid=source.generic_grid(rs,n)
            if not grid: continue
            aligned={tuple(sorted(k)):v for k,v in grid.items()}
            if len(aligned)!=len(grid) or set(aligned)!=exp["wide"]: continue
            if any(float(a)!=float(b) for a,b in aligned.values()) and aligned not in candidates:
                candidates.append(aligned)
        if len(candidates)!=1:
            raise ValueError("missing or ambiguous wide range")
        wide=candidates[0]
        records=[]
        for kind,book in (("wide",wide),("3renpuku",trio)):
            for k,(a,b) in sorted(book.items()):
                lo,hi=float(a),float(b)
                if not (math.isfinite(lo) and math.isfinite(hi) and 1<=lo<=hi):
                    raise ValueError("invalid numeric odds")
                if kind=="3renpuku" and lo!=hi:
                    raise ValueError("invalid trio interval")
                records.append((str(day),rid,rt,kind,"-".join(map(str,k)),a,b,stamp,"final"))
        return records
    raise ValueError("field not verified")

def check(path,day):
    groups=defaultdict(dict)
    with gzip.open(path,"rt",encoding="utf-8",newline="") as f:
        rd=csv.DictReader(f)
        if tuple(rd.fieldnames or ())!=FIELDS: raise ValueError("schema mismatch")
        for r in rd:
            if r["race_date"]!=str(day) or r["price_phase"]!="final": raise ValueError("wrong phase/day")
            k=(r["bet_type"],tuple(map(int,r["selection"].split("-"))))
            if k in groups[r["race_id"]]: raise ValueError("duplicate")
            lo,hi=float(r["odds_min"]),float(r["odds_max"])
            if not (math.isfinite(lo) and math.isfinite(hi) and 1<=lo<=hi): raise ValueError("bad price")
            if k[0]=="3renpuku" and lo!=hi: raise ValueError("bad trio")
            groups[r["race_id"]][k]=r
    for rid,rs in groups.items():
        cars=sorted({x for _,key in rs for x in key})
        expected={(t,k) for t,size in (("wide",2),("3renpuku",3)) for k in itertools.combinations(cars,size)}
        if set(rs)!=expected: raise ValueError("incomplete combinations")
    return len(groups),sum(len(x) for x in groups.values())

def api_session():
    s=requests.Session()
    s.headers.update({"Authorization":"Bearer "+os.environ["GH_TOKEN"],"Accept":"application/vnd.github+json","X-GitHub-Api-Version":"2022-11-28"})
    return s

def release_for(s,month):
    api="https://api.github.com/repos/"+os.environ["GH_REPO"]
    tag="haruq-wide-trio-"+month
    for page in range(1,101):
        r=s.get(api+"/releases",params={"per_page":100,"page":page},timeout=30);r.raise_for_status()
        items=r.json()
        for item in items:
            if item["tag_name"]==tag:
                if not item["draft"]: raise RuntimeError("refusing non-draft release")
                return item
        if len(items)<100: break
    r=s.post(api+"/releases",json={"tag_name":tag,"target_commitish":os.environ["GITHUB_SHA"],"name":tag,"draft":True,"body":"Authorized research; keep draft. Daily final odds, no public release."},timeout=30)
    r.raise_for_status()
    return r.json()

def upload(s,release,path):
    # Immutable per-run assets; no replacement or publication.
    url=release["upload_url"].split("{")[0]
    with path.open("rb") as f:
        r=s.post(url,params={"name":path.name},headers={"Content-Type":"application/octet-stream"},data=f,timeout=90)
    r.raise_for_status()

def list_assets(s,release):
    api="https://api.github.com/repos/"+os.environ["GH_REPO"]
    result={}
    for page in range(1,101):
        r=s.get(api+"/releases/"+str(release["id"])+"/assets",
                params={"per_page":100,"page":page},timeout=30)
        r.raise_for_status()
        items=r.json()
        result.update({x["name"]:x for x in items})
        if len(items)<100: return result
    raise RuntimeError("asset listing too large")

def read_asset(s,asset):
    api="https://api.github.com/repos/"+os.environ["GH_REPO"]
    reply=s.get(api+"/releases/assets/"+str(asset["id"]),
                headers={"Accept":"application/octet-stream"},timeout=90)
    reply.raise_for_status()
    raw=reply.content
    digest=asset.get("digest")
    if digest and digest!="sha256:"+hashlib.sha256(raw).hexdigest():
        raise RuntimeError("asset digest mismatch "+asset["name"])
    return raw

def month_days(month):
    day=date.fromisoformat(month+"-01")
    while day.strftime("%Y-%m")==month:
        yield day
        day+=timedelta(days=1)

def prior_month(s,release,month,work):
    assets=list_assets(s,release)
    manifests=sorted((a for name,a in assets.items()
                      if name.startswith("wide-trio-"+month+"-manifest-") and name.endswith(".json")),
                     key=lambda a:a["id"],reverse=True)
    for asset in manifests:
        obj=json.loads(read_asset(s,asset))
        if (obj.get("month")!=month or obj.get("draft_release_id")!=release["id"]
            or not obj.get("storage_readback_verified") or obj.get("status") not in ("complete","partial_verified")):
            continue
        days=obj.get("days",[])
        if ([x["date"] for x in days]!=[str(x) for x in month_days(month)]
            or any(x.get("blocked") or ("discovered" in x and not x["discovered"]) for x in days)):
            continue
        raw=read_asset(s,assets[obj["month_asset"]])
        if hashlib.sha256(raw).hexdigest()!=obj["month_sha256"]:
            raise RuntimeError("existing month checksum mismatch")
        groups=defaultdict(list)
        keys=set()
        import io
        with gzip.open(io.BytesIO(raw),"rt",encoding="utf-8",newline="") as f:
            rd=csv.DictReader(f)
            if tuple(rd.fieldnames or ())!=FIELDS: raise ValueError("month schema mismatch")
            for row in rd:
                key=(row["race_id"],row["bet_type"],row["selection"])
                if key in keys: raise ValueError("duplicate monthly key")
                keys.add(key)
                groups[row["race_date"]].append(tuple(row[k] for k in FIELDS))
        ids=set()
        for item in days:
            rows=groups.pop(item["date"],[])
            p=work/"check-prior.csv.gz"
            with gzip.open(p,"wt",encoding="utf-8",newline="") as f:
                w=csv.writer(f);w.writerow(FIELDS);w.writerows(rows)
            n,nrows=check(p,date.fromisoformat(item["date"]))
            if n!=item["complete_races"] or nrows!=item["rows"]: raise ValueError("prior counts mismatch")
            cur={r[1] for r in rows}
            if ids.intersection(cur): raise ValueError("duplicate race across days")
            ids.update(cur)
            p.unlink()
        if groups or len(ids)!=obj["complete_races"] or len(keys)!=obj["wide_rows"]+obj["trio_rows"]:
            raise ValueError("prior monthly totals mismatch")
        return obj
    return None

def resume_days(s,release,month,work):
    assets=list_assets(s,release)
    found=defaultdict(list)
    for name,a in assets.items():
        if name.startswith("checkpoint-"+month+"-") and name.endswith(".json"):
            item=json.loads(read_asset(s,a))
            d=item.get("date","")
            if d[:7]==month and item.get("storage_readback_verified") and not item.get("blocked") and not item.get("failures") and item.get("discovered",0)>0:
                found[d].append((a["id"],item))
    out={}
    for d,items in found.items():
        _,item=max(items,key=lambda x:x[0])
        raw=read_asset(s,assets[item["asset"]])
        if hashlib.sha256(raw).hexdigest()!=item["sha256"]: raise ValueError("checkpoint checksum mismatch")
        p=work/"checkpoint-read.csv.gz";p.write_bytes(raw)
        if check(p,date.fromisoformat(d))!=(item["complete_races"],item["rows"]):
            raise ValueError("checkpoint counts mismatch")
        with gzip.open(p,"rt",encoding="utf-8",newline="") as f:
            rows=[tuple(r[k] for k in FIELDS) for r in csv.DictReader(f)]
        p.unlink()
        if item["complete_races"]+item["girls_excluded"]!=item["discovered"]:
            raise ValueError("checkpoint counts do not reconcile")
        out[d]=(item,rows)
    return out

def download_verified(s,release,path):
    api="https://api.github.com/repos/"+os.environ["GH_REPO"]
    for page in range(1,101):
        r=s.get(api+"/releases/"+str(release["id"])+"/assets",
                params={"per_page":100,"page":page},timeout=30)
        r.raise_for_status()
        items=r.json()
        for asset in items:
            if asset["name"]==path.name:
                reply=s.get(api+"/releases/assets/"+str(asset["id"]),
                            headers={"Accept":"application/octet-stream"},timeout=90)
                reply.raise_for_status()
                if hashlib.sha256(reply.content).digest()!=hashlib.sha256(path.read_bytes()).digest():
                    raise RuntimeError("storage readback mismatch "+path.name)
                return reply.content
        if len(items)<100: break
    raise RuntimeError("stored asset not found "+path.name)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--month",required=True)
    p.add_argument("--accept-terms",action="store_true")
    a=p.parse_args()
    if not a.accept_terms: p.error("authorization required")
    if not re.fullmatch(r"20\d\d-(0[1-9]|1[0-2])",a.month) or not "2025-01"<=a.month<="2026-10": p.error("outside scope")
    day=date.fromisoformat(a.month+"-01")
    out=Path("monthly-audit");out.mkdir(exist_ok=True)
    work=Path("private-tmp");work.mkdir(exist_ok=True)
    gh=api_session()
    release=release_for(gh,a.month)
    prior=prior_month(gh,release,a.month,work)
    if prior:
        (out/"summary.json").write_text(json.dumps(prior,ensure_ascii=False,indent=2),encoding="utf-8")
        print("Already stored, verified month reused:",a.month,prior["complete_races"],prior["status"],flush=True)
        with open(os.environ["GITHUB_STEP_SUMMARY"],"a") as f:
            f.write(f"Reused {a.month}: {prior['complete_races']} races; {prior['status']}\n")
        return
    checkpoints=resume_days(gh,release,a.month,work)
    session=requests.Session()
    session.headers["User-Agent"]="Keirin-Research-Odds-Collector/1.0"
    report=[]
    month_rows=[]
    seen=set()
    run=os.environ["GITHUB_RUN_ID"]+"-"+os.environ["GITHUB_RUN_ATTEMPT"]
    while day.strftime("%Y-%m")==a.month:
        if str(day) in checkpoints:
            item,rows=checkpoints[str(day)]
            ids={r[1] for r in rows}
            if seen.intersection(ids): raise RuntimeError("duplicate race across resumed days")
            seen.update(ids)
            month_rows.extend(rows)
            report.append(item)
            print(str(day),"resumed",len(ids),flush=True)
            day+=timedelta(days=1)
            continue
        rows=[];fail=[];girls=0;blocked=False;discovered=0
        try:
            urls=source.discover(session,day,2)
            discovered=len(urls)
            for rid,url in sorted(urls.items()):
                try:
                    time.sleep(2)
                    r=session.get(url,timeout=35)
                    if r.status_code in (403,429):
                        blocked=True
                        fail.append({"race_id":rid,"reason":"source access restricted"})
                        break
                    r.raise_for_status()
                    rows.extend(parse_two(r.text,rid,day))
                except source.ExcludedRace:
                    girls+=1
                except Exception as e:
                    fail.append({"race_id":rid,"reason":type(e).__name__+": "+str(e)[:180]})
        except Exception as e:
            fail.append({"race_id":None,"reason":"index error: "+type(e).__name__})
            if isinstance(e,requests.HTTPError) and e.response is not None and e.response.status_code in (403,429): blocked=True
        # Retry failed race pages once, without retrying any access restriction.
        if fail and not blocked and discovered:
            unresolved=[]
            for f in fail:
                rid=f["race_id"]
                if blocked or rid not in urls:
                    unresolved.append(f)
                    continue
                try:
                    time.sleep(2)
                    r=session.get(urls[rid],timeout=35)
                    if r.status_code in (403,429): blocked=True
                    r.raise_for_status()
                    rows.extend(parse_two(r.text,rid,day))
                except source.ExcludedRace:
                    girls+=1
                except Exception as e:
                    unresolved.append({"race_id":rid,"reason":type(e).__name__+": "+str(e)[:180],"original_reason":f["reason"]})
            fail=unresolved
        item={"date":str(day),"discovered":discovered,"girls_excluded":girls,"complete_races":0,"rows":0,"failures":fail,"blocked":blocked,"asset":None}
        if rows:
            path=work/f"wide-trio-{day}-{run}.csv.gz"
            with gzip.open(path,"wt",encoding="utf-8",newline="") as f:
                wr=csv.writer(f);wr.writerow(FIELDS);wr.writerows(rows)
            count,nrows=check(path,day)
            ids={r[1] for r in rows}
            if len(ids)!=count or seen.intersection(ids):
                raise RuntimeError("duplicate race ID across days")
            if not blocked and count+girls+len(fail)!=discovered:
                raise RuntimeError("daily counts do not reconcile")
            upload(gh,release,path)
            download_verified(gh,release,path)
            seen.update(ids)
            month_rows.extend(rows)
            item.update(storage_readback_verified=True)
            item.update(complete_races=count,rows=nrows,asset=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            path.unlink()
            checkpoint=work/f"checkpoint-{day}-{run}.json"
            checkpoint.write_text(json.dumps(item,ensure_ascii=False,indent=2),encoding="utf-8")
            upload(gh,release,checkpoint)
            download_verified(gh,release,checkpoint)
            checkpoint.unlink()
        report.append(item)
        (out/"summary.json").write_text(json.dumps({"month":a.month,"draft_release_id":release["id"],"days":report},ensure_ascii=False,indent=2),encoding="utf-8")
        print(str(day),"discovered",discovered,"complete",item["complete_races"],"failures",len(fail),flush=True)
        if blocked or (len(report)>=3 and all(x["complete_races"]==0 for x in report[-3:])): break
        day+=timedelta(days=1)
    total=sum(x["complete_races"] for x in report)
    if total != len(seen): raise RuntimeError("monthly race count mismatch")
    remaining=sum(len(x["failures"]) for x in report)
    full_calendar=day.strftime("%Y-%m")!=a.month
    status="complete" if full_calendar and not remaining and total else "partial_verified"
    if not full_calendar or not total: status="incomplete_calendar"
    summary={"month":a.month,"draft_release_id":release["id"],"days":report,
             "complete_races":total,"remaining_failures":remaining,"status":status,
             "wide_rows":sum(r[3]=="wide" for r in month_rows),
             "trio_rows":sum(r[3]=="3renpuku" for r in month_rows),
             "storage_readback_verified":False}
    if total:
        month=work/f"wide-trio-{a.month}-verified-{run}.csv.gz"
        with gzip.open(month,"wt",encoding="utf-8",newline="") as f:
            wr=csv.writer(f);wr.writerow(FIELDS);wr.writerows(month_rows)
        with gzip.open(month,"rt",encoding="utf-8",newline="") as f:
            readback=list(csv.DictReader(f))
        if len(readback)!=len(month_rows) or len({(r["race_id"],r["bet_type"],r["selection"]) for r in readback})!=len(readback):
            raise RuntimeError("monthly duplicate or row mismatch")
        upload(gh,release,month)
        download_verified(gh,release,month)
        summary.update(month_asset=month.name,month_sha256=hashlib.sha256(month.read_bytes()).hexdigest(),storage_readback_verified=True)
        month.unlink()
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    manifest=work/f"wide-trio-{a.month}-manifest-{run}.json"
    manifest.write_bytes((out/"summary.json").read_bytes())
    upload(gh,release,manifest)
    download_verified(gh,release,manifest)
    print(json.dumps({k:v for k,v in summary.items() if k!="days"},ensure_ascii=False),flush=True)
    with open(os.environ["GITHUB_STEP_SUMMARY"],"a") as f:
        f.write(f"Month: {a.month} / verified races: {total} / processed days: {len(report)} / draft release ID: {release['id']}\n")
    if not full_calendar or not total or any(x["blocked"] or not x["discovered"] for x in report):
        raise SystemExit("Incomplete calendar/source access; verified checkpoints retained.")
    if remaining:
        print("::warning::Unresolved races remain; verified partial dataset saved. See manifest.")
if __name__=="__main__":
    main()
