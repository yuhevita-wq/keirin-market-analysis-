"""Nationwide historical daily-calendar census: enumerate every linked race ID.

This is an OUTCOME UNIVERSE AUDIT, not a predictor. A previously collected
dataset is not assumed representative merely because it has a large row count.

Uses historical KDreams daily event index only. Missing page/discovery days are
reported and must be reconciled before claiming nationwide coverage.
"""
from __future__ import annotations
import argparse,csv,gzip,json,re,time,calendar,unicodedata
from collections import Counter,defaultdict
from datetime import date,timedelta,datetime,timezone
from pathlib import Path
from urllib.parse import urljoin,urlsplit,urlunsplit

from bs4 import BeautifulSoup
from keirin_market_analysis.s_yosen_2025 import KDREAMS_DAILY,fetch_html,make_session,expand_row_cells
from scope import START, END, in_scope

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"data/nationwide_calendar"
ID_RE=re.compile(r"/racedetail/(\d{16})/")


def as_race_day(rid):
    try:
        start=date.fromisoformat(rid[2:6]+"-"+rid[6:8]+"-"+rid[8:10])
        day_no=int(rid[10:12])
        if day_no<1 or day_no>7:return None
        return start+timedelta(days=day_no-1)
    except (ValueError,IndexError):return None


def race_label(text):
    """A published programme label, never a model predictor."""
    norm=unicodedata.normalize("NFKC",str(text or "")).replace(" ","").strip()
    if norm.startswith(("S級","A級","L級","チャレンジ","ガールズ")):
        return norm if len(norm)<32 else ""
    return ""


def extract_day(html,base_url,day):
    soup=BeautifulSoup(html,"lxml")
    found={}
    outside=Counter()
    labels_by_id={}
    # A KDreams meeting programme uses table headers with colspan to group
    # consecutive race columns (e.g., A級予選 / A級特選 / S級予選).
    # Track the last declared label for each expanded race column and assign it
    # to its underlying racedetail links; do NOT guess by race number.
    for table in soup.find_all("table"):
        active={}
        for row in table.find_all("tr"):
            cells=expand_row_cells(row)
            for i,cell in enumerate(cells):
                label=race_label(cell.get_text(" ",strip=True))
                if label:
                    active[i]=label
            for i,cell in enumerate(cells):
                kind=active.get(i,"")
                if not kind:
                    continue
                for a in cell.find_all("a",href=True):
                    match=ID_RE.search(str(a["href"]))
                    if match and as_race_day(match.group(1))==day:
                        labels_by_id[match.group(1)]=kind
    for a in soup.find_all("a",href=True):
        href=str(a["href"])
        match=ID_RE.search(href)
        if not match:continue
        rid=match.group(1)
        real_date=as_race_day(rid)
        if real_date!=day:
            outside["other_meeting_day"]+=1
            continue
        url=urljoin(base_url,href)
        parts=urlsplit(url)
        if parts.scheme!="https" or parts.netloc!="keirin.kdreams.jp":
            outside["off_provider"]+=1
            continue
        link=urlunsplit(("https","keirin.kdreams.jp",parts.path,"pageType=odds",""))
        found[rid]={"odds_page_url":link,"race_type":labels_by_id.get(rid,"")}
    return found,outside

def days(start,end):
    d=start
    while d<=end:
        yield d
        d+=timedelta(days=1)


def execute(start,end,delay):
    if (end-start).days>30 or start>end:
        raise ValueError("bounded census must scan 1..31 consecutive days")
    if not in_scope(start.isoformat()) or not in_scope(end.isoformat()):
        raise ValueError("calendar census is restricted to 2021-01-01 through 2026-06-30")
    now=datetime.now(timezone(timedelta(hours=9))).date()
    if end>=now:
        raise ValueError("completed historical days only")
    folder=BASE/(f"{start.isoformat()}__{end.isoformat()}")
    if folder.exists():
        raise FileExistsError("refusing to overwrite completed census run "+str(folder))
    sess=make_session()
    found={}
    failures=[]
    page_stats=[]
    for day in days(start,end):
        url=KDREAMS_DAILY.format(year=day.year,month=day.month,day=day.day)
        try:
            html=fetch_html(sess,url)
            race_ids,extraneous=extract_day(html,url,day)
            page_stats.append({"day":day.isoformat(),"discovered_race_ids":len(race_ids),
                               "noncurrent_day_links":extraneous.get("other_meeting_day",0),
                               "daily_index_url":url})
            for rid,metadata in race_ids.items():
                if rid not in found:found[rid]={"race_id":rid,
                    "race_date":day.isoformat(),
                    "odds_page_url":metadata["odds_page_url"],
                    "race_type":metadata["race_type"],
                    "daily_index_url":url}
        except Exception as exc:
            failures.append({"day":day.isoformat(),"daily_index_url":url,
                             "error":f"{type(exc).__name__}: {exc}"[:500]})
        if delay:time.sleep(delay)
    folder.mkdir(parents=True,exist_ok=False)
    with gzip.open(folder/"race_index.csv.gz","wt",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["race_id","race_date","odds_page_url","race_type","daily_index_url"])
        w.writeheader()
        w.writerows(sorted(found.values(),key=lambda r:(r["race_date"],r["race_id"])))
    (folder/"page_status.json").write_text(json.dumps({"pages":page_stats,"errors":failures},
                                                       ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    summary={
        "status":"COMPLETE_DAILY_INDEX" if not failures else "PARTIAL_DAILY_INDEX",
        "start":start.isoformat(),"end":end.isoformat(),
        "days_attempted":(end-start).days+1,
        "days_succeeded":len(page_stats),"days_failed":len(failures),
        "unique_race_ids":len(found),
        "race_type_labels_found":sum(bool(v.get("race_type")) for v in found.values()),
        "no_links_days":[r["day"] for r in page_stats if r["discovered_race_ids"]==0],
        "source":"楽天Kドリームス public daily racing index",
        "discovery_limitations":"Calendar race links might be missing or filtered; this is an observed-link census, not an independently certified JKA universe",
        "model_features":"none: universe identification only"
    }
    (folder/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("DAILY_CALENDAR_CENSUS",json.dumps(summary,ensure_ascii=False),flush=True)
    return summary


def automatic_month():
    done=set()
    for p in BASE.glob("*/summary.json"):
        try:
            j=json.loads(p.read_text(encoding="utf-8"))
            if j["days_failed"]==0:
                done.add((j["start"],j["end"]))
        except (KeyError,ValueError):pass
    first=START
    yesterday=min(datetime.now(timezone(timedelta(hours=9))).date()-timedelta(days=1),END)
    now=first
    while now<=yesterday:
        last=date(now.year,now.month,calendar.monthrange(now.year,now.month)[1])
        if last>yesterday:break
        if (now.isoformat(),last.isoformat()) not in done:return now,last
        now=last+timedelta(days=1)
    return None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--auto",action="store_true")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--sleep",type=float,default=.55)
    a=ap.parse_args()
    if not .25<=a.sleep<=10:ap.error("sleep must be between .25 and 10")
    if a.auto:
        pair=automatic_month()
        if pair is None:
            print("NO_PENDING_FULL_HISTORICAL_MONTHS");return
        start,end=pair
    else:
        if not a.start or not a.end:ap.error("--start and --end required")
        start,end=date.fromisoformat(a.start),date.fromisoformat(a.end)
    execute(start,end,a.sleep)


if __name__=="__main__":
    main()
