#!/usr/bin/env python3
"""All seven keirin bet types; store ONLY odds and their lookup keys.

Source pages are published by third parties; obtain authorization for automated
collection/storage and honor their published terms before using --accept-terms.
The program never collects player, race result, grade, line or payout fields.
"""
import argparse
import csv
import gzip
import itertools
import math
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from collections import defaultdict
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE = "https://keirin.kdreams.jp"
URL_ID = re.compile(r"/racedetail/(\d{16})/")
DATE_RX = re.compile(r"(20\d\d)年\s*(\d{1,2})月\s*(\d{1,2})日\s*レース詳細")
RANK_TRIPLE = re.compile(r"(?<!\d)([1-9])\s*=\s*([1-9])\s*=\s*([1-9])\s+([\d,]+(?:\.\d+)?)")
WIDE_RX = re.compile(r"^\s*([\d,]+(?:\.\d+)?)\s*[～〜~\-]\s*([\d,]+(?:\.\d+)?)\s*$")
SINGLE_RX = re.compile(r"^\s*[\d,]+(?:\.\d+)?\s*$")
FIELDS = ("race_id", "race_type", "bet_type", "selection", "odds_min", "odds_max", "odds_time_jst")
ALL_TYPES = ("3rentan","3renpuku","2shatan","2shafuku","wide","2wakutan","2wakufuku")
JST = timezone(timedelta(hours=9))
START_DATE = date(2021,1,1)
LAST_DATE = date(2026,6,30)
GIRLS_RX = re.compile(r"ガールズ|女子|(?:^|\W)[ＬL]級")
RACE_TYPE_RX = re.compile(
    r"(?:[ＳＡSA]級|チャレンジ|[ＧG][123１２３]|[ＦF][12１２])"
    r"[^\s、。｜|（）()\[\]<>]{0,14}"
)


class ExcludedRace(ValueError):
    """A girls race, not a failed men's odds extraction."""


def race_type_from_page(soup):
    """Use race-specific headings, not general site menus or car-count guesses."""
    headings = [node.get_text(" ", strip=True) for node in
                soup.select("h1,h2,h3,h4,.raceTitle,.race_title,.race-title,.raceInfo,.race-info")]
    if any(GIRLS_RX.search(h) for h in headings):
        raise ExcludedRace("girls race excluded")
    for heading in headings:
        m = RACE_TYPE_RX.search(heading)
        if m:
            return m.group(0)
    raise ValueError("Race type not verified from race-specific heading")


def complete_saved_races(path):
    """Recover only already-verified full seven-bet books on restart."""
    if not path.exists():
        return {}
    grouped = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"Existing odds file schema differs: {path}")
        for row in reader:
            grouped[row["race_id"]].append(row)
    out = {}
    for rid, records in grouped.items():
        types = {r["race_type"] for r in records}
        if len(types) != 1 or not next(iter(types)):
            continue
        try:
            cars = {int(x) for r in records if r["bet_type"] == "3rentan"
                    for x in r["selection"].split("-")}
            exp = expected(len(cars))
            if cars != set(range(1, len(cars)+1)):
                continue
            found = defaultdict(set)
            for row in records:
                kind = row["bet_type"]
                key = tuple(map(int, row["selection"].split("-")))
                if kind not in exp or key not in exp[kind] or key in found[kind]:
                    raise ValueError("invalid or duplicate saved quote")
                if (float(row["odds_min"]) < 1 or
                    float(row["odds_max"]) < float(row["odds_min"]) or
                    not math.isfinite(float(row["odds_max"]))):
                    raise ValueError("invalid saved odds value")
                found[kind].add(key)
            if all(found[kind] == keys for kind, keys in exp.items()):
                out[rid] = [tuple(row[field] for field in FIELDS) for row in records]
        except (ValueError, TypeError, OverflowError):
            continue
    return out


def text_cells(tr):
    return [node.get_text(" ", strip=True).replace("\xa0", " ").strip()
            for node in tr.find_all(["th","td"], recursive=False)]


def digit(value):
    return int(value) if re.fullmatch("[1-9]", value) else None


def number(value):
    value = value.strip()
    if not SINGLE_RX.fullmatch(value):
        return None
    parsed = float(value.replace(",", ""))
    return value.replace(",", "") if parsed >= 1 else None


def odds_value(value):
    norm = re.sub(r"\s+", "", value.replace("〜", "～").replace("~", "～"))
    m = WIDE_RX.fullmatch(norm)
    if m:
        low, high = number(m.group(1)), number(m.group(2))
        if low and high and float(low) <= float(high):
            return low, high
        return None
    n = number(norm)
    return (n, n) if n else None


def expected(n):
    cars = tuple(range(1, n + 1))
    frame = tuple(range(1, 7))
    result = {
        "3rentan": set(itertools.permutations(cars, 3)),
        "3renpuku": set(itertools.combinations(cars, 3)),
        "2shatan": set(itertools.permutations(cars, 2)),
        "2shafuku": set(itertools.combinations(cars, 2)),
        "wide": set(itertools.combinations(cars, 2)),
    }
    if n == 9:
        result["2wakutan"] = {(a,b) for a in frame for b in frame
                                if a != b or a in (4,5,6)}
        result["2wakufuku"] = {(a,b) for a in frame for b in frame
                                if a < b or (a == b and a in (4,5,6))}
    return result


def rows(table):
    return [text_cells(tr) for tr in table.find_all("tr") if text_cells(tr)]


def record_map(value, key, odds):
    if key in value and value[key] != odds:
        raise ValueError("Contradictory quotes on the same selection: " + str(key))
    value[key] = odds


def find_headers(rs, width):
    for i, cells in enumerate(rs):
        nums = [digit(s) for s in cells if s]
        if len(nums) == width and all(nums) and len(set(nums)) == width:
            yield i, nums


def first_place_block(rs, n):
    """Decode one first-place block: n-1 columns, n-1 second-place rows."""
    for i, headings in find_headers(rs, n-1):
        complement = set(range(1, n+1)).difference(headings)
        if len(complement) != 1:
            continue
        first = complement.pop()
        quoted = {}
        for cell in rs[i+1:]:
            if len(cell) < n:
                continue
            second = digit(cell[0])
            if second is None or second not in headings:
                continue
            vals = cell[1:1+n-1]
            if len(vals) != n-1:
                continue
            if any((third != second and odds_value(v) is None) or
                   (third == second and odds_value(v) is not None)
                   for third,v in zip(headings,vals)):
                continue
            for third,v in zip(headings,vals):
                if third != second:
                    record_map(quoted,(first,second,third),odds_value(v))
        if len(quoted) == (n-1)*(n-2):
            return quoted
    return None


def generic_grid(rs, width):
    """Decode square table, including upper-empty triangular odds matrices."""
    for i, headings in find_headers(rs, width):
        if set(headings) != set(range(1, width+1)):
            continue
        quote = {}
        seen_rows = set()
        for row in rs[i+1:]:
            if len(row) < width+1:
                continue
            one = digit(row[0])
            if one is None or one > width or one in seen_rows:
                continue
            vals = row[1:1+width]
            if len(vals) != width:
                continue
            # Unavailable/diagonal cells contain printed markers, not empty HTML.
            # Ignore these markers here, and REQUIRE exact numeric coverage later.
            seen_rows.add(one)
            for two,v in zip(headings,vals):
                parsed = odds_value(v)
                if parsed is not None:
                    record_map(quote,(one,two),parsed)
            if len(seen_rows) == width:
                break
        if len(seen_rows) != width:
            continue
        return quote
    return None


def three_combinations_from_rank_tables(soup, n):
    all_odds = {}
    valid = set(itertools.combinations(range(1,n+1),3))
    # Only the 3renpuku odds ranking container; refund_table contains a payout\n    # for the winning same selection, which must NEVER be treated as an odds quote.\n    # Popularity top 50 + highest-payout top 50 covers 3renpuku's
    # at most 84 bets, without making the mistake of using 3rentan's top 50.
    for tr in soup.select("div.odds_table table tr"):
        content = " ".join(text_cells(tr))
        for m in RANK_TRIPLE.finditer(content):
            key = tuple(sorted(int(m.group(j)) for j in (1,2,3)))
            if key in valid:
                record_map(all_odds,key,odds_value(m.group(4)))
    return all_odds


def parse_all(html, race_id):
    soup = BeautifulSoup(html,"html.parser")
    label = soup.get_text(" ",strip=True)
    race_type = race_type_from_page(soup)
    if "確定オッズ" not in label:
        raise ValueError("Odds are not explicitly final")
    matched = DATE_RX.search(label)
    if not matched:
        raise ValueError("Missing race date label")
    actual_date = date(*map(int,matched.groups()))
    # race_id: venue(2)+meeting YYYYMMDD(8)+meeting_day(2)+race_no(4)
    first_day = date.fromisoformat(race_id[2:6]+"-"+race_id[6:8]+"-"+race_id[8:10])
    if actual_date != first_day + timedelta(days=int(race_id[10:12])-1):
        raise ValueError("Odds page date does not match requested race")
    tables = [rows(t) for t in soup.find_all("table")]
    for n in (9,8,7,6,5,4,3):
        all_blocks = {}
        for rs in tables:
            got = first_place_block(rs,n)
            if got:
                first = next(iter(got))[0]
                if first in all_blocks and all_blocks[first] != got:
                    raise ValueError("3rentan conflicts")
                all_blocks[first] = got
        market3t = {}
        for block in all_blocks.values():
            market3t.update(block)
        exp = expected(n)
        if set(market3t) != exp["3rentan"]:
            continue
        found = {"3rentan":market3t, "3renpuku":three_combinations_from_rank_tables(soup,n)}
        # Grid candidates are distinguished by their exact allowed combinations
        # and by whether values are numeric single odds or wide intervals.
        for rs in tables:
            for width in ({n,6} if n==9 else {n}):
                grid = generic_grid(rs,width)
                if not grid:
                    continue
                for kind in ("2shatan","2shafuku","wide","2wakutan","2wakufuku"):
                    if kind not in exp or kind in found:
                        continue
                    keys = set(grid)
                    expected_keys = exp[kind]
                    if kind in ("2shafuku","wide","2wakufuku"):
                        aligned = {(min(a,b),max(a,b)):v for (a,b),v in grid.items()}
                        if len(aligned) != len(grid):
                            continue
                    else:
                        aligned = grid
                    if set(aligned) != expected_keys:
                        continue
                    if kind=="wide" and not any(a!=b for a,b in aligned.values()):
                        continue
                    if kind!="wide" and any(a!=b for a,b in aligned.values()):
                        continue
                    found[kind]=aligned
        # 2shafuku / 2wakufuku can have ranges? No: these use single odds.
        # Ambiguous 2shafuku vs wide tables whose wide bounds happen to match:
        # never assert full coverage if either is missing.
        if any(k not in found or set(found[k]) != exp[k] for k in exp):
            missing={k:len(exp[k]-set(found.get(k,{}))) for k in exp}
            raise ValueError("Incomplete all-bet odds: "+str(missing))
        stamps = re.findall(r"(20\d\d/\d\d/\d\d)\s+(\d\d:\d\d)現在",label)
        stamp = (stamps[0][0].replace("/","-")+"T"+stamps[0][1]+":00+09:00") if stamps else ""
        if not stamp:
            raise ValueError("Missing source odds timestamp")
        result = []
        for kind in ALL_TYPES:
            if kind not in exp:
                continue
            for key, (low,high) in sorted(found[kind].items()):
                result.append((race_id,race_type,kind,"-".join(map(str,key)),low,high,stamp))
        if len(result) != sum(len(choices) for choices in exp.values()):
            raise AssertionError("Incomplete all-odds count")
        return result
    raise ValueError("No complete supported 3-to-9-car market found")


def discover(session, day, pause):
    url = BASE+"/kaisai/"+day.strftime("%Y/%m/%d")+"/"
    time.sleep(pause)
    reply = session.get(url,timeout=30)
    reply.raise_for_status()
    soup = BeautifulSoup(reply.text, "html.parser")
    urls = {}
    for a in soup.find_all("a",href=True):
        if a.get_text(" ",strip=True) != "オッズ":
            continue
        u=urljoin(BASE,a["href"])
        p=urlparse(u)
        m=URL_ID.search(p.path)
        if p.hostname != "keirin.kdreams.jp" or not m:
            continue
        rid=m.group(1)
        base_day=date.fromisoformat(rid[2:6]+"-"+rid[6:8]+"-"+rid[8:10])
        if base_day+timedelta(days=int(rid[10:12])-1)!=day:
            continue
        urls[rid]=p._replace(query="pageType=odds",fragment="").geturl()
    if not urls:
        raise ValueError("No odds links found; cannot equate this to zero races")
    return urls


def collect(start,end,output,pause,only_race_id=None):
    if start<START_DATE or end>LAST_DATE:
        raise ValueError("research scope ends 2026-06-30; 2026 second half excluded")
    session=requests.Session()
    session.headers.update({"User-Agent":"Keirin-Research-Odds-Collector/1.0"})
    failures=0
    for offset in range((end-start).days+1):
        day=start+timedelta(days=offset)
        path=output/day.strftime("%Y")/(day.isoformat()+".csv.gz")
        completed=complete_saved_races(path)
        success=[row for records in completed.values() for row in records]
        skipped_girls=0
        try:
            url_map=discover(session,day,pause)
        except Exception as exc:
            print(day,"INDEX_ERROR",str(exc),file=sys.stderr,flush=True)
            failures+=1
            continue
        for rid,url in sorted(url_map.items()):
            if only_race_id is not None and rid != only_race_id:
                continue
            if rid in completed:
                continue
            try:
                time.sleep(pause)
                response=session.get(url,timeout=35)
                response.raise_for_status()
                lines=parse_all(response.text,rid)
                success.extend(lines)
            except ExcludedRace:
                skipped_girls += 1
            except Exception as exc:
                failures+=1
                print(day,rid,"INCOMPLETE",str(exc),file=sys.stderr,flush=True)
        if success:
            path.parent.mkdir(parents=True,exist_ok=True)
            temp=path.with_suffix(".csv.gz.tmp")
            with gzip.open(temp,"wt",encoding="utf-8",newline="") as f:
                writer=csv.writer(f)
                writer.writerow(FIELDS)
                writer.writerows(success)
            temp.replace(path)
        print(day,"complete_races",len(set(r[0] for r in success)),
              "discovered_races",len(url_map),"girls_skipped",skipped_girls,
              "resumed_races",len(completed),"odds_rows",len(success),flush=True)
    if failures:
        raise SystemExit(f"Incomplete collection; {failures} failures. No fabricated odds.")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--start",type=date.fromisoformat,required=True)
    p.add_argument("--end",type=date.fromisoformat,required=True)
    p.add_argument("--out",type=Path,default=Path("data/odds_only"))
    p.add_argument("--pause",type=float,default=2)
    p.add_argument("--race-id",default=None,help="Optional single 16-digit ID for live smoke tests")
    p.add_argument("--accept-terms",action="store_true",
                   help="Confirm permission to collect and store source odds")
    args=p.parse_args()
    if args.race_id is not None and not re.fullmatch(r'\d{16}', args.race_id):
        p.error('Invalid race-id')
    if not args.accept_terms:
        p.error("Check and obtain source data collection/storage permission before running; specify --accept-terms")
    if (args.pause<2 or args.end<args.start or args.start<START_DATE
            or args.end>LAST_DATE or args.end>=datetime.now(JST).date()):
        p.error("Dates must be completed within 2021-01-01..2026-06-30 and pause >= 2 seconds")
    collect(args.start,args.end,args.out,args.pause,args.race_id)


if __name__=="__main__":
    main()
