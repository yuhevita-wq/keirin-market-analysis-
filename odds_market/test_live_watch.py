"""No-network tests for fresh complete market snapshots and isolation rankings."""
import csv
import itertools
import math
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from odds_market.collect_all_odds import JST
from odds_market.live_watch import decode_live, save_private_snapshot, validate_url, request_url
from odds_market.market_isolation import scan, normalize_quotes
from odds_market.test_collect_all_odds import market_fixture


RID = "7320240704010001"
STAMP = datetime(2024, 7, 4, 10, 51, tzinfo=JST)


def fixture():
    return (market_fixture(7).replace(" 確定オッズ", "")
            .replace("<span>2024/07/04 10:51現在</span>",
                     "<span>発売票数 20000 2024/07/04 10:51現在</span>"))


def test_gates():
    html = fixture()
    rows, t = decode_live(html, RID, now=STAMP + timedelta(minutes=2))
    assert len(rows) == 329
    assert t == STAMP
    assert len(scan(rows)["counts"]) == 5
    with TemporaryDirectory() as tmp:
        saved, path = save_private_snapshot(rows, Path(tmp))
        assert saved and path.is_file()
        fresh, _ = save_private_snapshot(rows, Path(tmp))
        assert not fresh
    for bad, message in [
        (html.replace("2024/07/04 10:51現在", "2024/07/04 10:40現在"), "STALE"),
        (html.replace("20000", "20000 2024/07/04 10:52現在"), "inconsistent"),
        (html.replace("発売票数 ", "発売総数 "), "vote-count"),
        (html.replace("<h3>A級一般</h3>", "<h3>ガールズ予選</h3>"), "excluded"),
        (html.replace('1=2=3', '1=2=4'), "Incomplete"),
    ]:
        try:
            decode_live(bad, RID, now=STAMP + timedelta(minutes=2))
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected fail-closed on {message}")
    try:
        decode_live(market_fixture(7).replace("<span>2024/", "<span>発売票数 1 2024/"),
                    RID, now=STAMP)
    except ValueError as e:
        assert "Final" in str(e)
    else:
        raise AssertionError("Final quotes must not silently count as live quotes")
    assert validate_url("https://keirin.kdreams.jp/foo/racedetail/" + RID + "/") == RID
    assert "pageType=odds" in request_url("https://keirin.kdreams.jp/foo/racedetail/" + RID + "/", 3)
    for invalid in ["http://keirin.kdreams.jp/foo/racedetail/" + RID,
                    "https://evil.org/foo/racedetail/" + RID,
                    "https://keirin.kdreams.jp/user/login"]:
        try:
            validate_url(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Unsafe URL accepted")


def synthetic_markets(n=7):
    weights = {(a,b,c): 1.3**a * 1.5**b * 1.6**c
               for a,b,c in itertools.permutations(range(1,n+1),3)}
    total=sum(weights.values())
    probs3={k: v/total for k,v in weights.items()}
    from odds_market.market_isolation import marginals_of_trifecta
    derived=marginals_of_trifecta(probs3,n)
    derived["3rentan"]=probs3
    rows=[]
    for kind, d in derived.items():
        if kind.startswith("2waku"):
            continue
        for key,prob in d.items():
            # Overprice 3-4-1 in only the 3rentan market.
            odd=(0.75/prob) * (8 if kind=="3rentan" and key==(3,4,1) else 1)
            rows.append(("7320240704010001","A級予選",kind,
                         "-".join(map(str,key)),round(odd,6),
                         round(odd*1.1,6) if kind=="wide" else round(odd,6),
                         "2024-07-04T10:51:00+09:00"))
    return rows


def test_isolation_and_actual_341():
    rows=synthetic_markets()
    report=scan(rows,limit=329)
    assert report["integrity"]["complete"]
    assert len(report["counts"])==5 and sum(report["counts"].values())==329
    match=[x for x in report["top"]
           if x["type"]=="3rentan" and x["selection"]=="3-4-1"]
    assert match and match[0]["support_ratio"]>3, match
    assert report["integrity"]["ipf_max_trio_margin_error"] < .0001
    # A false favorite from a missing ticket must fail rather than get ranked.
    try:
        scan(rows[:-1])
    except ValueError as e:
        assert "incomplete" in str(e).lower()
    else:
        raise AssertionError("Missing ticket accepted")


if __name__=="__main__":
    test_gates()
    test_isolation_and_actual_341()
    print("LIVE ODDS FRESHNESS, COMPLETENESS, AND MARKET ISOLATION TESTS PASSED")
