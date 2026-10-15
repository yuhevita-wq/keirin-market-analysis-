"""Offline fixtures for all sold odds, no network use."""
import itertools
import csv
import gzip
from pathlib import Path
from tempfile import TemporaryDirectory
from odds_market.collect_all_odds import parse_all, expected, complete_saved_races, FIELDS, ExcludedRace


def grid(kind, n):
    size = 6 if kind.startswith("2waku") else n
    heads = list(range(1,size+1))
    s=["<table><tr><th></th>"+''.join(f"<th>{x}</th>" for x in heads)+"</tr>"]
    for a in heads:
        if a==1 and kind in ('2shafuku','2wakufuku'):
            continue  # Live KDreams omits the all-empty first triangular row
        s.append(f"<tr><td>{a}</td>")
        for b in heads:
            if kind in ("2shatan","2wakutan"):
                valid = a!=b or kind=="2wakutan" and a in (4,5,6)
            elif kind in ("2shafuku","2wakufuku"):
                valid = a>b or kind=="2wakufuku" and a==b and a in (4,5,6)
            else:
                valid = a>b
            value = "5.4～8.2" if kind=="wide" else "42.2"
            placeholder="--" if kind in ("2shafuku","2wakufuku") else ""
            s.append(f"<td>{value if valid else placeholder}</td>")
        s.append("</tr>")
    s.append("</table>")
    return "".join(s)


def market_fixture(n):
    result=["<body><h1>2024年07月04日 レース詳細</h1>",
            "<h2>3連単 2車単 3連複 2車複 2枠単 2枠複 ワイド 確定オッズ</h2>",
            "<h3>A級一般</h3>",
            "<span>2024/07/04 10:51現在</span>"]
    for a in range(1,n+1):
        cars=[x for x in range(1,n+1) if x!=a]
        result.append("<table><tr><th></th>"+''.join(f"<th>{x}</th>" for x in cars)+"</tr>")
        for b in cars:
            result.append(f"<tr><td>{b}</td>")
            for c in cars:
                result.append(f"<td>{'' if b==c else '73.1'}</td>")
            result.append("</tr>")
        result.append("</table>")
    for kind in ("2shatan","2shafuku","2wakutan","2wakufuku","wide"):
        if kind.startswith("2waku") and n!=9:
            continue
        result.append(grid(kind,n))
    result.append("<div class=odds_table><table>")
    for a,b,c in itertools.combinations(range(1,n+1),3):
        result.append(f"<tr><td>1</td><td>{a}={b}={c}</td><td>88.8</td></tr>")
    result.append("</table></div><table class=refund_table><tr><td>1</td><td>1=2=3</td><td>99999.9</td></tr></table></body>")
    return "".join(result)


def test_complete():
    for n in (3,4,5,6,7,8,9):
        race_id="7320240704010001"
        rows=parse_all(market_fixture(n),race_id)
        kinds=expected(n)
        assert len(rows)==sum(map(len,kinds.values()))
        assert len(rows)==sum(map(len,kinds.values()))
        expected_counts={3:19,4:52,5:110,6:200,7:329,8:504,9:783}
        assert len(rows)==expected_counts[n],(n,len(rows))
        assert {r[2] for r in rows}==set(kinds)
        assert all(r[1]=="A級一般" for r in rows)
        for kind,choices in kinds.items():
            got={tuple(map(int,r[3].split("-"))) for r in rows if r[2]==kind}
            assert got==choices,(kind,len(got),len(choices))
        wide=[r for r in rows if r[2]=="wide"]
        assert all(row[4]=="5.4" and row[5]=="8.2" for row in wide)


def test_incomplete_rejected():
    html=market_fixture(7).replace("<td>1=2=3</td>","<td>1=2=4</td>")
    try:
        parse_all(html,"7320240704010001")
    except ValueError as exc:
        assert "Incomplete" in str(exc)
    else:
        raise AssertionError("Incomplete 3renpuku must fail")


def test_date_mismatch():
    try:
        parse_all(market_fixture(7),"7320240704020001")
    except ValueError as exc:
        assert "date" in str(exc).lower()
    else:
        raise AssertionError("Mismatched race date must fail")


def test_girls_rejected():
    html=market_fixture(7).replace("<h3>A級一般</h3>", "<h3>ガールズ予選</h3>")
    try:
        parse_all(html,"7320240704010001")
    except ExcludedRace:
        pass
    else:
        raise AssertionError("Girls market must never be stored")


def test_unknown_race_type_rejected():
    html=market_fixture(7).replace("<h3>A級一般</h3>", "")
    try:
        parse_all(html,"7320240704010001")
    except ValueError as exc:
        assert "Race type" in str(exc)
    else:
        raise AssertionError("Unverifiable race category must fail closed")


def test_resumable_full_books():
    with TemporaryDirectory() as tmp:
        path=Path(tmp)/"archive.csv.gz"
        odds=parse_all(market_fixture(7),"7320240704010001")
        with gzip.open(path,"wt",encoding="utf-8",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(FIELDS)
            writer.writerows(odds)
        complete=complete_saved_races(path)
        assert len(complete)==1
        assert len(complete["7320240704010001"])==329
        with gzip.open(path,"wt",encoding="utf-8",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(FIELDS)
            writer.writerows(odds[:-1])
        assert complete_saved_races(path)=={}, "Partial book cannot count as completed"


if __name__=="__main__":
    test_complete()
    test_incomplete_rejected()
    test_date_mismatch()
    test_girls_rejected()
    test_unknown_race_type_rejected()
    test_resumable_full_books()
    print("ALL 7 BET TYPE ODDS TESTS PASS")
