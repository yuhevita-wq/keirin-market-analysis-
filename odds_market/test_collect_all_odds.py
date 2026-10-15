"""Offline fixtures for all sold odds, no network use."""
import itertools
from odds_market.collect_all_odds import parse_all, expected


def grid(kind, n):
    size = 6 if kind.startswith("2waku") else n
    heads = list(range(1,size+1))
    s=["<table><tr><th></th>"+''.join(f"<th>{x}</th>" for x in heads)+"</tr>"]
    for a in heads:
        s.append(f"<tr><td>{a}</td>")
        for b in heads:
            if kind in ("2shatan","2wakutan"):
                valid = a!=b or kind=="2wakutan" and a in (4,5,6)
            elif kind in ("2shafuku","2wakufuku"):
                valid = a>b or kind=="2wakufuku" and a==b and a in (4,5,6)
            else:
                valid = a>b
            value = "5.4～8.2" if kind=="wide" else "42.2"
            s.append(f"<td>{value if valid else ''}</td>")
        s.append("</tr>")
    s.append("</table>")
    return "".join(s)


def market_fixture(n):
    result=["<body><h1>2024年07月04日 レース詳細</h1>",
            "<h2>3連単 2車単 3連複 2車複 2枠単 2枠複 ワイド 確定オッズ</h2>",
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
    result.append("<table>")
    for a,b,c in itertools.combinations(range(1,n+1),3):
        result.append(f"<tr><td>1</td><td>{a}={b}={c}</td><td>88.8</td></tr>")
    result.append("</table></body>")
    return "".join(result)


def test_complete():
    for n in (5,6,7,9):
        race_id="7320240704010001"
        rows=parse_all(market_fixture(n),race_id)
        kinds=expected(n)
        assert len(rows)==sum(map(len,kinds.values()))
        assert len(rows)==sum(map(len,kinds.values()))
        assert len(rows)==(110 if n==5 else 200 if n==6 else 329 if n==7 else 783)
        assert {r[1] for r in rows}==set(kinds)
        for kind,choices in kinds.items():
            got={tuple(map(int,r[2].split("-"))) for r in rows if r[1]==kind}
            assert got==choices,(kind,len(got),len(choices))
        wide=[r for r in rows if r[1]=="wide"]
        assert all(row[3]=="5.4" and row[4]=="8.2" for row in wide)


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


if __name__=="__main__":
    test_complete()
    test_incomplete_rejected()
    test_date_mismatch()
    print("ALL 7 BET TYPE ODDS TESTS PASS")
