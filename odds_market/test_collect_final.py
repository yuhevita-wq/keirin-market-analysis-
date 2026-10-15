"""Offline matrix/index regression tests. Does not call the betting website."""
import itertools
from datetime import date
from odds_market.collect_final import parse_final_html, discover_races


def make_fixture(n=7, missing=None):
    x = ['<body>確定オッズ 発売票数 123,456 2024/07/06 21:42現在']
    for first in range(1, n+1):
        riders = [r for r in range(1, n+1) if r != first]
        x.append('<table><tr><th></th>' + ''.join(f'<th>{k}</th>' for k in riders) + '</tr>')
        for second in riders:
            x.append(f'<tr><td>{second}</td>')
            for third in riders:
                value = '' if second == third or (first,second,third)==missing else '42.4'
                x.append(f'<td>{value}</td>')
            x.append('</tr>')
        x.append('</table>')
    return ''.join(x) + '</body>'


def test_complete_matrix():
    for n in (5,7,9):
        count, odds, stamp, pool = parse_final_html(make_fixture(n))
        assert count == n
        assert len(odds) == n*(n-1)*(n-2)
        assert set(odds) == set(itertools.permutations(range(1,n+1),3))
        assert odds[(1,2,3)] == '42.4'
        assert stamp == '2024-07-06 21:42'
        assert pool == 123456


def test_missing_odds_rejected():
    try:
        parse_final_html(make_fixture(7, missing=(1,2,3)))
    except ValueError:
        pass
    else:
        raise AssertionError("A partial matrix was accepted")


def test_date_index():
    html = (
        '<a href="/kurume/racedetail/8320240704030003/?pageType=odds">3R</a>'
        '<a href="/kurume/racedetail/8320240704020004/">previous day</a>'
        '<a href="/komatsushima/racedetail/7320240704040001/">next day</a>'
    )
    urls = discover_races(html, date(2024,7,6))
    assert set(urls) == {'8320240704030003'}
    assert urls['8320240704030003'].endswith('?pageType=odds')


if __name__ == "__main__":
    test_complete_matrix()
    test_missing_odds_rejected()
    test_date_index()
    print("Odds parser offline checks: OK")
