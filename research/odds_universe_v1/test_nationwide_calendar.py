"""Synthetic programme-table validation: the odds index must carry true race type."""
from datetime import date

from nationwide_calendar import extract_day
from scope import in_scope, girls_race

BASE="https://keirin.kdreams.jp/kaisai/2025/01/01/"
FIRST="3720250101010001"
SECOND="3720250101010002"
THIRD="3720250101010003"
html=f"""<html><body><table>
<tr><th colspan="2">Ａ級予選</th><th>Ｓ級予選</th></tr>
<tr><td><a href="/ito/racedetail/{FIRST}/?pageType=odds">オッズ</a></td>
    <td><a href="/ito/racedetail/{SECOND}/?pageType=odds">オッズ</a></td>
    <td><a href="/ito/racedetail/{THIRD}/?pageType=odds">オッズ</a></td></tr>
</table></body></html>"""

def main():
    got,_=extract_day(html,BASE,date(2025,1,1))
    assert len(got)==3,got
    assert got[FIRST]["race_type"]=="A級予選",got
    assert got[SECOND]["race_type"]=="A級予選",got
    assert got[THIRD]["race_type"]=="S級予選",got
    assert all(v["odds_page_url"].startswith("https://keirin.kdreams.jp/") for v in got.values())
    assert in_scope("2026-06-30") and not in_scope("2026-07-01")
    assert girls_race("ガールズ予選",["L1"])
    print("PROGRAMME_LABELS_AND_SCOPE_VALIDATED")

if __name__=="__main__":
    main()
