"""Correct sealed collector orientation, gated by independently labelled public rankings.
The original immutable ciphertext and original observation remain unchanged.
"""
import sys, json, math, re, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import requests
from bs4 import BeautifulSoup
from research.snowberg_observation_20261027 import observe as old
old.OUT=ROOT/"research/snowberg_observation_corrected_20261027"
VALIDATED={}
SLUGS=dict(zip(
"11 13 21 22 23 24 25 26 27 28 31 32 34 35 36 37 38 42 43 44 45 46 47 48 51 53 54 55 56 61 62 63 71 73 74 75 81 83 84 85 86 87".split(),
"hakodate iwakitaira yahiko maebashi toride utsunomiya omiya seibuen keiokaku tachikawa matsudo chiba kawasaki hiratsuka odawara ito shizuoka nagoya gifu ogaki toyohashi toyama matsusaka yokkaichi fukui nara mukomachi wakayama kishiwada tamano hiroshima hofu takamatsu komatsushima kochi matsuyama kokura kurume takeo sasebo beppu kumamoto".split()))
def swap_key(t):
    return (t[1],t[0]) if len(t)==2 else (t[0],t[2],t[1])
def verify_page(html,rid,books,finish):
    soup=BeautifulSoup(html,"html.parser")
    result=[]
    for tr in soup.select("table.result_table tr"):
        cells=tr.find_all("td",recursive=False)
        if len(cells)>2 and cells[1].get_text(strip=True) in ("1","2","3"):
            result.append((int(cells[1].get_text(strip=True)),int(cells[2].get_text(strip=True))))
    if tuple(car for rank,car in sorted(result))!=tuple(finish):
        raise RuntimeError("Public result vs stored result mismatch")
    evidence={}
    for kind,arity in (("2shatan",2),("3rentan",3)):
        refs={}
        for el in soup.select(f"#JS_ODDSCONTENTS_{kind} span.num"):
            selection=el.get_text(strip=True)
            val=el.find_next_sibling("span",class_="odds")
            if re.fullmatch(r"[1-9](?:-[1-9]){"+str(arity-1)+"}",selection) and val:
                try:v=float(val.get_text(strip=True).replace(",",""))
                except ValueError:continue
                refs[tuple(map(int,selection.split("-")))]=v
        if len(refs)<20:raise RuntimeError("Insufficient independently labelled ranked odds")
        identity=sum(math.isclose(books[kind].get(t,-1),v,abs_tol=1e-7) for t,v in refs.items())
        swapped=sum(math.isclose(books[kind].get(swap_key(t),-1),v,abs_tol=1e-7) for t,v in refs.items())
        if swapped!=len(refs) or identity==len(refs):
            raise RuntimeError("Archive does not match uniquely identified orientation correction")
        evidence[kind]={"labelled_prices_checked":len(refs),"original_matches":identity,"corrected_matches":swapped}
    return evidence
base_observe=old.observe
def corrected_observe(rid,day,books,finish,rtype,source):
    if source!="sealed_four_markets":
        raise RuntimeError("No unverified fallback allowed for corrected observation")
    n=len({c for t in books["3rentan"] for c in t})
    gate=(day[:4],n)
    if gate not in VALIDATED:
        slug=SLUGS.get(rid[:2])
        if not slug:raise RuntimeError("No verified venue slug for orientation check")
        url=f"https://keirin.kdreams.jp/{slug}/racedetail/{rid}/"
        last=None
        for attempt in range(3):
            try:
                response=requests.get(url,timeout=45);response.raise_for_status()
                evidence=verify_page(response.text,rid,books,finish)
                VALIDATED[gate]={"race_id":rid,"public_url":url,"cars":n,"year":day[:4],
                                 "result_match":True,"markets":evidence}
                print("INDEPENDENT_ORIENTATION_CHECK",json.dumps(VALIDATED[gate]),flush=True)
                break
            except Exception as e:
                last=e
                if attempt<2:time.sleep(2)
        else:raise RuntimeError("Independent orientation verification failed for "+rid) from last
    corrected={}
    for kind,book in books.items():
        corrected[kind]={swap_key(t):o for t,o in book.items()} if kind in ("2shatan","3rentan") else book
    base_observe(rid,day,corrected,finish,rtype,"sealed_four_markets_orientation_verified")
old.observe=corrected_observe
base_save=old.save
def save(status):
    base_save(status)
    path=old.OUT/"summary.json"
    d=json.loads(path.read_text())
    d["orientation"]={"original_observation_status":"SUPERSEDED_INVALID_ORDERED_PRICE_MAPPING",
       "two_car_correction":"archive (b,a) -> canonical (a,b)",
       "three_car_correction":"archive (a,c,b) -> canonical (a,b,c)",
       "method":"Explicit number labels in independent public ranking tables; result table verifies stored podium.",
       "sample_checks":list(VALIDATED.values()),
       "scope_limit":"At least one race per encountered year/car-count; not every archived quote independently refetched.",
       "original_ciphertexts_modified":False}
    # Compact public report for easy retrieval, separate from complete aggregate cells.
    compact={k:v for k,v in d.items() if k!="groups"}
    compact["price_groups"]=[r for r in d["groups"] if r["group"][3]=="price"]
    compact["conditional_order"]=[r for r in d["groups"] if r["group"][3]=="conditional_order"]
    path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n")
    (old.OUT/"compact.json").write_text(json.dumps(compact,ensure_ascii=False,indent=2)+"\n")
old.save=save
if __name__=="__main__":
    import os
    if not os.environ.get(old.bridge.SECRET_ENV):
        raise RuntimeError("Full source key unavailable; no silent partial substitute")
    try:
        old.sealed()
        save("COMPLETE_WITH_INDEPENDENT_ORIENTATION_CHECKS")
    except Exception:
        save("FAILED_OR_PARTIAL_NOT_RESEARCH_CONCLUSION")
        raise
