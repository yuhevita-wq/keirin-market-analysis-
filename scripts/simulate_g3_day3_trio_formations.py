from __future__ import annotations

import csv
import io
import itertools
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = ROOT / "data" / "grade_races" / "g3"
OUT = ROOT / "results" / "g3_day3_reality"


def decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp932", "shift_jis"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def read_member(zf: zipfile.ZipFile, basename: str) -> list[dict[str, str]]:
    names = [n for n in zf.namelist() if Path(n).name == basename]
    if not names:
        return []
    return list(csv.DictReader(io.StringIO(decode(zf.read(names[0])))))


def num(v: str | None) -> float:
    try:
        x = float(v or "")
        return x if math.isfinite(x) else float("-inf")
    except Exception:
        return float("-inf")


def meeting_day(race_id: str) -> int | None:
    if len(race_id) == 16 and race_id.isdigit():
        try:
            return int(race_id[10:12])
        except ValueError:
            return None
    return None


def choose_lines(entries: list[dict[str, str]]):
    by_line: dict[int, list[dict[str, str]]] = defaultdict(list)
    singles = []
    for e in entries:
        lid = e.get("line_id", "")
        lp = e.get("line_position", "")
        if lid.isdigit() and lp.isdigit():
            by_line[int(lid)].append(e)
        else:
            singles.append(e)
    ranked = []
    for lid, mem in by_line.items():
        mem = sorted(mem, key=lambda x: int(x.get("line_position", "99")))
        if len(mem) >= 2:
            key = (num(mem[0].get("score")) + num(mem[1].get("score")), num(mem[0].get("score")), 1 if len(mem)>=3 else 0, -lid)
            ranked.append((key, lid, mem))
    ranked.sort(reverse=True)
    return ranked


def trio_combo(cars) -> str:
    return "=".join(map(str, sorted(set(cars))))


def combos_from_groups(g1, g2, g3):
    out = set()
    for a in g1:
        for b in g2:
            for c in g3:
                if len({a,b,c}) == 3:
                    out.add(trio_combo((a,b,c)))
    return out


def top_scored(entries, exclude, n):
    xs = [e for e in entries if int(e['car_no']) not in exclude]
    xs.sort(key=lambda e: (-num(e.get('score')), int(e['car_no'])))
    return [int(e['car_no']) for e in xs[:n]]


def payout_map(rows):
    out = {}
    for p in rows:
        if p.get('ticket_type') == '3連複' and p.get('status') == 'paid':
            try:
                out[p['combination']] = int(float(p.get('payout_yen') or 0))
            except Exception:
                pass
    return out


def eval_rule(name, eligible_rows, selector):
    races=hits=stake=ret=0
    pts=[]
    off_races=off_hits=0
    main_races=main_hits=0
    by_year=defaultdict(lambda:[0,0,0,0]) # races,hits,stake,ret
    for row in eligible_rows:
        combos = selector(row)
        if not combos:
            continue
        races += 1; pts.append(len(combos)); stake += len(combos)*100
        pm = row['trio_payout_map']
        got = sum(pm.get(c,0) for c in combos)
        hit = got>0
        hits += int(hit); ret += got
        if row['winner_in_main']:
            main_races += 1; main_hits += int(hit)
        else:
            off_races += 1; off_hits += int(hit)
        y=row['race_date'][:4]; b=by_year[y]; b[0]+=1; b[1]+=int(hit); b[2]+=len(combos)*100; b[3]+=got
    return {
        'name':name,'races':races,'hit_races':hits,'hit_rate':hits/races if races else 0,
        'avg_points':sum(pts)/len(pts) if pts else 0,'stake':stake,'return':ret,'roi':ret/stake if stake else 0,
        'offmain_races':off_races,'offmain_hits':off_hits,'offmain_hit_rate':off_hits/off_races if off_races else 0,
        'main_races':main_races,'main_hits':main_hits,'main_hit_rate':main_hits/main_races if main_races else 0,
        'by_year':{y:{'races':v[0],'hits':v[1],'hit_rate':v[1]/v[0] if v[0] else 0,'roi':v[3]/v[2] if v[2] else 0} for y,v in sorted(by_year.items())}
    }


def main():
    races={}; entries_by=defaultdict(list); results_by=defaultdict(list); payouts_by=defaultdict(list)
    for path in sorted(ARCHIVE_ROOT.rglob('*.zip')):
        with zipfile.ZipFile(path) as zf:
            for r in read_member(zf,'races.csv'): races.setdefault(r['race_id'],r)
            for e in read_member(zf,'entries.csv'): entries_by[e['race_id']].append(e)
            for rr in read_member(zf,'results.csv'): results_by[rr['race_id']].append(rr)
            for p in read_member(zf,'payouts.csv'): payouts_by[p['race_id']].append(p)

    rows=[]
    for rid,r in races.items():
        if meeting_day(rid)!=3: continue
        es=entries_by[rid]; rs=results_by[rid]
        ranked=choose_lines(es)
        if not ranked: continue
        finish=[]
        for rr in rs:
            try:
                pos=int(rr.get('finish_position') or 99); car=int(rr.get('car_no') or 0)
            except: continue
            if pos<=3: finish.append((pos,car))
        if len(finish)!=3 or sorted(p for p,_ in finish)!=[1,2,3]: continue
        top3=[c for _,c in sorted(finish)]
        main=ranked[0][2]; maincars=[int(e['car_no']) for e in main]
        rival=ranked[1][2] if len(ranked)>1 else []
        rivalcars=[int(e['car_no']) for e in rival]
        a=maincars[0]; b=maincars[1]; c=maincars[2] if len(maincars)>=3 else None
        d=rivalcars[0] if len(rivalcars)>=1 else None; e=rivalcars[1] if len(rivalcars)>=2 else None; f=rivalcars[2] if len(rivalcars)>=3 else None
        score_order=sorted(es,key=lambda x:(-num(x.get('score')),int(x['car_no'])))
        scorecars=[int(x['car_no']) for x in score_order]
        pm=payout_map(payouts_by[rid])
        rows.append({'race_id':rid,'race_date':r.get('race_date',''),'race_type':r.get('race_type',''),'a':a,'b':b,'c':c,'d':d,'e':e,'f':f,'maincars':maincars,'rivalcars':rivalcars,'scorecars':scorecars,'entries':es,'winner_in_main':top3[0] in maincars,'top3':top3,'trio_payout_map':pm})

    # Candidate rules. Roles: A/B/C main line; D/E/F strongest rival line.
    def valid(xs): return [x for x in xs if x is not None]
    rules={
      # off-main capture family
      'OFF1_DE__ABDE__ABDEC': lambda r: combos_from_groups(valid([r['d'],r['e']]), valid([r['a'],r['b'],r['d'],r['e']]), valid([r['a'],r['b'],r['d'],r['e'],r['c']])),
      'OFF2_DE__ABDE__ABDE_plus_top1': lambda r: combos_from_groups(valid([r['d'],r['e']]), valid([r['a'],r['b'],r['d'],r['e']]), valid([r['a'],r['b'],r['d'],r['e']])+top_scored(r['entries'],set(valid([r['a'],r['b'],r['d'],r['e']])),1)),
      'OFF3_D_E_AB__ABDE__ABDE_top1': lambda r: combos_from_groups(valid([r['d'],r['e'],r['a'],r['b']]), valid([r['a'],r['b'],r['d'],r['e']]), valid([r['a'],r['b'],r['d'],r['e']])+top_scored(r['entries'],set(valid([r['a'],r['b'],r['d'],r['e']])),1)),
      'OFF4_DE__ABDE__all_score_top6': lambda r: combos_from_groups(valid([r['d'],r['e']]), valid([r['a'],r['b'],r['d'],r['e']]), r['scorecars'][:6]),
      'OFF5_D_or_E_key_with_AB_and_outside': lambda r: combos_from_groups(valid([r['d'],r['e']]), valid([r['a'],r['b'],r['d'],r['e']]), top_scored(r['entries'],set(),6)),
      # main narrow family
      'MAIN1_AB__ABC__ABC': lambda r: combos_from_groups(valid([r['a'],r['b']]), valid([r['a'],r['b'],r['c']]), valid([r['a'],r['b'],r['c']])),
      'MAIN2_AB__ABCD__ABCD': lambda r: combos_from_groups(valid([r['a'],r['b']]), valid([r['a'],r['b'],r['c'],r['d']]), valid([r['a'],r['b'],r['c'],r['d']])),
      'MAIN3_A_or_B_key__ABDE__ABDE': lambda r: combos_from_groups(valid([r['a'],r['b']]), valid([r['a'],r['b'],r['d'],r['e']]), valid([r['a'],r['b'],r['d'],r['e']])),
      'MAIN4_AB__ABCD__ABCDE': lambda r: combos_from_groups(valid([r['a'],r['b']]), valid([r['a'],r['b'],r['c'],r['d']]), valid([r['a'],r['b'],r['c'],r['d'],r['e']])),
      'MAIN5_AB_pair_plus_one_DE': lambda r: {trio_combo((r['a'],r['b'],x)) for x in valid([r['c'],r['d'],r['e']]) if len({r['a'],r['b'],x})==3},
    }
    res=[eval_rule(n,rows,fn) for n,fn in rules.items()]
    res.sort(key=lambda x:(x['roi'],x['hit_rate']),reverse=True)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'trio_formations.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    md=['# G3三日目 3連複フォーメーション比較','',f'- 対象: {len(rows)}R','', '## 全候補']
    for x in res:
        md.append(f"- {x['name']}: {x['hit_races']}/{x['races']}={x['hit_rate']:.1%}, 平均{x['avg_points']:.1f}点, ROI {x['roi']:.1%}, 外勝者捕捉 {x['offmain_hit_rate']:.1%}, 主力勝者捕捉 {x['main_hit_rate']:.1%}")
    md += ['', '注: 保存済みG3三日目の確定結果に対する100円均等バックテスト。将来収益を保証しない。']
    (OUT/'trio_formations.md').write_text('\n'.join(md)+'\n',encoding='utf-8')

if __name__=='__main__': main()
