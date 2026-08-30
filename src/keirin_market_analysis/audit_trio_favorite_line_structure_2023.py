from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data' / '2023' / 's_class_yosen'
OUT = ROOT / 'data' / 'audits' / 'trio_favorite_line_structure_2023.json'


def read_csv(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def parse_combo(s: str):
    s = s.strip().replace('=', '-').replace(',', '-')
    return tuple(sorted(int(x) for x in s.split('-') if x.strip()))


def parse_lines(s: str):
    out=[]
    for part in (s or '').split('/'):
        cars=[]
        for x in part.split('-'):
            x=x.strip()
            if x.isdigit(): cars.append(int(x))
        if cars: out.append(tuple(cars))
    return out


def main():
    races={r['race_id']:r for r in read_csv(DATA/'races.csv')}
    odds=read_csv(DATA/'trio_final_odds.csv')
    by={}
    for r in odds:
        if r.get('odds_status')!='available': continue
        try:o=float(r['odds'])
        except:continue
        if o<=0:continue
        rid=r['race_id']; c=parse_combo(r['combination'])
        if rid not in by or (o,c)<(by[rid][0],by[rid][1]): by[rid]=(o,c)
    counts=Counter(); examples=[]
    for rid,(o,fav) in sorted(by.items()):
        race=races.get(rid)
        if not race: continue
        lines=parse_lines(race.get('predicted_line_formation',''))
        favset=set(fav)
        exact3=any(len(line)==3 and set(line)==favset for line in lines)
        same_line=any(favset.issubset(set(line)) for line in lines)
        if exact3: cls='exact_3car_line'
        elif same_line: cls='same_line_but_not_exact3'
        else: cls='not_same_line'
        counts[cls]+=1
        if len(examples)<20:
            examples.append({'race_id':rid,'favorite':list(fav),'favorite_odds':o,'formation':race.get('predicted_line_formation',''),'class':cls})
    total=sum(counts.values())
    out={'status':'TRIO_FAVORITE_LINE_STRUCTURE_2023','year':2023,'years_read':[2023],'analyzable_races':total,'counts':dict(counts),'pct':{k:100*v/total for k,v in counts.items()},'definition':{'exact_3car_line':'3連複1人気3車が、楽天Kドリームス並び予想の1つのラインをちょうど3車で構成','same_line_but_not_exact3':'1人気3車が同一ライン内だが、そのラインは3車ちょうどではない','not_same_line':'1人気3車が同一ラインに揃っていない'},'examples_first20':examples,'note':'This is a descriptive audit only. Line data were not used in the market-only favorite-miss logic.'}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
