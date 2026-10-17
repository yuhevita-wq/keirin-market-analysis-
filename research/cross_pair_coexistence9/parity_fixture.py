#!/usr/bin/env python3
"""Generate one untouched real-race fixture for Python/JavaScript pair-model parity."""
import json
import math
import sys
from itertools import combinations
from pathlib import Path
import numpy as np
import train

def main(outpath):
    src=train.module()
    bundle=json.loads((train.OUT/"pair_model.json").read_text())
    race=None
    for r in src.load_races():
        if r.race_date[:4]!="2025" or r.grade not in {"G1","G2","G3"}:
            continue
        if any(str(e.get(k,"")).strip() in {"", "None"} for e in r.entries
            for k in train.RIDER_FIELDS):
            continue
        leaders={src.ino(e.get("line_id")) for e in r.entries if src.ino(e.get("line_position"))==1}
        if leaders != {src.ino(e.get("line_id")) for e in r.entries}:continue
        race=r;break
    if race is None:raise RuntimeError("No complete untouched real nine-car input")
    lineids=sorted({src.ino(e["line_id"]) for e in race.entries})
    lines=[
        [src.ino(e["car_no"]) for e in sorted(
            (e for e in race.entries if src.ino(e["line_id"])==lid),
            key=lambda e:src.ino(e["line_position"]))]
        for lid in lineids]
    input={"race_id":race.race_id,"race_type":race.race_type,
        "meeting_grade":race.grade,"lines":lines,"entries":race.entries}
    rows=[]
    pairs=list(combinations(range(1,10),2))
    b=[];m=[]
    for a,c in pairs:
        x,z=train.feature_row(race,a,c,src)
        b.append(x);m.append(z)
        rows.append({"pair":[a,c],"base":x,"mechanism":z})
    b=np.asarray(b);m=np.asarray(m)
    def calc(part,model):
        d=b if model=="rider_only" else np.concatenate((b,m),axis=1)
        obj=bundle["models"][part][model]
        means=np.asarray(obj["mean"]);scale=np.asarray(obj["scale"]);coef=np.asarray(obj["coef"])
        z=obj["intercept"]+((d-means)/scale)@coef
        return 1/(1+np.exp(-np.clip(z,-40,40)))
    t2=calc("top2","mechanism");t2base=calc("top2","rider_only")
    t3=calc("top3","mechanism");t3base=calc("top3","rider_only")
    t2=t2/t2.sum();t2base=t2base/t2base.sum()
    for i,row in enumerate(rows):
        row.update(top2=float(t2[i]),top2RiderOnly=float(t2base[i]),
            top3=float(t3[i]),top3RiderOnly=float(t3base[i]))
    Path(outpath).write_text(json.dumps({"input":input,"rows":rows},ensure_ascii=False))
    print("parity race",race.race_id,"lines",lines)

if __name__=="__main__":
    main(sys.argv[1])
