#!/usr/bin/env python3
from __future__ import annotations
import importlib.util,json,sys
from pathlib import Path
import joblib,numpy as np
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
V32=ROOT/"scripts/keirin_shogi_ninecar_v32.py"
SNAP=ROOT/"data/research/kyodo_2026_day1_snapshot.json"
MODEL=ROOT/"results/keirin_shogi/ninecar_v32/model.joblib"
OUT=ROOT/"results/keirin_shogi/kyodo_2026_day1_r3_spill_audit.json"
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    assert spec.loader is not None;sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
v32=load("v32_spill",V32);v2=v32.v2
def pk(a,b):return tuple(sorted((int(a),int(b))))
def main():
    data=json.loads(SNAP.read_text(encoding="utf-8"))
    x=next(r for r in data["races"] if r["race_no"]==3)
    bundle=joblib.load(MODEL);race=v32.live_race_from_payload(x)
    base=bundle["base_bundle"]
    feats,strong,p1,p2,joint=v32.state_features(race,base["models"],float(base["alpha"]),float(base["beta"]))
    X=np.asarray([[feats[f] for f in bundle["state_feature_names"]]],dtype=float)
    pp=bundle["state_model"].predict_proba(X)[0]
    scored={"race":race,"features":feats,"strong":strong,"p1":p1,"p2":p2,"joint":joint,
            "state_probs":{v32.STATE_NAMES[i]:float(pp[i]) for i in range(3)}}
    board,mass,participate,dominant,action=v32.apply_overlay(scored,bundle["rules"])
    m1,m2,m3=v2.marginals(joint)
    inc={n:m1[n]+m2[n]+m3[n] for n in m1}
    pair=defaultdict(float)
    for a,b,c,p in joint:
        pair[pk(a,b)]+=p;pair[pk(a,c)]+=p;pair[pk(b,c)]+=p
    # replay greedy 7 additions, storing candidate gains
    best=joint[0]; rows=[set([best[0]]),set([best[1]]),set([best[2]])]
    steps=[{"initial":list(best[:3]),"prob":best[3],"rows":[sorted(z) for z in rows]}]
    while sum(len(z) for z in rows)<7:
        cur=v2.board_mass(joint,rows); cand=[]
        for ri in range(3):
            for no in range(1,10):
                if no in rows[ri]:continue
                trial=[set(z) for z in rows];trial[ri].add(no)
                gain=v2.board_mass(joint,trial)-cur
                cand.append({"row":ri+1,"no":no,"gain":gain})
        cand.sort(key=lambda d:(-d["gain"],d["row"],d["no"]))
        ch=cand[0];rows[ch["row"]-1].add(ch["no"])
        steps.append({"chosen":ch,"top5":cand[:5],"rows":[sorted(z) for z in rows]})
    actual=(2,3,6)
    ordered_prob=next(p for a,b,c,p in joint if (a,b,c)==actual)
    unordered_mass=sum(p for a,b,c,p in joint if set((a,b,c))==set(actual))
    out={
      "race_no":3,"actual":[2,3,6],"board":[list(z) for z in board],"board_mass":mass,
      "participate":participate,"overlay_action":action,
      "marginals":{str(n):{"p1":m1[n],"p2":m2[n],"p3":m3[n],"top3":inc[n]} for n in range(1,10)},
      "top3_rank":[n for n in sorted(inc,key=lambda n:(-inc[n],n))],
      "pair_mass_rank":[{"pair":list(k),"mass":v} for k,v in sorted(pair.items(),key=lambda kv:(-kv[1],kv[0]))],
      "actual_pair_masses":{f"{a}-{b}":pair[pk(a,b)] for a,b in ((2,3),(2,6),(3,6))},
      "actual_ordered_triple_probability":ordered_prob,
      "actual_unordered_triple_mass":unordered_mass,
      "ordered_triple_rank":1+next(i for i,z in enumerate(joint) if z[:3]==actual),
      "unordered_triple_rank":None,
      "greedy_steps":steps
    }
    ums=defaultdict(float)
    for a,b,c,p in joint: ums[tuple(sorted((a,b,c)))]+=p
    urn=sorted(ums.items(),key=lambda kv:(-kv[1],kv[0]))
    out["unordered_triple_rank"]=1+next(i for i,(k,v) in enumerate(urn) if k==tuple(sorted(actual)))
    out["top_unordered_triples"]=[{"set":list(k),"mass":v} for k,v in urn[:20]]
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
