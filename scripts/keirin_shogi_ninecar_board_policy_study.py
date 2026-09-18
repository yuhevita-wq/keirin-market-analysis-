#!/usr/bin/env python3
from __future__ import annotations

"""Diagnostic study for nine-car board construction.

Keeps the v2 probabilistic model fixed and compares board extraction policies
without changing the learned probabilities. 2024 H1 chooses the policy; 2025
and 2026 are forward checks.
"""

import importlib.util, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"scripts/keirin_shogi_ninecar_v2.py"
OUT=ROOT/"results/keirin_shogi/ninecar_board_policy_study.json"

spec=importlib.util.spec_from_file_location("ninecar_v2_diag", BASE)
mod=importlib.util.module_from_spec(spec)
assert spec.loader
sys.modules[spec.name]=mod
spec.loader.exec_module(mod)

def marginal_board(joint, sizes):
    p1,p2,p3=mod.marginals(joint)
    rows=[]
    for probs,k in zip((p1,p2,p3),sizes):
        sel=sorted(probs,key=lambda n:(-probs[n],n))[:k]
        rows.append(tuple(sorted(sel)))
    rows=tuple(rows)
    return rows,mod.board_mass(joint,rows)

def evaluate(races,models,alpha,policy):
    rows=[]
    row_sizes=[]
    hit_patterns={}
    for race in races:
        joint=mod.predict_joint(race,models,alpha)
        if policy["type"]=="greedy":
            board,mass=mod.greedy_board(joint,policy["budget"])
        else:
            board,mass=marginal_board(joint,tuple(policy["sizes"]))
        hits=mod.captured(race,board)
        pat="".join("1" if x else "0" for x in hits)
        hit_patterns[pat]=hit_patterns.get(pat,0)+1
        row_sizes.append(tuple(len(x) for x in board))
        rows.append({"rows":board,"hits":hits,"special":mod.special_second_case(race,board)})
    met=mod.evaluate_predictions(rows)
    met["hit_patterns"]=hit_patterns
    counts={}
    for s in row_sizes:
        key="-".join(map(str,s)); counts[key]=counts.get(key,0)+1
    met["row_size_patterns"]=counts
    return met

def main():
    races=mod.load_races()
    tr24,ev24=mod.split_before(races,2024)
    m24=mod.fit_models(tr24)
    tune=[r for r in ev24 if r.race_date<="2024-06-30"]
    alpha,_=mod.choose_alpha(m24,tune)

    policies=[]
    for budget in (6,7,8,9):
        policies.append({"type":"greedy","budget":budget})
    for total in (6,7,8,9):
        for a in range(1,6):
            for b in range(1,6):
                for c in range(1,6):
                    if a+b+c==total:
                        policies.append({"type":"marginal","sizes":[a,b,c]})

    tuned=[]
    for pol in policies:
        met=evaluate(tune,m24,alpha,pol)
        tuned.append({"policy":pol,"metrics":met})
    tuned.sort(key=lambda x:(x["metrics"]["full_board_capture"],x["metrics"]["special_cross_line_second_capture"] or 0,x["metrics"]["second_capture"]),reverse=True)

    # Freeze the top policy separately for each exact total size, avoiding
    # silently winning by spending more pieces.
    selected={}
    for total in (6,7,8,9):
        candidates=[]
        for row in tuned:
            pol=row["policy"]
            t=pol.get("budget") if pol["type"]=="greedy" else sum(pol["sizes"])
            if t==total: candidates.append(row)
        selected[str(total)]=candidates[0]

    periods={}
    for year in (2025,2026):
        tr,ev=mod.split_before(races,year)
        models=mod.fit_models(tr)
        periods[str(year)]={}
        for total,chosen in selected.items():
            periods[str(year)][total]=evaluate(ev,models,alpha,chosen["policy"])

    report={
      "study":"ninecar_board_policy_fixed_probabilities",
      "alpha":alpha,
      "selection_period":"2024-01-01..2024-06-30",
      "selected_by_piece_total":selected,
      "forward":periods,
      "top20_tuning":tuned[:20]
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
