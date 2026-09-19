#!/usr/bin/env python3
from __future__ import annotations
import json, importlib.util, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
LIVE=ROOT/"docs/keirin-shogi/live-race-data.json"
V32=ROOT/"scripts/keirin_shogi_ninecar_v32.py"
OUT=ROOT/"results/keirin_shogi/today_2026_09_20_v32_trifecta.json"

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod

v32=load("v32_today",V32)

def trifecta_tickets(board):
    out=[]
    for a in board[0]:
        for b in board[1]:
            for c in board[2]:
                if len({a,b,c})==3:
                    out.append((int(a),int(b),int(c)))
    return sorted(set(out))

def main():
    live=json.loads(LIVE.read_text(encoding="utf-8"))
    races=[]
    for x in live.get("races",[]):
        if x.get("race_date")!="2026-09-20" or x.get("track")!="富山競輪":
            continue
        if len(x.get("entries",[]))!=9:
            continue
        try:
            p=v32.predict_live(x)
            board=(tuple(p["first_candidates"]),tuple(p["second_candidates"]),tuple(p["third_candidates"]))
            t=trifecta_tickets(board)
            races.append({
                "race_no":x.get("race_no"),
                "race_id":x.get("race_id"),
                "race_type":x.get("race_type"),
                "start_time":x.get("start_time"),
                "participate":p.get("participate"),
                "board":[list(z) for z in board],
                "unique_riders":len(set(board[0])|set(board[1])|set(board[2])),
                "ticket_count":len(t),
                "tickets":[list(z) for z in t],
                "joint_board_mass":p.get("joint_board_mass"),
                "overlay_action":p.get("overlay_action"),
                "strong_rider":p.get("strong_rider"),
                "state_probs":p.get("strong_state_probabilities"),
            })
        except Exception as e:
            races.append({"race_no":x.get("race_no"),"error":f"{type(e).__name__}: {e}"})
    report={
      "date":"2026-09-20",
      "track":"富山競輪",
      "strategy_note":"v32 board -> raw ordered trifecta formation; no odds/popularity/results/payouts",
      "race_count":len(races),
      "races":sorted(races,key=lambda z:z.get("race_no",99)),
      "total_tickets_all":sum(r.get("ticket_count",0) for r in races),
      "total_tickets_participants":sum(r.get("ticket_count",0) for r in races if r.get("participate")),
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
