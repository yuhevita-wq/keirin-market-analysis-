#!/usr/bin/env python3
from __future__ import annotations

import csv, json, math
from collections import defaultdict, Counter
from pathlib import Path
from statistics import mean, pstdev

BASE = Path("data/2024/s_class_f1_all_parts/2024_q1")
TRAIN_START, TRAIN_END = "2024-02-19", "2024-02-25"
TEST_START, TEST_END = "2024-02-26", "2024-03-03"
OUT_DIR = Path("results/keirin_shogi/v7_conditional_second/2024-02-26_2024-03-03")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_FEATURES=["score","win_rate","b_count","nige_count","makuri_count","sashi_count","top2_rate","top3_rate"]
LINE_FEATURES=["line_size_ctx","line_avg_score","line_max_score","line_b_sum","line_attack_sum",
               "self_gap_line_max_score","line_max_vs_other_best","line_strength_vs_other","solo_flag"]
FEATURES=BASE_FEATURES+LINE_FEATURES
PAIR_FEATURES=["score","win_rate","b_count","makuri_count","top2_rate","line_avg_score","line_max_vs_other_best","line_strength_vs_other"]
GAP_FEATURES=["score","win_rate","b_count","nige_count","makuri_count","sashi_count","top2_rate","top3_rate",
              "line_avg_score","line_max_score","line_b_sum","line_attack_sum","line_max_vs_other_best","line_strength_vs_other"]
COND_FEATURES=["same_line","score_diff_b_minus_a","win_diff_b_minus_a","a_attack","b_support",
               "line_strength_diff_b_minus_a","same_line_position_gap","b_gap_line_max_score",
               "b_line_strength_vs_other"]
RANK_LABELS={1:"first",2:"second",3:"third"}

def read_csv(name):
    with (BASE/name).open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def num(v):
    try:return float(v)
    except:return 0.0
def ino(v):
    try:return int(float(v))
    except:return 0

def get_target_races(races,start,end):
    return {r["race_id"]:r for r in races if start<=r.get("race_date","")<=end
            and ino(r.get("entry_count"))==7 and r.get("meeting_grade")=="F1" and "Ｓ級" in r.get("race_type","")}

def minmax_norm(vals):
    lo=min(vals); hi=max(vals)
    if hi==lo:return [50.0]*len(vals)
    return [(v-lo)/(hi-lo)*100.0 for v in vals]

def enrich_line_context(rows):
    groups=defaultdict(list)
    for r in rows:
        lid=(r.get("line_id") or "").strip() or f"solo_{r.get('car_no')}"
        groups[lid].append(r)
    stats={}
    for lid,rs in groups.items():
        scores=[num(x.get("score")) for x in rs]
        bsum=sum(num(x.get("b_count")) for x in rs)
        attack=sum(num(x.get("nige_count"))+num(x.get("makuri_count")) for x in rs)
        stats[lid]={"size":len(rs),"avg_score":mean(scores),"max_score":max(scores),
                    "b_sum":bsum,"attack_sum":attack,
                    "strength":mean(scores)+0.8*bsum+0.8*attack}
    out=[]
    for r in rows:
        lid=(r.get("line_id") or "").strip() or f"solo_{r.get('car_no')}"
        s=stats[lid]
        others=[v for k,v in stats.items() if k!=lid]
        other_best=max([x["max_score"] for x in others],default=s["max_score"])
        other_str=max([x["strength"] for x in others],default=s["strength"])
        x=dict(r)
        x["_lid"]=lid; x["_line_strength"]=s["strength"]
        x["line_size_ctx"]=s["size"]; x["line_avg_score"]=s["avg_score"]; x["line_max_score"]=s["max_score"]
        x["line_b_sum"]=s["b_sum"]; x["line_attack_sum"]=s["attack_sum"]
        x["self_gap_line_max_score"]=s["max_score"]-num(r.get("score"))
        x["line_max_vs_other_best"]=s["max_score"]-other_best
        x["line_strength_vs_other"]=s["strength"]-other_str
        x["solo_flag"]=1.0 if s["size"]==1 else 0.0
        out.append(x)
    return out

def race_rank(rows,f):
    o=sorted(rows,key=lambda r:(-num(r.get(f)),ino(r.get("car_no"))))
    return {ino(r["car_no"]):i+1 for i,r in enumerate(o)}

def rel_gaps(rows):
    out={}
    for f in GAP_FEATURES:
        nv=minmax_norm([num(r.get(f)) for r in rows]); o=sorted(nv,reverse=True)
        top=o[0]; second=o[1]; third=o[2]
        for i,r in enumerate(rows):
            no=ino(r["car_no"]); out.setdefault(no,{})
            out[no][f+"_gap_top"]=top-nv[i]
            out[no][f+"_gap_second"]=abs(second-nv[i])
            out[no][f+"_gap_third"]=abs(third-nv[i])
    return out

def relation_features(a,b):
    same=1.0 if a["_lid"]==b["_lid"] else 0.0
    apos=ino(a.get("line_position")); bpos=ino(b.get("line_position"))
    posgap=(bpos-apos) if same and apos and bpos else 0.0
    return {
        "same_line":same,
        "score_diff_b_minus_a":num(b.get("score"))-num(a.get("score")),
        "win_diff_b_minus_a":num(b.get("win_rate"))-num(a.get("win_rate")),
        "a_attack":num(a.get("nige_count"))+num(a.get("makuri_count")),
        "b_support":num(b.get("sashi_count"))+num(b.get("mark_count")),
        "line_strength_diff_b_minus_a":num(b.get("_line_strength"))-num(a.get("_line_strength")),
        "same_line_position_gap":posgap,
        "b_gap_line_max_score":num(b.get("self_gap_line_max_score")),
        "b_line_strength_vs_other":num(b.get("line_strength_vs_other")),
    }

def build_models(train_ids,entries_by,results_by):
    models={}
    for finish_pos,label in RANK_LABELS.items():
        tr={f:Counter() for f in FEATURES}; ar={f:Counter() for f in FEATURES}
        pt=Counter(); pa=Counter(); gt=defaultdict(list); ga=defaultdict(list); used=0
        for rid in train_ids:
            rows0=entries_by.get(rid,[])
            if len(rows0)!=7:continue
            rows=enrich_line_context(rows0)
            target=[x for x in results_by.get(rid,[]) if ino(x.get("finish_position"))==finish_pos]
            if len(target)!=1:continue
            tno=ino(target[0]["car_no"]); ranks={f:race_rank(rows,f) for f in FEATURES}; gaps=rel_gaps(rows); used+=1
            for r in rows:
                no=ino(r["car_no"])
                for f in FEATURES:
                    rk=ranks[f][no]; ar[f][rk]+=1
                    if no==tno:tr[f][rk]+=1
                for i,a in enumerate(PAIR_FEATURES):
                    for b in PAIR_FEATURES[i+1:]:
                        key=(a,b,ranks[a][no],ranks[b][no]); pa[key]+=1
                        if no==tno:pt[key]+=1
                for g,v in gaps[no].items():
                    ga[g].append(v)
                    if no==tno:gt[g].append(v)
        uni={}
        for f in FEATURES:
            uni[f]={}
            for rk in range(1,8):
                p=(tr[f][rk]+1.0)/(ar[f][rk]+7.0)
                uni[f][rk]=math.log(max(p,1e-9)/(1/7))
        pair={}
        for key,a in pa.items():
            if a>=4:
                p=(pt[key]+1.0)/(a+7.0); pair[key]=math.log(max(p,1e-9)/(1/7))
        gap={}
        for g,vals in ga.items():
            if vals and gt[g]:
                adv=max(0.0,mean(vals)-mean(gt[g]))
                if adv>0:gap[g]={"weight":adv/100.0,"target_avg_gap":mean(gt[g]),"all_avg_gap":mean(vals)}
        models[label]={"train_races":used,"univariate":uni,"pairwise":pair,"gap_model":gap}
    return models

def build_conditional_second_model(train_ids,entries_by,results_by):
    target=defaultdict(list); allv=defaultdict(list); used=0
    for rid in train_ids:
        rows0=entries_by.get(rid,[])
        if len(rows0)!=7:continue
        rows=enrich_line_context(rows0); byno={ino(r["car_no"]):r for r in rows}
        w=[x for x in results_by.get(rid,[]) if ino(x.get("finish_position"))==1]
        s=[x for x in results_by.get(rid,[]) if ino(x.get("finish_position"))==2]
        if len(w)!=1 or len(s)!=1:continue
        ano=ino(w[0]["car_no"]); bno=ino(s[0]["car_no"])
        if ano not in byno or bno not in byno:continue
        a=byno[ano]; used+=1
        for no,b in byno.items():
            if no==ano:continue
            rf=relation_features(a,b)
            for f,v in rf.items():
                allv[f].append(v)
                if no==bno:target[f].append(v)
    model={"train_races":used,"features":{}}
    for f in COND_FEATURES:
        av=allv[f]; tv=target[f]
        if not av or not tv:continue
        sd=pstdev(av) or 1.0
        separation=abs(mean(tv)-mean(av))/sd
        model["features"][f]={"target_mean":mean(tv),"all_mean":mean(av),"sd":sd,"weight":min(2.0,separation)}
    return model

def score_rows(rows0,model,label):
    rows=enrich_line_context(rows0); ranks={f:race_rank(rows,f) for f in FEATURES}; gaps=rel_gaps(rows)
    out=[]; raws=[]
    for r in rows:
        no=ino(r["car_no"]); parts={}; s=0.0
        for f in FEATURES:
            v=model["univariate"][f][ranks[f][no]]; s+=v; parts[f]=v
        ps=0.0
        for i,a in enumerate(PAIR_FEATURES):
            for b in PAIR_FEATURES[i+1:]:
                key=(a,b,ranks[a][no],ranks[b][no])
                if key in model["pairwise"]:ps+=model["pairwise"][key]
        s+=0.30*ps; parts["pairwise_30pct"]=0.30*ps
        gb=0.0
        for g,m in model["gap_model"].items():
            gb+=max(0.0,100.0-gaps[no][g])/100.0*m["weight"]
        mult={"first":1.5,"second":1.35,"third":1.0}[label]
        s+=mult*gb; parts["relative_gap_bonus"]=mult*gb
        raws.append(s); out.append({"no":no,"name":r.get("player_name",""),"raw_score":s,"parts":parts})
    lo=min(raws); hi=max(raws)
    for x in out:x["score"]=50.0 if hi==lo else (x["raw_score"]-lo)/(hi-lo)*100.0
    return out

def choose(scored,label):
    s=sorted(scored,key=lambda x:(-x["score"],x["no"]))
    if label=="first":count=1 if s[0]["score"]-s[1]["score"]>=18 else 2
    elif label=="second":count=2 if len(s)>=3 and s[1]["score"]-s[2]["score"]>=12 else 3
    else:count=3 if len(s)>=4 and s[2]["score"]-s[3]["score"]>=15 else 4
    return sorted(x["no"] for x in s[:count])

def conditional_second_scores(rows0,first_scored,first_candidates,base_second,cond_model):
    rows=enrich_line_context(rows0); byno={ino(r["car_no"]):r for r in rows}
    fscore={x["no"]:x["score"] for x in first_scored}
    weights={a:max(1.0,fscore.get(a,1.0)) for a in first_candidates}
    sw=sum(weights.values()) or 1.0
    base={x["no"]:x["score"] for x in base_second}
    raw={}
    for bno,b in byno.items():
        cond_total=0.0; active=0.0
        for ano,w in weights.items():
            if bno==ano or ano not in byno:continue
            rf=relation_features(byno[ano],b); rs=0.0
            for f,m in cond_model["features"].items():
                z=(rf[f]-m["target_mean"])/(m["sd"] or 1.0)
                rs += -m["weight"]*(z*z)
            cond_total += (w/sw)*rs; active += w/sw
        raw[bno]=cond_total/active if active>0 else -999.0
    valid=[v for v in raw.values() if v>-900]; lo=min(valid); hi=max(valid)
    cond_norm={no:(50.0 if hi==lo else (v-lo)/(hi-lo)*100.0) for no,v in raw.items() if v>-900}
    out=[]
    for no in byno:
        c=cond_norm.get(no,0.0)
        final=0.65*base.get(no,0.0)+0.35*c
        out.append({"no":no,"score":final,"base_second_score":base.get(no,0.0),"conditional_score":c})
    return out

def main():
    races=read_csv("races.csv"); entries=read_csv("entries.csv"); results=read_csv("results.csv")
    train=get_target_races(races,TRAIN_START,TRAIN_END); test=get_target_races(races,TEST_START,TEST_END); needed=set(train)|set(test)
    eb=defaultdict(list); rb=defaultdict(list)
    for e in entries:
        if e["race_id"] in needed:eb[e["race_id"]].append(e)
    for r in results:
        if r["race_id"] in needed:rb[r["race_id"]].append(r)

    models=build_models(train.keys(),eb,rb)
    cond=build_conditional_second_model(train.keys(),eb,rb)
    rows_out=[]; detail=[]
    for rid,race in sorted(test.items(),key=lambda kv:(kv[1]["race_date"],kv[1]["track"],ino(kv[1]["race_no"]))):
        er=eb.get(rid,[])
        if len(er)!=7:continue
        actual={}; ok=True
        for p in (1,2,3):
            xs=[x for x in rb.get(rid,[]) if ino(x.get("finish_position"))==p]
            if len(xs)!=1:ok=False;break
            actual[p]=ino(xs[0]["car_no"])
        if not ok:continue

        first_sc=score_rows(er,models["first"],"first"); first_c=choose(first_sc,"first")
        second_base=score_rows(er,models["second"],"second")
        second_sc=conditional_second_scores(er,first_sc,first_c,second_base,cond); second_c=choose(second_sc,"second")
        third_sc=score_rows(er,models["third"],"third"); third_c=choose(third_sc,"third")
        board={"first":first_c,"second":second_c,"third":third_c}
        hits={"first":actual[1] in first_c,"second":actual[2] in second_c,"third":actual[3] in third_c}
        complete=all(hits.values())
        rows_out.append({"race_id":rid,"race_date":race["race_date"],"track":race["track"],"race_no":race["race_no"],
                         "first_candidates":"-".join(map(str,first_c)),"second_candidates":"-".join(map(str,second_c)),
                         "third_candidates":"-".join(map(str,third_c)),"actual_1st":actual[1],"actual_2nd":actual[2],"actual_3rd":actual[3],
                         "hit_1st":int(hits["first"]),"hit_2nd":int(hits["second"]),"hit_3rd":int(hits["third"]),
                         "complete_capture":int(complete),"candidate_cells":len(first_c)+len(second_c)+len(third_c)})
        detail.append({"race":race,"board":board,"actual":actual,"hits":hits,
                       "second_scores":second_sc,"first_scores":first_sc,"third_scores":third_sc})

    n=len(rows_out)
    summary={"algorithm":"keirin_shogi_v7_conditional_second",
             "train_period":{"start":TRAIN_START,"end":TRAIN_END},"test_period":{"start":TEST_START,"end":TEST_END},
             "train_races":cond["train_races"],"test_races":n,
             "first_hit_rate":sum(r["hit_1st"] for r in rows_out)/n if n else 0,
             "second_hit_rate":sum(r["hit_2nd"] for r in rows_out)/n if n else 0,
             "third_hit_rate":sum(r["hit_3rd"] for r in rows_out)/n if n else 0,
             "complete_capture_rate":sum(r["complete_capture"] for r in rows_out)/n if n else 0,
             "avg_candidate_cells":mean(r["candidate_cells"] for r in rows_out) if n else 0,
             "avg_first_candidates":mean(len(r["first_candidates"].split("-")) for r in rows_out) if n else 0,
             "avg_second_candidates":mean(len(r["second_candidates"].split("-")) for r in rows_out) if n else 0,
             "avg_third_candidates":mean(len(r["third_candidates"].split("-")) for r in rows_out) if n else 0,
             "conditional_features":COND_FEATURES,
             "method":"v6を維持し、2着のみP(2着=B|1着=A)を近似する条件付き関係モデルを追加。学習時は実際の1着Aと2着Bの関係を学び、予測時はv6の1着候補AごとにBを評価して統合。固定の番手セオリーは使わず、同ライン・能力差・ライン強度差・位置差などをデータ特徴として使用。2/19-2/25学習、2/26-3/3固定検証。",
             "leakage_guard":"2024-02-26以降の結果は学習に未使用。オッズ不使用。"}
    (OUT_DIR/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT_DIR/"model.json").write_text(json.dumps({"conditional_second":cond},ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT_DIR/"detail.json").write_text(json.dumps(detail,ensure_ascii=False,indent=2),encoding="utf-8")
    with (OUT_DIR/"race_log.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows_out[0].keys()) if rows_out else ["race_id"]);w.writeheader();w.writerows(rows_out)
    (OUT_DIR/"README.md").write_text("# 競輪将棋 v7 条件付き2着モデル\n\n2着を1着候補との関係で評価。オッズ不使用。\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
