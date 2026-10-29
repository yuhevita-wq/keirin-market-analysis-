#!/usr/bin/env python3
"""Line-dependence mechanism audit, structurally NEW odds-free 2023 3ren-tan AI.

Frozen baseline: observe.py, git blob 19701f16c80fcc95dc8c032ed47b80dfcfe93bc6.
Jan-Oct 2023 alone fits the new group-first generative probability model.
Nov-Dec is strictly unseen until all training/feature/ablation choices are fixed.
No market prices, odds, payout, future-year data, or performance-driven holdout tuning.

Ablation: factor P(rider | prefix) as P(line | prefix) * P(rider | line,prefix).
Compare BASE, new LINE-CHOICE only, new RIDER-CHOICE only, and BOTH.
This tests *where* missing line dependence arises, not an AAA odds multiplier.
"""
from __future__ import annotations
import itertools
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/independent_trifecta_2023"))
import observe as original

OUT = ROOT / "data/audits/independent_trifecta_line_dependence_v2_2023.json"
VERSIONS = ("frozen_original", "new_line_only", "new_rider_only", "new_line_and_rider")
EPOCHS = 5
SEED = 20231031
BASE_LEARNING_RATE = 0.022
SHRINK_PER_EPOCH = 0.985
TRAIN_END = "2023-10-31"
EXPECTED = {"train": 19248, "holdout": 4192}
ROLE = ("leader", "second", "third", "fourth_plus", "solo")


def rlabel(e, c):
    _, pos, size = e["positions"][c]
    if size == 1:
        return "solo"
    if pos == 1:
        return "leader"
    if pos == 2:
        return "second"
    if pos == 3:
        return "third"
    return "fourth_plus"


def score_offset(e, value):
    avg = sum(e["scores"].values()) / len(e["scores"])
    return max(-3., min(3., (value - avg) / 12.))


def minrank(e, members):
    return min(e["ranks"][c] for c in members)


def line_shape(e):
    return "-".join(map(str, sorted(map(len, e["groups"]), reverse=True)))


def gfeatures(e, prefix, g, members):
    """Learn the LINE as an entity, not as a list of independently scored riders."""
    st = len(prefix) + 1
    whole = e["groups"][g - 1]
    size = len(whole)
    best = max(e["scores"][c] for c in members)
    rem = len(members)
    feats = {
        f"s{st}|size={min(size,4)}": 1.,
        f"s{st}|remaining={min(rem,4)}": 1.,
        f"s{st}|best_rank={min(minrank(e,members),4)}": 1.,
        f"s{st}|leader_alive={int(whole[0] in members)}": 1.,
        f"s{st}|second_alive={int(len(whole)>1 and whole[1] in members)}": 1.,
        f"s{st}|best_score": score_offset(e, best),
        f"s{st}|avg_score": score_offset(e, sum(e["scores"][c] for c in members)/rem)
    }
    if st == 2:
        first = prefix[0]
        if e["positions"][first][0] == g:
            feats["s2|revisit_first_line"] = 1.
            feats[f"s2|revisit_first_role={rlabel(e,first)}"] = 1.
            feats[f"s2|revisit_line_size={min(size,4)}"] = 1.
    if st == 3:
        a, b = prefix
        ga = e["positions"][a][0]
        gb = e["positions"][b][0]
        if ga == gb:
            if g == ga:
                feats["s3|complete_same_line_trio"] = 1.
                feats[f"s3|complete_size={min(size,4)}"] = 1.
                feats[f"s3|complete_role12={rlabel(e,a)}_{rlabel(e,b)}"] = 1.
        elif g == ga:
            feats["s3|return_to_first_line"] = 1.
        elif g == gb:
            feats["s3|stay_with_second_line"] = 1.
        else:
            feats["s3|third_new_line"] = 1.
    return feats


def cfeatures(e, prefix, c):
    """Within the selected line, learn lead/bante/third ordering independently."""
    st = len(prefix) + 1
    g, pos, size = e["positions"][c]
    rank = min(e["ranks"][c], 4)
    role = rlabel(e, c)
    members = [m for m in e["groups"][g-1] if m not in prefix]
    mean_g = sum(e["scores"][m] for m in members) / len(members)
    feats = {
        f"s{st}|role={role}": 1.,
        f"s{st}|rank={rank}": 1.,
        f"s{st}|role_rank={role}:{rank}": 1.,
        f"s{st}|field_score": score_offset(e, e["scores"][c]),
        f"s{st}|within_line_score": max(-3., min(3., (e["scores"][c]-mean_g)/12.))
    }
    if st >= 2:
        a = prefix[0]
        ga, pa, _ = e["positions"][a]
        if ga == g:
            order = "after" if pos > pa else "before"
            feats[f"s{st}|relative_to_first={order}"] = 1.
            feats[f"s{st}|first_role_{rlabel(e,a)}_then_{role}"] = 1.
    if st == 3:
        a, b = prefix
        gb, pb, _ = e["positions"][b]
        if gb == g:
            order = "after" if pos > pb else "before"
            feats[f"s3|relative_to_second={order}"] = 1.
            feats[f"s3|second_role_{rlabel(e,b)}_then_{role}"] = 1.
        if e["positions"][a][0] == gb == g:
            feats[f"s3|triple_role={rlabel(e,a)}_{rlabel(e,b)}_{role}"] = 1.
    return feats


def dot(w, feats):
    return sum(w.get(k, 0.) * v for k, v in feats.items())


def softmax(scores):
    top = max(scores.values())
    exp = {k: math.exp(min(0., v-top)) for k, v in scores.items()}
    z = sum(exp.values())
    return {k: v/z for k, v in exp.items()}


def line_members(e, prefix):
    used = set(prefix)
    return {i+1: [c for c in group if c not in used]
            for i, group in enumerate(e["groups"])
            if any(c not in used for c in group)}


def struct_components(e, prefix, wg, wr, return_features=False):
    groups = line_members(e, prefix)
    gf = {g: gfeatures(e, prefix, g, members) for g, members in groups.items()}
    # Log remaining gives neutral group probability proportional to rider count.
    gp = softmax({g: math.log(len(members)) + dot(wg, gf[g])
                  for g, members in groups.items()})
    cf = {c: cfeatures(e, prefix, c) for members in groups.values() for c in members}
    cp = {}
    for g, members in groups.items():
        within = softmax({c: dot(wr, cf[c]) for c in members})
        cp.update(within)
    if return_features:
        return gp, cp, gf, cf
    return gp, cp


def original_components(e, baseline, prefix):
    cars = [c for c in e["scores"] if c not in prefix]
    raw = {c: original.prob(e, c, prefix, baseline, True) for c in cars}
    z = sum(raw.values())
    if not z > 0:
        raise AssertionError("Original probability did not normalize")
    p = {c: v/z for c, v in raw.items()}
    groups = line_members(e, prefix)
    masses = {g: sum(p[c] for c in members) for g, members in groups.items()}
    within = {c: p[c]/masses[g] for g, members in groups.items() for c in members}
    return masses, within


def choices(e, prefix, baseline, wg, wr):
    bg, br = original_components(e, baseline, prefix)
    sg, sr = struct_components(e, prefix, wg, wr)
    groups = line_members(e, prefix)
    combos = {
        "frozen_original": (bg, br),
        "new_line_only": (sg, br),
        "new_rider_only": (bg, sr),
        "new_line_and_rider": (sg, sr)
    }
    out = {}
    for name, (gprob, rprob) in combos.items():
        vals = {c: gprob[g]*rprob[c] for g, members in groups.items() for c in members}
        if abs(sum(vals.values())-1.) > 1e-9 or any(x <= 0 for x in vals.values()):
            raise AssertionError(("Conditional probability conservation", name, prefix))
        out[name] = vals
    return out


def fit_structural(train):
    """Multinomial log likelihood of realized lines + realized rider within line.

    All learning is based on results from Jan-Oct only. Pre-fixed epochs/LR/seed.
    Critically, each stage trains on the ACTUAL earlier podium prefix, never
    on an independent candidate's hypothetical winner path.
    """
    wg, wr = {}, {}
    samples = list(train.values())
    rng = random.Random(SEED)
    for ep in range(EPOCHS):
        rng.shuffle(samples)
        lr = BASE_LEARNING_RATE / math.sqrt(ep+1)
        n, loss_g, loss_r = 0, 0., 0.
        for e in samples:
            for st in (1,2,3):
                prefix = e["order"][:st-1]
                winner = e["order"][st-1]
                target_g = e["positions"][winner][0]
                gp, cp, gf, cf = struct_components(e, prefix, wg, wr, True)
                loss_g -= math.log(max(gp[target_g], 1e-15))
                loss_r -= math.log(max(cp[winner], 1e-15))
                # Calculate entire gradient before applying updates.
                grads_g = defaultdict(float)
                for g, feats in gf.items():
                    err = (1. if g == target_g else 0.) - gp[g]
                    for key, value in feats.items():
                        grads_g[key] += err*value
                grads_r = defaultdict(float)
                for c in line_members(e, prefix)[target_g]:
                    err = (1. if c == winner else 0.) - cp[c]
                    for key, value in cf[c].items():
                        grads_r[key] += err*value
                for k, grad in grads_g.items():
                    wg[k] = wg.get(k, 0.) + lr * grad
                for k, grad in grads_r.items():
                    wr[k] = wr.get(k, 0.) + lr * grad
                n += 1
        # Fixed mild regularization, no late data involved.
        for w in (wg, wr):
            for k in w:
                w[k] *= SHRINK_PER_EPOCH
        print("STRUCTURAL_TRAIN_EPOCH", ep+1, "STAGES", n,
              "LINE_NLL", round(loss_g/n,6), "RIDER_NLL", round(loss_r/n,6),
              flush=True)
    return wg, wr


def pattern(e, t):
    l = [e["positions"][c][0] for c in t]
    d = {}
    return "".join(d.setdefault(g, "ABC"[len(d)]) for g in l)


def ticket_axes(e, t):
    a, b, c = t
    pat = pattern(e, t)
    r = [rlabel(e, c) for c in t]
    sizes = [e["positions"][c][2] for c in t]
    shape = line_shape(e)
    return {
        "line_pattern": pat,
        "line_shape_x_pattern": shape+"|"+pat,
        "line_size_first_x_pattern": str(sizes[0])+"|"+pat,
        "line_size_123": "-".join(map(str,sizes)),
        "role_first": r[0],
        "role_second": r[1],
        "role_third": r[2],
        "role_12": r[0]+"-"+r[1],
        "role_23": r[1]+"-"+r[2],
        "role_123": "-".join(r),
        "AAA_role_order": "-".join(r) if pat == "AAA" else "not_AAA",
        "AAA_line_size": str(sizes[0]) if pat == "AAA" else "not_AAA",
        "AAA_shape": shape if pat == "AAA" else "not_AAA"
    }


class Tally:
    def __init__(self):
        self.p = 0.
        self.actual = 0
        self.exposures = 0
        self.tickets = 0
        self.var = 0.
        self.day_residual = defaultdict(float)

    def add(self, p, actual, tickets, day):
        if not (-1e-8 <= p <= 1+1e-8):
            raise AssertionError("Invalid partition mass")
        p = min(1., max(0., p))
        self.p += p
        self.actual += int(actual)
        self.exposures += 1
        self.tickets += tickets
        self.var += p*(1-p)
        self.day_residual[day] += actual-p

    def result(self):
        delta = self.actual - self.p
        daily = list(self.day_residual.values())
        nday = len(daily)
        avg = sum(daily)/nday if nday else 0.
        sd = math.sqrt(nday/(nday-1)*sum((x-avg)**2 for x in daily)) if nday>1 else 0.
        return {
            "race_exposures": self.exposures,
            "trifecta_ticket_exposures": self.tickets,
            "predicted_events": round(self.p, 5),
            "actual_events": self.actual,
            "actual_minus_predicted": round(delta, 5),
            "actual_divided_by_predicted": round(self.actual/self.p, 6) if self.p else None,
            "race_z": round(delta/math.sqrt(self.var), 4) if self.var>0 else None,
            "day_cluster_z": round(delta/sd, 4) if sd>0 else None,
            "number_of_days": nday
        }


def summarize(stats):
    out = defaultdict(lambda: defaultdict(dict))
    for (version, axis, label), tally in stats.items():
        out[version][axis][label] = tally.result()
    return {m: {a: dict(sorted(d.items())) for a, d in sorted(v.items())}
            for m, v in sorted(out.items())}


def conditional_diagnostics(events, baseline, wg, wr):
    """Stage-specific errors, conditioned on *observed* earlier placings."""
    tally = defaultdict(Tally)
    for i, e in enumerate(events.values(), 1):
        a,b,c = e["order"]
        ga,gb,gc = [e["positions"][r][0] for r in (a,b,c)]
        shape = line_shape(e)
        first_role = rlabel(e,a)
        second_role = rlabel(e,b)
        for stage, prefix, target in ((1,(),a),(2,(a,),b),(3,(a,b),c)):
            options = choices(e, prefix, baseline, wg, wr)
            for version, vals in options.items():
                checks = {}
                if stage == 1:
                    for role in ROLE:
                        checks["s1_first_role="+role] = (
                            sum(p for rider,p in vals.items() if rlabel(e,rider)==role),
                            int(first_role==role))
                elif stage == 2:
                    mass = sum(p for rider,p in vals.items()
                               if e["positions"][rider][0]==ga)
                    obs = int(ga==gb)
                    checks = {
                        "s2_same_line12|ALL": (mass,obs),
                        "s2_same_line12|shape="+shape: (mass,obs),
                        "s2_same_line12|first_role="+first_role: (mass,obs),
                        "s2_same_line12|first_line_size="+str(e["positions"][a][2]): (mass,obs)
                    }
                else:
                    p13 = sum(p for rider,p in vals.items() if e["positions"][rider][0]==ga)
                    p23 = sum(p for rider,p in vals.items() if e["positions"][rider][0]==gb)
                    checks = {
                        "s3_same_line13|ALL": (p13,int(ga==gc)),
                        "s3_same_line23|ALL": (p23,int(gb==gc))
                    }
                    if ga == gb:
                        obs = int(ga==gc)
                        checks.update({
                            "s3_AAA_given_12_same|ALL": (p13,obs),
                            "s3_AAA_given_12_same|shape="+shape: (p13,obs),
                            "s3_AAA_given_12_same|line_size="+str(e["positions"][a][2]): (p13,obs),
                            "s3_AAA_given_12_same|roles12="+first_role+"-"+second_role: (p13,obs)
                        })
                    else:
                        checks["s3_ABC_given_12_different|ALL"] = (
                            sum(p for rider,p in vals.items() if e["positions"][rider][0] not in (ga,gb)),
                            int(gc not in (ga,gb))
                        )
                for label, (p,y) in checks.items():
                    tally[(version,"conditional",label)].add(p,y,0,e["date"])
        if i%5000==0:
            print("CONDITIONAL_PREFIX_AUDIT",i,flush=True)
    return summarize(tally)


def full_holdout(events, baseline, wg, wr):
    """All possible ordered triples, not winning combinations alone."""
    groupings = defaultdict(Tally)
    score = {k: {"logloss_sum": 0., "brier_sum": 0.,
                 "actual_probability_sum": 0., "probability_sum": 0.,
                 "races": 0, "tickets": 0} for k in VERSIONS}
    for i, (rid,e) in enumerate(events.items(),1):
        cars = sorted(e["scores"], key=int)
        actual = tuple(e["order"])
        top1 = choices(e, (), baseline, wg, wr)
        dists = {k:{} for k in VERSIONS}
        for a in cars:
            top2 = choices(e, (a,), baseline, wg, wr)
            for b in cars:
                if b == a:
                    continue
                top3 = choices(e, (a,b), baseline, wg, wr)
                for c in cars:
                    if c in (a,b):
                        continue
                    t = (a,b,c)
                    for version in VERSIONS:
                        dists[version][t] = (top1[version][a] *
                                             top2[version][b] *
                                             top3[version][c])
        ntriples = len(cars)*(len(cars)-1)*(len(cars)-2)
        local = defaultdict(lambda:[0.,0,0])
        # All axes are disjoint, exhaustive on each race.
        labels = {t:ticket_axes(e,t) for t in dists[VERSIONS[0]]}
        for version, dist in dists.items():
            if len(dist)!=ntriples or abs(sum(dist.values())-1)>1e-9:
                raise AssertionError(("3ren-tan probability != 100%", version,rid))
            if actual not in dist:
                raise AssertionError(("Missing actual triple",rid))
            pa = dist[actual]
            sc = score[version]
            sc["logloss_sum"] -= math.log(max(1e-15,pa))
            sc["brier_sum"] += 1 - 2*pa + sum(p*p for p in dist.values())
            sc["actual_probability_sum"] += pa
            sc["probability_sum"] += sum(dist.values())
            sc["races"] += 1
            sc["tickets"] += ntriples
            for t,prob in dist.items():
                hit = int(t==actual)
                for axis,label in labels[t].items():
                    x = local[(version,axis,label)]
                    x[0] += prob; x[1] += hit; x[2] += 1
        check = defaultdict(lambda:[0.,0,0])
        for (version,axis,label), (prob,win,n) in local.items():
            groupings[(version,axis,label)].add(prob,win,n,e["date"])
            z=check[(version,axis)]
            z[0]+=prob;z[1]+=win;z[2]+=n
        for (version,axis),(p,w,n) in check.items():
            if abs(p-1)>1e-8 or w!=1 or n!=ntriples:
                raise AssertionError(("Non-exhaustive 3ren-tan split",rid,version,axis,p,w,n))
        if i%500==0:
            print("FULL_HOLDOUT_ALL_TRIPLES",i,"RACES",flush=True)
    output = {}
    for m, sc in score.items():
        n=sc["races"]
        output[m] = {
            "races": n, "trifecta_ticket_combinations":sc["tickets"],
            "mean_3ren_tan_negative_log_likelihood":round(sc["logloss_sum"]/n,8),
            "mean_3ren_tan_brier":round(sc["brier_sum"]/n,8),
            "mean_probability_of_realized_ticket_pct":round(sc["actual_probability_sum"]*100/n,7),
            "sum_of_all_trifecta_probs_per_race":round(sc["probability_sum"]/n,10)
        }
    return output, summarize(groupings)


def check_selftest():
    e={"scores":{str(i):float(65+i*2) for i in range(1,8)},
       "positions":{"1":(1,1,3),"2":(1,2,3),"3":(1,3,3),
                    "4":(2,1,2),"5":(2,2,2),"6":(3,1,2),"7":(3,2,2)},
       "groups":[["1","2","3"],["4","5"],["6","7"]],
       "ranks":{str(i):i for i in range(1,8)},
       "order":("1","2","3"),"date":"2023-11-01"}
    base=original.train_model({"fake":e},True)
    wg,wr={},{}
    assert pattern(e,("1","2","3"))=="AAA"
    assert pattern(e,("1","2","4"))=="AAB"
    assert pattern(e,("1","4","2"))=="ABA"
    assert pattern(e,("4","1","2"))=="ABB"
    assert pattern(e,("1","4","6"))=="ABC"
    for prefix in [(),("1",),("1","2"),("4","1")]:
        probs=choices(e,prefix,base,wg,wr)
        for label,choice in probs.items():
            assert abs(sum(choice.values())-1)<1e-10, (label,prefix)
    assert ticket_axes(e,("1","2","3"))["AAA_role_order"] == "leader-second-third"
    assert ticket_axes(e,("4","1","2"))["AAA_role_order"] == "not_AAA"
    print("LINE_DEPENDENCE_V2_SELFTEST_PASSED",flush=True)


def main():
    check_selftest()
    assert original.TRAIN_END == TRAIN_END
    events, coverage = original.ingest()
    train = {rid:e for rid,e in events.items() if e["date"]<=TRAIN_END}
    hold = {rid:e for rid,e in events.items() if e["date"]>TRAIN_END}
    assert len(events)==23440 and len(train)==EXPECTED["train"] and len(hold)==EXPECTED["holdout"], (
        len(events),len(train),len(hold))
    baseline = original.train_model(train,True)
    print("BASELINE_FROZEN_AND_NEW_MODEL_TRAINING",len(train),len(hold),flush=True)
    wg,wr = fit_structural(train)
    train_cond = conditional_diagnostics(train,baseline,wg,wr)
    hold_cond = conditional_diagnostics(hold,baseline,wg,wr)
    metrics,alloc = full_holdout(hold,baseline,wg,wr)
    for model in VERSIONS:
        assert metrics[model]["races"]==4192
        assert abs(metrics[model]["sum_of_all_trifecta_probs_per_race"]-1.)<1e-9
        lp=alloc[model]["line_pattern"]
        assert set(lp)=={"AAA","AAB","ABA","ABB","ABC"}
        assert sum(q["actual_events"] for q in lp.values())==4192
        assert abs(sum(q["predicted_events"] for q in lp.values())-4192)<.01
    report={
        "state":"COMPLETED_INDEPENDENT_TRIFECTA_LINE_DEPENDENCE_V2_2023",
        "research_question":"Why does the fixed independent AI underallocate same-line AAA probability, and at which conditional stage?",
        "method":"Group-first conditional categorical likelihood: P(g1|X)*P(r1|g1,X)*P(g2|r1,X)*P(r2|g2,r1,X)*P(g3|r1,r2,X)*P(r3|g3,r1,r2,X); all candidate triples normalized.",
        "ablation_design":{
            "frozen_original":"P(original_line | prefix) x P(original_rider | original_line,prefix)",
            "new_line_only":"P(new_line | prefix) x P(original_rider | original_line,prefix)",
            "new_rider_only":"P(original_line | prefix) x P(new_rider | original_line,prefix)",
            "new_line_and_rider":"P(new_line | prefix) x P(new_rider | new_line,prefix)"
        },
        "train_period":"2023-01-01..2023-10-31",
        "unknown_test_period":"2023-11-01..2023-12-31",
        "train_races":len(train),
        "unknown_test_races":len(hold),
        "original_AI_modified":False,
        "market_odds_or_payout_in_features_labels_or_evaluation":False,
        "2024_or_later_data_accessed":False,
        "used_holdout_to_choose_hyperparameters":False,
        "fixed_training_settings":{
            "epochs":EPOCHS,"random_seed":SEED,
            "initial_learning_rate":BASE_LEARNING_RATE,
            "learning_rate_by_epoch":"initial / sqrt(epoch+1)",
            "parameter_shrink_each_epoch":SHRINK_PER_EPOCH
        },
        "baseline_model_source":"research/independent_trifecta_2023/observe.py",
        "baseline_model_blob_sha":"19701f16c80fcc95dc8c032ed47b80dfcfe93bc6",
        "fitted_feature_counts":{"group_choice":len(wg),"rider_within_line_choice":len(wr)},
        "learned_group_features_sorted":dict(sorted(wg.items())),
        "learned_role_features_sorted":dict(sorted(wr.items())),
        "training_actual_prefix_conditional_coupling":train_cond,
        "unknown_test_actual_prefix_conditional_coupling":hold_cond,
        "unknown_test_all_trifecta_proper_scoring_rules":metrics,
        "unknown_test_full_probability_allocation":alloc,
        "coverage":coverage,
        "audit_guarantees":[
            "Every eligible race enumerates n*(n-1)*(n-2) ordered triples; model sum == 1.",
            "No 2024/2025/2026 future labels, no odds as input or target.",
            "2023 Jan-Oct is training and not independent confirmation; Nov-Dec is strictly held out.",
            "Group/rider ablations isolate conditional line selection from intra-line rider placement.",
            "Conditional diagnostics use ACTUAL earlier podium as prefix and are not interchangeable with unconditional AAA counts.",
            "Scores and published line forecasts in historical archive are not independently cutoff-time verified.",
            "Predicted line is an archived PRE-RACE forecast, not certified as actual race-day lineup.",
            "Potential chronology or repeated rider dependence must be considered in inference.",
            "No automatic adoption of new model: only forward unknown score and calibration judge superiority."
        ]
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    summary={
      "train":len(train),"holdout":len(hold),
      "metrics":metrics,
      "holdout_AAA_counts":{m:alloc[m]["line_pattern"]["AAA"] for m in VERSIONS},
      "holdout_stage2_same_line":{m:hold_cond[m]["conditional"]["s2_same_line12|ALL"] for m in VERSIONS},
      "holdout_stage3_AAA_given12":{m:hold_cond[m]["conditional"]["s3_AAA_given_12_same|ALL"] for m in VERSIONS},
      "output":str(OUT.relative_to(ROOT))
    }
    print("LINE_DEPENDENCE_V2_EXPERIMENT_COMPLETE",json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
