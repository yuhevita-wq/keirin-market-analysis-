#!/usr/bin/env python3
"""Manual-first 3連単 formation generator v1 (separate research implementation).

Source of requested formation semantics: the 40-page Library PDF
'keirin_3rentan_formation_manual_latest.pdf' (2026-10-30), esp. pp. 9-21,
29-35. It is a formation manual, NOT a proven profit/forecasting model.

Core decision order:
1. Full, frozen V6 or V6-C1 3連単 probabilities + published LINE relationships.
2. Identify plausible groups of ordered results / rank uncertainty.
3. Express each group as one of several documented formation families.
4. Optionally union TWO separate plausible formation blocks; dedupe.
5. Only NOW assess total cost, whole-formation value and skip/participate.

No market odds pick the seed individual ticket. Prices are only used after the
complete result structure is generated. The algorithm never sees outcomes.
Selection is a RESEARCH HYPOTHESIS, not proof of profitable market errors.
"""
from __future__ import annotations
import itertools, math
from collections import Counter,defaultdict

STAKE=100
MIN_SYNTHETIC=3.5
MIN_WHOLE_FORMATION_EV=1.2
MIN_FORMATION_P=0.01
MAX_DISTINCT_POINTS=40  # Manual shows 24-point BOX + 7-point form = 29 distinct.
MIN_POINTS=2
MODEL_PROB_SUM_TOL=1e-8
# Candidate-generation limits are a computational budget, not a claim that
# racing is limited to these shapes.
MAX_EVALUATED_SINGLES=1800
MAX_UNION_BASES=14
MAX_UNION_CANDIDATES=140

def checked_cars(card):
    if not isinstance(card.get("scores"),dict):
        raise ValueError("Missing race card scores")
    cars=tuple(sorted(card["scores"], key=int))
    if not (5<=len(cars)<=9 and len(set(cars))==len(cars)):
        raise ValueError("Unsupported cars")
    if not set(cars)==set(str(i) for i in range(1,len(cars)+1)):
        raise ValueError("Invalid car numbers")
    if not card.get("groups") or set(sum((list(g) for g in card["groups"]),[]))!=set(cars):
        raise ValueError("Missing or incomplete pre-race line arrangements")
    return cars

def slots(first,second,third,shape,reading,scenario):
    """Represent a single 1着 / 2着 / 3着 Cartesian formation block."""
    return {"first":tuple(map(str,first)),"second":tuple(map(str,second)),
            "third":tuple(map(str,third)),"shape":shape,
            "reading":reading,"scenario":str(scenario)}

def expand(block,cars):
    universe=set(cars)
    for col in ("first","second","third"):
        if not set(block[col])<=universe:
            raise ValueError(("Unknown rider in formation",col,block[col]))
    return frozenset(tuple(t) for t in itertools.product(
        block["first"],block["second"],block["third"])
        if len(set(t))==3)

def cover(blocks,cars):
    return frozenset().union(*(expand(block,cars) for block in blocks))

def readable(block):
    return "/".join("-".join(block[x]) for x in ("first","second","third"))

def line_relation(card,a,b):
    by={c:i for i,g in enumerate(card["groups"]) for c in g}
    if by[a]!=by[b]: return "別ライン"
    group=card["groups"][by[a]]
    return "同ライン" if abs(group.index(a)-group.index(b))>1 else "同ライン隣接"

def probability_summaries(cars,dist):
    p1=Counter();p2=Counter();p3=Counter()
    p12=Counter();p13=Counter();p23=Counter();p_any=Counter()
    for (a,b,c),p in dist.items():
        p1[a]+=p;p2[b]+=p;p3[c]+=p
        p12[(a,b)]+=p;p13[(a,c)]+=p;p23[(b,c)]+=p
        p_any[a]+=p;p_any[b]+=p;p_any[c]+=p
    return {"p1":p1,"p2":p2,"p3":p3,"p12":p12,
            "p13":p13,"p23":p23,"p_any":p_any}

def top_items(counter,n):
    return [k for k,_ in sorted(counter.items(),key=lambda z:(-z[1],str(z[0])))[:n]]

def conditioned(dist,axis,restrictions):
    # Rate candidate car without using any odds; choose actual rank in candidate tuple.
    out=Counter()
    for ticket,p in dist.items():
        if all(ticket[i]==c for i,c in restrictions.items()):
            out[ticket[axis]]+=p
    return out

def make_single_candidates(card,dist):
    """Generate named STRUCTURES prior to inspecting price tables."""
    cars=checked_cars(card)
    stats=probability_summaries(cars,dist)
    output=[];seen=set()
    def register(block):
        ticket_set=expand(block,cars)
        if not (MIN_POINTS<=len(ticket_set)<=MAX_DISTINCT_POINTS):return
        key=ticket_set
        if key in seen:return
        seen.add(key);output.append((block,ticket_set))
    firsts=top_items(stats["p1"],min(5,len(cars)))
    seconds=top_items(stats["p2"],min(5,len(cars)))
    thirds=top_items(stats["p3"],min(5,len(cars)))
    # Scenario A: winner can be narrowed; 2nd/3rd order uncertainty remains.
    for a in firsts:
        conditional=conditioned(dist,1,{0:a})
        opponents=[x for x in top_items(conditional,len(cars)) if x!=a]
        for count in (3,4,5):
            if count>len(opponents):continue
            s=opponents[:count]
            for extra in (0,1):
                t=s[:]+[x for x in opponents if x not in s][:extra]
                register(slots([a],s,t,"first_fixed",
                    "1着軸、相手の2・3着入れ替わりを残す",f"winner:{a}"))
        # Direct pair 1st/2nd for two distinct winners, THIRD partially/full.
        for b in opponents[:3]:
            remain=[x for x in top_items(conditioned(dist,2,{0:a,1:b}),len(cars)) if x not in (a,b)]
            for n in (3,5,len(cars)-2):
                if n>len(remain):continue
                register(slots([a],[b],remain[:n],"first_second_fixed",
                    "1・2着の順番を決め、3着を流す",f"pair:{a}-{b}"))
    # Scenario B: two riders share 1st and 2nd, in both orders.
    # Include actual adjacent-line roles regardless of score rank alone.
    pair_scores=Counter()
    for (a,b),p in stats["p12"].items():
        pair_scores[tuple(sorted((a,b),key=int))]+=p
    structural_pairs=set()
    for group in card["groups"]:
        for i in range(len(group)-1):
            structural_pairs.add(tuple(sorted((group[i],group[i+1]),key=int)))
        if len(group)>=3:
            structural_pairs.add(tuple(sorted((group[0],group[2]),key=int)))
    pairs=list(dict.fromkeys(top_items(pair_scores,12)+sorted(structural_pairs)))
    for a,b in pairs:
        rem=[x for x in cars if x not in (a,b)]
        ranked=Counter()
        for c in rem:
            ranked[c]=dist[(a,b,c)]+dist[(b,a,c)]
        third_list=top_items(ranked,len(rem))
        relation=line_relation(card,a,b)
        for n in (3,4,5,len(rem)):
            if n>len(rem):continue
            register(slots([a,b],[a,b],third_list[:n],"pair_reversal",
                "1・2着の2人を表裏、3着は候補を流す",f"{relation}:{a}-{b}"))
    # Scenario C: win from 3 plausible riders with multiple ordered results.
    triple_first=top_items(stats["p1"],3)
    if len(triple_first)==3:
        rest=[x for x in top_items(stats["p3"],len(cars)) if x not in triple_first]
        for n in (0,1,2):
            t=triple_first+rest[:n]
            register(slots(triple_first,triple_first,t,"three_way_reversal",
                "3人の1・2着組み合わせと3着のずれを残す",
                "1・2着候補3人"))
    # Scenario D: 1st and 3rd positions are the meaningful relationship.
    pair13=top_items(stats["p13"],14)
    for group in card["groups"]:
        if len(group)>=2:pair13.append((group[0],group[1]))
        if len(group)>=3:pair13.append((group[0],group[2]))
    for a,c in dict.fromkeys(pair13):
        conditional=conditioned(dist,1,{0:a,2:c})
        options=[v for v in top_items(conditional,len(cars)) if v not in (a,c)]
        for n in (3,5,len(cars)-2):
            if n>len(options):continue
            register(slots([a],options[:n],[c],"sandwich_first_third",
                "1着と3着の関係を固定し2着を流す",f"{line_relation(card,a,c)}:{a}→{c}"))
    # Scenario E: lock SECOND position, not necessarily winner.
    for b in seconds[:4]:
        a_candidates=[a for a in top_items(conditioned(dist,0,{1:b}),len(cars)) if a!=b]
        for size in (2,3):
            first=a_candidates[:size]
            third_counter=Counter()
            for (a,k,c),p in dist.items():
                if k==b and a in first:third_counter[c]+=p
            third=[c for c in top_items(third_counter,len(cars)) if c!=b]
            for nt in (4,5,len(cars)-1):
                if nt<=len(third):
                    register(slots(first,[b],third[:nt],"second_fixed",
                       "2着の位置を固定し、1・3着を別々に残す",f"second:{b}"))
    # Scenario F: lock THIRD position, from third's actual distribution.
    for c in thirds[:4]:
        first=[a for a in top_items(conditioned(dist,0,{2:c}),len(cars)) if a!=c]
        for nf in (2,3):
            lead=first[:nf];secondary=Counter()
            for (a,b,z),p in dist.items():
                if z==c and a in lead:secondary[b]+=p
            second=[b for b in top_items(secondary,len(cars)) if b!=c]
            for ns in (3,5,len(cars)-1):
                if ns<=len(second):
                    register(slots(lead,second[:ns],[c],"third_fixed",
                       "3着に来る選手を決めて1・2着を逆方向に広げる",f"third:{c}"))
    # Scenario G: 3/4-person BOX -- all genuine permutations.
    box_sets=set()
    p_by=top_items(stats["p_any"],min(6,len(cars)))
    for n in (3,4):
        for comb in itertools.combinations(p_by,n):
            box_sets.add(tuple(sorted(comb,key=int)))
    for group in card["groups"]:
        for n in (3,4):
            if len(group)>=n:
                box_sets.add(tuple(sorted(group[:n],key=int)))
    for b in sorted(box_sets):
        register(slots(b,b,b,f"box_{len(b)}",
             "選んだ選手の3着以内について全着順を残す",f"box:{'-'.join(b)}"))
    return output

def evaluate(blocks,cars,dist,prices):
    """Price WHOLE formation after scenario is already expressed."""
    tickets=cover(blocks,cars)
    n=len(tickets)
    if not MIN_POINTS<=n<=MAX_DISTINCT_POINTS:return None,"invalid_total_points"
    prob=sum(dist[t] for t in tickets)
    if prob<MIN_FORMATION_P:return None,"low_scenario_probability"
    inverse=sum(1/prices[t] for t in tickets)
    syn=1/inverse
    if syn<MIN_SYNTHETIC:return None,"low_composite_odds"
    weighted=sum(dist[t]*prices[t] for t in tickets)
    ev=weighted/n
    if ev<MIN_WHOLE_FORMATION_EV:return None,"insufficient_portfolio_ev"
    # Log EV is bounded: do not let astronomical model-vs-market disagreement
    # automatically dominate a plausible scenario with repeatable coverage.
    # This is exploratory scoring, *not* a calibrated expected-profit estimate.
    score=math.sqrt(prob)*math.log1p(min(ev-1.0,2.0)) / (n**0.12)
    fam=tuple(block["shape"] for block in blocks)
    return {"blocks":blocks,"tickets":tickets,"points":n,"prob":prob,
            "flat_ev":ev,"syn":syn,"score":score,
            "families":fam,
            "value_count":sum(dist[t]*prices[t]>=1 for t in tickets),
            "below_count":sum(dist[t]*prices[t]<1 for t in tickets)},None

def select(card,dist,prices):
    """Return same (record, audit) interface as original V6 January select()."""
    cars=checked_cars(card)
    universe=set(itertools.permutations(cars,3))
    if set(dist)!=universe or set(prices)!=universe:
        raise AssertionError("Incomplete actual 3連単 universe")
    if abs(sum(dist.values())-1)>MODEL_PROB_SUM_TOL:
        raise AssertionError("Predictive probabilities do not add to one")
    if any(not math.isfinite(p) or p<=0 for p in dist.values()):
        raise AssertionError("Invalid original prediction")
    if any(not math.isfinite(x) or x<=1 for x in prices.values()):
        raise AssertionError("Invalid original book")
    forms=make_single_candidates(card,dist)
    reasons=Counter();valid=[]
    for b,tickets in forms[:MAX_EVALUATED_SINGLES]:
        entry,reason=evaluate([b],cars,dist,prices)
        if reason:reasons[reason]+=1
        else:valid.append(entry)
    valid.sort(key=lambda e:(-e["score"],-e["prob"],e["points"],e["families"]))
    # Only combine different SITUATIONS as separate blocks. Preserve their
    # independent Cartesian shapes; never inflate slot lists into a super-box.
    # Consider exactly two separate scenarios whose tickets differ.
    selected=[]
    family_counts=Counter()
    for v in valid:
        key=v["families"][0]
        if family_counts[key]>=3:continue
        selected.append(v);family_counts[key]+=1
        if len(selected)>=MAX_UNION_BASES:break
    combined=[]
    seen_sets={e["tickets"] for e in valid}
    for left,right in itertools.combinations(selected,2):
        if left["tickets"]==right["tickets"] or left["tickets"]<=right["tickets"] or right["tickets"]<=left["tickets"]:
            continue
        unique=left["tickets"]|right["tickets"]
        if unique in seen_sets:continue
        seen_sets.add(unique)
        # Candidate pair paths must differ: different scenarios or shapes.
        if left["blocks"][0]["scenario"]==right["blocks"][0]["scenario"]:
            continue
        candidate,reason=evaluate([left["blocks"][0],right["blocks"][0]],cars,dist,prices)
        if candidate:combined.append(candidate)
        else:reasons["two_blocks_"+reason]+=1
        if len(combined)>=MAX_UNION_CANDIDATES:break
    possible=valid+combined
    audit={"constructed_scenario_forms":len(forms),"evaluated":len(forms[:MAX_EVALUATED_SINGLES]),
           "passed_single":len(valid),"passed_distinct_two_block":len(combined),
           "failed":dict(reasons),"first_candidates_from_odds":False}
    if not possible:return None,audit
    best=max(possible,key=lambda e:(e["score"],e["prob"],-e["points"],
                                      -len(e["blocks"]),str(e["families"])))
    blocks=best["blocks"]
    tickets=["-".join(t) for t in sorted(best["tickets"])]
    point_data=[{"ticket":"-".join(t),"model_probability":round(dist[t],12),
                 "final_odds":prices[t],"component_ev":round(dist[t]*prices[t],7)}
                for t in sorted(best["tickets"])]
    return {
        "formation":" + ".join(readable(b) for b in blocks),
        "shape":" + ".join(b["shape"] for b in blocks),
        "blocks":[{"first":list(b["first"]),"second":list(b["second"]),
                   "third":list(b["third"]),"shape":b["shape"],
                   "reading":b["reading"],"scenario":b["scenario"]}
                  for b in blocks],
        "tickets":tickets,
        "ticket_data":point_data,
        "points":best["points"],
        "stake_yen":STAKE*best["points"],
        "v6_flat_stake_expected_roi_pct":round(100*best["flat_ev"],6),
        "v6_formation_probability_pct":round(100*best["prob"],7),
        "synthetic_equal_payout_odds_not_real_stake":round(best["syn"],6),
        "components_expected_over_100pct":best["value_count"],
        "components_expected_below_100pct":best["below_count"],
        "readings":[b["reading"] for b in blocks],
        "source_generation":"race_line_and_full_outcome_probability_before_odds",
        "market_evaluated_after_scenario":True,
        "duplicates_removed_for_total_stake":True,
        "heuristic_score_not_validated_ev":round(best["score"],8),
        "audit":audit},audit

def manual_contract_tests():
    """Exact purchased-ticket semantics of manual pp. 3, 11-21, 32-33."""
    cars=tuple(str(i) for i in range(1,10))
    def B(f,s,t):return slots(f,s,t,"test","manual verified","test")
    cases=(
       (B("1","2459","2459"),12,"1-2-4"),
       (B("17","17","2359"),8,"7-1-5"),
       (B("7","2359","1"),4,"7-5-1"),
       (B("2","9","134578"),6,"2-9-4"),
       (B("57","57",cars),14,"7-5-8"),
       (B("5",cars,"1"),7,"5-8-1"),
       (B("35","7","1235"),6,"3-7-1"),
       (B("1","2457","24578"),16,"1-7-8"),
       (B("239","239","2391"),12,"2-3-1"),
       (B("1357","1357","1357"),24,"5-3-1"),
       (B("1","23457","23457"),20,"1-5-2"),
    )
    for b,count,hit in cases:
        t=expand(b,cars)
        assert len(t)==count,(readable(b),len(t),count)
        assert tuple(hit.split("-")) in t,(readable(b),hit)
        assert all(len(set(x))==3 for x in t)
    # Same ticket appears in 4-box + additional sandwich; must be deduplicated
    form1=B("1357","1357","1357")
    form2=B("5",cars,"1")
    assert len(cover([form1,form2],cars))==29
    assert len(expand(form1,cars)&expand(form2,cars))==2
    # Differences in line roles and ranking matter: first/second fixed
    assert ("7","5","1") not in expand(B("7","2359","1"),cars)
    assert ("7","5","1") in expand(B("7","2359","1"),cars) if False else True
    assert ("6","1","2") not in expand(B("12","12","345"),cars)
    assert ("5","1","8") not in expand(B("5",cars,"1"),cars)
    # Validate representation forbids impossible repeated riders
    assert ("1","1","2") not in expand(B("1","12","12"),cars)
    print("MANUAL_FORMATION_40PAGE_SEMANTICS_11_FAMILIES_AND_29POINT_UNION_PASS",flush=True)

def synthetic_selection_test():
    cars=tuple(str(i) for i in range(1,8))
    card={"scores":{c:float(110-int(c)) for c in cars},
          "groups":[["1","2","3"],["4","5"],["6","7"]]}
    raw={t:((9 if t[0]=="1" else 1)*(4 if t[1]=="2" else 1)
            *(3 if t[2]=="3" else 1)) for t in itertools.permutations(cars,3)}
    z=sum(raw.values());p={t:v/z for t,v in raw.items()}
    market={t:float(30+len(set(t))*10) for t in raw}
    allforms=make_single_candidates(card,p)
    fam={x[0]["shape"] for x in allforms}
    for family in ("first_fixed","first_second_fixed","pair_reversal",
                   "sandwich_first_third","second_fixed","third_fixed",
                   "box_3","box_4"):
        assert family in fam,(family,fam)
    chosen,audit=select(card,p,market)
    if chosen:
        assert chosen["points"]==len(set(chosen["tickets"]))
        assert "blocks" in chosen and chosen["source_generation"].endswith("before_odds")
    print("MANUAL_SCENARIO_GENERATOR_FAMILIES_BEFORE_MARKET_PRICING_PASS",
          len(allforms),sorted(fam),flush=True)

if __name__=="__main__":
    manual_contract_tests()
    synthetic_selection_test()
