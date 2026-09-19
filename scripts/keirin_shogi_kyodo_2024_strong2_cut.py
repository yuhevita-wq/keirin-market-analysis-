#!/usr/bin/env python3
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"scripts/keirin_shogi_ninecar_wide_study.py"
OUT=ROOT/"results/keirin_shogi/kyodo_2024_strong2_cut.json"

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod

w=load("wide_base_strong2_cut",BASE)
START,END="2024-09-13","2024-09-14"
STAKE=100

def top3_inclusion(joint):
    p1,p2,p3=w.v2.marginals(joint)
    return {n:p1[n]+p2[n]+p3[n] for n in p1}

def summarize(rows,key):
    bets=[r for r in rows if r[key+"_tickets"]>0]
    tickets=sum(r[key+"_tickets"] for r in bets)
    payout=sum(r[key+"_payout"] for r in bets)
    hits=sum(r[key+"_payout"]>0 for r in bets)
    return {
        "races":len(bets),"tickets":tickets,"stake_yen":tickets*STAKE,
        "payout_yen":payout,"profit_yen":payout-tickets*STAKE,
        "roi_pct":100*payout/(tickets*STAKE) if tickets else None,
        "hit_races":hits,"hit_rate_pct":100*hits/len(bets) if bets else None,
        "avg_tickets":tickets/len(bets) if bets else None,
    }

def main():
    races=w.v2.load_races()
    payouts,_=w.load_wide_payouts()
    scored,rules,cal_n=w.build_forward_scored(races,2024)
    rows=[]
    for r in scored:
        race=r["race"]
        if race.grade!="G2" or not (START<=race.race_date<=END):
            continue
        paid=payouts.get(race.race_id)
        if not paid: continue
        board,mass,participate,dominant,action=w.v32.apply_overlay(r,rules)
        if not participate: continue
        base=w.union_pairs(board)
        inc=top3_inclusion(r["joint"])
        ranked=sorted(inc,key=lambda n:(-inc[n],n))
        strong_pair=tuple(sorted((ranked[0],ranked[1])))

        # Strongest "筋": strongest same-line unordered pair by the model's
        # own top3 co-survival probability from joint504. No odds/popularity.
        pair_probs=w.joint_pair_probabilities(r["joint"])
        line_of={int(e["car_no"]):int(float(e.get("line_id") or 0)) for e in race.entries}
        same_line=[
            p for p in base
            if line_of.get(p[0],0)>0 and line_of.get(p[0])==line_of.get(p[1])
        ]
        strong_line_pair=max(
            same_line,
            key=lambda p:(pair_probs.get(p,0.0),-p[0],-p[1]),
            default=None,
        )

        cut=set(base)
        cut.discard(strong_pair)

        cut_both=set(base)
        cut_both.discard(strong_pair)
        if strong_line_pair is not None:
            cut_both.discard(strong_line_pair)

        # H3: after strong2 + strongest-line cuts, remove exactly one more
        # remaining pair: the pair with the highest joint co-top3 probability.
        # This tests whether the most obvious surviving pair is still too cheap.
        h3_pair=max(
            cut_both,
            key=lambda p:(pair_probs.get(p,0.0),-p[0],-p[1]),
            default=None,
        )
        cut_h3=set(cut_both)
        if h3_pair is not None:
            cut_h3.discard(h3_pair)

        # H4: remove one additional remaining pair after H3,
        # again choosing the highest remaining joint co-top3 probability.
        h4_pair=max(
            cut_h3,
            key=lambda p:(pair_probs.get(p,0.0),-p[0],-p[1]),
            default=None,
        )
        cut_h4=set(cut_h3)
        if h4_pair is not None:
            cut_h4.discard(h4_pair)

        # Provisional H5: two overlapping top3 scenarios -> exactly five wide pairs.
        # IMPORTANT: H5 is applied AFTER H1-H4. Any pair removed by H1-H4
        # stays removed and cannot re-enter here.
        #
        # 1) Collapse joint504 ordered outcomes into unordered 3-rider sets.
        # 2) Keep only 3-rider sets whose three internal wide pairs all survive H4.
        # 3) Search every pair of such 3-rider sets sharing exactly two riders.
        #    Their union is exactly five wide pairs.
        # 4) Choose the pair of scenarios with maximum pre-race joint mass sum.
        # No odds/popularity/result/payout information is used for selection.
        from collections import defaultdict
        h5_set_mass=defaultdict(float)
        for a,b,c,prob in r["joint"]:
            s=tuple(sorted((a,b,c)))
            h5_set_mass[s] += prob

        def tri_edges(s):
            a,b,c=s
            return {
                tuple(sorted((a,b))),
                tuple(sorted((a,c))),
                tuple(sorted((b,c))),
            }

        h5_valid_sets=[
            s for s in h5_set_mass
            if tri_edges(s) <= cut_h4
        ]
        h5_best=None
        h5_best_key=None
        for i,s1 in enumerate(h5_valid_sets):
            e1=tri_edges(s1)
            for s2 in h5_valid_sets[i+1:]:
                if len(set(s1) & set(s2)) != 2:
                    continue
                tickets=e1 | tri_edges(s2)
                if len(tickets) != 5:
                    continue
                key=(
                    h5_set_mass[s1] + h5_set_mass[s2],
                    min(h5_set_mass[s1],h5_set_mass[s2]),
                    -sum(s1),
                    -sum(s2),
                )
                if h5_best_key is None or key > h5_best_key:
                    h5_best_key=key
                    h5_best=(s1,s2,tickets)

        provisional_h5=set()
        h5_s1=h5_s2=None
        h5_s1_mass=h5_s2_mass=None
        if h5_best is not None:
            h5_s1,h5_s2,provisional_h5=h5_best
            h5_s1_mass=h5_set_mass[h5_s1]
            h5_s2_mass=h5_set_mass[h5_s2]

        # H5: remove exactly one more surviving pair after H4,
        # again the highest remaining joint co-top3 probability pair.
        h5_pair=max(
            cut_h4,
            key=lambda p:(pair_probs.get(p,0.0),-p[0],-p[1]),
            default=None,
        )
        cut_h5=set(cut_h4)
        if h5_pair is not None:
            cut_h5.discard(h5_pair)

        # H7: "most expected lead line" public-bias cut from the H4 state.
        # Define the line mechanically from PRE-RACE entry data only.
        # Compare line leaders (line_position == 1) lexicographically by:
        #   1) B count desc
        #   2) nige_count desc
        #   3) line_size desc
        #   4) score desc
        #   5) car_no asc (deterministic tie-break)
        # Then cut exactly leader + line_position 2 if that wide pair still
        # survives after H4. Do NOT fall through to the next line.
        entries_by_no={int(e["car_no"]):e for e in race.entries}
        leaders=[
            e for e in race.entries
            if int(float(e.get("line_position") or 0))==1
            and int(float(e.get("line_id") or 0))>0
        ]
        def lead_bias_key(e):
            return (
                float(e.get("b_count") or 0),
                float(e.get("nige_count") or 0),
                float(e.get("line_size") or 0),
                float(e.get("score") or 0),
                -int(float(e.get("car_no") or 0)),
            )
        h7_leader=max(leaders,key=lead_bias_key,default=None)
        h7_pair=None
        h7_line_id=None
        if h7_leader is not None:
            h7_line_id=int(float(h7_leader.get("line_id") or 0))
            leader_no=int(float(h7_leader.get("car_no") or 0))
            second=next((
                e for e in race.entries
                if int(float(e.get("line_id") or 0))==h7_line_id
                and int(float(e.get("line_position") or 0))==2
            ),None)
            if second is not None:
                second_no=int(float(second.get("car_no") or 0))
                h7_pair=tuple(sorted((leader_no,second_no)))
        cut_h7=set(cut_h4)
        h7_pair_was_candidate=bool(h7_pair in cut_h4) if h7_pair else False
        if h7_pair_was_candidate:
            cut_h7.discard(h7_pair)

        # H8: "third-place only" public-bias cut from the H4 state.
        # Rider third-bias score = P3 - max(P1, P2).
        # Select the rider with the largest POSITIVE third-bias score.
        # Pair that rider with the strongest other rider by total top3 inclusion
        # P1+P2+P3, and cut exactly that wide pair if it survives H4.
        p1m,p2m,p3m=w.v2.marginals(r["joint"])
        third_bias={
            n: p3m[n] - max(p1m[n],p2m[n])
            for n in p1m
        }
        h8_third_rider=max(
            third_bias,
            key=lambda n:(third_bias[n],p3m[n],-n),
            default=None,
        )
        if h8_third_rider is not None and third_bias[h8_third_rider] <= 0:
            h8_third_rider=None

        h8_anchor=None
        h8_pair=None
        if h8_third_rider is not None:
            others=[n for n in inc if n != h8_third_rider]
            h8_anchor=max(
                others,
                key=lambda n:(inc[n],-n),
                default=None,
            )
            if h8_anchor is not None:
                h8_pair=tuple(sorted((h8_third_rider,h8_anchor)))

        cut_h8=set(cut_h4)
        h8_pair_was_candidate=bool(h8_pair in cut_h4) if h8_pair else False
        if h8_pair_was_candidate:
            cut_h8.discard(h8_pair)

        # H9A: line-third-position bias.
        # Among riders at line_position == 3, choose the rider with the highest
        # third-place marginal P3. Pair them with the strongest rider in the same
        # line by total top3 inclusion. Cut if that pair survives H4.
        pos3=[e for e in race.entries if int(float(e.get("line_position") or 0))==3]
        h9a_third=None
        h9a_anchor=None
        h9a_pair=None
        if pos3:
            h9a_entry=max(
                pos3,
                key=lambda e:(p3m.get(int(float(e["car_no"])),0.0), -int(float(e["car_no"]))),
            )
            h9a_third=int(float(h9a_entry["car_no"]))
            lid=int(float(h9a_entry.get("line_id") or 0))
            same=[int(float(e["car_no"])) for e in race.entries
                  if int(float(e.get("line_id") or 0))==lid and int(float(e["car_no"]))!=h9a_third]
            if same:
                h9a_anchor=max(same,key=lambda n:(inc[n],-n))
                h9a_pair=tuple(sorted((h9a_third,h9a_anchor)))
        cut_h9a=set(cut_h4)
        h9a_candidate=bool(h9a_pair in cut_h4) if h9a_pair else False
        if h9a_candidate:
            cut_h9a.discard(h9a_pair)

        # H9B: "mark/追 rider can hang on for 3rd" bias.
        # Eligible riders are style 追. Rank them by P3-P1, then P3, then mark_count.
        # Pair with the immediately preceding same-line rider (position-1).
        h9b_third=None
        h9b_anchor=None
        h9b_pair=None
        chase=[e for e in race.entries if str(e.get("style") or "").strip()=="追"]
        if chase:
            h9b_entry=max(
                chase,
                key=lambda e:(
                    p3m.get(int(float(e["car_no"])),0.0)-p1m.get(int(float(e["car_no"])),0.0),
                    p3m.get(int(float(e["car_no"])),0.0),
                    float(e.get("mark_count") or 0),
                    -int(float(e["car_no"])),
                ),
            )
            h9b_third=int(float(h9b_entry["car_no"]))
            lid=int(float(h9b_entry.get("line_id") or 0))
            pos=int(float(h9b_entry.get("line_position") or 0))
            prev=next((e for e in race.entries
                       if int(float(e.get("line_id") or 0))==lid
                       and int(float(e.get("line_position") or 0))==pos-1),None)
            if prev is not None:
                h9b_anchor=int(float(prev["car_no"]))
                h9b_pair=tuple(sorted((h9b_third,h9b_anchor)))
        cut_h9b=set(cut_h4)
        h9b_candidate=bool(h9b_pair in cut_h4) if h9b_pair else False
        if h9b_candidate:
            cut_h9b.discard(h9b_pair)

        # H9C: favourite + different-line "3rd-place only" bias.
        # Anchor = strongest rider by total top3 inclusion.
        # Opponent = rider from a DIFFERENT line with the highest positive
        # third-bias score P3-max(P1,P2). Cut if pair survives H4.
        h9c_anchor=max(inc,key=lambda n:(inc[n],-n),default=None)
        h9c_third=None
        h9c_pair=None
        if h9c_anchor is not None:
            anchor_line=line_of.get(h9c_anchor,0)
            candidates=[
                n for n in third_bias
                if n!=h9c_anchor
                and line_of.get(n,0)!=anchor_line
                and third_bias[n]>0
            ]
            if candidates:
                h9c_third=max(candidates,key=lambda n:(third_bias[n],p3m[n],-n))
                h9c_pair=tuple(sorted((h9c_anchor,h9c_third)))
        cut_h9c=set(cut_h4)
        h9c_candidate=bool(h9c_pair in cut_h4) if h9c_pair else False
        if h9c_candidate:
            cut_h9c.discard(h9c_pair)

        # New adopted H5 = H4 + H9C ("favourite + other-line third-bias") cut.
        adopted_h5=set(cut_h4)
        if h9c_candidate:
            adopted_h5.discard(h9c_pair)

        # H6 candidate: "explanation ease" bias.
        # For every pair still alive after adopted H5, count how many simple,
        # human-friendly reasons can be attached to buying it. Cut exactly the
        # highest-scoring pair. No odds/popularity/result fields are used.
        #
        # Explanation-ease components (1 point each):
        #  1 same line
        #  2 contains a top-3 rider by total top3 inclusion probability
        #  3 contains a rider on board row 1
        #  4 contains a rider on board row 3 with positive third-bias
        #  5 contains a top-3 rider by race score
        #  6 same-line and adjacent line positions
        #  7 both riders appear in 2+ board rows
        #  8 clean role split: one is P1-dominant, the other P3-dominant
        top3_inc=set(sorted(inc,key=lambda n:(-inc[n],n))[:3])
        score_rank=sorted(
            [int(float(e["car_no"])) for e in race.entries],
            key=lambda n:(-float(entries_by_no[n].get("score") or 0),n),
        )
        top3_score=set(score_rank[:3])
        row1=set(board[0]); row3=set(board[2])
        board_row_count={
            n:sum(n in set(row) for row in board)
            for n in range(1,10)
        }
        line_pos={
            int(float(e["car_no"])):int(float(e.get("line_position") or 0))
            for e in race.entries
        }

        def p1_dominant(n):
            return p1m[n] > max(p2m[n],p3m[n])

        def p3_dominant(n):
            return p3m[n] > max(p1m[n],p2m[n])

        def explanation_features(pair):
            a,b=pair
            same_line=(
                line_of.get(a,0)>0
                and line_of.get(a)==line_of.get(b)
            )
            adjacent=(
                same_line
                and abs(line_pos.get(a,0)-line_pos.get(b,0))==1
            )
            role_split=(
                (p1_dominant(a) and p3_dominant(b))
                or (p1_dominant(b) and p3_dominant(a))
            )
            feats={
                "same_line": same_line,
                "contains_top3_inclusion": a in top3_inc or b in top3_inc,
                "contains_row1": a in row1 or b in row1,
                "contains_row3_positive_third_bias": (
                    (a in row3 and third_bias[a]>0)
                    or (b in row3 and third_bias[b]>0)
                ),
                "contains_top3_score": a in top3_score or b in top3_score,
                "adjacent_line_positions": adjacent,
                "both_multirow": board_row_count.get(a,0)>=2 and board_row_count.get(b,0)>=2,
                "clean_p1_p3_role_split": role_split,
            }
            return feats

        h6e_features={p:explanation_features(p) for p in adopted_h5}
        h6e_scores={p:sum(bool(v) for v in fs.values()) for p,fs in h6e_features.items()}
        h6e_pair=max(
            adopted_h5,
            key=lambda p:(h6e_scores[p],pair_probs.get(p,0.0),-p[0],-p[1]),
            default=None,
        )
        cut_h6e=set(adopted_h5)
        if h6e_pair is not None:
            cut_h6e.discard(h6e_pair)

        # H6: redundancy peeling from the H4 state.
        # For each surviving wide pair, compute how much joint504 probability
        # mass would become completely uncovered if that single pair were removed.
        # Remove the pair with the smallest unique coverage mass.
        def uncovered_loss(pair, tickets):
            loss=0.0
            for a,b,c,prob in r["joint"]:
                tri_pairs={
                    tuple(sorted((a,b))),
                    tuple(sorted((a,c))),
                    tuple(sorted((b,c))),
                }
                if pair not in tri_pairs:
                    continue
                others=(tri_pairs & tickets) - {pair}
                if not others:
                    loss += prob
            return loss

        h6_losses={p:uncovered_loss(p,cut_h4) for p in cut_h4}
        h6_pair=min(
            cut_h4,
            key=lambda p:(h6_losses[p], -pair_probs.get(p,0.0), p[0], p[1]),
            default=None,
        )
        cut_h6=set(cut_h4)
        if h6_pair is not None:
            cut_h6.discard(h6_pair)

        def pay(ts):
            hp=sorted(set(ts)&set(paid))
            return sum(paid[p] for p in hp),hp
        bp,bh=pay(base); cp,ch=pay(cut); xp,xh=pay(cut_both); hp,hh=pay(cut_h3); qp,qh=pay(cut_h4); p5p,p5h=pay(provisional_h5); vp,vh=pay(cut_h5); rp,rh=pay(cut_h6); sp,sh=pay(cut_h7); tp,th=pay(cut_h8); a9p,a9h=pay(cut_h9a); b9p,b9h=pay(cut_h9b); c9p,c9h=pay(cut_h9c); ah5p,ah5h=pay(adopted_h5); e6p,e6h=pay(cut_h6e)
        rows.append({
            "race_id":race.race_id,"date":race.race_date,"race_type":race.race_type,
            "result":list(race.order),"board":[list(x) for x in board],
            "strongest_two":[ranked[0],ranked[1]],
            "strongest_two_top3_prob":[inc[ranked[0]],inc[ranked[1]]],
            "cut_pair":list(strong_pair),"cut_pair_was_candidate":strong_pair in base,
            "cut_pair_was_winner":strong_pair in paid,
            "cut_pair_payout_yen":paid.get(strong_pair,0),
            "strongest_line_pair":list(strong_line_pair) if strong_line_pair else None,
            "strongest_line_pair_prob":pair_probs.get(strong_line_pair,0.0) if strong_line_pair else None,
            "strongest_line_pair_was_winner":bool(strong_line_pair in paid) if strong_line_pair else False,
            "strongest_line_pair_payout_yen":paid.get(strong_line_pair,0) if strong_line_pair else 0,
            "base_tickets":len(base),"base_payout":bp,"base_hits":[list(x) for x in bh],
            "cut_tickets":len(cut),"cut_payout":cp,"cut_hits":[list(x) for x in ch],
            "both_tickets":len(cut_both),"both_payout":xp,"both_hits":[list(x) for x in xh],
            "h3_pair":list(h3_pair) if h3_pair else None,
            "h3_pair_prob":pair_probs.get(h3_pair,0.0) if h3_pair else None,
            "h3_pair_was_winner":bool(h3_pair in paid) if h3_pair else False,
            "h3_pair_payout_yen":paid.get(h3_pair,0) if h3_pair else 0,
            "h3_tickets":len(cut_h3),"h3_payout":hp,"h3_hits":[list(x) for x in hh],
            "h4_pair":list(h4_pair) if h4_pair else None,
            "h4_pair_prob":pair_probs.get(h4_pair,0.0) if h4_pair else None,
            "h4_pair_was_winner":bool(h4_pair in paid) if h4_pair else False,
            "h4_pair_payout_yen":paid.get(h4_pair,0) if h4_pair else 0,
            "h4_tickets":len(cut_h4),"h4_payout":qp,"h4_hits":[list(x) for x in qh],
            "provisional_h5_scenario1":list(h5_s1) if h5_s1 else None,
            "provisional_h5_scenario2":list(h5_s2) if h5_s2 else None,
            "provisional_h5_scenario1_mass":h5_s1_mass,
            "provisional_h5_scenario2_mass":h5_s2_mass,
            "provisional_h5_joint_mass_sum":(h5_s1_mass+h5_s2_mass) if h5_s1_mass is not None else None,
            "provisional_h5_tickets_list":[list(x) for x in sorted(provisional_h5)],
            "provisional_h5_tickets":len(provisional_h5),
            "provisional_h5_payout":p5p,
            "provisional_h5_hits":[list(x) for x in p5h],
            "h5_pair":list(h5_pair) if h5_pair else None,
            "h5_pair_prob":pair_probs.get(h5_pair,0.0) if h5_pair else None,
            "h5_pair_was_winner":bool(h5_pair in paid) if h5_pair else False,
            "h5_pair_payout_yen":paid.get(h5_pair,0) if h5_pair else 0,
            "h5_tickets":len(cut_h5),"h5_payout":vp,"h5_hits":[list(x) for x in vh],
            "h6_pair":list(h6_pair) if h6_pair else None,
            "h6_unique_mass_loss":h6_losses.get(h6_pair) if h6_pair else None,
            "h6_pair_prob":pair_probs.get(h6_pair,0.0) if h6_pair else None,
            "h6_pair_was_winner":bool(h6_pair in paid) if h6_pair else False,
            "h6_pair_payout_yen":paid.get(h6_pair,0) if h6_pair else 0,
            "h6_tickets":len(cut_h6),"h6_payout":rp,"h6_hits":[list(x) for x in rh],
            "h7_line_id":h7_line_id,
            "h7_leader":int(float(h7_leader.get("car_no") or 0)) if h7_leader else None,
            "h7_leader_b_count":float(h7_leader.get("b_count") or 0) if h7_leader else None,
            "h7_leader_nige_count":float(h7_leader.get("nige_count") or 0) if h7_leader else None,
            "h7_line_size":float(h7_leader.get("line_size") or 0) if h7_leader else None,
            "h7_leader_score":float(h7_leader.get("score") or 0) if h7_leader else None,
            "h7_pair":list(h7_pair) if h7_pair else None,
            "h7_pair_was_candidate":h7_pair_was_candidate,
            "h7_pair_was_winner":bool(h7_pair in paid) if h7_pair else False,
            "h7_pair_payout_yen":paid.get(h7_pair,0) if h7_pair else 0,
            "h7_tickets":len(cut_h7),"h7_payout":sp,"h7_hits":[list(x) for x in sh],
            "h8_third_rider":h8_third_rider,
            "h8_third_bias":third_bias.get(h8_third_rider) if h8_third_rider is not None else None,
            "h8_p1":p1m.get(h8_third_rider) if h8_third_rider is not None else None,
            "h8_p2":p2m.get(h8_third_rider) if h8_third_rider is not None else None,
            "h8_p3":p3m.get(h8_third_rider) if h8_third_rider is not None else None,
            "h8_anchor":h8_anchor,
            "h8_anchor_top3_prob":inc.get(h8_anchor) if h8_anchor is not None else None,
            "h8_pair":list(h8_pair) if h8_pair else None,
            "h8_pair_was_candidate":h8_pair_was_candidate,
            "h8_pair_was_winner":bool(h8_pair in paid) if h8_pair else False,
            "h8_pair_payout_yen":paid.get(h8_pair,0) if h8_pair else 0,
            "h8_tickets":len(cut_h8),"h8_payout":tp,"h8_hits":[list(x) for x in th],
            "h9a_third_rider":h9a_third,"h9a_anchor":h9a_anchor,
            "h9a_pair":list(h9a_pair) if h9a_pair else None,
            "h9a_pair_was_candidate":h9a_candidate,
            "h9a_pair_was_winner":bool(h9a_pair in paid) if h9a_pair else False,
            "h9a_pair_payout_yen":paid.get(h9a_pair,0) if h9a_pair else 0,
            "h9a_tickets":len(cut_h9a),"h9a_payout":a9p,"h9a_hits":[list(x) for x in a9h],
            "h9b_third_rider":h9b_third,"h9b_anchor":h9b_anchor,
            "h9b_pair":list(h9b_pair) if h9b_pair else None,
            "h9b_pair_was_candidate":h9b_candidate,
            "h9b_pair_was_winner":bool(h9b_pair in paid) if h9b_pair else False,
            "h9b_pair_payout_yen":paid.get(h9b_pair,0) if h9b_pair else 0,
            "h9b_tickets":len(cut_h9b),"h9b_payout":b9p,"h9b_hits":[list(x) for x in b9h],
            "h9c_third_rider":h9c_third,"h9c_anchor":h9c_anchor,
            "h9c_pair":list(h9c_pair) if h9c_pair else None,
            "h9c_pair_was_candidate":h9c_candidate,
            "h9c_pair_was_winner":bool(h9c_pair in paid) if h9c_pair else False,
            "h9c_pair_payout_yen":paid.get(h9c_pair,0) if h9c_pair else 0,
            "h9c_tickets":len(cut_h9c),"h9c_payout":c9p,"h9c_hits":[list(x) for x in c9h],
            "adopted_h5_tickets":len(adopted_h5),"adopted_h5_payout":ah5p,"adopted_h5_hits":[list(x) for x in ah5h],
            "h6_explanation_pair":list(h6e_pair) if h6e_pair else None,
            "h6_explanation_score":h6e_scores.get(h6e_pair) if h6e_pair else None,
            "h6_explanation_features":h6e_features.get(h6e_pair) if h6e_pair else None,
            "h6_explanation_pair_prob":pair_probs.get(h6e_pair,0.0) if h6e_pair else None,
            "h6_explanation_pair_was_winner":bool(h6e_pair in paid) if h6e_pair else False,
            "h6_explanation_pair_payout_yen":paid.get(h6e_pair,0) if h6e_pair else 0,
            "h6e_tickets":len(cut_h6e),"h6e_payout":e6p,"h6e_hits":[list(x) for x in e6h],
        })
    report={
        "study":"2024 Kyodo days1-2 board wide minus strongest-two pair",
        "dates":[START,END],
        "definition":{
            "h_order":"H1 -> H2 -> H3 -> H4 -> provisional H5; removed pairs never re-enter",
            "provisional_h5":"From H4 survivors only: choose two unordered 3-rider joint504 scenarios sharing two riders, maximizing their pre-race joint-mass sum; buy the union of their internal wide pairs (exactly 5 when available).",
            "board":"v3.2 forward board, calibrated only on prior data",
            "base_tickets":"all unique wide pairs among riders appearing anywhere on the 7-piece board",
            "strongest_two":"top two riders by model top3 inclusion probability = P1+P2+P3 from joint504",
            "cut":"remove exactly that strongest-two wide pair if it exists in base tickets",
            "stake":"100 yen per remaining pair","odds_used":False,"popularity_used":False,
        },
        "calibration_rows":cal_n,
        "baseline":summarize(rows,"base"),
        "strong2_cut":summarize(rows,"cut"),
        "strong2_plus_strongline_cut":summarize(rows,"both"),
        "h3_highest_remaining_pair_cut":summarize(rows,"h3"),
        "h4_second_remaining_pair_cut":summarize(rows,"h4"),
        "provisional_h5_two_overlapping_scenarios":summarize(rows,"provisional_h5"),
        "h5_third_remaining_pair_cut":summarize(rows,"h5"),
        "h6_redundancy_cut_from_h4":summarize(rows,"h6"),
        "h7_expected_lead_line_front_pair_cut_from_h4":summarize(rows,"h7"),
        "h8_third_place_bias_cut_from_h4":summarize(rows,"h8"),
        "h9a_line_third_position_cut_from_h4":summarize(rows,"h9a"),
        "h9b_mark_rider_third_bias_cut_from_h4":summarize(rows,"h9b"),
        "h9c_favourite_plus_otherline_third_bias_cut_from_h4":summarize(rows,"h9c"),
        "adopted_h5":summarize(rows,"adopted_h5"),
        "h6_explanation_ease_cut_from_adopted_h5":summarize(rows,"h6e"),
        "provisional_h5_effect":{
            "candidate_races":sum(r["provisional_h5_tickets"]==5 for r in rows),
            "no_valid_two_scenario_races":sum(r["provisional_h5_tickets"]!=5 for r in rows),
            "winning_races":sum(r["provisional_h5_payout"]>0 for r in rows),
            "total_selected_tickets":sum(r["provisional_h5_tickets"] for r in rows),
            "h4_to_h5_removed_tickets":sum(r["h4_tickets"]-r["provisional_h5_tickets"] for r in rows),
        },
        "cut_effect":{
            "strong2_candidate_cut_races":sum(r["cut_pair_was_candidate"] for r in rows),
            "strong2_winning_cut_pair_races":sum(r["cut_pair_was_winner"] for r in rows),
            "strong2_removed_stake_yen":sum(r["base_tickets"]-r["cut_tickets"] for r in rows)*100,
            "strong2_removed_winning_payout_yen":sum(r["cut_pair_payout_yen"] for r in rows if r["cut_pair_was_candidate"] and r["cut_pair_was_winner"]),
            "strongline_cut_races":sum(r["strongest_line_pair"] is not None and r["strongest_line_pair"]!=r["cut_pair"] for r in rows),
            "strongline_winning_cut_pair_races":sum(r["strongest_line_pair_was_winner"] and r["strongest_line_pair"]!=r["cut_pair"] for r in rows),
            "both_removed_stake_yen":sum(r["base_tickets"]-r["both_tickets"] for r in rows)*100,
            "both_removed_winning_payout_yen":sum(
                r["base_payout"]-r["both_payout"] for r in rows
            ),
            "h3_cut_races":sum(r["h3_pair"] is not None for r in rows),
            "h3_winning_cut_pair_races":sum(r["h3_pair_was_winner"] for r in rows),
            "h3_removed_stake_yen":sum(r["both_tickets"]-r["h3_tickets"] for r in rows)*100,
            "h3_removed_winning_payout_yen":sum(r["both_payout"]-r["h3_payout"] for r in rows),
            "h4_cut_races":sum(r["h4_pair"] is not None for r in rows),
            "h4_winning_cut_pair_races":sum(r["h4_pair_was_winner"] for r in rows),
            "h4_removed_stake_yen":sum(r["h3_tickets"]-r["h4_tickets"] for r in rows)*100,
            "h4_removed_winning_payout_yen":sum(r["h3_payout"]-r["h4_payout"] for r in rows),
            "h5_cut_races":sum(r["h5_pair"] is not None for r in rows),
            "h5_winning_cut_pair_races":sum(r["h5_pair_was_winner"] for r in rows),
            "h5_removed_stake_yen":sum(r["h4_tickets"]-r["h5_tickets"] for r in rows)*100,
            "h5_removed_winning_payout_yen":sum(r["h4_payout"]-r["h5_payout"] for r in rows),
            "h6_cut_races":sum(r["h6_pair"] is not None for r in rows),
            "h6_winning_cut_pair_races":sum(r["h6_pair_was_winner"] for r in rows),
            "h6_removed_stake_yen":sum(r["h4_tickets"]-r["h6_tickets"] for r in rows)*100,
            "h6_removed_winning_payout_yen":sum(r["h4_payout"]-r["h6_payout"] for r in rows),
            "h7_candidate_cut_races":sum(r["h7_pair_was_candidate"] for r in rows),
            "h7_winning_cut_pair_races":sum(r["h7_pair_was_candidate"] and r["h7_pair_was_winner"] for r in rows),
            "h7_removed_stake_yen":sum(r["h4_tickets"]-r["h7_tickets"] for r in rows)*100,
            "h7_removed_winning_payout_yen":sum(r["h4_payout"]-r["h7_payout"] for r in rows),
            "h8_candidate_cut_races":sum(r["h8_pair_was_candidate"] for r in rows),
            "h8_winning_cut_pair_races":sum(r["h8_pair_was_candidate"] and r["h8_pair_was_winner"] for r in rows),
            "h8_removed_stake_yen":sum(r["h4_tickets"]-r["h8_tickets"] for r in rows)*100,
            "h8_removed_winning_payout_yen":sum(r["h4_payout"]-r["h8_payout"] for r in rows),
            "h9a_candidate_cut_races":sum(r["h9a_pair_was_candidate"] for r in rows),
            "h9a_winning_cut_pair_races":sum(r["h9a_pair_was_candidate"] and r["h9a_pair_was_winner"] for r in rows),
            "h9a_removed_stake_yen":sum(r["h4_tickets"]-r["h9a_tickets"] for r in rows)*100,
            "h9a_removed_winning_payout_yen":sum(r["h4_payout"]-r["h9a_payout"] for r in rows),
            "h9b_candidate_cut_races":sum(r["h9b_pair_was_candidate"] for r in rows),
            "h9b_winning_cut_pair_races":sum(r["h9b_pair_was_candidate"] and r["h9b_pair_was_winner"] for r in rows),
            "h9b_removed_stake_yen":sum(r["h4_tickets"]-r["h9b_tickets"] for r in rows)*100,
            "h9b_removed_winning_payout_yen":sum(r["h4_payout"]-r["h9b_payout"] for r in rows),
            "h9c_candidate_cut_races":sum(r["h9c_pair_was_candidate"] for r in rows),
            "h9c_winning_cut_pair_races":sum(r["h9c_pair_was_candidate"] and r["h9c_pair_was_winner"] for r in rows),
            "h9c_removed_stake_yen":sum(r["h4_tickets"]-r["h9c_tickets"] for r in rows)*100,
            "h9c_removed_winning_payout_yen":sum(r["h4_payout"]-r["h9c_payout"] for r in rows),
            "h6e_cut_races":sum(r["adopted_h5_tickets"]>r["h6e_tickets"] for r in rows),
            "h6e_winning_cut_pair_races":sum(r["h6_explanation_pair_was_winner"] for r in rows),
            "h6e_removed_stake_yen":sum(r["adopted_h5_tickets"]-r["h6e_tickets"] for r in rows)*100,
            "h6e_removed_winning_payout_yen":sum(r["adopted_h5_payout"]-r["h6e_payout"] for r in rows),
        },
        "races":rows,
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("baseline","strong2_cut","strong2_plus_strongline_cut","h3_highest_remaining_pair_cut","h4_second_remaining_pair_cut","provisional_h5_two_overlapping_scenarios","provisional_h5_effect")},ensure_ascii=False,indent=2))

if __name__=="__main__": main()
