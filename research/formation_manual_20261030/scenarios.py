"""Result-blind formation candidates: structure first, pricing/decision second.

Manual pp. 5-8, 11-20, 29-35:
  primary line -> which riders survive -> uncertain finish ranks -> formation.
This is a scenario *proposal* engine, not proof a specific line will lead.
Odds, results, payout, and race truth are not accepted as input.
All candidate groupings remain separately labeled; combining two groupings
never makes their Cartesian cross product.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations,permutations
from typing import Mapping

try:
    from .formation import Shape, Package, form, box, top_two_flip, third_all, sandwich, second_fixed, third_fixed, head_fixed, together, normalize_cars
except ImportError:
    from formation import Shape, Package, form, box, top_two_flip, third_all, sandwich, second_fixed, third_fixed, head_fixed, together, normalize_cars

Ticket=tuple[str,str,str]


@dataclass(frozen=True)
class Candidate:
    """A scenario hypothesis, NOT an adopted bet."""
    scenario: str
    explanation: str
    pack: Package


def _sorted_probs(items,probabilities):
    return tuple(sorted(items,key=lambda a:(-probabilities[a],tuple(int(x) if isinstance(x,str) else 0 for x in (a if isinstance(a,tuple) else (a,))))))


def _probability_margins(cars,p):
    first=defaultdict(float)
    second=defaultdict(float)
    third=defaultdict(float)
    twop=defaultdict(float)
    firstthird=defaultdict(float)
    flip=defaultdict(float)
    triple=defaultdict(float)
    for (a,b,c),v in p.items():
        first[a]+=v
        second[b]+=v
        third[c]+=v
        twop[(a,b)]+=v
        firstthird[(a,c)]+=v
        flip[tuple(sorted((a,b),key=int))]+=v
        triple[tuple(sorted((a,b,c),key=int))]+=v
    return first,second,third,twop,firstthird,flip,triple


def _top_third(p,pair,cars):
    a,b=pair
    x={c:p[(a,b,c)]+p[(b,a,c)] for c in cars if c not in pair}
    return _sorted_probs(x,x)


def _head_others(p,head,cars):
    scores=defaultdict(float)
    for (a,b,c),v in p.items():
        if a==head:
            scores[b]+=v
            scores[c]+=v
    return _sorted_probs({x:scores[x] for x in cars if x!=head},scores)


def _follow_others(p,head,second,cars):
    x={c:p[(head,second,c)] for c in cars if c not in (head,second)}
    return _sorted_probs(x,x)


def _middle_others(p,head,third,cars):
    x={b:p[(head,b,third)] for b in cars if b not in (head,third)}
    return _sorted_probs(x,x)


def _dominant_firsts(p,second,cars):
    x={a:sum(p[(a,second,c)] for c in cars if c not in (a,second))
       for a in cars if a!=second}
    return _sorted_probs(x,x)


def _dominant_thirds(p,second,cars):
    x={c:sum(p[(a,second,c)] for a in cars if a not in (second,c))
       for c in cars if c!=second}
    return _sorted_probs(x,x)


def _dominant_first_for_third(p,third,cars):
    x={a:sum(p[(a,b,third)] for b in cars if b not in (a,third))
       for a in cars if a!=third}
    return _sorted_probs(x,x)


def _dominant_second_for_third(p,third,cars):
    x={b:sum(p[(a,b,third)] for a in cars if a not in (b,third))
       for b in cars if b!=third}
    return _sorted_probs(x,x)


def _line_list(cars,positions):
    if not positions:
        return []
    line=defaultdict(list)
    for car in cars:
        if car not in positions:
            raise ValueError(f"missing position for {car}")
        row=positions[car]
        if not (isinstance(row,(tuple,list)) and len(row)>=2):
            raise ValueError("need (line_number,role_position,line_size)")
        key,pos=row[:2]
        line[key].append((int(pos),car))
    for items in line.values():
        if len({rank for rank,_ in items})!=len(items):
            raise ValueError("duplicate line role position")
    return [tuple(c for _,c in sorted(v)) for _,v in sorted(line.items(),key=lambda item:str(item[0]))]


def construct_candidates(cars,probabilities:Mapping[Ticket,float],positions=None,
                         *,max_scenario_groups=160):
    """Propose diverse relations; never select the financially optimal outcome.

    Every valid race ordered combo must be in model probabilities. The selector
    uses ONLY the outcome distribution and pre-race line roles. No market
    price can shape the slots; prices are evaluated later on complete packages.
    """
    import math
    cars=normalize_cars(cars)
    all_t=set(permutations(cars,3))
    if set(probabilities)!=all_t:
        raise ValueError("incomplete 3連単 prediction universe")
    if abs(sum(probabilities.values())-1)>1e-8 or any(not math.isfinite(v) or v<0 for v in probabilities.values()):
        raise ValueError("invalid model probabilities")
    if not isinstance(max_scenario_groups,int) or max_scenario_groups<20:
        raise ValueError("max_scenario_groups must be >= 20")

    first,second,third,twop,firstthird,flip,triple=_probability_margins(cars,probabilities)
    scenarios=[]
    seen=set()

    def append(scenario,explanation,*shapes):
        pack=together(shapes)
        key=pack.tickets
        if key in seen:return
        seen.add(key)
        scenarios.append(Candidate(scenario,explanation,pack))

    # Rank the structural hypotheses by unconditional outcomes before looking
    # at prices. Examples use several alternative counts rather than imposing
    # the old universal 4..12 ticket cap.
    for a in _sorted_probs(first,first)[:3]:
        others=_head_others(probabilities,a,cars)
        for k in (3,4,5):
            k=min(k,len(others))
            if k>=2:
                append("first_fixed_swap",f"1着{a}固定、2・3着は入れ替わりを残す（候補{k}人）",
                       head_fixed(cars,a,others[:k],why="first fixed, shared 2nd/3rd candidates"))

    # Ordered 1-2 pair: fixed order, 3rd candidate set. Independent all-3rd.
    for (a,b) in _sorted_probs(twop,twop)[:5]:
        cs=_follow_others(probabilities,a,b,cars)
        for k in (2,3,4):
            append("fixed12",f"{a}→{b}の1・2着を固定、3着候補{k}人",
                   form(cars,a,b,cs[:min(k,len(cs))],type="fixed_first_second",why="ordered 1-2 conditional mass"))
        append("fixed12_all",f"{a}→{b}固定、3着総流し",
               third_all(cars,a,b,why="ordered pair, unrestricted 3rd"))

    # First and second interchange. Use same-line and cross-line pairs alike,
    # but preserve the actual line-relational rationale separately.
    for pair in _sorted_probs(flip,flip)[:6]:
        a,b=pair
        ordered_thirds=_top_third(probabilities,pair,cars)
        for k in (2,4):
            append("top_two_flip",f"{a}/{b}のワンツー表裏、3着候補{k}人",
                   top_two_flip(cars,pair,ordered_thirds[:min(k,len(ordered_thirds))],
                                why="P(a-b-any)+P(b-a-any)"))
        append("top_two_flip_all",f"{a}/{b}ワンツー表裏、3着総流し",
               top_two_flip(cars,pair,why="pair conditional, all remaining riders"))

    # In some races, keeping the 2nd or the 3rd position fixed is the most
    # rational structural encoding; the former engine could not do this.
    for b in _sorted_probs(second,second)[:3]:
        aa=_dominant_firsts(probabilities,b,cars)
        cc=_dominant_thirds(probabilities,b,cars)
        append("second_fixed",f"2着{b}固定、1着/3着は各候補",
               second_fixed(cars,aa[:min(2,len(aa))],b,cc[:min(4,len(cc))],
                            why="second-position probability mass"))
    for c in _sorted_probs(third,third)[:3]:
        aa=_dominant_first_for_third(probabilities,c,cars)
        bb=_dominant_second_for_third(probabilities,c,cars)
        append("third_fixed",f"3着{c}固定、1着/2着は各候補",
               third_fixed(cars,aa[:min(3,len(aa))],bb[:min(3,len(bb))],c,
                           why="third-position probability mass"))

    # Hasa: 1st and 3rd pinned, middle open; sometimes a lineup partner
    # survives in 3rd while another line inserts at 2nd.
    for (a,c) in _sorted_probs(firstthird,firstthird)[:5]:
        middles=_middle_others(probabilities,a,c,cars)
        append("sandwich",f"{a}1着→{c}3着、2着を広げる",
               sandwich(cars,a,c,middles[:min(4,len(middles))],
                        why="first-and-third conditional relation"))
        append("sandwich_all",f"{a}1着→{c}3着、2着総流し",
               sandwich(cars,a,c,why="first-and-third relation, all intervening"))

    # Favor actual unordered leading-3 sets for BOX, not 'top three race score'.
    for trio in _sorted_probs(triple,triple)[:4]:
        append("three_box",f"{''.join(trio)}の全着順BOX",
               box(cars,trio,why="unordered podium trio joint mass"))
    if len(cars)>=4:
        combos=list(combinations(cars,4))
        mass4={x:sum(v for t,v in probabilities.items() if set(t).issubset(x))
               for x in combos}
        for four in _sorted_probs(mass4,mass4)[:2]:
            append("four_box",f"{''.join(four)}の4車BOX",
                   box(cars,four,why="unordered podium mass for four riders"))

    # A three-person top-1/top-2 relation plus an extra 3rd rider.
    leaders=_sorted_probs(first,first)[:min(3,len(cars))]
    cands=tuple(x for x in _sorted_probs(third,third) if x not in leaders)
    if len(leaders)==3 and cands:
        append("three_head_swap",f"{''.join(leaders)}の1・2着表裏＋3着候補",
               form(cars,leaders,leaders,leaders+cands[:1],
                    type="three_top_two_swap",why="multiple possible winners"))

    # A line role is NOT the finishing order. Explicitly include the leader,
    # bante, third rider cases and their appropriate alternatives.
    line_cases=_line_list(cars,positions)
    for line in line_cases:
        if len(line)<2:continue
        leader,bante=line[:2]
        cs=_top_third(probabilities,(leader,bante),cars)
        append("same_line_one_two_flip",
               f"同ライン先頭{leader}と番手{bante}の表裏、他ライン等が3着",
               top_two_flip(cars,(leader,bante),cs[:min(4,len(cs))],
                            why="line leader/bante may interchange at goal"))
        append("same_line_one_two_all",
               f"同ライン{leader}・{bante}上位を固定し3着総流し",
               top_two_flip(cars,(leader,bante),why="line 1-2 relation"))
        append("same_line_sandwich",
               f"先頭{leader}1着・番手{bante}3着（ハサミ）",
               sandwich(cars,leader,bante,why="other line can split the ordered partners"))
        if len(line)>=3:
            thirdman=line[2]
            cs2=_top_third(probabilities,(bante,thirdman),cars)
            append("line_leader_exhausts",
                   f"先頭{leader}が脚を使い番手{bante}・3番手{thirdman}が上位",
                   top_two_flip(cars,(bante,thirdman),cs2[:min(4,len(cs2))],
                                why="bante and third rider move past leader"))
            append("line_three_mixed",
                   f"{leader}/{bante}/{thirdman}ライン全員の着順を入れ替える",
                   box(cars,line[:3],why="three-rider line, all 6 orders"))

    # Each scenario remains a separate shape. Distinct scenario unions are
    # deliberately created without merging slots (which would add invented
    # tickets). A cost cap is NEVER silently imposed.
    roots=[x for x in scenarios if x.scenario in (
        "top_two_flip","first_fixed_swap","second_fixed",
        "third_fixed","sandwich","same_line_one_two_flip","line_leader_exhausts")]
    for left in roots[:4]:
        for right in roots[4:8]:
            if left.pack.tickets==right.pack.tickets:continue
            append("multiple_readings",
                   f"独立した2つの読みを併用：{left.scenario} + {right.scenario}",
                   *left.pack.shapes,*right.pack.shapes)

    # Resource cap on number of listed alternatives only. NO cap on ticket count.
    return tuple(scenarios[:max_scenario_groups])


def rank_packages(candidates,distribution,full_market_prices,*,stakes=100):
    """Score after structural generation, NEVER declare adoption on EV alone."""
    try:
        from .formation import evaluate_set
    except ImportError:
        from formation import evaluate_set
    out=[]
    for candidate in candidates:
        metrics=evaluate_set(candidate.pack.tickets,distribution,full_market_prices,stakes)
        out.append((candidate,metrics))
    return tuple(sorted(out,key=lambda x:(-x[1]["model_expected_roi_pct"],
                                         -x[1]["model_hit_probability"],
                                         x[1]["points"],x[0].scenario)))


def format_decision(candidate,metrics,*,approved_market_edge=False):
    """Default fail-closed. A large p*odds is not proof of market error.

    The caller must *independently* supply an approved validation criterion.
    This helper does not grant approval or optimize thresholds on outcomes.
    """
    return {
        "adoption":"採用" if approved_market_edge else "見送り",
        "adoption_basis":"external_evidence_required" if not approved_market_edge else "externally_validated_market_edge",
        "scenario":candidate.scenario,
        "explanation":candidate.explanation,
        "forms":[x.notation for x in candidate.pack.shapes],
        "points_after_deduplication":candidate.pack.points,
        "additional_points_if_double_betting_overlaps":len(candidate.pack.overlap),
        "model_metrics":metrics,
    }
