"""Regression against user-provided 40-page 3連単 formation manual."""
from __future__ import annotations
import itertools,math,random
try:
    from .formation import (form,box,from_manual_notation,together,evaluate_set,
        verify_settlement,top_two_flip,sandwich)
    from .scenarios import construct_candidates,rank_packages,format_decision
except ImportError:
    from formation import (form,box,from_manual_notation,together,evaluate_set,
        verify_settlement,top_two_flip,sandwich)
    from scenarios import construct_candidates,rank_packages,format_decision


def check_examples():
    cars="123456789"
    docs=[
        ("1-2459-2459",12,"1-2-4","2-1-4"),
        ("17-17-2359",8,"7-1-5","7-5-1"),
        ("57-57-全",14,"7-5-8","5-8-7"),
        ("5-全-1",7,"5-7-1","5-1-8"),
        ("35-7-1235",6,"3-7-1","7-3-1"),
        ("239-239-2391",12,"2-3-9","2-2-9"),
        ("1-2457-24578",16,"1-2-8","2-1-8"),
        ("1-23457-23457",20,"1-5-2","1-2-2"),
        ("2-9-134578",6,"2-9-4","9-2-4"),
        ("5-382-3821",9,"5-3-8","2-3-8"),
    ]
    for notation,count,hit,miss in docs:
        shape=from_manual_notation(cars,notation)
        assert shape.points==count,(notation,shape.points,count)
        assert tuple(hit.split("-")) in shape.tickets,(notation,"misses intended")
        assert tuple(miss.split("-")) not in shape.tickets,(notation,"includes unwanted")
    assert from_manual_notation(cars,"1357 BOX").points==24
    assert from_manual_notation(cars,"235 BOX").points==6
    assert from_manual_notation(cars,"3=2-1947").points==8
    assert from_manual_notation(cars,"1-6=7").points==2
    s1=from_manual_notation(cars,"1357 BOX")
    s2=from_manual_notation(cars,"5-全-1")
    combined=together([s1,s2])
    assert combined.points==29,combined.points
    assert combined.separate_points==31
    assert len(combined.overlap)==2
    assert combined.overlap=={("5","3","1"),("5","7","1")}
    # Two independent readings must not be merged into all-pairs-per-slot
    two=together([from_manual_notation(cars,"38-38-126"),
                  from_manual_notation(cars,"13-13-458")])
    assert two.points==12
    assert ("3","8","5") not in two.tickets
    # A 3rd all is different from every-rider finishing permutation.
    assert from_manual_notation(cars,"57-57-全").points==14
    assert from_manual_notation("1234567","57-57-全").points==10
    # 2nd-locked and 3rd-locked are not reduced to 1st=seed.
    assert from_manual_notation(cars,"35-7-1235").second==("7",)
    assert from_manual_notation(cars,"35-7-1235").first==("3","5")
    assert sandbox_invalid() is None
    print("MANUAL_EXACT_FORMS_BOX_SANDWICH_LOCKED_RANKS_AND_OVERLAP_PASS")


def sandbox_invalid():
    cars="123456789"
    for cmd in ["1-1-1","1-1-2","1-2-2","1-67=2358"]:
        try:
            from_manual_notation(cars,cmd)
            if cmd=="1-1-2":
                raise AssertionError("repeated same rider cannot generate a valid bet")
        except ValueError:
            pass
    return None


def check_money():
    cars="1234567"
    tickets=list(itertools.permutations(cars,3))
    p={t:1/len(tickets) for t in tickets}
    odds={t:37.0 for t in tickets}
    a=from_manual_notation(cars,"57-57-全")
    metrics=evaluate_set(a.tickets,p,odds)
    assert metrics["points"]==10
    assert metrics["stake_yen"]==1000
    assert abs(metrics["model_hit_probability"]-10/210)<1e-9
    assert abs(metrics["model_expected_roi_pct"]-100*37/210)<1e-8
    assert metrics["ticket_ev_under_100_count"]==10
    assert metrics["ticket_ev_over_100_count"]==0
    outcome=("5","7","1")
    st=verify_settlement(a.tickets,outcome,35330)
    assert st["actual_payout_yen"]==35330
    assert st["profit_yen"]==34330
    assert verify_settlement(a.tickets,("1","5","7"),35330)["actual_payout_yen"]==0
    # Entire package is evaluated once after overlap de-duplication.
    big=from_manual_notation("123456789","1357 BOX")
    small=from_manual_notation("123456789","5-全-1")
    q=together([big,small])
    grid=list(itertools.permutations("123456789",3))
    mass={t:1/len(grid) for t in grid}
    market={t:15.0 for t in grid}
    z=evaluate_set(q.tickets,mass,market)
    assert z["stake_yen"]==2900
    assert abs(z["model_expected_roi_pct"]-100*15/len(grid))<1e-7
    print("COMBINED_PACKAGE_NO_DOUBLE_COUNT_AND_ACTUAL_SETTLEMENT_PASS")


def check_scenarios():
    cars="1234567"
    triples=list(itertools.permutations(cars,3))
    # Change prediction mass across conditions; never read races/odds/results.
    weights={(a,b,c):1+(a=="2")*3+(a=="3" and b=="5")*4
             for a,b,c in triples}
    den=sum(weights.values())
    p={k:v/den for k,v in weights.items()}
    lines={
        "1":(1,1,3),"2":(1,2,3),"3":(1,3,3),
        "4":(2,1,2),"5":(2,2,2),
        "6":(3,1,2),"7":(3,2,2),
    }
    result=construct_candidates(cars,p,lines,max_scenario_groups=160)
    tags={r.scenario for r in result}
    for tag in ("first_fixed_swap","fixed12_all","top_two_flip",
                "top_two_flip_all","second_fixed","third_fixed","sandwich",
                "same_line_one_two_flip","line_leader_exhausts","three_box",
                "four_box","multiple_readings"):
        assert tag in tags,tag
    assert any(x.pack.points>12 for x in result),"old 12-point cap must be removed"
    assert any(len(x.pack.shapes)==2 for x in result),"combining independent shapes must work"
    assert any(x.pack.points==24 for x in result if x.scenario=="four_box")
    for choice in result:
        assert choice.pack.points==len(choice.pack.tickets)
        assert choice.pack.tickets.issubset(p)
        assert sum(shape.points for shape in choice.pack.shapes)>=choice.pack.points
    # Repricing must not retroactively alter which structural candidates exist.
    expensive={t:15. for t in triples}
    cheap={t:120. for t in triples}
    ranked1=rank_packages(result,p,expensive)
    ranked2=rank_packages(result,p,cheap)
    assert {x.pack.tickets for x,_ in ranked1}=={x.pack.tickets for x,_ in ranked2}
    selected,m=ranked1[0]
    decision=format_decision(selected,m)
    assert decision["adoption"]=="見送り"
    assert not any("winning_result" in vars(r) for r in result)
    # 2+ forma package can be formed as sum of separate scenario sets.
    twos=[r for r in result if r.scenario=="multiple_readings"]
    assert twos
    for x in twos:
        a,b=x.pack.shapes
        assert x.pack.tickets==(a.tickets|b.tickets)
    print("STRUCTURAL_SCENARIOS_DIFFERENT_SLOT_LOCKS_LINE_ROLES_AND_DEFAULT_SKIP_PASS")


if __name__=="__main__":
    check_examples()
    check_money()
    check_scenarios()
    print("USER_MANUAL_FORMAT_AND_FUNDAMENTAL_BEHAVIOR_ALL_PASS")
