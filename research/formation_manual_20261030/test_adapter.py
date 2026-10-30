"""The public adapter must not repeat old all-buy or result leakage."""
from __future__ import annotations
from itertools import permutations
try:from .adapter import generate_preview,authorize_purchase
except ImportError:from adapter import generate_preview,authorize_purchase

def main():
    cars="1234567"
    dist={x:1/210 for x in permutations(cars,3)}
    prices={x:40.0 for x in dist}
    event={
        "scores":{c:80+int(c) for c in cars},
        "positions":{"1":(1,1,3),"2":(1,2,3),"3":(1,3,3),
                     "4":(2,1,2),"5":(2,2,2),"6":(3,1,2),"7":(3,2,2)}
    }
    prev=generate_preview(event,dist,prices)
    assert prev["decision"]=="見送り"
    assert prev["alternatives"],"all candidates should not be filtered"
    assert len(set(c.label for c in prev["alternatives"]))==len(prev["alternatives"])
    assert any(c.points>12 for c in prev["alternatives"])
    assert any(len(c.forms)>1 for c in prev["alternatives"])
    assert any(c.label=="second_fixed" for c in prev["alternatives"])
    assert any(c.label=="third_fixed" for c in prev["alternatives"])
    assert any(c.label=="sandwich" for c in prev["alternatives"])
    assert authorize_purchase(prev)["decision"]=="見送り"
    assert authorize_purchase(prev,external_approval=True)["decision"]=="見送り"
    assert authorize_purchase(prev,external_approval={"approved":True})["decision"]=="見送り"
    assert authorize_purchase(prev,external_approval={
        "approved":True,"candidate_label":"does_not_exist",
        "pre_declared_validation_id":"audit-2026-001"})["decision"]=="見送り"
    target=prev["alternatives"][0]
    adopted=authorize_purchase(prev,external_approval={
        "approved":True,"candidate_label":target.label,
        "pre_declared_validation_id":"external-audit-id-001"})
    assert adopted["decision"]=="採用"
    assert adopted["ticket_points"]==target.points
    assert adopted["ticket_set"]==target.ticket_set
    # Only non-result input may be accepted.
    try:
        generate_preview({**event,"order":("1","2","3")},dist,prices)
    except ValueError:
        pass
    else:raise AssertionError("leak of official result into model decision")
    before=tuple((x.forms,x.points) for x in prev["alternatives"])
    changed_odds=generate_preview(event,dist,{x:80.0 for x in dist})
    # No post-hoc odds-seeded formation generation: same structure types.
    assert set(x.label for x in prev["alternatives"])==set(x.label for x in changed_odds["alternatives"])
    assert all(x.points>0 for x in changed_odds["alternatives"])
    print("MANUAL_ADAPTER_PRICE_AFTER_SHAPE_NO_ALL_BUY_AND_NO_LABEL_LEAK_PASS")
if __name__=="__main__":main()
