"""Market quote canonicalization for the specifically audited sealed 7-market archives.

CAUTION: Original encrypted all_seven_bet_odds_YYYY.jsonl.gz.cms stores 3rentan
quotes under (first, THIRD, SECOND) relative to official payout combinations.
This was established with 47,380 independent official payout rows, not guessed:
47,375 exact parity after the 2/3 swap, five payouts above 999,990 yen capped
at 9,999.9 odds in the archive. The 2025 true heldout 6,865/6,865 match.
3renpuku has no such mapping error (47,380/47,380 direct parity).

Only use for this sealed archive schema. Do not rewrite official winning order,
WINTICKET wt_odds, other market sources, or permanent historical raw archives.
"""
from __future__ import annotations
from decimal import Decimal

def sealed_source_key_for_official_combo(ticket_type,official_order):
    cars=tuple(int(x) for x in official_order)
    if len(cars)!=3 or len(set(cars))!=3 or any(c not in range(1,10) for c in cars):
        raise ValueError("expected three distinct valid car numbers")
    if ticket_type in ("3rentan","trifecta"):
        cars=(cars[0],cars[2],cars[1])
    elif ticket_type in ("3renpuku","trio"):
        cars=tuple(sorted(cars))
    else:
        raise ValueError("this helper handles ONLY the two audited three-car markets")
    return "-".join(map(str,cars))

def archived_final_odds_for_official_combo(market,kind,official_order):
    """Return final odds Decimal or None; do not infer official dividend from this alone."""
    key=sealed_source_key_for_official_combo(kind,official_order)
    target={"trifecta":"3rentan","trio":"3renpuku",
            "3rentan":"3rentan","3renpuku":"3renpuku"}[kind]
    matches=[v for v in market.get(target,()) if v[0]==key]
    if len(matches)!=1:return None
    row=matches[0]
    if len(row)<3:return None
    a=Decimal(str(row[1]));b=Decimal(str(row[2]))
    return a if a==b and a>0 else None

def selftest():
    a=(3,2,1)
    assert sealed_source_key_for_official_combo("trifecta",a)=="3-1-2"
    assert sealed_source_key_for_official_combo("trio",a)=="1-2-3"
    market={"3rentan":[["3-1-2","906.7","906.7"],["3-2-1","1202.4","1202.4"]],
            "3renpuku":[["1-2-3","153.5","153.5"]]}
    assert archived_final_odds_for_official_combo(market,"3rentan",a)==Decimal("906.7")
    assert archived_final_odds_for_official_combo(market,"trio",a)==Decimal("153.5")
    print("SEALED_TRIFECTA_OFFICIAL_COMBO_PARITY_SELFTEST_OK")
if __name__=="__main__":selftest()
