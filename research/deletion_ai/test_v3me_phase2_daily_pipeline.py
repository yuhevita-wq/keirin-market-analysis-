"""Phase 2 learning order, confidentiality and paired receipt regression."""
from __future__ import annotations

import copy
import itertools
from collections import defaultdict

import numpy as np

from deletion_core import MARKETS, model_digest, new_weights
from v3me_phase2_daily_pipeline import (
    MODEL, EXACT_SETTINGS, SETTINGS_HASH, _book_pre_result,
    day_predictions, settle_and_fit_day, weights_from_raw, weight_dump,
    chronological_days,
)


def fixture(race_id, day, shift=0, finish=(1,2,3)):
    cars=list(range(1,8))
    ordered=list(itertools.permutations(cars,3))
    triples=list(itertools.combinations(cars,3))
    t={"-".join(map(str,key)):float(2+sum(key)+i%83) for i,key in enumerate(ordered)}
    u={"=".join(map(str,key)):float(2+sum(key)+i%57) for i,key in enumerate(triples)}
    return {
        "race_id":race_id, "race":{"race_date":day},
        "entries":{c:{} for c in cars},
        "odds":{"3連単":t,"3連複":u},
        "outcome":tuple(finish),
    }


def w0():
    w=new_weights()
    return {MODEL:{m:w[MODEL][m].copy() for m in MARKETS}}


def containers():
    return (
        defaultdict(lambda:defaultdict(float)),
        defaultdict(lambda:defaultdict(lambda:defaultdict(float))),
        defaultdict(lambda:defaultdict(float)),
        defaultdict(lambda:defaultdict(float)),
    )


def check():
    assert SETTINGS_HASH and EXACT_SETTINGS["probability_coverage"]==.25
    a=fixture("7320230112010001","2023-01-12",finish=(1,2,3))
    b=fixture("7420230112010001","2023-01-12",finish=(7,5,4))
    c=fixture("7520230113010001","2023-01-13",finish=(1,5,2))
    originals=[a,b]
    weights=w0()
    opening=model_digest(weights,MODEL)
    decisions, snaps=day_predictions(originals,weights)
    assert len(decisions)==2 and snaps["phase2"]==opening
    # Labels may change, yet all pre-result decisions and SHA receipts must
    # remain EXACTLY unchanged. Exclude labels from feature construction.
    corrupted=copy.deepcopy(originals)
    corrupted[0]["outcome"]=(7,6,5)
    corrupted[1]["outcome"]=(3,1,2)
    again, snaps2=day_predictions(list(reversed(corrupted)),weights)
    assert snaps==snaps2
    assert [x[3] for x in decisions]==[x[3] for x in again]
    assert model_digest(weights,MODEL)==opening
    assert all(_book_pre_result(x)["cars"]==tuple(range(1,8)) for x in originals)

    # First full day is sealed before any result updates.
    score,month,loss,control=containers()
    settle_and_fit_day(decisions, weights, True, score, loss, control, month)
    trained=model_digest(weights,MODEL)
    assert trained!=opening
    assert score["phase2"]["races"]==2
    assert score["full_sheet"]["races"]==2
    assert all(control["phase2"][k]>0 for k in ("same_stake_yen",))
    assert set(month)=={"2023-01"}
    assert all(np.isfinite(weights[MODEL][m]).all() for m in MARKETS)
    # Resuming from serialized end-of-year weights must reproduce exact hash.
    restored=weights_from_raw(weight_dump(weights))
    assert model_digest(restored,MODEL)==trained
    after_day,_=day_predictions([c],restored)
    after_day2,_=day_predictions([c],weights)
    assert [x[3] for x in after_day]==[x[3] for x in after_day2]

    # 2026 is frozen; settle with 'training=False' must not mutate weights.
    a2026=fixture("7320260601010001","2026-06-01",finish=(4,3,2))
    before=model_digest(restored,MODEL)
    sealed,_=day_predictions([a2026],restored)
    another=containers()
    settle_and_fit_day(sealed,restored,False,another[0],another[2],another[3],another[1])
    assert model_digest(restored,MODEL)==before

    # An invalid mixed day must be rejected instead of quietly interleaving.
    try:
        day_predictions([a,c],w0())
    except RuntimeError as ex:
        assert "CROSS_DAY" in str(ex)
    else:
        raise AssertionError("Mixed days accepted")
    print("V3ME_PHASE2_DAY_FROZEN_NO_LEAK_AND_RESUME_TEST_PASS")


if __name__=="__main__":
    check()
