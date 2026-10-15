"""Research data acquisition boundary.

Male keirin markets only; all categories and all fields within the window.
2026 ends at June 30, with no exceptions for individual collectors,
manual retries, archived ZIPs, or aggregation.
Class information is used ONLY to exclude girls, never in probabilities.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date

START = date(2021, 1, 1)
END = date(2026, 6, 30)
GIRLS = re.compile(r"ガールズ|女子|(?:^|[^A-Z])L級|ガ予|ガ決|ガ一般|ガ選", re.I)


def in_scope(day):
    """Strictly accept real ISO race dates in 2021-01-01 through 2026-06-30."""
    try:
        d = date.fromisoformat(str(day))
    except (TypeError, ValueError):
        return False
    return START <= d <= END


def girls_race(race_type, rider_classes=()):
    label = unicodedata.normalize("NFKC", str(race_type or "")).strip()
    if GIRLS.search(label):
        return True
    for klass in rider_classes:
        value = unicodedata.normalize("NFKC", str(klass or "")).upper().replace(" ", "")
        if value.startswith("L1") or value.startswith("L2") or value == "L級":
            return True
    return False


def eligible(race, rider_classes=()):
    return in_scope(race.get("race_date")) and not girls_race(race.get("race_type"), rider_classes)


if __name__ == "__main__":
    assert in_scope("2021-01-01")
    assert in_scope("2026-06-30")
    assert not in_scope("2026-07-01")
    assert not in_scope("2026-10-15")
    assert not in_scope("2020-12-31")
    assert not in_scope("2026-02-30")
    assert not eligible({"race_date":"2025-06-15","race_type":"ガールズ予選"},["L1"])
    assert not eligible({"race_date":"2025-06-15","race_type":"A級予選"},["L1"])
    assert eligible({"race_date":"2026-06-30","race_type":"S級決勝"},["S1"])
    print("ODDS_COLLECTION_SCOPE_VALIDATED", START, END)
