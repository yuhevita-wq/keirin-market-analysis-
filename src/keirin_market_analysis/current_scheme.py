from __future__ import annotations

"""Current active scheme pointer.

Restored after rejecting v9.x/v10.x as the active production/research baseline.
Those versions remain in the repository as experiment history only.
"""

from v8_25_f26_final_set_lock_only import build_v8_25_f26

CURRENT_SCHEME_VERSION = "v8.25-F26"
CURRENT_SCHEME_STATUS = "RESTORED_ACTIVE_BASELINE"
CURRENT_SCHEME_REASON = (
    "Restore the last market-psychology hierarchy baseline. "
    "v9.x and v10.x are retained as rejected/diagnostic experiments and are not current."
)


def build_current_scheme(trio_odds, trifecta_odds, predicted_line_formation: str, race_type: str):
    return build_v8_25_f26(
        trio_odds,
        trifecta_odds,
        predicted_line_formation,
        race_type,
    )
