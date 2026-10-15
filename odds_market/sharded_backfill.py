"""Non-overlapping parallel collection windows, independent per-shard cursors.

All output here is progress metadata. Only seal_verified_odds stores odds,
and only after authenticated encryption.
"""
import argparse
import json
from datetime import date, timedelta
from pathlib import Path

from . import backfill_cursor as core

SHARD_START = date(2021, 1, 3)
SHARD_END = date(2026, 6, 30)
SHARD_COUNT = 8
SHARD_DIR = Path("data/audits/odds_seven_bet_shards")


def bounds(shard):
    if not 0 <= shard < SHARD_COUNT:
        raise ValueError("Invalid shard")
    total_days = (SHARD_END - SHARD_START).days + 1
    size, extra = divmod(total_days, SHARD_COUNT)
    offset = shard * size + min(shard, extra)
    length = size + (1 if shard < extra else 0)
    begin = SHARD_START + timedelta(days=offset)
    return begin, begin + timedelta(days=length - 1)


def state_path(shard):
    return SHARD_DIR / f"shard_{shard:02d}.json"


def state_for(shard):
    first, last = bounds(shard)
    path = state_path(shard)
    if not path.exists():
        return {
            "schema": 1,
            "next_date": first.isoformat(),
            "zero_success_streak": 0,
            "blocked": False,
            "pending_retries": {},
            "history": [],
        }
    state = core.load(path)
    current = date.fromisoformat(state["next_date"])
    if not first <= current <= last + timedelta(days=1):
        raise ValueError("Shard cursor is outside its non-overlapping range")
    if any(not first <= date.fromisoformat(d) <= last
           for d in state["pending_retries"]):
        raise ValueError("Shard retry date is outside its range")
    return state


def choose(shard, days):
    if not 1 <= days <= 7:
        raise ValueError("Window days must be 1..7")
    state = state_for(shard)
    if state["blocked"] or not core.has_real_world_proof():
        return None
    begin, last = bounds(shard)
    cursor = date.fromisoformat(state["next_date"])
    if cursor <= last:
        return cursor, min(cursor + timedelta(days=days - 1), last), "forward"
    candidates = sorted(
        (info.get("attempts", 0), day)
        for day, info in state["pending_retries"].items()
        if info.get("attempts", 0) < 3
    )
    if candidates:
        d = date.fromisoformat(candidates[0][1])
        return d, d, "retry"
    return None


def finish(shard, start, end, mode, audit, log, run_id):
    bound_start, bound_end = bounds(shard)
    if not bound_start <= start <= end <= bound_end:
        raise ValueError("Collection not within requested shard")
    state = state_for(shard)
    if mode == "forward" and date.fromisoformat(state["next_date"]) != start:
        raise ValueError("Stale cursor: refusing to overwrite another batch")
    state = core.finalize(state, start, end, mode, audit, log, run_id)
    path = state_path(shard)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    print("SHARD", shard, "CURSOR", state["next_date"],
          "BLOCKED", state["blocked"],
          "UNRESOLVED_DATES", len(state["pending_retries"]), flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="action", required=True)
    pick = sub.add_parser("pick")
    pick.add_argument("--shard", type=int, required=True)
    pick.add_argument("--days", type=int, default=4)
    pick.add_argument("--github-output", type=Path)
    done = sub.add_parser("finish")
    done.add_argument("--shard", type=int, required=True)
    done.add_argument("--start", type=date.fromisoformat, required=True)
    done.add_argument("--end", type=date.fromisoformat, required=True)
    done.add_argument("--mode", choices=("forward", "retry"), required=True)
    done.add_argument("--audit", type=Path, required=True)
    done.add_argument("--log", type=Path, required=True)
    done.add_argument("--run-id", required=True)
    args = p.parse_args()
    if args.action == "pick":
        window = choose(args.shard, args.days)
        first, last = bounds(args.shard)
        values = {
            "ready": "true" if window else "false",
            "bounds_start": first.isoformat(),
            "bounds_end": last.isoformat(),
        }
        if window:
            start, end, mode = window
            values.update(start=start.isoformat(), end=end.isoformat(),
                          mode=mode)
        payload = "\n".join(f"{k}={v}" for k, v in values.items()) + "\n"
        if args.github_output:
            with args.github_output.open("a", encoding="utf-8") as out:
                out.write(payload)
        print(payload, end="", flush=True)
    else:
        finish(args.shard, args.start, args.end, args.mode,
               args.audit, args.log, args.run_id)


if __name__ == "__main__":
    main()
