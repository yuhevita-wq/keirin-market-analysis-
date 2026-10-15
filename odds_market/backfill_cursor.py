"""Resumable date-census cursor for authorized, encrypted seven-bet backfills.

Only metadata (dates, completion counts and failed dates) is stored here.
The odds themselves belong exclusively in encrypted .p7m archives.
"""
import argparse
import json
import os
import re
from datetime import date, timedelta, datetime, timezone
from pathlib import Path

START = date(2021, 1, 1)
LAST = date(2026, 6, 30)
DEFAULT = Path("data/audits/odds_seven_bet_cursor.json")
ARCHIVES = Path("data/audits/odds_seven_bet_runs")
DAY_LOG = re.compile(
    r"^(20\d\d-\d\d-\d\d) complete_races (\d+) discovered_races (\d+) "
    r"girls_skipped (\d+)"
)
FAIL_LOG = re.compile(r"^(20\d\d-\d\d-\d\d) (?:\d{16} INCOMPLETE|INDEX_ERROR) ")


def load(path):
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
    else:
        state = {}
    if int(state.get("schema", 1)) != 1:
        raise ValueError("Unsupported cursor schema")
    state.setdefault("schema", 1)
    state.setdefault("next_date", START.isoformat())
    state.setdefault("zero_success_streak", 0)
    state.setdefault("blocked", False)
    state.setdefault("pending_retries", {})
    state.setdefault("history", [])
    if not START <= date.fromisoformat(state["next_date"]) <= LAST + timedelta(days=1):
        raise ValueError("Invalid persisted collection cursor")
    return state


def has_real_world_proof():
    for path in ARCHIVES.glob("*.json"):
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("complete_races", 0) > 0:
                return True
        except (ValueError, OSError):
            pass
    return False


def choose(state, days):
    if state["blocked"] or not has_real_world_proof():
        return None
    cursor = date.fromisoformat(state["next_date"])
    if cursor <= LAST:
        return cursor, min(cursor + timedelta(days=days - 1), LAST), "forward"
    pending = sorted((info.get("attempts", 0), day)
                     for day, info in state["pending_retries"].items()
                     if info.get("attempts", 0) < 3)
    if pending:
        retry_day = date.fromisoformat(pending[0][1])
        return retry_day, retry_day, "retry"
    return None


def parse_attempt_log(path, start, end):
    data = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    reports = {}
    errors = set()
    for line in data.splitlines():
        m = DAY_LOG.match(line.strip())
        if m:
            day = m.group(1)
            if start.isoformat() <= day <= end.isoformat():
                complete, total, girls = map(int, m.groups()[1:])
                reports[day] = {"complete": complete, "discovered": total,
                                "girls_excluded": girls}
                if complete < total - girls:
                    errors.add(day)
        f = FAIL_LOG.match(line.strip())
        if f and start.isoformat() <= f.group(1) <= end.isoformat():
            errors.add(f.group(1))
    return reports, sorted(errors)


def finalize(state, start, end, mode, audit_path, log_path, run_id):
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if str(audit["run_id"]) != str(run_id):
        raise ValueError("Immutable archive run ID does not match")
    completed = int(audit["complete_races"])
    reports, errors = parse_attempt_log(log_path, start, end)
    if completed > 0:
        state["zero_success_streak"] = 0
        if mode == "forward":
            state["next_date"] = (end + timedelta(days=1)).isoformat()
    else:
        state["zero_success_streak"] += 1
    if state["zero_success_streak"] >= 3:
        state["blocked"] = True  # stop repeated bad fetches; require code repair
    pending = state["pending_retries"]
    if mode == "retry":
        key = start.isoformat()
        if key in errors or completed == 0:
            pending.setdefault(key, {"attempts": 0})
            pending[key]["attempts"] += 1
        else:
            pending.pop(key, None)
    for day in errors:
        pending.setdefault(day, {"attempts": 0})
    state["history"].append({
        "run_id": str(run_id), "start": start.isoformat(), "end": end.isoformat(),
        "mode": mode, "complete_races": completed,
        "odds_rows": int(audit["quote_rows"]),
        "date_reports": reports, "incomplete_dates": errors,
        "audited_utc": datetime.now(timezone.utc).isoformat()
    })
    # Store only a compact trail; per-run immutable summaries are separate.
    state["history"] = state["history"][-40:]
    return state


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("pick")
    select.add_argument("--days", type=int, default=2)
    select.add_argument("--state", type=Path, default=DEFAULT)
    select.add_argument("--github-output", type=Path, default=None)
    finish = sub.add_parser("finish")
    finish.add_argument("--state", type=Path, default=DEFAULT)
    finish.add_argument("--start", type=date.fromisoformat, required=True)
    finish.add_argument("--end", type=date.fromisoformat, required=True)
    finish.add_argument("--mode", choices=["forward", "retry"], required=True)
    finish.add_argument("--audit", type=Path, required=True)
    finish.add_argument("--log", type=Path, required=True)
    finish.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.command == "pick":
        if not 1 <= args.days <= 7:
            parser.error("days must be within 1..7")
        selection = choose(load(args.state), args.days)
        values = {"ready": "true" if selection else "false"}
        if selection:
            start, end, mode = selection
            values.update({"start": start.isoformat(),
                           "end": end.isoformat(), "mode": mode})
        lines = [f"{k}={v}" for k,v in values.items()]
        if args.github_output is not None:
            args.github_output.open("a", encoding="utf-8").write("\n".join(lines)+"\n")
        print("\n".join(lines))
    else:
        if not START <= args.start <= args.end <= LAST:
            parser.error("Date window is outside the confirmed scope")
        state = load(args.state)
        state = finalize(state, args.start, args.end, args.mode,
                         args.audit, args.log, args.run_id)
        args.state.parent.mkdir(parents=True, exist_ok=True)
        args.state.write_text(json.dumps(state, ensure_ascii=False, indent=2)+"\n",
                              encoding="utf-8")
        print("NEXT_WINDOW", state["next_date"], "BLOCKED", state["blocked"],
              "PENDING_RETRIES", len(state["pending_retries"]))


if __name__ == "__main__":
    main()
