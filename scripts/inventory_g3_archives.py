from __future__ import annotations

import csv
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = ROOT / "data" / "grade_races" / "g3"
OUT = ROOT / "results" / "g3_day3_reality"


def decode_text(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp932", "shift_jis"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def inspect_csv(zf: zipfile.ZipFile, name: str) -> dict:
    raw = zf.read(name)
    text = decode_text(raw)
    reader = csv.reader(io.StringIO(text))
    rows = []
    for i, row in enumerate(reader):
        rows.append(row)
        if i >= 3:
            break
    line_count = text.count("\n")
    return {
        "name": name,
        "bytes": len(raw),
        "line_count_approx": line_count,
        "header": rows[0] if rows else [],
        "sample_rows": rows[1:4] if len(rows) > 1 else [],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    archives = sorted(ARCHIVE_ROOT.rglob("*.zip"))
    result = {"archive_count": len(archives), "archives": []}
    all_headers = Counter()
    all_names = Counter()

    for path in archives:
        rec = {"path": str(path.relative_to(ROOT)), "members": [], "csvs": []}
        try:
            with zipfile.ZipFile(path) as zf:
                names = [n for n in zf.namelist() if not n.endswith("/")]
                rec["members"] = names
                for n in names:
                    all_names[Path(n).name] += 1
                    if n.lower().endswith(".csv"):
                        c = inspect_csv(zf, n)
                        rec["csvs"].append(c)
                        all_headers[tuple(c["header"])] += 1
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        result["archives"].append(rec)

    result["member_basename_counts"] = dict(all_names)
    result["distinct_csv_headers"] = [
        {"count": count, "header": list(header)}
        for header, count in all_headers.most_common()
    ]
    (OUT / "inventory.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# G3 archive inventory", "", f"archives: {len(archives)}", "", "## Common member basenames"]
    for name, count in all_names.most_common():
        lines.append(f"- `{name}`: {count}")
    lines += ["", "## CSV schemas"]
    for item in result["distinct_csv_headers"]:
        lines.append(f"- {item['count']} archive/file(s): `{','.join(item['header'])}`")
    lines += ["", "## Archive samples"]
    for rec in result["archives"]:
        lines.append(f"### {rec['path']}")
        if rec.get("error"):
            lines.append(f"ERROR: {rec['error']}")
            continue
        lines.append(f"members: {len(rec['members'])}")
        for c in rec["csvs"]:
            lines.append(f"- `{c['name']}` ~{c['line_count_approx']} lines")
            lines.append(f"  - header: `{','.join(c['header'])}`")
            if c["sample_rows"]:
                lines.append(f"  - sample: `{c['sample_rows'][0]}`")
    (OUT / "inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"archive_count": len(archives), "member_basename_counts": dict(all_names)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
