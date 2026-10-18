#!/usr/bin/env python3
"""Audit original G3-only ZIP source layout BEFORE choosing model/training inputs."""
from __future__ import annotations
import collections
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"data/grade_races/g3"
OUT=ROOT/"research/g3_independent_v1/output/source_archive_audit.json"

def main():
    paths=sorted(BASE.rglob("*.zip"))
    if not paths:
        raise RuntimeError("No historical G3 ZIP archives!")
    out={"scope":"G3 only, archive original bytes","zip_count":len(paths),
         "source_repo":"yuhevita-wq/keirin-market-analysis-",
         "research_branch":"research/g3-independent-first-20261018",
         "status":"SOURCE_LAYOUT_DISCOVERY_BEFORE_MODEL_FITTING",
         "archives":[]}
    for path in paths:
        summary={"path":str(path.relative_to(ROOT)),"zip_bytes":path.stat().st_size,
                 "zip_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                 "files":[],"csv_samples":[],"json_samples":[]}
        with zipfile.ZipFile(path) as z:
            summary["files"]=[{"name":f.filename,"size":f.file_size}
                              for f in z.infolist() if not f.is_dir()][:45]
            summary["n_zip_members"]=sum(not f.is_dir() for f in z.infolist())
            for entry in z.infolist():
                if entry.is_dir():continue
                name=entry.filename.lower()
                if name.endswith(".csv") and len(summary["csv_samples"])<7:
                    with z.open(entry) as f:
                        reader=csv.DictReader(io.TextIOWrapper(f,encoding="utf-8-sig",errors="replace"))
                        first=next(reader,None)
                        summary["csv_samples"].append({
                            "name":entry.filename,"columns":reader.fieldnames,
                            "first_record":first
                        })
                if name.endswith((".json",".jsonl")) and len(summary["json_samples"])<3:
                    with z.open(entry) as f:
                        line=f.readline(50000).decode("utf-8-sig",errors="replace")
                        summary["json_samples"].append({
                            "name":entry.filename,
                            "first_line":line[:1200]
                        })
        out["archives"].append(summary)
        print("G3_ARCHIVE",summary["path"],"members",summary["n_zip_members"],
              "csv_headers",[(i["name"],i["columns"]) for i in summary["csv_samples"]],
              flush=True)
    OUT.parent.mkdir(exist_ok=True,parents=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("G3_SOURCE_LAYOUT_AUDIT",json.dumps({
        "zip_count":len(paths),
        "total_zip_bytes":sum(a["zip_bytes"] for a in out["archives"]),
        "total_members":sum(a["n_zip_members"] for a in out["archives"]),
        "extensions":dict(collections.Counter(
            Path(f["name"]).suffix for a in out["archives"] for f in a["files"]))
    },ensure_ascii=False),flush=True)
if __name__=="__main__":main()
