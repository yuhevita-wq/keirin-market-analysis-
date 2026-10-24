#!/usr/bin/env python3
"""Extract official already archived payout CSV rows for 2023-2025.

Data belongs to source repository and is saved under owner's CMS public cert.
Do NOT substitute final odds for official settlement or invent rows when absent.
"""
from __future__ import annotations
import collections, csv, gzip, io, json, re, subprocess, zipfile, hashlib, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"data"
CERT=ROOT/"odds_market/vault_public_cert.pem"
VALID=re.compile(r"^\d{16}$")
TICKET=re.compile(r".",re.I)
PAYOUT_FIELDS=("payout_yen","payout_amount","payout","amount","refund","dividend","return","pay","money","value","payout_text")
class Writer:
 def __init__(self,out):self.path=out;self.n=0
 def __enter__(self):
  self.p=subprocess.Popen(["openssl","cms","-encrypt","-binary","-aes-256-gcm","-stream",
     "-outform","DER","-out",str(self.path),str(CERT)],
     stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
  self.gz=gzip.GzipFile(fileobj=self.p.stdin,mode="wb",mtime=0)
  return self
 def write(self,obj):
  self.gz.write((json.dumps(obj,ensure_ascii=False,separators=(",",":"))+"\n").encode())
  self.n+=1
 def __exit__(self,kind,val,tb):
  self.gz.close();self.p.stdin.close()
  rc=self.p.wait(timeout=120)
  if kind is not None or rc:
   self.path.unlink(missing_ok=True)
   if kind is None:raise RuntimeError("ENCRYPTION_ERROR")
def source_tables():
 for file in sorted(DATA.rglob("payouts.csv")):
  with file.open("r",encoding="utf-8-sig",newline="") as f:
   yield str(file.relative_to(ROOT)),csv.DictReader(f)
 for file in sorted((DATA/"grade_races").rglob("*.zip")):
  with zipfile.ZipFile(file) as z:
   for n in sorted(z.namelist()):
    if n.split("/")[-1]!="payouts.csv":continue
    with z.open(n) as raw,io.TextIOWrapper(raw,encoding="utf-8-sig",newline="") as f:
     yield str(file.relative_to(ROOT))+":"+n,csv.DictReader(f)

def run(out):
 out.mkdir(parents=True,exist_ok=True)
 count=collections.Counter();keys=collections.Counter();examples=[]
 with Writer(out/"all_markets_official_2023_2025.jsonl.gz.cms") as writer:
  for source,rows in source_tables():
   count["payout_sources"]+=1
   for r in rows:
    race=str(r.get("race_id") or "")
    if not VALID.fullmatch(race) or race[2:6] not in {"2023","2024","2025"}:continue
    count["payout_rows_2023_2025"]+=1
    for k in r:keys[k]+=1
    typ=" ".join(str(r.get(k,"")) for k in ("ticket_type","bet_code"))
    if not typ.strip():continue
    count["all_market_payout_rows"]+=1
    if len(examples)<5:
     examples.append({"columns":list(r.keys()),"bet_type":typ[:40],
                      "combination":str(r.get("combination"))[:30],
                      "nonempty_payout_fields":{x:r[x] for x in PAYOUT_FIELDS if r.get(x) not in (None,"")}})
    writer.write({"race_id":race,"ticket_type":str(r.get("ticket_type") or ""),
                  "bet_code":str(r.get("bet_code") or ""),
                  "combination":str(r.get("combination") or ""),
                  "all_payout_fields":{x:str(r[x]) for x in PAYOUT_FIELDS if r.get(x) not in (None,"")},
                  "status":str(r.get("status") or ""),
                  "source_name":source})
 report={"counts":dict(count),"columns":dict(keys),"examples_redacted":examples,
  "sealed_output":"all_markets_official_2023_2025.jsonl.gz.cms",
  "official_settlement_file_rows":writer.n,
  "notes":["These are directly archived official payout.csv rows where available.",
           "Coverage does NOT imply all 49016 races have official payout data.",
           "Do not fill absent payout values with odds."],
  "encrypted_sha256":hashlib.sha256((out/"all_markets_official_2023_2025.jsonl.gz.cms").read_bytes()).hexdigest()}
 (out/"all_markets_official_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
 print("TYPE_LAB_OFFICIAL_PAYOUTS_CENSUS",json.dumps(report,ensure_ascii=False)[:7000],flush=True)
if __name__=="__main__":
 run(Path(sys.argv[1] if len(sys.argv)>1 else "official-payout-audit"))
