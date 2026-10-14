from pathlib import Path
import json,zipfile,collections
ROOT=Path(__file__).resolve().parent
with zipfile.ZipFile(ROOT/'race_details_all_pairs.zip') as z:
 rows=[json.loads(line) for line in z.read('race_details_all_pairs.jsonl').decode().splitlines()]
assert len(rows)==len({r['race_id'] for r in rows})
for pair,positions in [('12',('first','second')),('23',('second','third')),('13',('first','third'))]:
 counts=collections.defaultdict(lambda:[0,0])
 for r in rows:
  value=r['same'+pair]
  if value is None:continue
  a,b=(r[pos] for pos in positions)
  groups=[set(map(int,g.split('-'))) for g in r['line'].split('/')]
  assert value==any(a in g and b in g for g in groups),(pair,r['race_id'])
  key=(r['class'],r['cars'],r['formation']);counts[key][0]+=1;counts[key][1]+=int(value)
 summary=json.loads((ROOT/('pair'+pair)/'summary.json').read_text())
 assert len(counts)==len(summary['formations'])
 for g in summary['formations']:
  n,same=counts[(g['class'],g['cars'],g['formation'])]
  assert (n,same,n-same)==(g['n'],g['suji_count'],g['non_suji_count'])
 print(f'{pair}: all classifications and {len(counts)} grouped counts verified')
print(f'{len(rows):,} unique races verified')
