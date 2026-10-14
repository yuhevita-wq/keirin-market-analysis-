from pathlib import Path
import pandas as pd, zipfile, io, re, json, collections, math
REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'data'; OUT=REPO/'research/line_finish_linkage/pair12'
OUT.mkdir(parents=True,exist_ok=True)
START='2023-01-01'; END='2026-06-30'
races={}; results={}; conflicts=[]; sources=[]
def loadpair(label,rb,sb,group):
 r=pd.read_csv(rb,dtype=str).fillna(''); s=pd.read_csv(sb,dtype=str).fillna('')
 sources.append({'source':label,'races':len(r),'results_rows':len(s)})
 for x in r.to_dict('records'):
  if not START<=x['race_date']<=END:continue
  rid=x['race_id']; x['group']=group;x['dataset']=label
  if group=='S-grade':x['group']='S級'
  if rid in races:
   old=races[rid]
   if old.get('predicted_line_formation') and x.get('predicted_line_formation') and old['predicted_line_formation']!=x['predicted_line_formation']:conflicts.append(rid)
   if old.get('predicted_line_formation'):continue
  races[rid]=x
 local={}
 for x in s.to_dict('records'):local.setdefault(x['race_id'],[]).append(x)
 for rid,group_rows in local.items():
  if rid not in results:results[rid]=group_rows
paths=sorted(ROOT.rglob('races.csv'))
# primary all-stage collections first; supplemental preliminary collections second
paths.sort(key=lambda p:('s_class_yosen' in str(p), str(p)))
for p in paths:
 name=str(p); q=p.with_name('results.csv')
 if not q.exists():continue
 if '/a_class_12/' in name:group='A級1・2班'
 elif '/a3_challenge/' in name:group='A級チャレンジ'
 elif 's_class_f1' in name or 's_class_yosen' in name:group='S級'
 else:continue
 loadpair(name,p,q,group)
for p in sorted((ROOT/'grade_races').rglob('*.zip')):
 if not any(str(y) in str(p) for y in [2023,2024,2025,2026]):continue
 with zipfile.ZipFile(p) as z:
  rn=[n for n in z.namelist() if n.endswith('races.csv')];sn=[n for n in z.namelist() if n.endswith('results.csv')]
  if len(rn)!=1 or len(sn)!=1:continue
  loadpair(str(p),io.BytesIO(z.read(rn[0])),io.BytesIO(z.read(sn[0])),'S-grade')
rows=[];excluded=collections.Counter(); conflict_set=set(conflicts)
for rid,r in races.items():
 def reject(msg):excluded[(r['group'],msg)]+=1
 if rid in conflict_set:reject('duplicate_line_conflict');continue
 line=r.get('predicted_line_formation','')
 if not re.fullmatch(r'\d+(?:-\d+)*(?:/\d+(?:-\d+)*)*',line):reject('line_missing_or_complex');continue
 lines=[[int(x) for x in l.split('-')] for l in line.split('/')];flat=sum(lines,[])
 if len(flat)!=len(set(flat)):reject('line_duplicate_car');continue
 try:n=int(r['entry_count'])
 except:reject('entry_count_invalid');continue
 if len(flat)!=n:reject('line_car_count_mismatch');continue
 s=results.get(rid,[])
 if not s:reject('result_missing');continue
 sr=[(x['car_no'],x['finish_position']) for x in s]
 if len(set(a for a,b in sr))!=len(sr):reject('duplicate_result_car');continue
 winners=[x for x in s if x['finish_position']=='1']; seconds=[x for x in s if x['finish_position']=='2']
 if len(winners)!=1 or len(seconds)!=1:reject('top2_missing_or_tie');continue
 # 2nd-place dead heats are captured above; any tied placing above 2 is excluded as well.
 a=int(winners[0]['car_no']);b=int(seconds[0]['car_no'])
 if a not in flat or b not in flat:reject('top2_not_in_line');continue
 mapping={car:(i,j+1,len(l)) for i,l in enumerate(lines) for j,car in enumerate(l)}
 same=mapping[a][0]==mapping[b][0]
 accident=any(re.search('落|失|棄|欠',x.get('finish_text','')+x.get('result_comment','')) for x in s)
 shape='-'.join(map(str,sorted(map(len,lines),reverse=True)))
 rows.append({'race_id':rid,'date':r['race_date'],'year':r['race_date'][:4],'class':r['group'],'cars':n,'formation':shape,'line':line,'first':a,'second':b,'suji':same,'accident':accident,'first_pos':mapping[a][1],'second_pos':mapping[b][1],'first_line_size':mapping[a][2],'second_line_size':mapping[b][2],'race_type':r['race_type'],'source':r['dataset']})
df=pd.DataFrame(rows)
assert df.race_id.is_unique
df.to_json(OUT/'race_details.jsonl',orient='records',lines=True,force_ascii=False)

def summarize(keys,data=df):
 ans=[]
 for key,g in data.groupby(keys,sort=True):
  key=key if isinstance(key,tuple) else (key,)
  key=tuple(x.item() if hasattr(x,'item') else x for x in key); row=dict(zip(keys,key)); n=len(g);s=int(g.suji.sum());p=s/n;z=1.96;den=1+z*z/n;c=(p+z*z/(2*n))/den;d=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
  row.update(n=n,suji_count=s,non_suji_count=n-s,suji_pct=round(100*p,3),non_suji_pct=round(100*(1-p),3),suji_ci_low=round(100*(c-d),3),suji_ci_high=round(100*(c+d),3));ans.append(row)
 return ans
summary={'overall':summarize(['class','cars']),'formations':summarize(['class','cars','formation']),'years':summarize(['class','cars','year']),'formation_years':summarize(['class','cars','formation','year']),'race_types':summarize(['class','cars','race_type']),'no_accident_formations':summarize(['class','cars','formation'],df[~df.accident]),'excluded':[{'class':k[0],'reason':k[1],'n':v} for k,v in excluded.items()],'sources':sources,'period':[START,END],'source_commit':'91713712edc4c5c21c20d8aea66d999cbb0de1f4'}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
df.to_json(OUT/'race_details.jsonl',orient='records',lines=True,force_ascii=False)
print(json.dumps({'overall':summary['overall'],'7car': [x for x in summary['formations'] if x['cars']==7 and x['class']!='A級チャレンジ'],'years':[x for x in summary['years'] if x['cars']==7],'excluded':summary['excluded']},ensure_ascii=False,indent=2))
