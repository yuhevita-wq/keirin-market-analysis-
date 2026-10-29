#!/usr/bin/env python3
"""Find where every 2023 3連単 probability is over/under allocated, odds-free.
All original-model eligible 2023 races; Jan-Oct TRAIN and Nov-Dec untouched.
2024-H1 calibrated AI on 2023 is future-fitted BACKCAST, not a forward test.
All the grouping axes below are fixed BEFORE observing any outcomes.
"""
from __future__ import annotations
import json,math,sys
from pathlib import Path
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2023"))
import observe as fixed

CALIBRATION=ROOT/"data/audits/independent_trifecta_recalibration_2024.json"
OUTPUT=ROOT/"data/audits/independent_trifecta_allocation_errors_2023.json"
MODELS=("original_2023_AI","2024H1_calibrated_backcast")
AXES=("line_pattern","role1","role2","role3","role12","role23","roles123",
      "score_rank1","score_rank2","score_rank3","score_rank12","score_rank23",
      "score_ranks123","line_size1","line_size2","line_size3",
      "line_pattern_x_role1","line_pattern_x_role2","line_pattern_x_role3",
      "line_pattern_x_role12","line_pattern_x_role23",
      "line_pattern_x_rank1","line_pattern_x_rank2","line_pattern_x_rank3",
      "line_pattern_x_role123","line_pattern_x_score_rank12",
      "predicted_probability_band","predicted_trifecta_rank_band",
      "field_size","line_shape","line_shape_x_line_pattern",
      "line_pattern_x_probability_band","line_pattern_x_predicted_rank_band")
PBANDS=((0,.001,"p<0.1%"),(.001,.005,"0.1-0.5%"),(.005,.01,"0.5-1%"),
        (.01,.02,"1-2%"),(.02,.05,"2-5%"),(.05,.1,"5-10%"),
        (.1,.2,"10-20%"),(.2,2,"20%+"))
RBANDS=((1,2,"rank1"),(2,4,"rank2-3"),(4,11,"rank4-10"),
        (11,31,"rank11-30"),(31,101,"rank31-100"),
        (101,301,"rank101-300"),(301,10000,"rank301+"))
PERIODS=("JanOct_in_sample","NovDec_original_AI_holdout","all_2023")

class Tally:
 def __init__(self):
  self.tickets=0;self.races=0;self.p=0.;self.wins=0;self.var=0.
  self.day=defaultdict(float)
 def add(self,prob,win,n,day):
  if not -1e-8<=prob<=1+1e-8:raise AssertionError("Invalid group probability mass")
  prob=max(0.,min(1.,prob))
  if win not in (0,1):raise AssertionError("Winner group invalid")
  self.p+=prob;self.wins+=win;self.tickets+=n;self.races+=1
  self.var+=prob*(1-prob);self.day[day]+=win-prob
 def result(self):
  diff=self.wins-self.p
  sd=math.sqrt(self.var)
  ds=list(self.day.values());nd=len(ds)
  av=sum(ds)/nd
  day_se=(math.sqrt(nd/(nd-1)*sum((z-av)**2 for z in ds)) if nd>1 else None)
  return {"trifecta_tickets":self.tickets,"race_exposures":self.races,
    "AI_predicted_event_count":round(self.p,5),
    "actual_event_count":self.wins,"actual_minus_AI_expected":round(diff,5),
    "AI_mean_probability_per_ticket_pct":round(100*self.p/self.tickets,8),
    "observed_win_frequency_per_ticket_pct":round(100*self.wins/self.tickets,8),
    "actual_over_predicted_count":round(self.wins/self.p,6) if self.p else None,
    "race_event_z":round(diff/sd,4) if sd else None,
    "day_cluster_z":round(diff/day_se,4) if day_se and day_se>0 else None,
    "dates":nd}

def band(x,b):
 for lo,hi,v in b:
  if lo<=x<hi:return v
 raise AssertionError(("No band",x))

def role(e,c):
 _,pos,n=e["positions"][c]
 if n==1:return "solo"
 if pos==1:return "leader"
 if pos==2:return "second"
 if pos==3:return "third"
 return "fourth_plus"

def line_pattern(e,a,b,c):
 ids=[e["positions"][x][0] for x in (a,b,c)]
 d={};s=""
 for x in ids:
  if x not in d:d[x]="ABC"[len(d)]
  s+=d[x]
 return s

def static_labels(e,t,shape):
 a,b,c=t
 r=[role(e,x) for x in t]
 rk=[str(e["ranks"][x]) if e["ranks"][x]<4 else "4plus" for x in t]
 size=[str(e["positions"][x][2]) if e["positions"][x][2]<4 else "4plus" for x in t]
 pat=line_pattern(e,*t)
 r12="_".join(r[:2]);r23="_".join(r[1:]);rs="_".join(r)
 k12="_".join(rk[:2]);k23="_".join(rk[1:]);ks="_".join(rk)
 return pat,(pat,r[0],r[1],r[2],r12,r23,rs,
  rk[0],rk[1],rk[2],k12,k23,ks,
  size[0],size[1],size[2],
  pat+"|"+r[0],pat+"|"+r[1],pat+"|"+r[2],
  pat+"|"+r12,pat+"|"+r23,
  pat+"|"+rk[0],pat+"|"+rk[1],pat+"|"+rk[2],
  pat+"|"+rs,pat+"|"+k12)

def classify(e,t,p,rank,static,shape):
 pat,base=static[t]
 probband=band(p,PBANDS)
 rankband=band(rank,RBANDS)
 labels=base+(probband,rankband,str(len(e["scores"])),shape,
              shape+"|"+pat,pat+"|"+probband,pat+"|"+rankband)
 assert len(labels)==len(AXES)
 return zip(AXES,labels)

def corrected(e,dist,coeff):
 alpha=float(coeff["alpha"]);betas=[float(q) for q in coeff["line_coefs"]]
 lv=[]
 for (a,b,c),p in dist.items():
  ga,gb,gc=(e["positions"][v][0] for v in (a,b,c))
  fs=(ga==gb,ga==gc,gb==gc,ga==gb==gc)
  lv.append(((a,b,c),alpha*math.log(max(p,1e-15))+
             sum(beta*flag for beta,flag in zip(betas,fs))))
 top=max(z for _,z in lv)
 w=[(key,math.exp(z-top)) for key,z in lv]
 norm=sum(v for _,v in w)
 out={key:v/norm for key,v in w}
 assert abs(sum(out.values())-1)<1e-7
 return out

def scan_race(e,dists,stats,period):
 shape="-".join(map(str,sorted(map(len,e["groups"]),reverse=True)))
 keys=list(dists[0][1])
 st={t:static_labels(e,t,shape) for t in keys}
 actual=e["order"]
 if actual not in st:raise AssertionError("Actual ticket not in model")
 for model,dist in dists:
  assert len(dist)==len(keys)
  assert abs(sum(dist.values())-1)<1e-7
  # Ranking is model-derived only, never reconstructed from odds or outcomes.
  ranked=sorted(dist,key=lambda x:(-dist[x],x))
  ranks={t:i+1 for i,t in enumerate(ranked)}
  local={}
  for t,p in dist.items():
   win=int(t==actual)
   for ax,label in classify(e,t,p,ranks[t],st,shape):
    key=(ax,label)
    item=local.get(key)
    if item is None:item=[0.,0,0];local[key]=item
    item[0]+=p;item[1]+=win;item[2]+=1
  masses=defaultdict(float);hits=defaultdict(int);counts=defaultdict(int)
  for (ax,label),(p,win,n) in local.items():
   masses[ax]+=p;hits[ax]+=win;counts[ax]+=n
   for time_group in (period,"all_2023"):
    stats[(model,time_group,ax,label)].add(p,win,n,e["date"])
  for ax in AXES:
   if not abs(masses[ax]-1)<1e-7 or hits[ax]!=1 or counts[ax]!=len(keys):
    raise AssertionError(("Group not a complete disjoint ticket partition",ax))

def table(stats,model,period,axis):
 return dict(sorted((label,x.result()) for (m,t,a,label),x in stats.items()
                    if m==model and t==period and a==axis))

def diagnostics(stats,model):
 # Jan-Oct values alone select structural error hypotheses; holdout only checks.
 training=[]
 for (m,period,axis,label),x in stats.items():
  if m!=model or period!="JanOct_in_sample":continue
  d=x.result()
  if d["trifecta_tickets"]<300 or d["race_exposures"]<120:continue
  if d["AI_predicted_event_count"]<40:continue
  if d["race_event_z"] is None:continue
  training.append((axis,label,d))
 def pack(rows):
  out=[]
  for axis,label,x in rows[:40]:
   hold=stats.get((model,"NovDec_original_AI_holdout",axis,label))
   h=hold.result() if hold else None
   out.append({"axis":axis,"winning_trifecta_structure":label,
       "JanOct_training_diagnostic":x,"NovDec_frozen_bucket_check":h,
       "same_direction_in_NovDec":bool(h and
        x["actual_minus_AI_expected"]*h["actual_minus_AI_expected"]>0),
       "NovDec_has_at_least_20_predicted_events":bool(h and
        h["AI_predicted_event_count"]>=20)})
  return out
 return {"candidate_groups_preselected_on_training":len(training),
  "probability_taken_too_much_from":pack(sorted(
   (x for x in training if x[2]["actual_minus_AI_expected"]>0),
   key=lambda x:-x[2]["race_event_z"])),
  "probability_allocated_too_much_to":pack(sorted(
   (x for x in training if x[2]["actual_minus_AI_expected"]<0),
   key=lambda x:x[2]["race_event_z"])),
  "selection_warning":"Jan-Oct is original AI training data, not independent discovery. Holdout anomalies are not selected on holdout results."}

def selftest():
 test={"scores":{str(i):70+i for i in range(1,5)},
       "positions":{"1":(1,1,2),"2":(1,2,2),"3":(2,1,1),"4":(3,1,1)},
       "groups":[["1","2"],["3"],["4"]],
       "ranks":{"4":1,"3":2,"2":3,"1":4},
       "order":("1","2","3"),"date":"2023-11-05"}
 import itertools
 ts=list(itertools.permutations(test["scores"],3))
 p={t:1/len(ts) for t in ts}
 assert line_pattern(test,"1","2","3")=="AAB"
 assert line_pattern(test,"1","3","2")=="ABA"
 assert line_pattern(test,"3","1","2")=="ABB"
 assert line_pattern(test,"1","3","4")=="ABC"
 assert role(test,"3")=="solo" and role(test,"2")=="second"
 assert band(.1,PBANDS)=="10-20%"
 st=defaultdict(Tally)
 scan_race(test,[(MODELS[0],p)],st,"NovDec_original_AI_holdout")
 assert sum(z.wins for (m,period,ax,label),z in st.items()
            if ax=="line_pattern" and period=="NovDec_original_AI_holdout")==1
 print("ALL_TICKET_DISJOINT_PROBABILITY_PARTITION_TEST_PASS",flush=True)

def main():
 selftest()
 assert fixed.TRAIN_END=="2023-10-31"
 j=json.loads(CALIBRATION.read_text("utf-8"))
 assert j["accepted"] and j["model_adopted"]=="joint"
 assert j["odds_or_market_data_accessed"] is False
 assert j["2025_or_later_accessed"] is False
 events,coverage=fixed.ingest()
 assert len(events)==23440,("All 2023 eligible races",len(events))
 train={k:v for k,v in events.items() if v["date"]<=fixed.TRAIN_END}
 held={k:v for k,v in events.items() if v["date"]>fixed.TRAIN_END}
 assert len(train)==19248 and len(held)==4192
 model=fixed.train_model(train,True)
 stats=defaultdict(Tally)
 n=0;tickets=0
 for rid,e in events.items():
  per="JanOct_in_sample" if e["date"]<=fixed.TRAIN_END else "NovDec_original_AI_holdout"
  orig=fixed.distribution(e,model,True)
  calib=corrected(e,orig,j["fitted_coefficients"]["joint"])
  scan_race(e,[(MODELS[0],orig),(MODELS[1],calib)],stats,per)
  n+=1;tickets+=len(orig)
  if n%1000==0:print("2023_ALL_RACE_PROBABILITY_ALLOCATIONS",n,"RACES",tickets,"TRIFECTA_TICKETS",flush=True)
 assert n==23440
 out={"state":"COMPLETED_FROZEN_AI_ALL_2023_TRIFECTA_ALLOCATION_ERROR_AUDIT",
  "purpose":"For every independently predicted 3連単, find predicted versus observed winning events in predefined line/role/score/rank groups.",
  "races_2023_all_eligible":n,
  "races_JanOct_original_AI_training":len(train),
  "races_NovDec_original_AI_unused":len(held),
  "all_2023_trifecta_ticket_combinations_per_model":tickets,
  "market_odds_or_payout_used":False,
  "AI_retrained":False,
  "AI_parameters_changed":False,
  "group_axes_frozen_before_labels":list(AXES),
  "probability_allocation_rule":"For each disjoint ticket group within a race, sum AI's probabilities; count actual winning triple as 1 in exactly one group. Across races compare sum expected and actual. The disparity identifies where probability was taken from or allocated to.",
  "line_pattern_definitions":{"AAA":"1-2-3 all same line",
   "AAB":"1st+2nd same line, 3rd other","ABA":"1st+3rd same line, 2nd other",
   "ABB":"2nd+3rd same line, 1st other","ABC":"all three distinct lines"},
  "position_roles":"leader/second/third/fourth_plus by predicted line, solo separate",
  "score_rank":"Top1,2,3 and 4plus by original training-era score ranking",
  "model_warning":"Original AI trained 2023 Jan-Oct; corrected AI coefficients fitted 2024-H1 and retrospectively applied to ALL 2023, not chronological 2023 out of sample.",
  "source_coverage":coverage,
  "models":{},
  "limitations":[
    "Not 5226 odds-matched races: uses ALL 23440 available original AI 2023 full-card and result races.",
    "Original 2023 model Jan-Oct same sample as training; Nov-Dec 4192 races unseen in original fitting.",
    "Corrected model uses 2024 future data, so all its 2023 numbers are backcasts, not independent forward evaluations.",
    "Published projected lines and score timestamps have not been audited against historical cutoff.",
    "Ticket probabilities add to exactly 100% each race; grouping axes each partition that 100%, but different axes overlap.",
    "A low expected frequency bucket with few winners is uncertain; train-phase biggest errors are diagnostic, not validated without unused years.",
    "No model modification, odds, purchases, payouts or market score was used.",
    "Race-level z accounts for mutually exclusive tickets in the same race. Day-cluster z is approximate, not multiple-test-corrected."
  ]}
 for name in MODELS:
  s={}
  for per in PERIODS:
   s[per]={ax:table(stats,name,per,ax) for ax in AXES}
   expected={"JanOct_in_sample":19248,"NovDec_original_AI_holdout":4192,"all_2023":23440}[per]
   for ax,vals in s[per].items():
    if sum(x["actual_event_count"] for x in vals.values())!=expected:
     raise AssertionError(("Wrong actual group event sum",name,per,ax))
    if abs(sum(x["AI_predicted_event_count"] for x in vals.values())-expected)>.01:
     raise AssertionError(("Wrong predicted event sum",name,per,ax))
  s["training_anomalies_checked_against_original_unused_NovDec"]=diagnostics(stats,name)
  out["models"][name]=s
 OUTPUT.parent.mkdir(parents=True,exist_ok=True)
 OUTPUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 print("ALL_2023_TRIFECTA_ALLOCATIONS_COMPLETE",
       json.dumps({"races":n,"trifecta_ticket_combinations_per_model":tickets,
                    "original_model_holdout_line_patterns":
                      out["models"][MODELS[0]]["NovDec_original_AI_holdout"]["line_pattern"],
                    "selected_train_groups":
                      out["models"][MODELS[0]]["training_anomalies_checked_against_original_unused_NovDec"]["candidate_groups_preselected_on_training"]
                    },ensure_ascii=False),flush=True)

if __name__=="__main__":main()
