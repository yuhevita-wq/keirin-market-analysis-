import fs from 'node:fs';
import {analyzeCross,normalizeRace} from '../../docs/cross-lab/cross_engine.mjs';
import {predictJoint} from '../../docs/cross-lab/joint_model.mjs';
import {roleAdjustedForecast} from '../../docs/cross-lab/role_conditions.mjs';
const root=new URL('../../../',import.meta.url),read=p=>JSON.parse(fs.readFileSync(new URL(p,root))),cards=read('role_results/cards.json'),bundle=read('system/docs/cross-lab/joint_model_frozen.json'),roles=read('system/docs/cross-lab/role_conditions_frozen.json');
const out={},counts={eligible:0,invalid:0,excludedClass:0},records=[];
const same=(a,b)=>[...a].sort().join() === [...b].sort().join();
for(const [i,row] of cards.entries()){
 if(row.date<'2025-01-01')continue;
 row.input.race_date=row.date;
 if(row.class==='A級チャレンジ'){counts.excludedClass++;continue;}
 let f;try{f=predictJoint(row.input,bundle);}catch(e){counts.invalid++;continue;}counts.eligible++;
 const r=normalizeRace(row.input),all=analyzeCross(row.input).candidates,base=all.filter(c=>[...c.firstPair,...c.remaining].every(n=>r.position.get(n).pos<3)),adj=roleAdjustedForecast(row.input,f,roles),key=row.target.join('-');
 const best=(cs,fc)=>cs.reduce((best,c)=>{const v=c.tickets.reduce((s,t)=>s+fc.probabilities[t.join('-')],0);return !best||v>best.p?{c,p:v}:best;},null);
 const old=best(base,f),expanded=best(all,f),adjusted=best(all,adj),within=best(all.filter(c=>same(c.lines,expanded.c.lines)),adj),pair=expanded.c.lines;
 const actualPos=row.target.map(n=>r.position.get(n)),opp=actualPos[0].line!==actualPos[1].line&&actualPos.every(x=>pair.includes(x.line));
 const pp=row.date<='2025-06-30'?'2025H1':row.date<='2025-12-31'?'2025H2':row.date<='2026-06-30'?'2026H1':'2026partialH2';
 for(const [policy,z] of Object.entries({live_front_bante12:old,live_expanded40:expanded,role_adjusted40:adjusted,role_adjusted_fixed_pair:within})){
  const id=row.class+'|'+pp+'|'+policy,a=out[id]??={n:0,hits:0,stake:0,payout:0,withinPairOpportunities:0,withinPairHits:0,headCorrect:0,predictedHits:0};
  const hit=z.c.tickets.some(t=>t.join('-')===key);a.n++;a.hits+=+hit;a.stake+=400;a.payout+=hit?row.payout:0;a.predictedHits+=z.p;a.withinPairOpportunities+=+opp;a.withinPairHits+=+(opp&&hit);a.headCorrect+=+(opp&&same(z.c.firstPair,row.target.slice(0,2)));
  records.push({race_id:row.input.race_id,period:pp,class:row.class,policy,hit:+hit,payout:hit?row.payout:0,opp:+opp,heads:z.c.firstPair,thirds:z.c.remaining,lines:z.c.lines,probability:z.p});
 }
 if(i%2000===0)console.log('PROGRESS',i,cards.length);
}
for(const a of Object.values(out)){a.returnPct=a.payout/a.stake*100;a.capturePct=a.withinPairHits/a.withinPairOpportunities*100;}
fs.writeFileSync(new URL('role_results/live_model_validation.json',root),JSON.stringify({counts,results:out},null,2));fs.writeFileSync(new URL('role_results/live_selections.json',root),JSON.stringify(records));console.log('DONE',JSON.stringify({counts,results:out}));
