import fs from 'node:fs';
import {analyzeCross} from '../../docs/cross-lab/cross_engine.mjs';
import {predictJoint} from '../../docs/cross-lab/joint_model.mjs';
import {classifyOutcome} from '../../docs/cross-lab/cross_comparison.mjs';
const [cardsPath,bundlePath,outPath]=process.argv.slice(2),cards=JSON.parse(fs.readFileSync(cardsPath)),bundle=JSON.parse(fs.readFileSync(bundlePath));
const aggregates={};
for(const name of ['joint','additiveBaseline'])aggregates[name]={races:0,selected:0,hits:0,coreCrossActual:0,hitsOnCoreCross:0,priced:0,stake:0,payout:0,predictedHits:0,ties:0,bins:Array.from({length:10},()=>({n:0,predicted:0,hits:0}))};
for(const row of cards){
  const result=analyzeCross(row.input,{limit:1000}),actual=row.target.join('-'),core=classifyOutcome(row.input,row.target).coreCross;
  for(const name of ['joint','additiveBaseline']){
    const b=name==='joint'?bundle:structuredClone(bundle);
    if(name!=='joint')for(const m of Object.values(b.models))m.pair=m.baselinePair;
    const f=predictJoint(row.input,b),ranked=result.candidates.map(c=>({c,p:c.tickets.reduce((s,t)=>s+f.probabilities[t.join('-')],0)})).sort((a,b)=>b.p-a.p);
    const a=aggregates[name];a.races++;if(core)a.coreCrossActual++;
    if(!ranked.length)continue;
    if(ranked[1]&&Math.abs(ranked[0].p-ranked[1].p)<1e-12){a.ties++;continue;}
    const {c,p}=ranked[0],hit=c.tickets.some(t=>t.join('-')===actual);a.selected++;a.hits+=+hit;a.predictedHits+=p;
    if(core&&hit)a.hitsOnCoreCross++;
    const bin=a.bins[Math.min(9,Math.floor(p*10))];bin.n++;bin.predicted+=p;bin.hits+=+hit;
    if(row.payout!=null){a.priced++;a.stake+=400;if(hit)a.payout+=row.payout;}
  }
}
for(const a of Object.values(aggregates)){
  a.hitRate=a.selected?a.hits/a.selected:null;a.returnRatio=a.stake?a.payout/a.stake:null;
  a.coreCrossCaptureRate=a.coreCrossActual?a.hitsOnCoreCross/a.coreCrossActual:null;
  a.meanPredictedHit=a.selected?a.predictedHits/a.selected:null;
}
fs.writeFileSync(outPath,JSON.stringify(aggregates));
