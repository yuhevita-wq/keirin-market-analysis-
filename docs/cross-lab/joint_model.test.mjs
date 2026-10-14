import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {featurePack,predictJoint,jointDiagnostics,JOINT_VERSION} from './joint_model.mjs';
import {analyzeCross} from './cross_engine.mjs';
import {compareJointForecast} from './cross_comparison.mjs';
const card=()=>({race_id:'1320261012030007',race_type:'Ａ級特選',lines:[[1,3,4],[7],[2,5,6]],entries:Array.from({length:7},(_,i)=>({no:i+1,score:90-i,age:25+i,style:i%2?'追':'逃',win_rate:20,top3_rate:50,b:i%2?0:8,nige:i%2?0:4,makuri:2,sashi:3,mark:1}))});
const blank=()=>{const p=featurePack(card());return {version:JOINT_VERSION,trainThrough:'2024-12-31',models:Object.fromEntries(['A','S'].map(c=>[c,{pair:Array(p.pairFeatures[0].length).fill(0),third:Array(p.thirdFeatures[0][0].length).fill(0)}]))};};

test('same individual winner marginals can have entirely different simultaneous top-two probabilities',()=>{
  const c={firstPair:[1,2],remaining:[3,5]};
  const f=e=>({probabilities:{'1-2-3':e,'2-1-5':e,'1-3-2':0.5-e,'2-5-1':0.5-e}});
  const low=jointDiagnostics(c,f(0.01)),high=jointDiagnostics(c,f(0.4));
  assert.equal(low[0].winProbability,high[0].winProbability);
  assert.equal(low[1].winProbability,high[1].winProbability);
  assert.equal(low[0].secondGivenFirst,0.02);
  assert.equal(high[0].secondGivenFirst,0.8);
  assert.notEqual(low[0].orderedPairProbability,low[0].winProbability*low[1].winProbability);
});

test('pair competition and conditional thirds form 210 distinct exhaustive outcomes summing to one',()=>{
  const f=predictJoint(card(),blank());
  assert.equal(Object.keys(f.probabilities).length,210);
  assert.ok(Math.abs(Object.values(f.probabilities).reduce((s,p)=>s+p,0)-1)<1e-12);
  const r=analyzeCross(card(),{limit:1000});
  const cmp=compareJointForecast(card(),r.candidates,f,{}, {objective:'HIT_PROBABILITY'});
  assert.ok(Math.abs(Object.values(cmp.mass).reduce((s,p)=>s+p,0)-1)<1e-12);
  assert.equal(cmp.selectedSignature,null); // no car-number tie break
  assert.ok(cmp.rows.every(row=>Math.abs(row.hitProbability-4/210)<1e-12));
});

test('training feature shape, observed pair targets and front-vs-front interaction are explicit',()=>{
  const race=card();race.entries[1].b=8;race.entries[1].nige=4;
  const p=featurePack(race);
  assert.equal(p.basePairFeatureCount,35);assert.equal(p.pairFeatures[0].length,53);
  assert.equal(p.thirdFeatures[0][0].length,29);
  assert.ok(p.pairFeatures[p.pairs.findIndex(t=>t.join()==='1,2')][35]>0);
  assert.equal(p.pairFeatures[p.pairs.findIndex(t=>t.join()==='3,5')][35],0);
});

test('A/S coefficient sets stay separate, missing or zero-score fields fail, race results are not inputs',()=>{
  const r=card(),m=blank(),before=predictJoint(r,m);
  r.result=[5,3,1];r.entries.forEach(e=>{e.finish_position=1;e.result_comment='未来の結果';});
  assert.deepEqual(predictJoint(r,m),before);
  m.models.S.pair[0]=2;
  assert.deepEqual(predictJoint(r,m),before);
  r.entries[0].score=0;assert.throws(()=>predictJoint(r,m),/0得点/);
  r.entries[0].score=90;r.entries[0].mark=null;assert.throws(()=>predictJoint(r,m),/不足/);
});

test('probability comparison has a main choice without odds; expected return still requires every price',()=>{
  const r=card(),result=analyzeCross(r,{limit:1000}),f=predictJoint(r,blank()),selected=result.candidates[0];
  for(const key of Object.keys(f.probabilities))f.probabilities[key]=0;
  selected.tickets.forEach(t=>f.probabilities[t.join('-')]=0.25);
  const c=compareJointForecast(r,result.candidates,f,{}, {objective:'HIT_PROBABILITY'});
  assert.equal(c.selectedSignature,selected.signature);assert.equal(c.priceComplete,false);
  assert.equal(compareJointForecast(r,result.candidates,f,{}).selectedSignature,null);
  assert.ok(c.rows[0].directions.every(d=>d.secondGivenFirst===1&&d.coreThirdGivenPair===1));
});

test('frozen fit contains A and S, and is independent of October 14 results',()=>{
  const model=JSON.parse(fs.readFileSync(new URL('./joint_model_frozen.json',import.meta.url)));
  assert.equal(model.trainThrough,'2024-12-31');
  assert.deepEqual(Object.keys(model.models).sort(),['A','S']);
  for(const cls of ['A','S']){assert.ok(model.models[cls].trainRaces>=1000);assert.equal(model.models[cls].pair.length,53);assert.equal(model.models[cls].third.length,29);}
});

test('midnight answers remain frozen: core-cross misses, same-line outcomes and illegal old core stay separate',()=>{
  const rows=JSON.parse(fs.readFileSync(new URL('./fixtures/midnight_20261014_frozen.json',import.meta.url)));
  let coreMisses=0,sameLine=0,illegal=0;
  for(const {input,frozenAnswer:f,gradingOnly:g} of rows){
    const result=analyzeCross(input,{limit:1000});
    const t=f.firstPair.flatMap((a,i)=>f.remaining.map(c=>[a,f.firstPair[1-i],c]));
    assert.equal(t.some(x=>x.join()===g.actual.join()),false);
    const belongs=result.candidates.some(c=>c.tickets.map(x=>x.join('-')).sort().join()===t.map(x=>x.join('-')).sort().join());
    if(!belongs)illegal++;
    const p=new Map(input.lines.flatMap((l,i)=>l.map((n,j)=>[n,{line:i,pos:j+1}])));
    const ps=g.actual.map(n=>p.get(n));
    if(ps[0].line===ps[1].line)sameLine++;
    else if(ps.every(x=>x.pos<=2)&&new Set(ps.map(x=>x.line)).size===2)coreMisses++;
    assert.equal(g.fitEligible,false);
  }
  assert.equal(rows.length,9);assert.equal(coreMisses,2);assert.equal(sameLine,6);assert.equal(illegal,1);
});
