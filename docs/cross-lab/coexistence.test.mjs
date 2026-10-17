import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {predictJoint} from './joint_model.mjs';
import {coexistenceFeatures,applyCoexistence,auditCoexistence,FEATURE_NAMES} from './coexistence.mjs';
const bundle=JSON.parse(fs.readFileSync(new URL('./joint_model_frozen.json',import.meta.url)));
const input={race_id:'8120261017010010',race_type:'Ａ級特選',lines:[[1,3,4],[2,5],[6,7]],entries:Array.from({length:7},(_,i)=>({no:i+1,score:90-i,age:25+i,style:i%2?'追':'逃',win_rate:20,top3_rate:50,b:i%2?0:8,nige:i%2?0:4,makuri:2,sashi:3,mark:1}))};
test('line and rider joint events are exact sums; never complement of strong-strong',()=>{
  const f=predictJoint(input,bundle),audit=auditCoexistence(input,f);
  assert.ok(Math.abs(audit.lineRows.reduce((s,z)=>s+z.probability,0)+audit.sameLine-1)<1e-10);
  assert.equal(audit.lineRows.length,3);
  for(const row of audit.lineRows){
    assert.ok(Math.abs(row.playerPairs.reduce((s,z)=>s+z.probability,0)-row.probability)<1e-12);
    assert.equal(row.playerSwitch,row.playerStrengthUnique&&row.bestStrongWeak.probability>row.strongPair.probability+1e-12);
  }
  assert.equal(audit.lineSwitch,audit.bestStrongWeak.probability>audit.strongStrong.probability+1e-12);
});
test('learner redistributes all 21 pairs, preserves ordered direction and conditional third',()=>{
  const f=predictJoint(input,bundle),model={intercept:-3,trees:[[{leaf:false,feature:0,threshold:.03,left:1,right:2},
    {leaf:true,value:-.5},{leaf:true,value:.5}]]};
  const b={version:'coexistence-v1',featureNames:FEATURE_NAMES,trainThrough:'2024-12-31',models:{A:model,S:model}};
  const out=applyCoexistence(input,f,b);
  assert.equal(Object.keys(out.probabilities).length,210);
  assert.ok(Math.abs(Object.values(out.probabilities).reduce((s,x)=>s+x,0)-1)<1e-10);
  const groups=new Map();for(const [k,v] of Object.entries(f.probabilities)){
    const pair=k.split('-').slice(0,2).sort().join('-');
    const ratio=out.probabilities[k]/v;if(groups.has(pair))assert.ok(Math.abs(groups.get(pair)-ratio)<1e-10);else groups.set(pair,ratio);
  }
  assert.equal(groups.size,21);
  assert.throws(()=>applyCoexistence(input,f,{...b,trainThrough:'2026-10-17'}),/当日/);
  assert.throws(()=>applyCoexistence(input,{...f,source:{...f.source,id:'other'}},b),/v3専用/);
  const polluted={...input,target:[7,6,5],payout:999999,odds:{'1-2-3':999},result:[7,6,5]};
  assert.deepEqual(coexistenceFeatures(polluted,f).records,coexistenceFeatures(input,f).records);
});
test('browser tree evaluator matches independently computed Python pairs for A and S',()=>{
  const frozen=JSON.parse(fs.readFileSync(new URL('./coexistence_frozen.json',import.meta.url)));
  const fixtures=JSON.parse(fs.readFileSync(new URL('./fixtures/coexistence_python_reference.json',import.meta.url)));
  for(const row of fixtures){
    const f=predictJoint(row.input,bundle),out=applyCoexistence(row.input,f,frozen),pack=coexistenceFeatures(row.input,out);
    assert.equal(out.coexistenceGateEnabled,false);
    for(let i=0;i<21;i++)assert.ok(Math.abs(pack.records[i].probability-row.expectedPairs[i])<1e-12,`${row.input.race_type}: pair ${i}`);
  }
});
