import test from 'node:test';
import assert from 'node:assert/strict';
import {analyzeCross} from './cross_engine.mjs';
import {resolveDecision} from './cross_comparison.mjs';
const race={lines:[[1,2,3],[4,5],[6,7]],entries:Array.from({length:7},(_,i)=>({no:i+1,score:90,b:5,nige:2,makuri:3,sashi:4,mark:5}))};
test('zero, missing and large aggregates never turn a question into support or rejection',()=>{
 const shape=r=>analyzeCross(r).candidates.map(({observations,...c})=>c);
 for(const value of [0,null,999]){
  const altered={...race,entries:race.entries.map(e=>({...e,score:value,b:value,nige:value,makuri:value,sashi:value,mark:value}))};
  assert.deepEqual(shape(altered),shape(race));
 }
 const c=analyzeCross(race).candidates[0];
 assert.equal(c.reasoning.length,2);
 for(const d of c.reasoning)assert.equal(d.rivals.length,5);
});
test('analyst facts, provenance, inference and counterevidence remain distinct; missing is unknown',()=>{
 const result=analyzeCross(race),c=result.candidates[0];
 const d={signature:c.signature,interaction:'joint reasoning',alternativeCross:'other cross',sameLine:'line alternative',outside:'outside comparison',
  directions:c.firstPair.map((first,i)=>({first,second:c.firstPair[1-i],coexistence:'compatible actions',secondCompetition:'all rivals'})),
  observedFacts:'observed position',sourceContext:'source, date and opponent',counterEvidence:'opponent can block course',uncertainty:'current execution unknown'};
 const out=resolveDecision(result,d);
 assert.equal(out.basis.observedFacts,d.observedFacts);
 assert.equal(out.basis.counterEvidence,d.counterEvidence);
 assert.deepEqual(out.basis.unrecorded,[]);
 delete d.sourceContext;
 assert.deepEqual(resolveDecision(result,d).basis.unrecorded,['sourceContext']);
 assert.ok(resolveDecision(result,d).main);
});
