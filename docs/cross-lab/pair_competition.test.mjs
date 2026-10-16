import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {predictJoint,pairCompetition} from './joint_model.mjs';
import {analyzeCross} from './cross_engine.mjs';
import {compareJointForecast,resolveDecision} from './cross_comparison.mjs';
const frozen=JSON.parse(fs.readFileSync(new URL('./fixtures/nara_20261015_withdrawn.json',import.meta.url)));
const model=JSON.parse(fs.readFileSync(new URL('./joint_model_frozen.json',import.meta.url)));
const race=frozen.races.find(r=>r.race_no===10);
const candidate=()=>analyzeCross(race).candidates.find(c=>c.firstPair.join()==='7,1');

test('identical either-winner probability can mean zero or certain simultaneous top two',()=>{
  const r={...race,lines:[[1,3,4],[2,5],[6,7]]},c={firstPair:[1,2],remaining:[3,5],tickets:[[1,2,3],[1,2,5],[2,1,3],[2,1,5]]};
  const low=pairCompetition(c,{probabilities:{'1-3-4':0.5,'2-5-4':0.5}},r);
  const high=pairCompetition(c,{probabilities:{'1-2-3':0.5,'2-1-5':0.5}},r);
  assert.equal(low.eitherWins,1);assert.equal(high.eitherWins,1);
  assert.equal(low.bothTopTwo,0);assert.equal(high.bothTopTwo,1);
  assert.equal(low.exactlyOneTopTwo,1);assert.equal(high.exactlyOneTopTwo,0);
  assert.equal(low.directions[0].ownLineSecondGivenFirst,1);
  assert.equal(high.directions[0].secondGivenFirst,1);
});

test('Nara 10R: own bante outranks cross partner in each winning direction',()=>{
  const f=predictJoint(race,model),a=pairCompetition(candidate(),f,race);
  assert.ok(Math.abs(a.eitherWins-0.5345447623306172)<1e-12);
  assert.ok(Math.abs(a.bothTopTwo-0.07730853450022765)<1e-12);
  assert.deepEqual(a.directions.map(d=>d.secondAlternatives[0].no),[2,5]);
  assert.deepEqual(a.directions.map(d=>d.partnerSecondRank),[2,3]);
  assert.ok(a.interactionDelta<0);
  assert.ok(Math.abs(a.bothTopTwo+a.exactlyOneTopTwo+a.neitherTopTwo-1)<1e-12);
  for(const d of a.directions)
    assert.ok(Math.abs(d.secondAlternatives.reduce((s,x)=>s+x.conditionalProbability,0)-1)<1e-12);
});

test('withdrawn cards retain 12 strict four-ticket crosses and a provisional model main',()=>{
  for(const r of frozen.races){
    const result=analyzeCross(r),f=predictJoint(r,model);
    const c=compareJointForecast(r,result.candidates,f,{}, {objective:'HIT_PROBABILITY'});
    assert.equal(result.candidates.length,12);
    assert.ok(result.candidates.every(x=>x.tickets.length===4));
    assert.ok(c.rankedSignature);assert.equal(c.selectedSignature,null);
    assert.equal(resolveDecision(result,null,c).main.signature,c.rankedSignature);
    assert.ok(c.rows.every(x=>x.competition&&x.competition.additiveCoreProbability!==null));
    // Legacy selection field is ignored: the ranked candidate alone determines the provisional main.
    assert.equal(resolveDecision(result,null,{...c,selectedSignature:'legacy-forgery'}).main.signature,c.rankedSignature);
  }
});

test('manual adoption requires separate coexistence and second-place competition for BOTH directions',()=>{
  const result=analyzeCross(race),c=candidate(),[a,b]=c.firstPair;
  const d={signature:c.signature,interaction:'common condition',alternativeCross:'alternative',sameLine:'compare line finish',outside:'outside risk'};
  assert.throws(()=>resolveDecision(result,d),/表裏/);
  d.directions=[{first:a,second:b,coexistence:'survival A',secondCompetition:'bante A'}];
  assert.throws(()=>resolveDecision(result,d),/表裏/);
  d.directions.push({first:b,second:a,coexistence:'survival B',secondCompetition:'bante B'});
  assert.deepEqual(resolveDecision(result,d).main,c);
  d.directions[1].secondCompetition='';
  assert.throws(()=>resolveDecision(result,d),/表裏/);
});

test('audit does not suppress all leader pairs, favor bante pairs, or alter forecast probabilities',()=>{
  const result=analyzeCross(race),f=predictJoint(race,model),before=structuredClone(f.probabilities);
  assert.ok(result.candidates.some(c=>c.firstPair.join()==='7,1'));
  assert.ok(result.candidates.some(c=>c.firstPair.join()==='2,5'));
  for(const c of result.candidates)pairCompetition(c,f,race);
  assert.deepEqual(f.probabilities,before);
  assert.equal(Object.keys(f.probabilities).length,210);
});
