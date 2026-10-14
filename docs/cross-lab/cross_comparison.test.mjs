import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {analyzeCross} from './cross_engine.mjs';
import {allOutcomes,classifyOutcome,ticketMarket,compareJointForecast,resolveDecision} from './cross_comparison.mjs';
const fixture=JSON.parse(readFileSync(new URL('./fixtures/beppu_20261014_frozen.json',import.meta.url)));
const card=()=>{const {gradingOnly,frozenAnswer,...input}=fixture[5];return input;};

test('all four types from each two-line pair, including both bante; exactly four core tickets',()=>{
  const result=analyzeCross(card(),{limit:1000});
  assert.equal(result.candidates.length,12);
  assert.equal(result.candidates.filter(c=>c.lines.join() === '0,1').length,4);
  assert.ok(result.candidates.some(c=>c.firstPair.includes(4)&&c.firstPair.includes(7)));
  for(const c of result.candidates){assert.equal(c.tickets.length,4);assert.equal(new Set(c.signature.split('|')).size,4);}
});

test('no winning method threshold silently deletes legal cross types',()=>{
  const race=card();race.entries=race.entries.map(e=>({...e,nige:0,makuri:0,sashi:0,mark:0}));
  const result=analyzeCross(race,{limit:1000});
  assert.equal(result.candidates.length,12);
  assert.equal(result.recommended,null);
  assert.ok(result.candidates.every(c=>c.evidenceStatus==='UNRESOLVED'));
});

test('renumbering riders cannot change the set of candidate tickets',()=>{
  const race=card(), swap=n=>8-n;
  const changed={...race,lines:race.lines.map(l=>l.map(swap)),entries:race.entries.map(e=>({...e,no:swap(e.no)}))};
  const set=r=>analyzeCross(r,{limit:1000}).candidates.flatMap(c=>c.tickets).map(t=>t.join('-')).sort();
  assert.deepEqual(set(changed),set(race).map(t=>t.split('-').map(Number).map(swap).join('-')).sort());
});

test('retrospective coverage: all core-cross outcomes present; neither same-line nor three-line counted as core hit',()=>{
  let core=0;
  for(const r of fixture){
    const {gradingOnly,frozenAnswer,...input}=r;
    const c=classifyOutcome(input,gradingOnly.actual);
    const result=analyzeCross(input,{limit:1000});
    const found=result.candidates.some(x=>x.tickets.some(t=>t.join()===gradingOnly.actual.join()));
    assert.equal(found,c.coreCross);
    if(c.coreCross)core++;
    assert.deepEqual(analyzeCross(r,{limit:1000}),result); // outcomes never enter selection
  }
  assert.equal(core,5);
  assert.equal(classifyOutcome(fixture[7],fixture[7].gradingOnly.actual).kind,'THREE_LINES');
});

test('market metrics distinguish four equal stakes from inverse-price normalization',()=>{
  const c=analyzeCross(card()).candidates[0];
  const odds=Object.fromEntries(c.tickets.map((t,i)=>[t.join('-'),[2,4,8,16][i]]));
  const m=ticketMarket(c,odds);
  assert.equal(m.minRealizedReturnRatio,0.5);
  assert.equal(m.inverseOddsSum,0.9375);
  assert.equal(ticketMarket(c,{}).complete,false);
  assert.equal(resolveDecision({candidates:[c]}).main,null);
});

test('forecast requires complete distribution, matching race and strictly historical training',()=>{
  const race=card(), result=analyzeCross(race,{limit:1000}), outcomes=allOutcomes(race);
  const f={raceId:race.race_id,source:{id:'synthetic-test-only',trainThrough:'2025-12-31'},
    probabilities:Object.fromEntries(outcomes.map(t=>[t.join('-'),1/outcomes.length]))};
  const odds=Object.fromEntries(outcomes.map(t=>[t.join('-'),100]));
  const equal=compareJointForecast(race,result.candidates,f,odds);
  assert.equal(equal.selectedSignature,null); // no car-number tie breaker
  assert.ok(Math.abs(Object.values(equal.mass).reduce((s,p)=>s+p,0)-1)<1e-8);
  assert.throws(()=>compareJointForecast(race,result.candidates,{...f,raceId:'other'},odds));
  assert.throws(()=>compareJointForecast(race,result.candidates,{...f,source:{...f.source,trainThrough:'2026-10-14'}},odds));
  const missing=structuredClone(f);delete missing.probabilities[outcomes[0].join('-')];
  assert.throws(()=>compareJointForecast(race,result.candidates,missing,odds));
  const altered=structuredClone(f);altered.probabilities[outcomes[0].join('-')]=1;
  assert.throws(()=>compareJointForecast(race,result.candidates,altered,odds));
});

test('forecast chooses maximal four-point equal-stake return; partial prices never pick the priced subset',()=>{
  const race=card(), result=analyzeCross(race,{limit:1000}), outcomes=allOutcomes(race);
  const f={raceId:race.race_id,source:{id:'synthetic-test-only',trainThrough:'2025-12-31'},
    probabilities:Object.fromEntries(outcomes.map(t=>[t.join('-'),1/outcomes.length]))};
  const chosen=result.candidates.at(-1), odds=Object.fromEntries(outcomes.map(t=>[t.join('-'),2]));
  chosen.tickets.forEach(t=>odds[t.join('-')]=100);
  const comparison=compareJointForecast(race,result.candidates,f,odds);
  assert.equal(comparison.rankedSignature,chosen.signature);assert.equal(comparison.selectedSignature,null);
  assert.ok(Math.abs(comparison.rows.find(x=>x.signature===chosen.signature).returnRatio-100/210)<1e-10);
  assert.equal(resolveDecision(result,null,comparison).main,null);
  delete odds[result.candidates[0].tickets[0].join('-')];
  assert.equal(compareJointForecast(race,result.candidates,f,odds).selectedSignature,null);
});
