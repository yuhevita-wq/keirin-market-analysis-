import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {predictMarket} from './market_model.mjs';
import {allOutcomes} from './cross_comparison.mjs';
const model=JSON.parse(readFileSync(new URL('./market_model_frozen.json',import.meta.url)));
function sample(){
  const input={race_id:'8620261014010006',race_type:'S級予選',meeting_grade:'F1',lines:[[1,2,3],[4,5],[6,7]],
    entries:Array.from({length:7},(_,i)=>({no:i+1,score:95+i,win_rate:10+i*3,top3_rate:30+i*5,
      b:i,nige:i%3,makuri:(i+1)%4,sashi:(i+2)%5,mark:i%2,age:25+i*3,style:['逃','両','追'][i%3]}))};
  const outcomes=allOutcomes(input),odds=Object.fromEntries(outcomes.map((t,i)=>[t.join('-'),10+i]));
  const trioOdds={};outcomes.forEach(t=>trioOdds[[...t].sort().join('=')]=20+t.reduce((s,v)=>s+v,0));
  return {input,odds,trioOdds};
}
test('frozen model produces all 210 normalized outcomes, positive probabilities and separate scope',()=>{
  const x=sample(),forecast=predictMarket(x.input,x.odds,x.trioOdds,model);
  assert.equal(Object.keys(forecast.probabilities).length,210);
  assert.ok(Math.abs(Object.values(forecast.probabilities).reduce((s,p)=>s+p,0)-1)<1e-12);
  assert.ok(Object.values(forecast.probabilities).every(p=>p>0));
  assert.equal(forecast.source.trainThrough,'2025-12-31');
});
test('cannot silently apply S F1 model to A, G races, missing card data or partial markets',()=>{
  const x=sample();
  for(const change of [{race_type:'A級予選'},{meeting_grade:'G3'},{race_id:''}])
    assert.throws(()=>predictMarket({...x.input,...change},x.odds,x.trioOdds,model));
  const missing=structuredClone(x);delete missing.input.entries[0].age;
  assert.throws(()=>predictMarket(missing.input,missing.odds,missing.trioOdds,model));
  delete missing.odds['1-2-3'];assert.throws(()=>predictMarket(x.input,missing.odds,x.trioOdds,model));
});
test('zero correction is exactly the market baseline; result fields never affect the prediction',()=>{
  const x=sample(),zero={...model,coefficients:model.coefficients.map(()=>0)};
  const out=predictMarket(x.input,x.odds,x.trioOdds,zero);
  const z=Object.values(x.odds).reduce((s,o)=>s+1/o,0);
  for(const [k,p]of Object.entries(out.probabilities))assert.ok(Math.abs(p-1/x.odds[k]/z)<1e-12);
  assert.deepEqual(predictMarket({...x.input,result:[7,6,5],payout:999},x.odds,x.trioOdds,model),predictMarket(x.input,x.odds,x.trioOdds,model));
});
export {sample};

test('JS frozen-model port matches independently computed Python probabilities',()=>{
  const x=sample(),expected=JSON.parse(readFileSync(new URL('./fixtures/market_model_python_reference.json',import.meta.url))).expected;
  const actual=predictMarket(x.input,x.odds,x.trioOdds,model).probabilities;
  for(const key of Object.keys(expected))assert.ok(Math.abs(expected[key]-actual[key])<1e-8,key);
});
