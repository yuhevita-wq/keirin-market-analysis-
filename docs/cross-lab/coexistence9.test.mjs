import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,existsSync} from 'node:fs';
import {predictObservedPair9,pairFeaturePack9} from './coexistence9.mjs';
const bundle=JSON.parse(readFileSync(new URL('./coexistence9_frozen.json',import.meta.url),'utf8'));
const baseRace={
  race_id:'2720261017010003',race_type:'Ｓ級一次予選',meeting_grade:'G3',
  lines:[[9,2,4],[3,5,7],[8,1,6]],
  entries:Array.from({length:9},(_,i)=>({
    car_no:i+1,score:105-(i%4)*3,win_rate:10+2*i,top3_rate:35+4*i,
    b_count:i%3?i+1:0,nige_count:i%4,makuri_count:(i+2)%5,
    sashi_count:(i+1)%6,mark_count:(i+3)%4,
  }))
};
test('actual coexistence model returns 36 joint pairs, not a four-ticket selection',()=>{
  const out=predictObservedPair9(baseRace,bundle);
  assert.equal(out.rows.length,36);
  assert.equal(out.top2Ranking.length,12);
  assert.equal(out.allCrossTop2.length,27);
  assert.equal(out.status,'PAIR_ONLY_NOT_4TICKET');
  assert.ok(Math.abs(out.rows.reduce((s,r)=>s+r.top2,0)-1)<1e-10);
  assert.ok(out.rows.every(r=>r.top3>0&&r.top3<1&&r.top2>0&&r.top2<1));
  assert.ok(out.rows.some(r=>Math.abs(r.interactionDeltaTop2)>1e-5));
});
test('never treat missing historical ability or a post-training race as an observed forecast',()=>{
  const missing=structuredClone(baseRace);delete missing.entries[0].top3_rate;
  assert.throws(()=>predictObservedPair9(missing,bundle),/不足/);
  const old=structuredClone(baseRace);old.race_id='2720241017010003';
  assert.throws(()=>predictObservedPair9(old,bundle),/学習期間/);
  const bad=structuredClone(bundle);bad.evaluationStatus='HOLDOUT_FAILED';
  assert.throws(()=>predictObservedPair9(baseRace,bad),/未検証/);
});
test('Python learned features match JS inference on a real untouched race',()=>{
  const path=process.env.CROSS9_PARITY_FIXTURE;
  if(!path||!existsSync(path))return;
  const sample=JSON.parse(readFileSync(path,'utf8'));
  const actual=pairFeaturePack9(sample.input);
  assert.equal(actual.pairs.length,36);
  for(let i=0;i<36;i++){
    assert.deepEqual(actual.pairs[i].pair,sample.rows[i].pair);
    for(const kind of ['base','mechanism']){
      const got=actual.pairs[i][kind],expected=sample.rows[i][kind];
      assert.equal(got.length,expected.length);
      got.forEach((v,j)=>assert.ok(Math.abs(v-expected[j])<1e-5,`${kind} ${i},${j}: ${v} vs ${expected[j]}`));
    }
  }
  const predicted=predictObservedPair9(sample.input,bundle);
  for(let i=0;i<36;i++){
    for(const key of ['top2','top3','top2RiderOnly','top3RiderOnly'])
      assert.ok(Math.abs(predicted.rows[i][key]-sample.rows[i][key])<1e-5,key+' '+i);
  }
});
