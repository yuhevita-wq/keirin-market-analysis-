import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {roleFeatures,predictRoleConditions,roleAdjustedForecast} from './role_conditions.mjs';
import {analyzeCross,normalizeRace} from './cross_engine.mjs';
import {allOutcomes} from './cross_comparison.mjs';
const bundle=JSON.parse(fs.readFileSync(new URL('./role_conditions_frozen.json',import.meta.url))),fixtures=JSON.parse(fs.readFileSync(new URL('./fixtures/role_conditions_python_reference.json',import.meta.url)));
test('JS card features and role probabilities match independent Python reference across leaves, grades and 3x2/2x2',()=>{
 for(const c of fixtures){const f=roleFeatures(c.input,c.indexes),p=predictRoleConditions(c.input,c.indexes,bundle);
  f.features.forEach((v,i)=>{if(c.features[i]===null)assert.ok(!Number.isFinite(v));else assert.ok(Math.abs(v-c.features[i])<1e-9);});
  assert.equal(p.leaf,c.leaf);p.probabilities.forEach((v,i)=>assert.ok(Math.abs(v-c.probabilities[i])<1e-10));assert.equal(p.trainThrough,'2024-12-31');
 }
});
test('all 210 outcomes have exact structural coverage; 18/18/4 candidate sets stay four tickets each',()=>{
 const input=fixtures[0].input,r=normalizeRace(input),cs=analyzeCross(input).candidates;
 assert.equal(cs.length,40);
 for(let a=0;a<3;a++)for(let b=a+1;b<3;b++)assert.equal(cs.filter(c=>c.lines.join() === [a,b].join()).length,r.lines[a].length*r.lines[b].length===6?18:4);
 for(const t of allOutcomes(input)){
  const p=t.map(n=>r.position.get(n)),valid=p[0].line!==p[1].line&&new Set(p.map(x=>x.line)).size===2;
  assert.equal(cs.some(c=>c.tickets.some(x=>x.join()===t.join())),valid);
 }
 assert.ok(cs.every(c=>c.tickets.length===4&&new Set(c.tickets.map(t=>t.join())).size===4));
});
test('research role adjustment preserves total and each pair event, with no change to same-line/three-line outcomes',()=>{
 const input=fixtures[0].input,r=normalizeRace(input),p=Object.fromEntries(allOutcomes(input).map((t,i)=>[t.join('-'),i+1])),sum=Object.values(p).reduce((s,v)=>s+v,0);for(const k in p)p[k]/=sum;
 const f={raceId:input.race_id,source:{id:'test',trainThrough:'2024-12-31'},probabilities:p},q=roleAdjustedForecast(input,f,bundle).probabilities;
 assert.ok(Math.abs(Object.values(q).reduce((s,v)=>s+v,0)-1)<1e-10);
 const mass={},after={};
 for(const [k,v]of Object.entries(p)){const ps=k.split('-').map(Number).map(n=>r.position.get(n));if(ps[0].line===ps[1].line||new Set(ps.map(x=>x.line)).size===3)assert.equal(q[k],v);else{const id=ps.slice(0,2).map(x=>x.line).sort().join();mass[id]=(mass[id]||0)+v;after[id]=(after[id]||0)+q[k];}}
 for(const k in mass)assert.ok(Math.abs(mass[k]-after[k])<1e-10);
});
test('result/odds fields never affect conditions, missing inputs remain disclosed and unsupported formations fail',()=>{
 const c=fixtures[0],p=predictRoleConditions(c.input,c.indexes,bundle);assert.deepEqual(predictRoleConditions({...c.input,actual:[7,6,5],odds:{'7-6-5':9999}},c.indexes,bundle),p);
 const input=structuredClone(c.input);input.entries.forEach(e=>e.score=null);assert.ok(predictRoleConditions(input,c.indexes,bundle).missingFeatures.length>0);
 assert.throws(()=>predictRoleConditions({...c.input,race_date:'2024-12-31'},c.indexes,bundle));
 assert.throws(()=>roleFeatures({...c.input,lines:[[1,2],[3,4],[5],[6,7]]},[0,1]));
});
