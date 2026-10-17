import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {normalizeRace} from './cross_engine.mjs';
import {resolveRaceClass} from './race_scope.mjs';
import {predictExtendedJoint} from './extended_joint.mjs';
const races=JSON.parse(fs.readFileSync(new URL('./fixtures/keiokaku_yokkaichi_20261017.json',import.meta.url))).races;
const bundle=JSON.parse(fs.readFileSync(new URL('./extended_joint_frozen.json',import.meta.url)));
test('separate size/class models reject out-of-scope inputs and preserve full outcome mass',()=>{
 for(const r of races.filter(r=>r.entries.length===9||r.entries.every(e=>e.class==='A3'))){
  const forecast=predictExtendedJoint(r,bundle),n=r.entries.length;
  assert.equal(Object.keys(forecast.probabilities).length,n*(n-1)*(n-2));
  assert.ok(Math.abs(Object.values(forecast.probabilities).reduce((s,v)=>s+v,0)-1)<1e-10);
  assert.ok(Object.values(forecast.probabilities).every(v=>Number.isFinite(v)&&v>0));
  assert.equal(forecast.source.trainThrough,'2024-12-31');
 }
 assert.throws(()=>predictExtendedJoint(races.find(r=>r.entries[0].class==='A1'),bundle));
 assert.equal(resolveRaceClass({race_type:'予選',entries:[{class:'S1'},{class:'S2'}]}),'S');
 assert.equal(resolveRaceClass({race_type:'準決勝',entries:[{class:'A1'},{class:'A2'}]}),'A');
 assert.equal(resolveRaceClass({race_type:'チャレンジ一般',entries:[{class:'A3'}]}),'A3');
 assert.equal(resolveRaceClass({race_type:'予選',entries:[{class:'L1'}]}),'unknown');
});
test('actual public page load/analyze produces a unique valid four-ticket answer for all 23 cards',async()=>{
 const html=fs.readFileSync(new URL('./index.html',import.meta.url),'utf8');
 const source=html.match(/<script type="module">([\s\S]*?)<\/script>/)[1].replace(/from "(\.\/[^\"]+)"/g,(_,p)=>'from '+JSON.stringify(new URL(p,import.meta.url).href));
 const nodes=new Map();let current;
 const make=()=>({value:'',checked:false,innerHTML:'',textContent:'',className:'',ownerDocument:null,addEventListener(){},append(){},replaceChildren(){}});
 const document={getElementById(id){if(!nodes.has(id))nodes.set(id,make());const n=nodes.get(id);n.ownerDocument=document;return n;},createElement(){return make();},querySelectorAll(){return [];},querySelector(selector){const no=+selector.match(/data-car="(\d+)"/)[1],field=selector.match(/data-stat="([^\"]+)"/)?.[1]||'style';return {value:String(current.entries.find(e=>e.no===no)[field])};}};
 const oldDocument=globalThis.document,oldFetch=globalThis.fetch;
 globalThis.document=document;globalThis.fetch=async path=>({ok:true,json:async()=>String(path).includes('live-race-data')?{races}:JSON.parse(fs.readFileSync(new URL(path,import.meta.url),'utf8'))});
 try{
  const page=await import('data:text/javascript;base64,'+Buffer.from(source+'\nexport {load,analyze};export const record=()=>lastRecord;').toString('base64'));
  for(const r of races){
   current=normalizeRace(r);document.getElementById('raceUrl').value='https://keirin.kdreams.jp/racedetail/'+r.race_id+'/';
   await page.load();document.getElementById('useMarketModel').checked=true;document.getElementById('autoModel').value='joint';document.getElementById('comparisonObjective').value='EXPECTED_RETURN';
   assert.equal(document.getElementById('classSelect').value,resolveRaceClass(r));
   await page.analyze();const record=page.record(),rendered=document.getElementById('analysis').innerHTML;
   assert.ok(record?.selection.signature,r.race_id+': '+document.getElementById('analysis').textContent);
   assert.match(rendered,/主採用（モデル暫定1位）/);assert.doesNotMatch(rendered,/主採用判定不能/);
   const selected=record.comparison.rows.find(c=>c.signature===record.selection.signature);
   assert.ok(selected.hitProbability>0);assert.equal(record.comparison.priceComplete,false);
   if(r.entries.length===9)assert.equal(record.forecast.source.id,'conditional-pair-extended-v1_S9');
   else if(r.entries.every(e=>e.class==='A3'))assert.equal(record.forecast.source.id,'conditional-pair-extended-v1_A3');
   else assert.equal(record.forecast.source.id,'coexistence-v1_A');
  }
 }finally{globalThis.document=oldDocument;globalThis.fetch=oldFetch;}
});
