import test from "node:test";
import assert from "node:assert/strict";
import {analyzeCross, crossTickets, parseLines, normalizeRace} from "./cross_engine.mjs";

function example() {
  const entries = [
    {car_no:1,score:80,b_count:5,nige:2,makuri:0,sashi:0,mark:0},
    {car_no:2,score:80,b_count:0,nige:0,makuri:0,sashi:1,mark:1},
    {car_no:3,score:80,b_count:3,nige:0,makuri:1,sashi:0,mark:0},
    {car_no:4,score:80,b_count:0,nige:0,makuri:0,sashi:1,mark:1},
    {car_no:5,score:80,b_count:0,nige:0,makuri:0,sashi:0,mark:0},
    {car_no:6,score:80,b_count:0,nige:0,makuri:0,sashi:0,mark:0},
    {car_no:7,score:80,b_count:0,nige:0,makuri:0,sashi:0,mark:0}
  ];
  return {entries,lines:[[1,2],[3,4],[5,6,7]],race_type:"A級予選"};
}
const tickets = t => t.map(x=>x.join("-")).join("|");

test("four cross tickets are 13-13-24, not arbitrary third-place ranking",()=>{
  assert.equal(tickets(crossTickets(1,3,2,4)),"1-3-2|1-3-4|3-1-2|3-1-4");
  assert.equal(crossTickets(1,3,2,4).length,4);
  assert.throws(()=>crossTickets(1,3,1,4));
});

test("never select because of race points or B alone",()=>{
  const race=example();
  for(const e of race.entries){e.nige=null;e.makuri=null;e.sashi=null;e.mark=null;}
  race.entries[4].score=5000;
  race.entries[4].b_count=300;
  assert.equal(analyzeCross(race).status,"NO_EVIDENCE");
  assert.equal(analyzeCross(race).recommended,null);
});

test("changing only scores never affects any pair or ticket",()=>{
  const race=example();
  const baseline=analyzeCross(race);
  race.entries.forEach((e,i)=>e.score=1000000-(i*10000));
  const changed=analyzeCross(race);
  assert.deepEqual(changed.candidates,baseline.candidates);
});

test("cross's 1st and 2nd are strictly from different lines",()=>{
  const race=example(),normal=normalizeRace(race);
  const result=analyzeCross(race);
  assert.ok(result.qualified>0);
  for(const c of result.candidates){
    assert.notEqual(normal.position.get(c.firstPair[0]).line,
      normal.position.get(c.firstPair[1]).line);
    assert.equal(c.tickets.length,4);
    assert.deepEqual(c.tickets.map(t=>t[2]),[
      c.remaining[0],c.remaining[1],c.remaining[0],c.remaining[1]]);
  }
});

test("322: crossing two two-rider lines explicitly warns of three-line extinction",()=>{
  const race=example(),result=analyzeCross(race);
  const twoByTwo=result.candidates.find(x=>x.firstPair.join("-")==="1-3");
  assert.ok(twoByTwo);
  assert.ok(twoByTwo.warnings.some(s=>s.includes("3車ライン全滅")));
  assert.equal(twoByTwo.needsVerification,true);
});

test("missing rider winning methods cannot create a confident recommendation",()=>{
  const race=example();
  race.entries[6].mark=null;
  const result=analyzeCross(race);
  assert.ok(result.incompleteCars.includes(7));
  assert.equal(result.recommended,null);
});

test("preserve specified line order and reject invented line membership",()=>{
  assert.deepEqual(parseLines("1-2-3 / 4-5 / 6-7"),[[1,2,3],[4,5],[6,7]]);
  assert.throws(()=>parseLines("1-2 / 2-3"));
  assert.throws(()=>parseLines("1-2 / 3-4",[1,2,3,5]));
});

test("front + opposing makuri requires actual winning-method evidence",()=>{
  const race=example();
  let result=analyzeCross(race);
  assert.ok(result.candidates.some(x=>x.firstPair.join("-")==="1-3"));
  race.entries[2].makuri=0;
  result=analyzeCross(race);
  assert.ok(!result.candidates.some(x=>x.firstPair.join("-")==="1-3"));
});
