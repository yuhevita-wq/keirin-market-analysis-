import test from "node:test";
import assert from "node:assert/strict";
import {selectMainCross,buildAuxiliaryThirds} from "./cross_presentation.mjs";

const entries = [
  {car_no:1, b_count:4, nige:2, makuri:0, sashi:0, mark:0},
  {car_no:2, b_count:0, nige:0, makuri:0, sashi:1, mark:2},
  {car_no:3, b_count:0, nige:0, makuri:0, sashi:0, mark:1},
  {car_no:4, b_count:5, nige:1, makuri:1, sashi:0, mark:0},
  {car_no:5, b_count:0, nige:0, makuri:0, sashi:1, mark:1},
  {car_no:6, b_count:2, nige:0, makuri:2, sashi:0, mark:0},
  {car_no:7, b_count:0, nige:0, makuri:0, sashi:1, mark:1},
];
const race = {entries,lines:[[1,2,3],[4,5],[6,7]]};
const example = (lines,firstPair,remaining) => ({
  lines, firstPair, remaining,
  tickets:[
    [firstPair[0],firstPair[1],remaining[0]],
    [firstPair[0],firstPair[1],remaining[1]],
    [firstPair[1],firstPair[0],remaining[0]],
    [firstPair[1],firstPair[0],remaining[1]],
  ]
});

test("main requires a comparison record, never the first candidate or warning count",()=>{
  const candidates=[
    {...example([1,2],[4,6],[5,7]),signature:"a",mechanismCount:99,exclusionCount:0},
    {...example([0,1],[2,4],[1,5]),signature:"b",mechanismCount:0,exclusionCount:99}
  ];
  const result={candidates,ties:2,incompleteCars:[]};
  assert.equal(selectMainCross(result).main,null);
  const record={signature:"b",interaction:"interaction",alternativeCross:"other cross",sameLine:"same line",outside:"outside",directions:[{first:2,second:4,coexistence:"2 wins, 4 survives",secondCompetition:"compare 1 and 5"},{first:4,second:2,coexistence:"4 wins, 2 survives",secondCompetition:"compare 1 and 5"}]};
  assert.equal(selectMainCross(result,record).main,candidates[1]);
  assert.throws(()=>selectMainCross(result,{...record,outside:""}));
});

test("bante in the 3-line CROSS's first pair adds actual third-position rider",()=>{
  const c=example([0,2],[2,6],[1,7]);
  const out=buildAuxiliaryThirds(c,race);
  assert.deepEqual(out.thirds,[3]);
  assert.equal(out.thirdDisplay,"1・7（3）");
  assert.deepEqual(out.additionalTickets,[[2,6,3],[6,2,3]]);
  assert.deepEqual(out.coreTickets,c.tickets);
});

test("if third-position rider was already core third, show parentheses but no duplicate",()=>{
  const out=buildAuxiliaryThirds(example([0,2],[2,6],[3,7]),race);
  assert.deepEqual(out.thirds,[3]);
  assert.equal(out.additionalTickets.length,0);
  assert.deepEqual(out.alreadyInCore,[3]);
});

test("2x2 cross draws one third from outside 3-line by third-specific evidence, not automatic bante",()=>{
  const out=buildAuxiliaryThirds(example([1,2],[4,6],[5,7]),race);
  assert.equal(out.thirds.length,1);
  assert.equal(out.thirds[0],2); // two independent third-place clues (sashi + mark)
  assert.deepEqual(out.additionalTickets,[[4,6,2],[6,4,2]]);
});

test("2x2 cross can pick the 3-line leader if its third-entry evidence is stronger",()=>{
  const updated={...race,entries:race.entries.map(x=>({...x}))};
  updated.entries[0].makuri=2; // nige/B and makuri: two qualifying mechanisms
  updated.entries[1].sashi=0;
  updated.entries[1].mark=0;
  const out=buildAuxiliaryThirds(example([1,2],[4,6],[5,7]),updated);
  assert.deepEqual(out.thirds,[1]);
});

test("no invented 3-line auxiliary when third-place data cannot support a rider",()=>{
  const empty={...race,entries:race.entries.map(x=>({...x,b_count:null,nige:null,makuri:null,sashi:null,mark:null}))};
  const out=buildAuxiliaryThirds(example([1,2],[4,6],[5,7]),empty);
  assert.deepEqual(out.thirds,[]);
  assert.deepEqual(out.additionalTickets,[]);
  assert.match(out.explanations.join(" "),/根拠が不足/);
});

test("cross not involving a 3-line bante and not 2x2 has no arbitrary auxiliary",()=>{
  const out=buildAuxiliaryThirds(example([0,1],[1,4],[2,5]),race);
  assert.deepEqual(out.thirds,[]);
});
