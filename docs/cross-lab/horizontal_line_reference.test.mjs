import test from "node:test";
import assert from "node:assert/strict";
import {buildHorizontalLineReference as build} from "./horizontal_line_reference.mjs";

const input = {
  lines:[[5,6,7],[1,2],[3,4]],
  selectedLineIndexes:[1,2],
  odds:{"1-2-3":12.5,"1-2-4":16.6,"2-1-3":33,"3-4-5":27.1,"4-3-2":87.7},
};
test("shows both horizontal pairs, not the crossover pairs", () => {
  const result = build(input);
  assert.equal(result.available, true);
  assert.deepEqual(result.lines.map(x => x.display), ["1－2", "3－4"]);
  assert.deepEqual(result.paths.map(x => x.pair), ["1－2","2－1","3－4","4－3"]);
  assert.equal(result.paths[0].thirds.length, 5);
  assert.equal(result.paths[0].minOdds, 12.5);
  assert.equal(result.paths[0].maxOdds, 16.6);
  assert.equal(result.paths[0].marketShare, null);
});
test("does not change cross ticket array or force 3-car third selection", () => {
  const tickets = [[1,3,2],[1,3,4],[3,1,2],[3,1,4]];
  const before = JSON.stringify(tickets);
  build({...input, crossTickets:tickets});
  assert.equal(JSON.stringify(tickets), before);
});
test("reversed order is optional", () => {
  assert.deepEqual(build({...input,includeReverse:false}).paths.map(x => x.pair),["1－2","3－4"]);
});
test("odds are not invented", () => {
  const result = build({...input,odds:{}});
  assert.equal(result.paths[0].minOdds,null);
  assert.equal(result.paths[0].marketShare,null);
});
test("complete 7-car trifecta market produces normalized reference weight only", () => {
  const odds = {};
  for(let a=1;a<=7;a++) for(let b=1;b<=7;b++) for(let c=1;c<=7;c++)
    if (a!==b && b!==c && a!==c) odds[a+"-"+b+"-"+c]=20;
  const result = build({...input,odds});
  assert.equal(result.completeMarket,true);
  assert.equal(result.availableMarketOddsCount,210);
  assert.ok(Math.abs(result.paths[0].marketShare-1/42)<1e-12);
});
test("rejects invalid rider assignments", () => {
  assert.equal(build({lines:[[1,2],[2,3]],selectedLineIndexes:[0,1]}).available,false);
  assert.equal(build({lines:[[1],[2,3]],selectedLineIndexes:[0,1]}).available,false);
});
