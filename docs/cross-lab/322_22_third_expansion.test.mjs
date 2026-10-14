import test from "node:test";
import assert from "node:assert/strict";
import { expand322TwoTwoCrossThird as expand } from "./322_22_third_expansion.mjs";

const base = {
  lines: [[5, 6, 7], [1, 2], [3, 4]],
  selectedLineIndexes: [1, 2],
  tickets: [[1, 3, 2], [1, 3, 4], [3, 1, 2], [3, 1, 4]],
  thirdPlaceScores: { 5: 0.09, 6: 0.14, 7: 0.05 },
};
test("322: selected 2x2 becomes four plus two, not another four", () => {
  const result = expand(base);
  assert.equal(result.applied, true);
  assert.equal(result.candidate, 6);
  assert.equal(result.tickets.length, 6);
  assert.deepEqual(result.addedTickets, [[1, 3, 6], [3, 1, 6]]);
  assert.deepEqual(result.tickets.slice(0, 4), base.tickets);
});
test("explicit third-place candidate is allowed only from 3 line", () => {
  assert.equal(expand({ ...base, thirdPlaceCandidate: 7 }).candidate, 7);
  assert.equal(expand({ ...base, thirdPlaceCandidate: 2 }).applied, false);
});
test("does not affect 3x2 selected CROSS", () => {
  const result = expand({ ...base, selectedLineIndexes: [0, 1] });
  assert.equal(result.applied, false);
  assert.deepEqual(result.tickets, base.tickets);
});
test("does not affect other formations", () => {
  assert.equal(expand({ ...base, lines: [[5, 6], [1, 2], [3, 4]] }).applied, false);
});
test("will not choose a 3-line rider without a third-place model score", () => {
  assert.equal(expand({ ...base, thirdPlaceScores: {} }).applied, false);
});
