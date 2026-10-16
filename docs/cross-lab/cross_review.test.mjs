import test from 'node:test';
import assert from 'node:assert/strict';
import {candidateMatrix,reviewSubmission} from './cross_review.mjs';
const race=lines=>({lines,entries:lines.flat().map(no=>({no}))});
test('322 compares every line pair and every vertical/diagonal cross without scores',()=>{
 const rows=candidateMatrix(race([[1,5],[2,6],[4,7,3]]));
 assert.equal(rows.length,3);assert.ok(rows.map(r=>r.candidates.length).sort((a,b)=>a-b).join()==='4,18,18');
});
test('Tamano 6: line selection miss; alternate held 17 is coverage, not adopted hit',()=>{
 const r=race([[1,5],[2,6],[4,7,3]]);
 assert.equal(reviewSubmission(r,{literal:'12-12-56',status:'ADOPTED'},[1,7,4]).kind,'LINE_PAIR_MISS');
 const alternate=reviewSubmission(r,{literal:'17-17-45',status:'HOLD',auxiliaryThirds:[3]},[1,7,4]);
 assert.equal(alternate.coreHit,true);assert.equal(alternate.adoptedHit,false);
});
test('Tamano 10: 17 and 57 miss roles, 13 covers result with unchanged thirds',()=>{
 const r=race([[1,5,4],[6,2],[7,3]]);
 for(const literal of ['17-17-53','57-57-13'])assert.equal(reviewSubmission(r,{literal},[3,1,7]).kind,'HEAD_ROLE_MISS');
 assert.equal(reviewSubmission(r,{literal:'13-13-57'},[3,1,7]).coreHit,true);
});
test('Tamano 3 separates eligible auxiliary from actually submitted auxiliary',()=>{
 const r=race([[4,1],[2,7,5],[6,3]]);
 const miss=reviewSubmission(r,{literal:'12-12-47'},[6,7,5]);
 assert.equal(miss.kind,'LINE_PAIR_MISS');assert.ok(miss.expected.some(c=>c.coverage==='CORE'&&c.literal.endsWith('25')));
 assert.ok(miss.expected.some(c=>c.coverage==='AUXILIARY'));
 assert.equal(reviewSubmission(r,{literal:'67-67-23'},[6,7,5]).auxiliaryHit,false);
 assert.equal(reviewSubmission(r,{literal:'67-67-23',auxiliaryThirds:[5]},[6,7,5]).auxiliaryHit,true);
});
test('Nara invalid published third is not silently repaired',()=>{
 const r=race([[7,2,4],[1,5],[3,6]]);
 assert.equal(reviewSubmission(r,{literal:'23-23-17'},[7,2,4]).kind,'INVALID_SUBMISSION');
 assert.equal(reviewSubmission(r,{literal:'23-23-67'},[7,2,4]).kind,'SAME_LINE_12');
});
test('Nara 3 does not add third of line when leader wins',()=>{
 const r=race([[4,1],[7,2],[5,3,6]]), out=reviewSubmission(r,{literal:'45-45-13'},[5,4,6]);
 assert.equal(out.kind,'THIRD_MISS');assert.ok(out.expected.some(c=>c.coverage==='CORE'));
});
test('malformed actual and result-picked illegal auxiliary are rejected',()=>{
 const r=race([[1,5],[2,6],[4,7,3]]);
 assert.throws(()=>reviewSubmission(r,{literal:'12-12-56'},[1,1,4]));
 assert.equal(reviewSubmission(r,{literal:'12-12-56',auxiliaryThirds:[7]},[1,7,4]).kind,'INVALID_AUXILIARY');
});
