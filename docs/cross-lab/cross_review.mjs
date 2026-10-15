/** Structural review. Results diagnose a frozen answer, never select a forecast. */
import {analyzeCross, normalizeRace} from './cross_engine.mjs';
import {classifyOutcome} from './cross_comparison.mjs';
import {buildAuxiliaryThirds} from './cross_presentation.mjs';
const key = xs => [...xs].sort((a,b)=>a-b).join(',');
export const crossLabel = c => `${c.firstPair.join('')}-${c.firstPair.join('')}-${c.remaining.join('')}`;
export function candidateMatrix(input) {
  const r=normalizeRace(input), candidates=analyzeCross(input).candidates;
  return [...new Set(candidates.map(c=>key(c.lines)))].map(id=>{
    const indexes=id.split(',').map(Number);
    return {id,lines:indexes.map(i=>r.lines[i]),excluded:r.lines.filter((_,i)=>!indexes.includes(i)),
      candidates:candidates.filter(c=>key(c.lines)===id).map(c=>({signature:c.signature,label:crossLabel(c),
        roles:c.firstPair.map(n=>r.position.get(n).pos),remaining:c.remaining}))};
  });
}
export function resolveLiteral(input, literal) {
  const m=String(literal).replace(/[－ー−–—]/g,'-').match(/^([1-9])([1-9])-([1-9])([1-9])-([1-9])([1-9])$/);
  if(!m || key([+m[1],+m[2]])!==key([+m[3],+m[4]]))return null;
  return analyzeCross(input).candidates.find(c=>key(c.firstPair)===key([+m[1],+m[2]]) && key(c.remaining)===key([+m[5],+m[6]])) || null;
}
export function reviewSubmission(input, submission, actual) {
  const classification=classifyOutcome(input,actual), race=normalizeRace(input);
  const candidates=analyzeCross(input).candidates;
  const selected=resolveLiteral(input,submission.literal);
  const includes=tickets=>tickets.some(t=>t.join('-')===actual.join('-'));
  // Only automatic third-of-bante auxiliaries are used for retrospective coverage.
  // Never choose an excluded-line auxiliary using the result.
  const structural=candidates.filter(c=>includes(c.tickets)||includes(buildAuxiliaryThirds(c,input).additionalTickets));
  const expected=structural.map(c=>({literal:crossLabel(c),coverage:includes(c.tickets)?'CORE':'AUXILIARY'}));
  if(!selected)return {kind:'INVALID_SUBMISSION',classification,expected,coreHit:false,auxiliaryHit:false,
    adoptedHit:false,note:'提示した3着が残り2車と一致しない。元答案を修正して採点しない'};
  const legalAux=buildAuxiliaryThirds(selected,input,submission.auxiliaryAssessment).thirds;
  const submittedAux=submission.auxiliaryThirds ?? [];
  if(submittedAux.some(n=>!legalAux.includes(n)))return {kind:'INVALID_AUXILIARY',classification,expected,
    coreHit:false,auxiliaryHit:false,adoptedHit:false};
  const coreHit=includes(selected.tickets);
  const auxiliaryHit=includes(submittedAux.flatMap(n=>[[...selected.firstPair,n],[...selected.firstPair].reverse().concat(n)]));
  const actualLines=key(actual.slice(0,2).map(n=>race.position.get(n).line));
  const actualHeads=key(actual.slice(0,2));
  let kind;
  if(coreHit)kind='CORE_HIT';
  else if(auxiliaryHit)kind='AUXILIARY_HIT';
  else if(classification.kind==='SAME_LINE_12')kind='SAME_LINE_12';
  else if(!candidates.some(c=>key(c.firstPair)===actualHeads))kind='OUTSIDE_HEAD_ROLES';
  else if(key(selected.lines)!==actualLines)kind='LINE_PAIR_MISS';
  else if(key(selected.firstPair)!==actualHeads)kind='HEAD_ROLE_MISS';
  else kind='THIRD_MISS';
  return {kind,classification,expected,coreHit,auxiliaryHit,
    adoptedHit:submission.status==='ADOPTED' && (coreHit||auxiliaryHit),
    status:submission.status,actual:actual.slice(),literal:submission.literal,
    note:'結果が示す構造上の差。実際の仕掛け・位置取りの原因は映像等で別途確認する'};
}
export const reviewLabels={CORE_HIT:'基本4点内',AUXILIARY_HIT:'提示済み補助内',SAME_LINE_12:'同ライン1・2着',
  LINE_PAIR_MISS:'選ぶ2ラインを誤った',HEAD_ROLE_MISS:'同じ2ラインの先頭・番手を取り違えた',
  THIRD_MISS:'1・2着の組は一致、3着が提示外',OUTSIDE_HEAD_ROLES:'1・2着に基本役割外の選手',
  INVALID_SUBMISSION:'基本クロスの定義違反',INVALID_AUXILIARY:'補助3着の定義・選定記録を確認'};
