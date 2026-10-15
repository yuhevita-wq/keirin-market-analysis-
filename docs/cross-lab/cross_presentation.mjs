/**
 * CROSS LAB presentation policy.
 * The base 4 tickets stay unchanged. A parenthesized third is an auxiliary
 * scenario, never a silent replacement of the original CROSS.
 */
import { normalizeRace } from "./cross_engine.mjs";

const uniqueTickets = tickets => [...new Map(tickets.map(t => [t.join("-"),t])).values()];

export {resolveDecision as selectMainCross} from './cross_comparison.mjs';

export function buildAuxiliaryThirds(candidate, input, assessment=null) {
  const race = normalizeRace(input);
  const [ai,bi] = candidate.lines;
  const chosenLines = [race.lines[ai],race.lines[bi]];
  const extras = [];
  const explanations = [];
  const known = new Set(candidate.remaining);
  const crossHeads = candidate.firstPair;
  const push = (car,reason) => {
    if (!extras.includes(car)) extras.push(car);
    explanations.push(reason);
  };

  // If the bante of a three-rider line is in 1st/2nd, its actual third
  // rider is an auxiliary third; never automatically use the line's front.
  for (const first of crossHeads) {
    const p = race.position.get(first);
    const line = race.lines[p.line];
    if (p.pos === 2 && line.length >= 3) {
      push(line[2],first+"番が3車ライン番手で1・2着候補 → 同ライン3番手"+line[2]+"番を3着補助");
    }
  }

  // Two two-rider lines cross: choose ONE rider from the excluded three line.
  // An explicit contextual judgment replaces winning-method-count ranking.
  if (!extras.length && chosenLines.every(line => line.length === 2)) {
    const outside = race.lines.map((line,i) => ({line,i}))
      .filter(({line,i}) => i !== ai && i !== bi && line.length === 3);
    if (outside.length === 1) {
      const {line} = outside[0];
      const no=Number(assessment?.no);
      if(assessment && (!line.includes(no)||!assessment.reason?.trim()))
        throw Error('補助3着は選外3車ラインの1人と、今回の相手関係で残る理由を指定してください');
      if(assessment)push(no,'選外3車ラインから'+no+'番：'+assessment.reason+'（分析者判断）');
      else explanations.push('選外3車ラインの'+line.join('・')+'から1人を比較選定。未選定であり、不成立でも根拠なしでもない。決まり手の項目数では選ばない');
    }
  }

  const existing = new Set(candidate.tickets.map(t=>t.join("-")));
  const additions = uniqueTickets(extras.flatMap(third => [
    [crossHeads[0],crossHeads[1],third],
    [crossHeads[1],crossHeads[0],third]
  ])).filter(t=>!existing.has(t.join("-")));
  return {
    thirds:extras,
    thirdDisplay:candidate.remaining.join("・") +
      extras.map(third=>"（"+third+"）").join(""),
    additionalTickets:additions,
    alreadyInCore:extras.filter(third=>known.has(third)),
    explanations,
    coreTickets:candidate.tickets
  };
}


