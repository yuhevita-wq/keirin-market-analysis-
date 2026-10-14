/**
 * CROSS LAB presentation policy.
 * The base 4 tickets stay unchanged. A parenthesized third is an auxiliary
 * scenario, never a silent replacement of the original CROSS.
 */
import { normalizeRace } from "./cross_engine.mjs";

const positive = (r, key) => Number.isFinite(r?.[key]) && r[key] > 0;
const uniqueTickets = tickets => [...new Map(tickets.map(t => [t.join("-"),t])).values()];

export {resolveDecision as selectMainCross} from './cross_comparison.mjs';

function thirdEntryEvidence(race, line, position) {
  const rider = race.riders.get(line[position]);
  const front = race.riders.get(line[0]);
  const signals = [];
  if (position === 0) {
    if (positive(rider,"nige") && positive(rider,"b")) signals.push("逃げ・B実績から先行粘り");
    if (positive(rider,"makuri")) signals.push("捲り実績から別線間への流入");
  } else if (position === 1) {
    if (positive(rider,"sashi") && (positive(front,"nige") || positive(front,"makuri") || positive(front,"b")))
      signals.push("前の自力実績と本人の差し");
    if (positive(rider,"mark")) signals.push("追走実績からの3着残り");
  } else {
    if (positive(rider,"mark") && (positive(front,"nige") || positive(front,"makuri") || positive(front,"b")))
      signals.push("3番手追走と前方の自力実績");
    if (positive(rider,"sashi")) signals.push("後位からの差し実績");
  }
  return {no:rider.no, signals, position:position+1};
}

export function buildAuxiliaryThirds(candidate, input) {
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
  // This is based on third-place-specific mechanisms, not race score or B alone.
  if (!extras.length && chosenLines.every(line => line.length === 2)) {
    const outside = race.lines.map((line,i) => ({line,i}))
      .filter(({line,i}) => i !== ai && i !== bi && line.length === 3);
    if (outside.length === 1) {
      const {line} = outside[0];
      const ranked = line.map((_,p) => thirdEntryEvidence(race,line,p))
        .sort((a,b)=>b.signals.length-a.signals.length || a.position-b.position);
      const best = ranked[0];
      if (best.signals.length) {
        push(best.no,"選外3車ラインから"+best.no+"番（"+best.signals.join("・")+"）");
        if (ranked[1]?.signals.length === best.signals.length)
          explanations.push("3着の根拠が同数。選択は暫定であり他車を否定しない");
      } else {
        explanations.push("選外3車ラインの3着根拠が不足。車番を機械的に追加しない");
      }
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

