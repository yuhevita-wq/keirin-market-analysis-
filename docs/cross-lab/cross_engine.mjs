/**
 * CROSS LAB - interaction-first selector.
 *
 * Contract:
 * - CROSS means two riders from DIFFERENT lines contest 1st/2nd in BOTH orders.
 * - Third place is either of the OTHER TWO riders chosen from those same lines.
 * - Four tickets, always. All legal third-place pairs are compared.
 * - Race points and B alone NEVER produce a recommended pair.
 * - No invented "probabilities" or value/EV without calibration and odds.
 */
export const ENGINE_VERSION = "cross-main-selection-v5.1";
const own = (obj, key) => Object.prototype.hasOwnProperty.call(obj || {}, key);
const numeric = value => value === null || value === undefined || value === "" ? null :
  (Number.isFinite(Number(value)) && Number(value) >= 0 ? Number(value) : null);
const carNum = value => Number(value);
const distinct = values => new Set(values).size === values.length;

export function readKimarite(entry, kind) {
  // An explicit edit, including null (unknown), overrides stale source aliases.
  if (own(entry,kind)) return numeric(entry[kind]);
  const aliases = {
    nige:["nige","nige_count","escape_count","kimarite_nige"],
    makuri:["makuri","makuri_count","kimarite_makuri"],
    sashi:["sashi","sashi_count","kimarite_sashi"],
    mark:["mark","mark_count","kimarite_mark"]
  };
  const candidates = [entry,entry?.kimarite,entry?.wins_by,entry?.winning_methods];
  for (const group of candidates) {
    if (!group || typeof group !== "object") continue;
    for (const name of aliases[kind]) {
      if (own(group,name) && numeric(group[name]) !== null) return numeric(group[name]);
    }
  }
  return null;
}

export function parseLines(text, allowed = []) {
  // Only explicit separated digit car numbers; never deduce a formation from score.
  const normalized = String(text || "").replace(/[－ー−–—]/g,"-")
    .replace(/[／｜|；;,]+/g,"/").trim();
  const pieces = normalized.split(/\s*\/\s*|\s{2,}/).filter(Boolean);
  const lines = pieces.map(chunk => chunk.trim().split(/\s*-\s*|\s+/).filter(Boolean)
    .map(x => /^\d+$/.test(x) ? Number(x) : NaN));
  const flat = lines.flat();
  if (lines.length < 2 || lines.some(x=>x.length === 0) ||
      flat.some(x=>!Number.isInteger(x) || x < 1 || x > 9) || !distinct(flat)) {
    throw new Error("ライン並びを確認できません。例：1-2-3 / 4-5 / 6-7");
  }
  if (allowed.length && (!distinct(allowed) || allowed.length !== flat.length ||
      allowed.some(x=>!flat.includes(Number(x))))) {
    throw new Error("ライン並びと出走選手の車番が一致しません");
  }
  return lines;
}

function reconstructLines(race, entries) {
  const ids = entries.map(e=>e.line_id);
  if (ids.every(x=>x !== "" && x !== null && x !== undefined)) {
    const m = new Map();
    entries.forEach(e=>{
      const id = String(e.line_id);
      if (!m.has(id)) m.set(id,[]);
      m.get(id).push(e);
    });
    const groups = [...m.values()].map(arr=>arr.sort((a,b)=>
      (numeric(a.line_position) ?? 99)-(numeric(b.line_position) ?? 99))
      .map(e=>e.no));
    if (groups.length > 1 && groups.flat().length === entries.length && distinct(groups.flat())) return groups;
  }
  return parseLines(race.predicted_line_formation || race.line_formation || race.lines_text,
    entries.map(e=>e.no));
}

export function normalizeRace(race) {
  if (!race || typeof race !== "object") throw new Error("レース情報がありません");
  const raw = race.entries || race.riders || [];
  const entries = raw.map(e=>({
    ...e,
    no: carNum(e.no ?? e.car_no ?? e.car ?? e.number),
    name: String(e.name ?? e.player_name ?? ""),
    score: numeric(e.score ?? e.race_score),
    b: numeric(e.b ?? e.b_count),
    nige: readKimarite(e,"nige"),
    makuri: readKimarite(e,"makuri"),
    sashi: readKimarite(e,"sashi"),
    mark: readKimarite(e,"mark")
  }));
  if (entries.length < 5 || entries.length > 9 ||
      entries.some(e=>!Number.isInteger(e.no) || e.no < 1 || e.no > 9) ||
      !distinct(entries.map(e=>e.no))) throw new Error("5〜9車の重複しない選手データが必要です");
  const lines = Array.isArray(race.lines) && race.lines.every(Array.isArray)
    ? race.lines.map(line=>line.map(Number)) : reconstructLines(race,entries);
  const members = lines.flat();
  if (members.length !== entries.length || !distinct(members) ||
      !entries.every(e=>members.includes(e.no))) throw new Error("ラインと選手の対応が不正です");
  const riders = new Map(entries.map(e=>[e.no,e]));
  const position = new Map();
  lines.forEach((line,li)=>line.forEach((no,i)=>position.set(no,{line:li,pos:i+1})));
  return { entries, lines, riders, position,
    race_id:race.race_id || "", track:race.track || "", race_no:race.race_no || "",
    race_type:race.race_type || "", meeting_grade:race.meeting_grade || "" };
}

export function crossTickets(firstA, firstB, thirdA, thirdB) {
  const n = [firstA,firstB,thirdA,thirdB].map(Number);
  if (n.some(x=>!Number.isInteger(x) || x < 1 || x > 9) || !distinct(n))
    throw new Error("クロスは重複しない4車が必要です");
  return [[n[0],n[1],n[2]], [n[0],n[1],n[3]],
          [n[1],n[0],n[2]], [n[1],n[0],n[3]]];
}

/** Structure generates questions, never a numeric proof or a veto. */
function interactionEvidence(x,y,race) {
  const directions=[x,y].map((winner,i)=>{
    const partner=[x,y][1-i];
    const rivals=race.entries.filter(e=>e.no!==winner.no&&e.no!==partner.no);
    return {first:winner.no,second:partner.no,
      questions:[
        `${winner.no}が勝つ動きの中で、${partner.no}はどの位置・コースから2着を確保するか`,
        `${partner.no}が残るための条件は、${winner.no}が勝つための条件と両立するか`,
        `2着を争う${rivals.map(e=>e.no).join('・')}の動きに対し、何が有利／不利になるか`,
        'その推論に使った実際の走り・相手関係・並び・コメントは何か。確認できた事実と推測を分ける',
        '同じ材料で筋決着や別クロスになる説明と比べ、こちらを選ぶ差は何か'
      ], rivals:rivals.map(e=>({no:e.no,sameLineAsWinner:
        race.position.get(e.no).line===race.position.get(winner.no).line}))};
  });
  return {cases:[],veto:[],directions};
}

function outsideRisks(race, chosen) {
  const missing=race.entries.filter(e=>!chosen.has(e.no));
  const excludedLines=race.lines.filter(line=>line.every(no=>!chosen.has(no)));
  const issues=[];
  if(excludedLines.some(line=>line.length>=3))
    issues.push('基本4点の成立条件は3車ライン全滅。これは着順の条件であり、このレースで起こりにくいという証拠ではない（補助は別）');
  for(const r of missing)
    issues.push(`選外${r.no}番：どの動きなら入着し、どの相手関係なら届かないか。得点や決まり手の多寡だけで消さない`);
  return issues;
}

export function analyzeCross(input, {limit=1000}={}) {
  const race = normalizeRace(input);
  const found = [];
  let evaluated = 0;
  for(let i=0;i<race.lines.length;i++)for(let j=i+1;j<race.lines.length;j++){
    const a=race.lines[i], b=race.lines[j];
    if(a.length<2 || b.length<2)continue;
    // The CROSS core is exactly two riders from each line: front and bante.
    // A three-rider line's third stays an explicit parenthesized auxiliary.
    const coreA=a.slice(0,2), coreB=b.slice(0,2);
    for(const firstA of coreA)for(const firstB of coreB){
      const thirdA=coreA.find(no=>no!==firstA);
      const thirdB=coreB.find(no=>no!==firstB);
      evaluated++;
      const proof = interactionEvidence(race.riders.get(firstA),race.riders.get(firstB),race);
      // No hard gate: a legal type with weak evidence must remain inspectable.
      const risks=outsideRisks(race,new Set([firstA,firstB,thirdA,thirdB]));
      const tickets=crossTickets(firstA,firstB,thirdA,thirdB);
      const serious=risks.filter(s=>s.includes("3車ライン全滅"));
      found.push({lines:[i,j],firstPair:[firstA,firstB],
        remaining:[thirdA,thirdB],tickets,cases:proof.cases,reasoning:proof.directions,
        warnings:[...proof.veto,...risks],needsVerification:serious.length>0,
        // Compatibility fields: no mechanism counts are used as evidence.
        mechanismCount:proof.cases.length,exclusionCount:risks.length,
        evidenceStatus:"UNASSESSED",
        observations:[firstA,firstB,thirdA,thirdB].map(no=>{
          const r=race.riders.get(no);return {no,position:race.position.get(no).pos,
            b:r.b,nige:r.nige,makuri:r.makuri,sashi:r.sashi,mark:r.mark};
        }),
        signature:tickets.map(t=>t.join("-")).join("|")});
    }
  }
  const unique=[...new Map(found.map(x=>[x.signature,x])).values()];
  // Enumeration order is not a strength ranking or a default main choice.
  const top=unique.slice(0,Math.max(1,limit));
  const supported=unique.filter(c=>c.cases.length>0);
  const tied=unique;
  const incomplete=race.entries.filter(e=>["nige","makuri","sashi","mark"]
    .some(k=>e[k]===null)).map(e=>e.no);
  return {
    version:ENGINE_VERSION,race:{track:race.track,race_no:race.race_no,
      race_type:race.race_type,meeting_grade:race.meeting_grade,lines:race.lines},
    status:"COMPARISON_REQUIRED",
    recommended:null,
    candidates:top,evaluated,qualified:unique.length,supported:supported.length,ties:tied.length,
    incompleteCars:incomplete,
    notes:[
      "各型は4点。各ラインの先頭・番手から2車ずつ。3×2も2×2も各4型、3番手は（）補助です",
      "実績の回数は観測値。今回の成立・不成立の証拠に自動変換しません",
      incomplete.length ? "決まり手欠損："+incomplete.join("・")+"番。数値欠損はクロス不成立を意味しません" : "",
      "列挙順・根拠件数・警告件数は順位ではありません。同時入着モデルの4点確率と、全価格があるときの期待回収で主採用を比較します"
    ].filter(Boolean)
  };
}

