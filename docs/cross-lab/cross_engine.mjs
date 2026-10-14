/**
 * CROSS LAB - interaction-first selector.
 *
 * Contract:
 * - CROSS means two riders from DIFFERENT lines contest 1st/2nd in BOTH orders.
 * - Third place is either of the OTHER TWO riders chosen from those same lines.
 * - Four tickets, always. No score-based third-place selection.
 * - Race points and B alone NEVER produce a recommended pair.
 * - No invented "probabilities" or value/EV without calibration and odds.
 */
export const ENGINE_VERSION = "cross-conditional-v3";
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

function lineFront(line, riders) { return riders.get(line[0]); }
const has = (r,k,threshold=1) => r && r[k] !== null && r[k] !== undefined && r[k] >= threshold;
const allKnown = (r,keys) => keys.every(k=>r?.[k] !== null && r?.[k] !== undefined);

function interactionEvidence(x,y,race) {
  const px = race.position.get(x.no);
  const py = race.position.get(y.no);
  const ax = race.lines[px.line], by = race.lines[py.line];
  const frontX = lineFront(ax,race.riders), frontY = lineFront(by,race.riders);
  const cases = [];
  const veto = [];
  const lacking = new Set();
  const req = (...checks)=>{
    for (const [r,k] of checks) if (!allKnown(r,[k])) lacking.add(String(r.no)+":"+k);
    return checks.every(([r,k])=>allKnown(r,[k]));
  };

  // "race points" deliberately absent: these are JOINT arrival mechanisms.
  if (px.pos === 1 && py.pos === 1) {
    if (req([x,"b"],[x,"nige"],[y,"makuri"]) &&
        has(x,"b",2) && has(x,"nige") && has(y,"makuri")) {
      cases.push({key:"A_LEAD_B_MAKURI",label:String(x.no)+"先行 × "+String(y.no)+"別線捲り",
        evidence:[String(x.no)+"のB/逃げ",String(y.no)+"の捲り"]});
    }
    if (req([y,"b"],[y,"nige"],[x,"makuri"]) &&
        has(y,"b",2) && has(y,"nige") && has(x,"makuri")) {
      cases.push({key:"B_LEAD_A_MAKURI",label:String(y.no)+"先行 × "+String(x.no)+"別線捲り",
        evidence:[String(y.no)+"のB/逃げ",String(x.no)+"の捲り"]});
    }
    if (allKnown(x,["b","nige"]) && allKnown(y,["b","nige"]) &&
        has(x,"b",3) && has(y,"b",3) && has(x,"nige") && has(y,"nige")) {
      veto.push("両先行の踏み合い。両者残りを当然視できない");
    }
  }
  if (px.pos === 2 && py.pos === 1) {
    if (req([x,"sashi"],[frontX,"b"],[frontX,"nige"],[y,"makuri"]) &&
      has(x,"sashi") && has(frontX,"b",2) && has(frontX,"nige") && has(y,"makuri")) {
      cases.push({key:"A_BANTE_B_MAKURI",label:String(x.no)+"番手差し × "+String(y.no)+"別線捲り",
        evidence:[String(frontX.no)+"の先行力",String(x.no)+"の差し",String(y.no)+"の捲り"]});
    }
  }
  if (py.pos === 2 && px.pos === 1) {
    if (req([y,"sashi"],[frontY,"b"],[frontY,"nige"],[x,"makuri"]) &&
      has(y,"sashi") && has(frontY,"b",2) && has(frontY,"nige") && has(x,"makuri")) {
      cases.push({key:"B_BANTE_A_MAKURI",label:String(y.no)+"番手差し × "+String(x.no)+"別線捲り",
        evidence:[String(frontY.no)+"の先行力",String(y.no)+"の差し",String(x.no)+"の捲り"]});
    }
  }
  // The other line may launch a makuri while its bante closes onto the lead-line.
  // Sashi is required because both 1st/2nd orders must remain plausible.
  if (px.pos === 1 && py.pos === 2) {
    if (req([x,"b"],[x,"nige"],[frontY,"makuri"],[y,"sashi"]) &&
      has(x,"b",2) && has(x,"nige") && has(frontY,"makuri") && has(y,"sashi")) {
      cases.push({key:"A_LEAD_B_MAKURI_BANTE",label:String(x.no)+"先行 × "+String(y.no)+"別線捲り番手",
        evidence:[String(x.no)+"の先行",String(frontY.no)+"の捲り",String(y.no)+"の差し"]});
    }
  }
  if (py.pos === 1 && px.pos === 2) {
    if (req([y,"b"],[y,"nige"],[frontX,"makuri"],[x,"sashi"]) &&
      has(y,"b",2) && has(y,"nige") && has(frontX,"makuri") && has(x,"sashi")) {
      cases.push({key:"B_LEAD_A_MAKURI_BANTE",label:String(y.no)+"先行 × "+String(x.no)+"別線捲り番手",
        evidence:[String(y.no)+"の先行",String(frontX.no)+"の捲り",String(x.no)+"の差し"]});
    }
  }
  if (px.pos > 2 || py.pos > 2) veto.push("3番手以降の1・2着は追加根拠が必要");
  if (px.pos === 2 && py.pos === 2) {
    // A role is not a reason to delete a legal CROSS. Inspect both lines.
    const signal = r => has(r,"sashi") || has(r,"mark");
    if (signal(x) && signal(y) &&
        (has(frontX,"nige") || has(frontX,"makuri")) &&
        (has(frontY,"nige") || has(frontY,"makuri"))) {
      cases.push({key:"BOTH_BANTE_RETENTION",label:x.no+"・"+y.no+"の異線番手残存・切替",
        evidence:[frontX.no+"・"+frontY.no+"の自力実績",x.no+"・"+y.no+"の差し／追走実績"]});
    }
    veto.push("番手同士は両先頭の消耗・切替・コースを比較。役割だけで優遇しない");
  }
  if (!cases.length) veto.push("異線2人の同時残りを裏付ける決まり手の組合せが未成立");
  return {cases,veto,lacking:[...lacking]};
}

function outsideRisks(race, chosen) {
  const missing = race.entries.filter(e=>!chosen.has(e.no));
  const excludedLines = race.lines.filter(line=>line.every(no=>!chosen.has(no)));
  const issues = [];
  if (excludedLines.some(line=>line.length >= 3)) {
    issues.push("基本4点では3車ライン全滅が必要。3-2-2の2車×2車クロスでは特に厳しい（（）付き補助では緩和）");
  }
  if (excludedLines.some(line=>line.length === 2)) {
    issues.push("別の2車ライン2人とも着外が必要");
  }
  // Rivals left outside the four tickets are examined individually, not just counted.
  for (const r of missing) {
    if (has(r,"b",3) && has(r,"nige"))
      issues.push("選外"+r.no+"番にB/逃げ実績。ラインを消せる根拠が別途必要");
    if (has(r,"makuri"))
      issues.push("選外"+r.no+"番に捲り実績。別線への侵入リスク");
  }
  if (missing.length >= 3) issues.push("選外"+String(missing.length)+"人が全員着外になる必要");
  return issues;
}

export function analyzeCross(input, {limit=24}={}) {
  const race = normalizeRace(input);
  const found = [];
  let evaluated = 0;
  for(let i=0;i<race.lines.length;i++)for(let j=i+1;j<race.lines.length;j++){
    const a=race.lines[i], b=race.lines[j];
    if(a.length<2 || b.length<2)continue;
    // Core: each line's front and bante. Actual third rider stays auxiliary.
    for(const firstA of a.slice(0,2))for(const firstB of b.slice(0,2))
    for(const thirdA of a.slice(0,2).filter(no=>no!==firstA))
    for(const thirdB of b.slice(0,2).filter(no=>no!==firstB)){
      evaluated++;
      const proof = interactionEvidence(race.riders.get(firstA),race.riders.get(firstB),race);
      // No hard gate: a legal type with weak evidence must remain inspectable.
      const risks=outsideRisks(race,new Set([firstA,firstB,thirdA,thirdB]));
      const tickets=crossTickets(firstA,firstB,thirdA,thirdB);
      const serious=risks.filter(s=>s.includes("3車ライン全滅"));
      found.push({lines:[i,j],firstPair:[firstA,firstB],
        remaining:[thirdA,thirdB],tickets,cases:proof.cases,
        warnings:[...proof.veto,...risks],needsVerification:serious.length>0,
        // Sorting describes quality of identified MECHANISMS, not rider merit.
        mechanismCount:proof.cases.length,exclusionCount:risks.length,
        evidenceStatus:proof.cases.length ? "SUPPORTED_HYPOTHESIS" : "UNRESOLVED",
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
    status:supported.length === 0 ? "NO_EVIDENCE" : "COMPARISON_REQUIRED",
    recommended:null,
    candidates:top,evaluated,qualified:unique.length,supported:supported.length,ties:tied.length,
    incompleteCars:incomplete,
    notes:[
      "強い順の並べ替え・得点加点・基本4点の3着再選定は行いません（補助は別判定）",
      "成立メカニズムの条件照合であり、的中確率や期待回収率ではありません",
      incomplete.length ? "決まり手欠損："+incomplete.join("・")+"番。欠損時は確定推奨しません" : "",
      "列挙順・根拠件数・警告件数を採用順位に変換しません。主採用には比較記録が必要"
    ].filter(Boolean)
  };
}

