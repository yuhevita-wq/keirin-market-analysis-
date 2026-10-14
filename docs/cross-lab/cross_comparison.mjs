/** Exact ticket aggregation and explicit decision records. No fitted coefficients. */
import {normalizeRace} from './cross_engine.mjs';
import {jointDiagnostics,pairCompetition} from './joint_model.mjs';

export function allOutcomes(input) {
  const race=normalizeRace(input), cars=race.entries.map(e=>e.no), out=[];
  for(const a of cars)for(const b of cars)for(const c of cars)
    if(a!==b && a!==c && b!==c)out.push([a,b,c]);
  return out;
}

export function classifyOutcome(input, ticket) {
  const r=normalizeRace(input), t=ticket.map(Number);
  if(t.length!==3 || new Set(t).size!==3 || t.some(n=>!r.position.has(n)))
    throw Error('不正な確定着順');
  const p=t.map(n=>r.position.get(n));
  if(p[0].line===p[1].line)return {kind:'SAME_LINE_12',coreCross:false};
  if(new Set(p.map(x=>x.line)).size===3)return {kind:'THREE_LINES',coreCross:false};
  if(p.some(x=>x.pos>2 || r.lines[x.line].length<2))return {kind:'OUTSIDE_CORE_ROLE',coreCross:false};
  return {kind:'CORE_CROSS',coreCross:true};
}

// Four equal stakes: E(return / investment)=sum(p_i * odds_i)/4.
// min paying odds / 4 is a realized break-even property, not a forecast.
export function ticketMarket(candidate, odds={}) {
  const keys=candidate.tickets.map(t=>t.join('-'));
  const prices=keys.map(k=>odds[k]);
  const valid=prices.every(x=>typeof x==='number' && Number.isFinite(x) && x>=1);
  return {complete:valid,priced:prices.filter(x=>typeof x==='number' && Number.isFinite(x) && x>=1).length,
    tickets:keys.map((key,i)=>({key,odds:prices[i]??null})),
    minOdds:valid?Math.min(...prices):null,maxOdds:valid?Math.max(...prices):null,
    minRealizedReturnRatio:valid?Math.min(...prices)/keys.length:null,
    inverseOddsSum:valid?prices.reduce((s,o)=>s+1/o,0):null};
}

export function compareJointForecast(input, candidates, forecast, odds={}, {objective='EXPECTED_RETURN'}={}) {
  if(!forecast)return null;
  if(!['EXPECTED_RETURN','HIT_PROBABILITY'].includes(objective))throw Error('比較目的が不正');
  const race=normalizeRace(input), expected=allOutcomes(input).map(t=>t.join('-'));
  const p=forecast.probabilities, source=forecast.source;
  if(!p || !source?.id || !source?.trainThrough || !forecast.raceId ||
      !race.race_id || String(forecast.raceId)!==String(race.race_id))
    throw Error('予測のレースID・モデルID・学習終了日が必要');
  const match=String(race.race_id).match(/^\d{2}(\d{4})(\d{2})(\d{2})/);
  const date=match?match.slice(1).join('-'):null;
  if(!/^\d{4}-\d{2}-\d{2}$/.test(source.trainThrough) || !date || source.trainThrough>=date)
    throw Error('当日以降を学習した予測は採用できません');
  if(Object.keys(p).length!==expected.length || expected.some(k=>typeof p[k]!=='number' ||
      !Number.isFinite(p[k]) || p[k]<0 || p[k]>1))throw Error('全着順の同時確率が必要');
  const total=expected.reduce((s,k)=>s+p[k],0);
  if(Math.abs(total-1)>1e-8)throw Error('全着順の確率合計は1が必要');
  const mass={SAME_LINE_12:0,THREE_LINES:0,OUTSIDE_CORE_ROLE:0,CORE_CROSS:0};
  for(const key of expected)mass[classifyOutcome(input,key.split('-')).kind]+=p[key];
  const rows=candidates.map(c=>{
    const market=ticketMarket(c,odds), ks=c.tickets.map(t=>t.join('-'));
    return {signature:c.signature,hitProbability:ks.reduce((s,k)=>s+p[k],0),
      directions:jointDiagnostics(c,forecast),competition:pairCompetition(c,forecast,input),
      returnRatio:market.complete?ks.reduce((s,k)=>s+p[k]*odds[k],0)/ks.length:null};
  });
  const metric=objective==='EXPECTED_RETURN'?'returnRatio':'hitProbability';
  const ranked=rows.filter(r=>r[metric]!==null).sort((a,b)=>b[metric]-a[metric]);
  // Partial candidate prices can reverse the choice: never optimize only priced subset.
  const complete=ranked.length===candidates.length && candidates.length>0;
  const tie=complete && ranked.length>1 && Math.abs(ranked[0][metric]-ranked[1][metric])<=1e-12;
  return {source,mass,rows,complete,objective,priceComplete:rows.every(r=>r.returnRatio!==null),
    // Argmax answers "which candidate ranks first", never "should adopt".
    rankedSignature:complete&&!tie?ranked[0].signature:null,
    selectedSignature:null,
    selectionStatus:'COMPARISON_ONLY',
    reason:!complete?'全候補の基本4点オッズが不足':tie?'モデル基準で同値':objective==='EXPECTED_RETURN'?'入力モデル基準の4点等額期待回収を比較':'入力モデル基準の基本4点の合計的中確率を比較（利益順位ではない）',
    note:'入力モデルの較正・収益性をこの計算器は証明しません。市場重みを予測確率で代用しません'};
}

export function resolveDecision(result, decision=null, comparison=null) {
  if(!decision)return {main:null,label:'主採用未決定',provisional:false,
    reason:'候補内のモデル順位だけでは主採用にしません。表裏それぞれの同時残存・2着競合と筋・別クロス・選外車を比較してください'};
  const main=result.candidates.find(c=>c.signature===decision.signature);
  if(!main)throw Error('採用クロスが候補集合に存在しません');
  for(const k of ['interaction','alternativeCross','sameLine','outside'])
    if(typeof decision[k]!=='string'||!decision[k].trim())
      throw Error('採用理由・別クロス比較・筋比較・選外車の反証をすべて記入してください');
  const directions=decision.directions;
  if(!Array.isArray(directions)||directions.length!==2||
      !main.firstPair.every((first,i)=>directions.some(d=>d.first===first&&
        d.second===main.firstPair[1-i]&&['coexistence','secondCompetition'].every(k=>
          typeof d[k]==='string'&&d[k].trim()))))
    throw Error('表裏それぞれに、勝つ走りで相手が2着に残る理由と、同線番手を含む2着競合の比較が必要です');
  return {main,label:'主採用（分析者の比較判断）',provisional:true,
    reason:[decision.interaction,decision.alternativeCross,decision.sameLine,decision.outside].join('。'),
    record:{...decision}};
}

export function gradeFrozenDecision(input, decision, actual, payoutPer100, result) {
  const selected=result.candidates.find(c=>c.signature===decision.signature);
  if(!selected)throw Error('固定答案が候補集合に存在しません');
  const key=actual.join('-'), classification=classifyOutcome(input,actual);
  const candidateContains=result.candidates.some(c=>c.tickets.some(t=>t.join('-')===key));
  const hit=selected.tickets.some(t=>t.join('-')===key);
  return {classification,hit,candidateContains,
    failure:hit?'HIT':classification.coreCross?(candidateContains?'SELECTION_MISS':'GENERATION_MISS'):classification.kind,
    stake:400,payout:hit?payoutPer100:0};
}
